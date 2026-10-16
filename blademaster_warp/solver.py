"""Single-body TLMPM and single rigid blade on Warp/CUDA, all mechanics on GPU."""
from dataclasses import dataclass, asdict
import time
import numpy as np
import warp as wp
from . import material as m, contact as c, tool
from .geometry import points, stencil


@dataclass(frozen=True)
class Config:
    size: tuple = (.06, .03, .03)
    grid_spacing: float = .0075
    dt: float = .0005
    density: float = 1100.
    young_modulus: float = 6000.
    poisson_ratio: float = .35
    dynamic_viscosity: float = 0.
    yield_stress: float = 0.  # Disabled, not zero-strength plasticity.
    flip_fraction: float = .95
    normal_scale: float = .1
    friction_static: float = .3
    friction_dynamic: float = .2
    tangent_stiffness: float = .05
    tangent_damping: float = .1
    # 0: s_n and gamma_k are per-step constants (paper Eqs. 11, 12, 15), so the
    # contact stiffness s_n*m/dt^2 changes with dt. >0: normal_scale and
    # tangent_stiffness are the values at this dt and are rescaled by
    # (dt/contact_reference_dt)^2, keeping the stiffness fixed across dt.
    contact_reference_dt: float = 0.
    cutting_energy: float = .5
    sharpness: float = .9
    blade_mass: float = .2
    blade_half_size: tuple = (.0005, .025, .02)
    drive_stiffness: float = 250.
    drive_damping: float = 14.
    drive_force_limit: float = 10.
    shape: str = 'block'
    blade_bevel_height: float = 0.
    free_blade: bool = False
    pose_servo: bool = False
    external_tool: bool = False
    cache_topology: bool = False
    grid_storage: str = 'hash'  # compact is validated against the original hash path
    # 'all' checks every pair (cheaper than a hash-grid build for small n),
    # 'auto' picks 'all' up to ALL_PAIRS_MAX_PARTICLES.
    pair_search: str = 'auto'
    # External tool within a coupling interval: 'hold' (paper III-G) or
    # 'predict' (mobility-based twist update, see tool.finish_external_step).
    tool_substep: str = 'hold'
    accumulate_tool_impulses: bool = False
    rotational_stiffness: float = .05
    rotational_damping: float = .003

    def __post_init__(self):
        if self.grid_storage not in ('hash', 'compact'):
            raise ValueError('grid_storage must be hash or compact')
        if self.pair_search not in ('auto', 'hash', 'all'):
            raise ValueError('pair_search must be auto, hash or all')
        if self.tool_substep not in ('hold', 'predict'):
            raise ValueError('tool_substep must be hold or predict')
        if self.tool_substep == 'predict' and not self.external_tool:
            raise ValueError('tool_substep=predict requires external_tool')
        numeric = [float(v) for k, v in asdict(self).items() if isinstance(v, (int, float))]
        if not np.isfinite(numeric).all():
            raise ValueError('Nonfinite configuration')
        positive = ('dt', 'grid_spacing', 'density', 'young_modulus', 'normal_scale',
                    'tangent_stiffness', 'blade_mass', 'cutting_energy', 'drive_force_limit')
        if any(getattr(self, k) <= 0 for k in positive):
            raise ValueError('Invalid positive parameter')
        if not (0 < self.poisson_ratio < .5 and 0 < self.sharpness < 1):
            raise ValueError('Invalid material/sharpness')
        if not (0 <= self.flip_fraction <= 1 and 0 <= self.friction_dynamic <= self.friction_static):
            raise ValueError('Invalid transfer/friction')
        if min(self.dynamic_viscosity, self.yield_stress, self.tangent_damping, self.contact_reference_dt) < 0:
            raise ValueError('Negative damping/plasticity/reference step')
        if max(self.contact_scales()) > 1:
            raise ValueError('Rescaled contact factor exceeds 1 (one-step overshoot); reduce dt')
        if not 0 <= self.blade_bevel_height < 2*self.blade_half_size[2]:
            raise ValueError('Invalid blade geometry')
        if sum((self.free_blade, self.pose_servo, self.external_tool)) > 1:
            raise ValueError('Choose free, pose-servo, or externally prescribed tool')
        cells = np.asarray(self.size)/self.grid_spacing
        if min(cells) < 2 or not np.allclose(cells, np.rint(cells)):
            raise ValueError('Size must contain integer cells, at least two per axis')

    def contact_scales(self):
        """Effective (s_n, gamma_k) used by the kernels at this dt."""
        ratio = 1. if self.contact_reference_dt == 0 else (self.dt/self.contact_reference_dt)**2
        return self.normal_scale*ratio, self.tangent_stiffness*ratio


ALL_PAIRS_MAX_PARTICLES = 4096


class Solver:
    def __init__(self, config=Config(), device='cuda:0'):
        self.config = config
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError('This runner requires CUDA; no silent CPU fallback')
        cfg = config
        self.reference, self.origin, cells = points(cfg.size, cfg.grid_spacing, cfg.shape)
        self.n = len(self.reference)
        nodes, weights, gradients = stencil(self.reference, self.origin, cfg.grid_spacing, cells)
        self.node_count = int(np.prod(cells+1))
        self.node_numpy, self.weight_numpy = nodes, weights
        # Local hash-table sizing choice, NOT a paper parameter. There are at
        # most 8*n distinct (reference node, cut code) keys. Reserving the next
        # power of two >=16*n keeps open-addressing occupancy below one half.
        # The compact backend uses this only for key lookup on topology changes;
        # physical fields have dense active indices and device-side loop bounds.
        self.capacity = 1 << (max(16*self.n, 16)-1).bit_length()
        # Up to 64 active contact pairs/particle before an explicit overflow error.
        # This is storage capacity, never a force/contact truncation rule.
        self.pair_capacity = 1 << (max(64*self.n, 16)-1).bit_length()
        self.volume = (cfg.grid_spacing/2)**3
        self.mass = cfg.density*self.volume
        self.radius = cfg.grid_spacing/4
        self.mu = cfg.young_modulus/(2*(1+cfg.poisson_ratio))
        self.lam = cfg.young_modulus*cfg.poisson_ratio/((1+cfg.poisson_ratio)*(1-2*cfg.poisson_ratio))
        self.threshold = cfg.cutting_energy*self.volume**(2/3)*(1-cfg.sharpness)/cfg.sharpness
        self.normal_scale, self.tangent_stiffness = cfg.contact_scales()
        self.half = wp.vec3(*cfg.blade_half_size)
        hx, hy, hz = cfg.blade_half_size
        self.inertia_inv = wp.vec3(*(3/(cfg.blade_mass*np.array([hy*hy+hz*hz,hx*hx+hz*hz,hx*hx+hy*hy]))))
        def array(data, dtype):
            return wp.array(np.asarray(data), dtype=dtype, device=self.device)
        def zeros(shape, dtype):
            return wp.zeros(shape, dtype=dtype, device=self.device)
        self.ref = array(self.reference, wp.vec3)
        self.x = array(self.reference, wp.vec3)
        self.v = zeros(self.n, wp.vec3)
        self.F = array(np.tile(np.eye(3), (self.n,1,1)), wp.mat33)
        self.Fp = array(np.tile(np.eye(3), (self.n,1,1)), wp.mat33)
        self.codes, self.damage = zeros(self.n,int), zeros(self.n,float)
        self.nodes = array(nodes, int)
        self.weights = array(weights,float)
        self.gradients = array(gradients,wp.vec3)
        self.slots = zeros((self.n,8),int)
        self.keys = zeros(self.capacity,wp.int64)
        self.compact = cfg.grid_storage == 'compact'
        self.grid_capacity = 8*self.n if self.compact else self.capacity
        self.grid_threads = min(2048, self.grid_capacity)
        self.active_count = zeros(1,int)
        self.grid_keys = zeros(self.grid_capacity,wp.int64) if self.compact else self.keys
        self.key_mapping = zeros(self.capacity,int) if self.compact else None
        self.heads = zeros(self.node_count,int)
        self.links = zeros(self.grid_capacity,int)
        self.gm = zeros(self.grid_capacity,float)
        for name in ('momentum','gi','coupled_momentum','coupled_impulse','old_v','new_v','remapped','coupled_remapped'):
            setattr(self,name,zeros(self.grid_capacity,wp.vec3))
        self.impulses = zeros(self.n,wp.vec3)
        self.histories = zeros((2,self.n),wp.vec3)
        self.pair_keys = [zeros(self.pair_capacity,wp.int64) for _ in range(2)]
        self.pair_history = [zeros(self.pair_capacity,wp.vec3) for _ in range(2)]
        for a in self.pair_keys:
            a.fill_(-1)
        # Hash-bucket dimensions, NOT spatial boundaries. Positions outside
        # the reference box are still inserted and queried without clipping.
        extent=np.ceil(np.asarray(cfg.size)/(2*self.radius)).astype(int)+2
        self.hash_grid_shape=tuple(1 << (max(8,int(v))-1).bit_length() for v in extent)
        self.all_pairs = cfg.pair_search == 'all' or (cfg.pair_search == 'auto' and self.n <= ALL_PAIRS_MAX_PARTICLES)
        if not self.all_pairs:
            self.pair_grid = wp.HashGrid(*self.hash_grid_shape,device=self.device)
            self.pair_grid.reserve(self.n)
        self.speed = zeros(1,float)
        self.bx = array([[0.,0.,cfg.size[2]+.012+hz]],wp.vec3)
        self.bv, self.bw = zeros(1,wp.vec3), zeros(1,wp.vec3)
        self.bq = array([[0.,0.,0.,1.]],wp.quat)
        self.tool_mobility = zeros((6,6),float)
        self.reaction, self.torque, self.board = [zeros(1,wp.vec3) for _ in range(3)]
        self.work = zeros(1,float)
        self.tool_table_impulse = zeros(1,wp.vec3)
        self.interval_impulse, self.interval_angular_impulse = zeros(1,wp.vec3), zeros(1,wp.vec3)
        self.single_target = zeros(1,wp.vec3)
        self.single_rotation = array([[0.,0.,0.,1.]],wp.quat)
        self.errors = zeros(3,int)
        self.topology_dirty = array([1],int)
        self.step_count = 0
        self.cut_index = 0
        self._batches = {}
        self.graph_build_s = 0.

    def launch(self, kernel, dim, args):
        wp.launch(kernel, dim, args, device=self.device)

    def material_step(self):
        cfg = self.config
        if cfg.cache_topology:
            # Fixed reference stencils and unchanged side codes imply the
            # same particle-to-logical-grid map. No contact step is skipped.
            wp.capture_if(self.topology_dirty, self._register_topology)
        else:
            self._register_topology()
        if self.compact:
            self.launch(m.clear_compact_grid,self.grid_threads,[self.active_count,self.grid_threads,
                self.gm,self.momentum,self.gi,self.remapped])
        else:
            for a in (self.gm,self.momentum,self.gi,self.remapped):
                a.zero_()
        self.launch(m.p2g,self.n,[self.v,self.F,self.Fp,self.impulses,self.slots,self.weights,self.gradients,
                                  self.mass,self.volume,self.mu,self.lam,cfg.dt,self.gm,self.momentum,self.gi,self.errors])
        self._material_transfers()

    def _register_topology(self):
        self.keys.fill_(-1)
        self.heads.fill_(-1)
        self.launch(m.register,(self.n,8),[self.nodes,self.codes,self.keys,self.slots,self.node_count,self.capacity,self.errors])
        if self.compact:
            self.active_count.zero_()
            self.launch(m.compact_keys,self.capacity,[self.keys,self.key_mapping,self.grid_keys,self.active_count])
            self.launch(m.remap_slots,(self.n,8),[self.slots,self.key_mapping])
            self.launch(m.compact_node_lists,self.grid_threads,[self.active_count,self.grid_threads,
                self.grid_keys,self.heads,self.links,self.node_count])
        else:
            self.launch(m.node_lists,self.capacity,[self.keys,self.heads,self.links,self.node_count])
        self.topology_dirty.zero_()

    def _material_transfers(self):
        cfg = self.config
        if cfg.dynamic_viscosity:
            self._couple(self.momentum,self.coupled_momentum)
            self._grid_launch(m.grid_vel,m.grid_vel_compact,[self.gm,self.coupled_momentum,self.gi,self.old_v,self.new_v])
            self.launch(m.viscous_impulse,self.n,[self.F,self.slots,self.gradients,self.old_v,
                cfg.dynamic_viscosity,self.volume,cfg.dt,self.gi])
            self._couple(self.gi,self.coupled_impulse)
            self._grid_launch(m.grid_vel,m.grid_vel_compact,[self.gm,self.coupled_momentum,self.coupled_impulse,self.old_v,self.new_v])
        else:
            self._grid_launch(m.couple_and_velocity,m.couple_and_velocity_compact,[self.grid_keys,self.heads,self.links,self.gm,
                self.momentum,self.gi,self.old_v,self.new_v,self.node_count])
        self.launch(m.g2p_remap,self.n,[self.x,self.v,self.slots,self.weights,self.old_v,self.new_v,
            cfg.dt,cfg.flip_fraction,self.mass,self.remapped])
        self._couple(self.remapped,self.coupled_remapped)
        self.launch(m.update_F_plastic,self.n,[self.F,self.Fp,self.slots,self.gradients,self.gm,
            self.coupled_remapped,cfg.dt,self.mu,cfg.yield_stress,self.errors])

    def _couple(self, src, dst):
        self._grid_launch(m.couple,m.couple_compact,[self.grid_keys,self.heads,self.links,self.gm,src,dst,self.node_count])

    def _grid_launch(self, baseline, compact, args):
        if self.compact:
            # GPU count controls the grid-stride loop; no host readback or
            # physical node truncation. 2048 is scheduling, not a node limit.
            self.launch(compact,self.grid_threads,[self.active_count,self.grid_threads,*args])
        else:
            self.launch(baseline,self.capacity,args)

    def step(self, target_edge=(0.,0.,.042), cutting=True, cut_index=0, orientation=None):
        if self.config.external_tool:
            raise ValueError('Use ExternalToolPort.advance for an externally owned tool')
        self._set_cut(cut_index)
        self._step_kernels(target_edge, cutting, cut_index, orientation=orientation)

    def _set_cut(self, cut_index):
        if cut_index < self.cut_index or cut_index > 18:
            raise ValueError('Persistent ternary int32 codes require monotonic cut indices in [0,18]')
        if cut_index != self.cut_index:
            self.damage.zero_()
            self.cut_index = cut_index

    def _step_kernels(self, target_edge, cutting, cut_index, buffer=None, index=0, rotations=None, orientation=None):
        cfg = self.config
        sn, gk = self.normal_scale, self.tangent_stiffness
        if not cfg.external_tool:
            # External mode resets these at the end of the previous step.
            self.launch(c.clear_step,1,[self.reaction,self.torque,self.board,self.speed])
        self.launch(c.rigid_and_speed,self.n,[self.speed,self.x,self.v,self.bx,self.bv,self.bq,self.bw,self.half,cfg.blade_bevel_height,
            self.radius,cfg.grid_spacing,self.mass,cfg.blade_mass,self.inertia_inv,self.tool_mobility,
            -1 if cfg.external_tool else int(cfg.free_blade or cfg.pose_servo),
            cfg.dt,sn,cfg.friction_static,cfg.friction_dynamic,gk,cfg.tangent_damping,
            self.histories,self.damage,self.threshold,int(cutting),self.impulses,self.reaction,self.torque,self.board,self.work])
        old,new = self.step_count%2,(self.step_count+1)%2
        self.pair_keys[new].fill_(-1)
        contact = [self.x,self.ref,self.v,self.codes,self.radius,cfg.grid_spacing,
            self.mass,cfg.dt,sn,cfg.friction_static,cfg.friction_dynamic,gk,cfg.tangent_damping]
        history = [self.pair_keys[old],self.pair_history[old],self.pair_keys[new],self.pair_history[new],
            self.pair_capacity,self.n,self.impulses,self.errors]
        if self.all_pairs:
            self.launch(c.pairs_all,self.n,[*contact,*history])
        else:
            self.pair_grid.build(self.x,2*self.radius)
            self.launch(c.pairs,self.n,[self.pair_grid.id,*contact,self.speed,*history])
        self.material_step()
        if cfg.external_tool:
            # The external engine owns tool integration between intervals.
            self.launch(tool.finish_external_step,1,[self.reaction,self.torque,self.board,self.speed,
                self.bx,self.bv,self.bq,self.bw,self.tool_mobility,int(cfg.tool_substep=='predict'),cfg.dt,
                self.interval_impulse,self.interval_angular_impulse])
        elif cfg.accumulate_tool_impulses:
            # Use the contact-time origin, before integrating the rigid body.
            self.launch(tool.accumulate_reaction, 1, [self.reaction, self.torque, self.bx,
                self.interval_impulse, self.interval_angular_impulse])
        if cfg.external_tool:
            pass
        elif cfg.pose_servo or cfg.free_blade:
            if buffer is None:
                self.single_target.assign([target_edge])
                self.single_rotation.assign([orientation if orientation is not None else [0.,0.,0.,1.]])
                buffer, rotations, index = self.single_target, self.single_rotation, 0
            self.launch(tool.step,1,[self.bx,self.bv,self.bq,self.bw,buffer,rotations,index,
                self.reaction,self.torque,self.tool_table_impulse,cfg.dt,cfg.blade_mass,self.half,
                cfg.blade_bevel_height,cfg.drive_stiffness,cfg.drive_damping,cfg.drive_force_limit,
                cfg.rotational_stiffness,cfg.rotational_damping,self.inertia_inv,cfg.friction_dynamic,int(cfg.free_blade)])
        else:
            control = [wp.vec3(*target_edge)] if buffer is None else [buffer, index]
            self.launch(c.blade_step if buffer is None else c.blade_step_buffer,1,[self.bx,self.bv,self.bq,self.bw,*control,self.reaction,self.torque,
                cfg.dt,cfg.blade_mass,self.half,cfg.drive_stiffness,cfg.drive_damping,cfg.drive_force_limit,
                int(cfg.free_blade),self.inertia_inv])
        self.launch(c.assign,self.n,[self.x,self.v,self.codes,self.bx,self.bv,self.bq,self.bw,
            self.half,cfg.grid_spacing,cfg.dt,3**cut_index,int(cutting),self.topology_dirty])
        self.step_count += 1

    def advance(self, targets, cutting=True, cut_index=0, graph=True, orientations=None):
        """Submit exact per-substep targets, including every intermediate contact.

        A graph batches scheduling only: dt, forces, contact checks, and target
        samples remain identical. Contact-history ping-pong parity is in the key.
        """
        if self.config.external_tool:
            raise ValueError('Use ExternalToolPort.advance for an externally owned tool')
        targets = np.ascontiguousarray(targets, dtype=np.float32)
        if targets.ndim != 2 or targets.shape[1] != 3 or not np.isfinite(targets).all() or not len(targets):
            raise ValueError('targets must be finite (substeps, 3) edge positions')
        self._set_cut(cut_index)
        orientations = np.tile([0.,0.,0.,1.], (len(targets),1)) if orientations is None else np.asarray(orientations)
        if orientations.shape != (len(targets),4) or not np.isfinite(orientations).all():
            raise ValueError('orientations must be finite (substeps, 4) xyzw quaternions')
        lengths = np.linalg.norm(orientations,axis=1)
        if np.any(lengths < 1.e-6):raise ValueError('Zero quaternion')
        orientations = np.ascontiguousarray(orientations/lengths[:,None],dtype=np.float32)
        if not graph:
            for target, orientation in zip(targets,orientations):
                self._step_kernels(target, cutting, cut_index, orientation=orientation)
            return
        buffer, rotations, captured = self.prepare(targets, cutting, cut_index, orientations)
        buffer.assign(targets)
        if self.config.pose_servo or self.config.free_blade:rotations.assign(orientations)
        wp.capture_launch(captured)
        self.step_count += len(targets)

    def prepare(self, targets, cutting=True, cut_index=0, orientations=None):
        """Capture scheduling without advancing physics or changing cut state."""
        targets = np.ascontiguousarray(targets,dtype=np.float32)
        if orientations is None:orientations=np.tile([0.,0.,0.,1.],(len(targets),1))
        key = (len(targets), bool(cutting), cut_index, self.step_count % 2)
        if key not in self._batches:
            start = time.perf_counter()
            # Eager warmup must have loaded all modules before capture.
            wp.load_module(module=m, device=self.device)
            wp.load_module(module=c, device=self.device)
            if (self.config.pose_servo or self.config.free_blade or self.config.external_tool
                    or self.config.accumulate_tool_impulses):
                wp.load_module(module=tool, device=self.device)
            buffer = wp.array(targets, dtype=wp.vec3, device=self.device)
            rotations = wp.array(orientations, dtype=wp.quat, device=self.device)
            count = self.step_count
            try:
                with wp.ScopedCapture(device=self.device) as capture:
                    for i in range(len(targets)):
                        self._step_kernels(targets[i], cutting, cut_index, buffer, i, rotations)
            finally:
                self.step_count = count
            self._batches[key] = buffer, rotations, capture.graph
            self.graph_build_s += time.perf_counter() - start
        return self._batches[key]

    def state(self):
        errors = self.errors.numpy()
        if errors.any():
            raise RuntimeError(f'GPU errors [storage overflow, inverted material, coincident proxies]: {errors.tolist()}')
        state = {k: getattr(self,a).numpy() for k,a in [('position','x'),('velocity','v'),('deformation','F'),
            ('plastic_deformation','Fp'),('sides','codes'),('damage','damage'),('blade_position','bx'),
            ('blade_velocity','bv'),('blade_orientation','bq'),('blade_omega','bw')]}
        state['time'] = self.step_count*self.config.dt
        if any(not np.isfinite(v).all() for v in state.values()):
            raise RuntimeError('Nonfinite state; run is invalid')
        if np.linalg.det(state['deformation']).min() <= 0:
            raise RuntimeError('Inverted material; run is invalid')
        return state
