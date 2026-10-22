"""Single-field MLS-MPM cutting baseline after TopoCut (Wang et al., CoRL 2025, Sec. 3.1, App. 7.5-7.6).

Updated-Lagrangian MLS-MPM (Hu et al. 2018): quadratic B-splines, APIC
transfer, one velocity field, so material on both sides of a cut shares grid
nodes. Damage (Eqs. 6, 7) marks particles; separation is found afterwards by
`topocut_topology`.

Particles, material, time step, and the gravity, board and knife contact
impulses (`contact.rigid_and_speed` with cutting off) are the BladeMaster ones,
so the two solvers differ only in the field representation and in how
separation is detected.
"""

from dataclasses import dataclass
import numpy as np
import warp as wp
from . import contact as c
from .coupling import Reaction
from .geometry import points
from .material import corotated_stress


@dataclass(frozen=True)
class TopoCutParams:
    compression: float = 0.025  # epsilon_c, Eq. 6
    stretch: float = 0.01  # epsilon_s
    sensitivity: float = 1.0  # m
    softening: float = 0.0  # gamma in Eq. 7, Pa per unit Hencky plastic strain
    # Shear modulus factor of damaged particles; the bulk term is kept.
    damaged_shear: float = 0.0
    # Domain margin around the material and height above it, metres; cut pieces
    # pushed or tipped over must stay inside the grid.
    margin: float = 0.06
    headroom: float = 0.06


# PlasticineLab clamps singular values before the return map, so inverted
# particles are pushed back instead of stopping the simulation.
MIN_SINGULAR_VALUE = wp.constant(0.05)


@wp.func
def spline_weights(fx: wp.vec3):
    """Row a holds the quadratic B-spline weights of node offset a for each axis."""
    w = wp.mat33(0.0)
    for axis in range(3):
        f = fx[axis]
        w[0, axis] = 0.5 * (1.5 - f) * (1.5 - f)
        w[1, axis] = 0.75 - (f - 1.0) * (f - 1.0)
        w[2, axis] = 0.5 * (f - 0.5) * (f - 0.5)
    return w


@wp.func
def cell_base(x: wp.vec3, origin: wp.vec3, dx: float):
    local = (x - origin) / dx
    return wp.vec3i(
        int(wp.floor(local[0] - 0.5)), int(wp.floor(local[1] - 0.5)), int(wp.floor(local[2] - 0.5))
    )


@wp.func
def inside_domain(base: wp.vec3i, dims: wp.vec3i):
    # base < dims - 2 rather than base + 2 < dims: a diverged position saturates to INT_MAX.
    return (
        base[0] >= 0
        and base[1] >= 0
        and base[2] >= 0
        and base[0] < dims[0] - 2
        and base[1] < dims[1] - 2
        and base[2] < dims[2] - 2
    )


@wp.kernel
def p2g(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    C: wp.array(dtype=wp.mat33),
    F: wp.array(dtype=wp.mat33),
    damage: wp.array(dtype=float),
    impulse: wp.array(dtype=wp.vec3),
    mass: float,
    volume: float,
    mu: float,
    lam: float,
    damaged_shear: float,
    dx: float,
    dt: float,
    origin: wp.vec3,
    dims: wp.vec3i,
    grid_m: wp.array3d(dtype=float),
    grid_p: wp.array3d(dtype=wp.vec3),
):
    p = wp.tid()
    if not wp.isfinite(x[p]):
        return
    base = cell_base(x[p], origin, dx)
    if not inside_domain(base, dims):
        return
    fx = (x[p] - origin) / dx - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    w = spline_weights(fx)
    shear = mu
    if damage[p] > 0.0:
        shear = mu * damaged_shear
    U, sigma, V = wp.svd3(F[p])
    P = corotated_stress(F[p], U, V, wp.identity(n=3, dtype=float), shear, lam)
    stress = (-dt * volume * 4.0 / (dx * dx)) * P * wp.transpose(F[p])
    affine = stress + mass * C[p]
    momentum = mass * v[p] + impulse[p]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w[i, 0] * w[j, 1] * w[k, 2]
                dpos = (wp.vec3(float(i), float(j), float(k)) - fx) * dx
                wp.atomic_add(
                    grid_p, base[0] + i, base[1] + j, base[2] + k, weight * (momentum + affine * dpos)
                )
                wp.atomic_add(grid_m, base[0] + i, base[1] + j, base[2] + k, weight * mass)


@wp.kernel
def grid_update(
    grid_m: wp.array3d(dtype=float),
    grid_p: wp.array3d(dtype=wp.vec3),
    grid_v: wp.array3d(dtype=wp.vec3),
    dims: wp.vec3i,
):
    """Nodal velocity; sticky walls one node inside the domain boundary."""
    i, j, k = wp.tid()
    m = grid_m[i, j, k]
    v = wp.vec3(0.0)
    if m > 0.0:
        v = grid_p[i, j, k] / m
    if i < 1 or j < 1 or k < 1 or i >= dims[0] - 1 or j >= dims[1] - 1 or k >= dims[2] - 1:
        v = wp.vec3(0.0)
    grid_v[i, j, k] = v


@wp.kernel
def accumulate_reaction(
    reaction: wp.array(dtype=wp.vec3),
    torque: wp.array(dtype=wp.vec3),
    board: wp.array(dtype=wp.vec3),
    speed: wp.array(dtype=float),
    impulse: wp.array(dtype=wp.vec3),
    angular_impulse: wp.array(dtype=wp.vec3),
):
    impulse[0] = impulse[0] + reaction[0]
    angular_impulse[0] = angular_impulse[0] + torque[0]
    reaction[0] = wp.vec3(0.0)
    torque[0] = wp.vec3(0.0)
    board[0] = wp.vec3(0.0)
    speed[0] = 0.0


@wp.kernel
def g2p(
    grid_v: wp.array3d(dtype=wp.vec3),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    C: wp.array(dtype=wp.mat33),
    F: wp.array(dtype=wp.mat33),
    yield_stress: wp.array(dtype=float),
    damage: wp.array(dtype=float),
    damage_reference: wp.array(dtype=float),
    plastic: int,
    mu: float,
    softening: float,
    compressed: float,
    stretched: float,
    dx: float,
    dt: float,
    origin: wp.vec3,
    dims: wp.vec3i,
    inversions: wp.array(dtype=int),
):
    p = wp.tid()
    if not wp.isfinite(x[p]):
        return
    base = cell_base(x[p], origin, dx)
    if not inside_domain(base, dims):
        return
    fx = (x[p] - origin) / dx - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    w = spline_weights(fx)
    new_v = wp.vec3(0.0)
    new_C = wp.mat33(0.0)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w[i, 0] * w[j, 1] * w[k, 2]
                dpos = (wp.vec3(float(i), float(j), float(k)) - fx) * dx
                gv = grid_v[base[0] + i, base[1] + j, base[2] + k]
                new_v += weight * gv
                new_C += (4.0 * weight / (dx * dx)) * wp.outer(gv, dpos)
    v[p] = new_v
    C[p] = new_C
    x[p] = x[p] + dt * new_v
    f = (wp.identity(n=3, dtype=float) + dt * new_C) * F[p]
    U, s, V = wp.svd3(f)
    if wp.min(s) <= 0.0:
        wp.atomic_add(inversions, 0, 1)
    if wp.min(s) < MIN_SINGULAR_VALUE:
        s = wp.vec3(
            wp.max(s[0], MIN_SINGULAR_VALUE), wp.max(s[1], MIN_SINGULAR_VALUE), wp.max(s[2], MIN_SINGULAR_VALUE)
        )
        f = U * wp.diag(s) * wp.transpose(V)
    # 0 until damage is enabled; then J at that moment, so Eq. 6 measures the volume
    # change from the settled state and gravity settling is not damage.
    reference_J = damage_reference[p]
    if plastic != 0:
        # Hencky von Mises return map; Eq. 7 softens the yield stress by the plastic increment.
        e = wp.vec3(wp.log(s[0]), wp.log(s[1]), wp.log(s[2]))
        dev = e - wp.vec3((e[0] + e[1] + e[2]) / 3.0)
        norm = wp.length(dev)
        limit = yield_stress[p] / (2.0 * mu) * wp.sqrt(2.0 / 3.0)
        if norm > limit:
            e = e - (1.0 - limit / norm) * dev
            s = wp.vec3(wp.exp(e[0]), wp.exp(e[1]), wp.exp(e[2]))
            f = U * wp.diag(s) * wp.transpose(V)
            if softening > 0.0 and reference_J > 0.0:
                yield_stress[p] = wp.max(0.0, yield_stress[p] - softening * (norm - limit))
                if yield_stress[p] <= 0.0:
                    damage[p] = 1.0
    F[p] = f
    if reference_J > 0.0:
        J = s[0] * s[1] * s[2] / reference_J
        if J <= compressed or J >= stretched:
            damage[p] = 1.0


class MlsSolver:
    """damage_from_start=True applies Eq. 6 to J of the reference configuration from the
    first step (the literal reading); False leaves damage off until `start_damage`, which
    takes the current configuration (e.g. settled under gravity) as the reference."""

    def __init__(self, cfg, params=TopoCutParams(), device=None, damage_from_start=True):
        self.config, self.params = cfg, params
        self.device = wp.get_device(device)
        dx = cfg.grid_spacing
        reference, _, _ = points(cfg.size, dx, cfg.shape, cfg.sampling_seed)
        self.reference = reference
        self.n = len(reference)
        self.volume = (dx / 2) ** 3
        self.mass = cfg.density * self.volume
        self.radius = dx / 4
        self.blade_offset = max(cfg.blade_contact_radius or self.radius, self.radius - cfg.blade_half_size[0])
        self.normal_scale, self.tangent_stiffness = cfg.contact_scales()
        self.mu = cfg.young_modulus / (2 * (1 + cfg.poisson_ratio))
        self.lam = (
            cfg.young_modulus * cfg.poisson_ratio / ((1 + cfg.poisson_ratio) * (1 - 2 * cfg.poisson_ratio))
        )
        pad = params.margin
        low = reference.min(0) - pad
        high = reference.max(0) + pad
        low[2] = -2 * dx
        high[2] = reference[:, 2].max() + params.headroom
        self.origin = low
        self.dims = np.ceil((high - low) / dx).astype(int) + 1
        self.compressed = (1 - params.compression) ** params.sensitivity
        self.stretched = (1 + params.stretch) ** params.sensitivity
        with wp.ScopedDevice(self.device):
            self.x = wp.array(reference, dtype=wp.vec3)
            self.v = wp.zeros(self.n, dtype=wp.vec3)
            self.C = wp.zeros(self.n, dtype=wp.mat33)
            self.F = wp.array(np.tile(np.eye(3), (self.n, 1, 1)), dtype=wp.mat33)
            self.yield_stress = wp.full(self.n, cfg.yield_stress, dtype=float)
            self.damage = wp.zeros(self.n, dtype=float)
            self.damage_reference = wp.full(self.n, 1.0 if damage_from_start else 0.0, dtype=float)
            shape = tuple(int(d) for d in self.dims)
            self.grid_m = wp.zeros(shape, dtype=float)
            self.grid_p = wp.zeros(shape, dtype=wp.vec3)
            self.grid_v = wp.zeros(shape, dtype=wp.vec3)
            self.tool_pos = wp.zeros(1, dtype=wp.vec3)
            self.tool_vel = wp.zeros(1, dtype=wp.vec3)
            self.tool_rot = wp.array([[0.0, 0.0, 0.0, 1.0]], dtype=wp.quat)
            self.tool_omega = wp.zeros(1, dtype=wp.vec3)
            self.tool_mobility = wp.zeros((6, 6), dtype=float)
            self.histories = wp.zeros((2, self.n), dtype=wp.vec3)
            self.contact_impulse = wp.zeros(self.n, dtype=wp.vec3)
            self.unused_damage = wp.zeros(self.n, dtype=float)
            self.reaction = wp.zeros(1, dtype=wp.vec3)
            self.torque = wp.zeros(1, dtype=wp.vec3)
            self.board = wp.zeros(1, dtype=wp.vec3)
            self.speed = wp.zeros(1, dtype=float)
            self.work = wp.zeros(1, dtype=float)
            self.contact_parts = wp.zeros(4, dtype=wp.vec3)
            self.impulse = wp.zeros(1, dtype=wp.vec3)
            self.angular_impulse = wp.zeros(1, dtype=wp.vec3)
            self.inversions = wp.zeros(1, dtype=int)
        self.step_count = 0
        self.graphs = {}

    def _substep(self):
        cfg, dx = self.config, self.config.grid_spacing
        origin, dims = wp.vec3(*self.origin), wp.vec3i(*self.dims)
        # Knife contact as an external, kinematic tool: tool_model -1 with zero mobility.
        wp.launch(
            c.rigid_and_speed,
            self.n,
            [
                self.speed,
                self.x,
                self.v,
                self.tool_pos,
                self.tool_vel,
                self.tool_rot,
                self.tool_omega,
                wp.vec3(*cfg.blade_half_size),
                cfg.blade_bevel_height,
                self.radius,
                self.blade_offset,
                dx,
                cfg.cut_band_cells,
                self.mass,
                cfg.blade_mass,
                wp.vec3(0.0),
                self.tool_mobility,
                -1,
                wp.vec2(*cfg.board_half_extent),
                cfg.board_normal_scale or self.normal_scale,
                cfg.dt,
                self.normal_scale,
                cfg.friction_static,
                cfg.friction_dynamic,
                self.tangent_stiffness,
                cfg.tangent_damping,
                self.histories,
                self.unused_damage,
                1.0,
                0,
                self.contact_impulse,
                self.reaction,
                self.torque,
                self.board,
                self.work,
                self.contact_parts,
            ],
        )
        wp.launch(
            accumulate_reaction,
            1,
            [self.reaction, self.torque, self.board, self.speed, self.impulse, self.angular_impulse],
        )
        self.grid_m.zero_()
        self.grid_p.zero_()
        wp.launch(
            p2g,
            self.n,
            [
                self.x,
                self.v,
                self.C,
                self.F,
                self.damage,
                self.contact_impulse,
                self.mass,
                self.volume,
                self.mu,
                self.lam,
                self.params.damaged_shear,
                dx,
                cfg.dt,
                origin,
                dims,
                self.grid_m,
                self.grid_p,
            ],
        )
        wp.launch(
            grid_update, tuple(int(d) for d in self.dims), [self.grid_m, self.grid_p, self.grid_v, dims]
        )
        wp.launch(
            g2p,
            self.n,
            [
                self.grid_v,
                self.x,
                self.v,
                self.C,
                self.F,
                self.yield_stress,
                self.damage,
                self.damage_reference,
                int(cfg.yield_stress > 0),
                self.mu,
                self.params.softening,
                self.compressed,
                self.stretched,
                dx,
                cfg.dt,
                origin,
                dims,
                self.inversions,
            ],
        )

    def advance(self, state, duration_s):
        """Hold the knife pose over duration_s (as ExternalToolPort with tool_substep='hold')."""
        state = state.checked()
        dt = self.config.dt
        steps = int(round(duration_s / dt))
        if steps < 1 or not np.isclose(steps * dt, duration_s, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("Duration must be an integer number of substeps")
        self.tool_pos.assign([state.position])
        self.tool_rot.assign([state.orientation_xyzw])
        self.tool_vel.assign([state.linear_velocity])
        self.tool_omega.assign([state.angular_velocity])
        self.impulse.zero_()
        self.angular_impulse.zero_()
        with wp.ScopedDevice(self.device):
            if steps not in self.graphs:
                with wp.ScopedCapture() as capture:
                    for _ in range(steps):
                        self._substep()
                self.graphs[steps] = capture.graph
            wp.capture_launch(self.graphs[steps])
        self.step_count += steps
        lost = self.diagnostics()["left_domain"]
        if lost:
            raise RuntimeError(
                f"MLS-MPM diverged by t = {self.step_count * dt:.3f} s: {lost} particles non-finite or outside the grid"
            )
        return Reaction(
            duration_s=steps * dt,
            impulse_ns=self.impulse.numpy()[0].astype(float),
            angular_impulse_nms=self.angular_impulse.numpy()[0].astype(float),
            origin_world_m=np.asarray(state.position, dtype=float),
        )

    def start_damage(self):
        """Enable Eq. 6 with each particle's current J as its reference volume."""
        self.damage_reference.assign(np.linalg.det(self.F.numpy()))

    def diagnostics(self):
        """Particles non-finite or outside the grid, and clamped inversions (particle-steps)."""
        local = (self.x.numpy() - self.origin) / self.config.grid_spacing
        base = np.floor(local - 0.5)
        outside = ~np.all(np.isfinite(base) & (base >= 0) & (base + 2 < self.dims), axis=1)
        return dict(left_domain=int(outside.sum()), clamped_inversions=int(self.inversions.numpy()[0]))

    def state(self):
        return dict(
            position=self.x.numpy(),
            velocity=self.v.numpy(),
            damage=self.damage.numpy(),
            deformation=self.F.numpy(),
            time=self.step_count * self.config.dt,
        )
