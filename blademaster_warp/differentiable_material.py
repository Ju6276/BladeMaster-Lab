"""Variational plasticity and fixed cut-field transfers with independent step states."""

from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from . import material as m
from . import plasticity
from .differentiable import ElasticTrajectory, State, elastic_stress


@wp.func
def yield_active(stretches: wp.vec3, limit: float, branch: int):
    stress = wp.cw_mul(stretches, stretches - wp.vec3(1.0))
    dev = stress - wp.vec3((stress[0] + stress[1] + stress[2]) / 3.0)
    return branch == 1 or (branch < 0 and wp.length(dev) > limit)


@wp.func
def plastic_project(f: wp.mat33, limit: float):
    U, s, V = wp.svd3(f)
    projected, converged = plasticity.corotated_return(wp.vec3d(s), wp.float64(limit))
    return U * wp.diag(wp.vec3(projected)) * wp.transpose(V), converged


@wp.func_grad(plastic_project)
def plastic_project_grad(f: wp.mat33, limit: float, adj_projected: wp.mat33, adj_converged: int):
    derivative, radius_derivative = plasticity.projection_vjp(wp.mat33d(f), wp.float64(limit),
                                                              wp.mat33d(adj_projected))
    wp.adjoint[f] += wp.mat33(derivative)
    wp.adjoint[limit] += float(radius_derivative)


@wp.func
def constitutive_response(F: wp.mat33, Fp: wp.mat33, mu: float, lam: float,
                          limit: float, branch: int):
    f = F * wp.inverse(Fp)
    U, s, V = wp.svd3(f)
    next_Fp = Fp
    converged = int(1)
    for i in range(3):
        if s[i] <= 0.0 or not wp.isfinite(s[i]):
            converged = int(0)
    if yield_active(s, limit, branch):
        updated, converged = plasticity.corotated_return(wp.vec3d(s), wp.float64(limit))
        s = wp.vec3(updated)
        projected = U * wp.diag(s) * wp.transpose(V)
        next_Fp = wp.inverse(projected) * F
        f = projected
    stress = m.corotated_stress(f, U, V, wp.inverse(next_Fp), mu, lam)
    return next_Fp, stress, converged


@wp.func_grad(constitutive_response)
def constitutive_response_grad(F: wp.mat33, Fp: wp.mat33, mu: float, lam: float,
                               limit: float, branch: int, adj_Fp: wp.mat33, adj_stress: wp.mat33, adj_converged: int):
    old_inverse = wp.inverse(Fp)
    f = F * old_inverse
    U, s, V = wp.svd3(f)
    active = yield_active(s, limit, branch)
    projected = f
    next_Fp = Fp
    if active:
        projected, converged = plastic_project(f, limit)
        next_Fp = wp.inverse(projected) * F
    inverse = wp.inverse(next_Fp)
    inverse_t = wp.transpose(inverse)
    elastic = elastic_stress(projected, mu, lam)
    adj_elastic = adj_stress * inverse
    adj_plastic = adj_Fp - inverse_t * wp.transpose(adj_stress) * elastic * inverse_t
    # Polar rotation derivative is regular at equal positive singular values.
    Q, sigma, R = wp.svd3(projected)
    local = wp.transpose(Q) * adj_elastic * R
    skew = wp.mat33(0.0)
    for i in range(3):
        for j in range(3):
            if i != j:
                skew[i, j] = (local[i, j] - local[j, i]) / (sigma[i] + sigma[j])
    projected_inverse_t = wp.transpose(wp.inverse(projected))
    J = wp.determinant(projected)
    contraction = wp.ddot(adj_elastic, projected_inverse_t)
    adj_projected = (
        2.0 * mu * (adj_elastic - Q * skew * wp.transpose(R))
        + lam * (2.0 * J - 1.0) * J * contraction * projected_inverse_t
        - lam * (J - 1.0) * J * projected_inverse_t * wp.transpose(adj_elastic) * projected_inverse_t
    )
    wp.adjoint[mu] += 2.0 * wp.ddot(adj_elastic, projected - Q * wp.transpose(R))
    wp.adjoint[lam] += (J - 1.0) * J * contraction
    if active:
        wp.adjoint[F] += projected_inverse_t * adj_plastic
        adj_projected -= projected_inverse_t * adj_plastic * wp.transpose(F) * projected_inverse_t
        derivative, adj_limit = plasticity.projection_vjp(wp.mat33d(f), wp.float64(limit),
                                                          wp.mat33d(adj_projected))
        adj_f = wp.mat33(derivative)
        wp.adjoint[limit] += float(adj_limit)
        wp.adjoint[F] += adj_f * wp.transpose(old_inverse)
        wp.adjoint[Fp] -= wp.transpose(old_inverse) * wp.transpose(F) * adj_f * wp.transpose(old_inverse)
    else:
        wp.adjoint[F] += adj_projected * wp.transpose(old_inverse)
        wp.adjoint[Fp] += adj_plastic - wp.transpose(old_inverse) * wp.transpose(F) * adj_projected * wp.transpose(old_inverse)


@wp.kernel
def compute_stress(
    F: wp.array(dtype=wp.mat33),
    Fp: wp.array(dtype=wp.mat33),
    young: wp.array(dtype=float),
    poisson: wp.float64,
    stress: wp.array(dtype=wp.mat33),
):
    p = wp.tid()
    mu = float(wp.float64(young[0]) / (wp.float64(2.0) * (wp.float64(1.0) + poisson)))
    lam = float(wp.float64(young[0]) * poisson / ((wp.float64(1.0) + poisson)
                                                * (wp.float64(1.0) - wp.float64(2.0) * poisson)))
    inverse = wp.inverse(Fp[p])
    stress[p] = elastic_stress(F[p] * inverse, mu, lam) * wp.transpose(inverse)


@wp.kernel
def project_plastic(
    F: wp.array(dtype=wp.mat33),
    Fp: wp.array(dtype=wp.mat33),
    young: wp.array(dtype=float),
    yield_stress: wp.array(dtype=float),
    poisson: wp.float64,
    fixed_yielded: wp.array(dtype=int),
    next_Fp: wp.array(dtype=wp.mat33),
    yielded: wp.array(dtype=int),
    next_stress: wp.array(dtype=wp.mat33),
    errors: wp.array(dtype=int),
):
    p = wp.tid()
    mu = float(wp.float64(young[0]) / (wp.float64(2.0) * (wp.float64(1.0) + poisson)))
    lam = float(wp.float64(young[0]) * poisson / ((wp.float64(1.0) + poisson)
                                                * (wp.float64(1.0) - wp.float64(2.0) * poisson)))
    limit = yield_stress[0] / (2.0 * mu) * wp.sqrt(2.0 / 3.0)
    branch = fixed_yielded[p]
    if yield_stress[0] <= 0.0:
        branch = 0
    plastic, stress, converged = constitutive_response(F[p], Fp[p], mu, lam, limit, branch)
    if converged == 0:
        wp.atomic_add(errors, 0, 1)
    next_Fp[p] = plastic
    next_stress[p] = stress
    yielded[p] = int(wp.ddot(plastic - Fp[p], plastic - Fp[p]) > 0.0)


@dataclass
class PlasticState(State):
    Fp: wp.array
    yielded: wp.array | None = None
    stress: wp.array | None = None


@dataclass
class FixedTopology:
    slots: wp.array
    keys: wp.array
    offsets: wp.array
    sources: wp.array
    coefficients: wp.array
    count: int
    particle_offsets: wp.array
    particles: wp.array
    corners: wp.array


@wp.kernel
def ordered_p2g(
    offsets: wp.array(dtype=int), particles: wp.array(dtype=int), corners: wp.array(dtype=int),
    weights: wp.array2d(dtype=float), gradients: wp.array2d(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3), stress: wp.array(dtype=wp.mat33), impulses: wp.array(dtype=wp.vec3),
    mass: float, volume: float, duration: wp.array(dtype=float), grid_mass: wp.array(dtype=float),
    momentum: wp.array(dtype=wp.vec3), grid_impulse: wp.array(dtype=wp.vec3),
):
    slot = wp.tid()
    dt = duration[0]
    total_mass = float(0.0)
    total_momentum = wp.vec3(0.0)
    total_impulse = wp.vec3(0.0)
    for edge in range(offsets[slot], offsets[slot + 1]):
        p, a = particles[edge], corners[edge]
        w = weights[p, a]
        total_mass += mass * w
        total_momentum += mass * w * v[p]
        total_impulse += w * impulses[p] - dt * volume * stress[p] * gradients[p, a]
    grid_mass[slot] = total_mass
    momentum[slot] = total_momentum
    grid_impulse[slot] = total_impulse


@wp.func
def gather_at(
    p: int,
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), slots: wp.array2d(dtype=int),
    weights: wp.array2d(dtype=float), old_v: wp.array(dtype=wp.vec3), new_v: wp.array(dtype=wp.vec3),
    duration: wp.array(dtype=float), flip: float, reference_dt: float,
):
    dt = duration[0]
    alpha = 1.0 - (1.0 - flip) * dt / reference_dt
    pic, dv = wp.vec3(0.0), wp.vec3(0.0)
    for a in range(8):
        slot = slots[p, a]
        pic += weights[p, a] * new_v[slot]
        dv += weights[p, a] * (new_v[slot] - old_v[slot])
    return x[p] + dt * pic, (1.0 - alpha) * pic + alpha * (v[p] + dv)


@wp.kernel
def gather_particles(
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), slots: wp.array2d(dtype=int),
    weights: wp.array2d(dtype=float), old_v: wp.array(dtype=wp.vec3), new_v: wp.array(dtype=wp.vec3),
    duration: wp.array(dtype=float), flip: float, reference_dt: float,
    next_x: wp.array(dtype=wp.vec3), next_v: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    xp, vp = gather_at(p, x, v, slots, weights, old_v, new_v, duration, flip, reference_dt)
    next_x[p] = xp
    next_v[p] = vp


@wp.kernel
def correct_kinematics(
    free_x: wp.array(dtype=wp.vec3), free_v: wp.array(dtype=wp.vec3),
    displacement: wp.array(dtype=wp.vec3), duration: wp.array(dtype=float),
    next_x: wp.array(dtype=wp.vec3), next_v: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    next_x[p] = free_x[p] + displacement[p]
    next_v[p] = free_v[p] + displacement[p] / duration[0]


@wp.kernel
def gather_stage(
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), stage_v: wp.array(dtype=wp.vec3),
    slots: wp.array2d(dtype=int), weights: wp.array2d(dtype=float),
    old_v: wp.array(dtype=wp.vec3), new_v: wp.array(dtype=wp.vec3),
    duration: wp.array(dtype=float), pic_rate: float,
    next_x: wp.array(dtype=wp.vec3), next_v: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    pic, dv = wp.vec3(0.0), wp.vec3(0.0)
    for a in range(8):
        slot = slots[p, a]
        pic += weights[p, a] * old_v[slot]
        dv += weights[p, a] * (new_v[slot] - old_v[slot])
    next_x[p] = x[p] + duration[0] * pic
    next_v[p] = v[p] + dv + duration[0] * pic_rate * (pic - stage_v[p])


@wp.kernel
def stage_grid_candidate(base_momentum: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
                         old_v: wp.array(dtype=wp.vec3), new_v: wp.array(dtype=wp.vec3),
                         candidate: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if mass[i] > 0.0:
        candidate[i] = base_momentum[i] / mass[i] + new_v[i] - old_v[i]


@wp.kernel
def add_stage_projection(new_v: wp.array(dtype=wp.vec3), candidate: wp.array(dtype=wp.vec3),
                         projected: wp.array(dtype=wp.vec3), result: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    result[i] = new_v[i] + projected[i] - candidate[i]


@wp.kernel
def timed_body_impulse(gravity: wp.vec3, mass: float, duration: wp.array(dtype=float),
                       impulse: wp.array(dtype=wp.vec3)):
    impulse[wp.tid()] = mass * duration[0] * gravity


@wp.kernel
def timed_p2g(
    v: wp.array(dtype=wp.vec3), stress: wp.array(dtype=wp.mat33), impulses: wp.array(dtype=wp.vec3),
    slots: wp.array2d(dtype=int), weights: wp.array2d(dtype=float), gradients: wp.array2d(dtype=wp.vec3),
    mass: float, volume: float, duration: wp.array(dtype=float), grid_mass: wp.array(dtype=float),
    momentum: wp.array(dtype=wp.vec3), grid_impulse: wp.array(dtype=wp.vec3),
):
    p, a = wp.tid()
    slot = slots[p, a]
    if slot >= 0:
        w = weights[p, a]
        wp.atomic_add(grid_mass, slot, mass * w)
        wp.atomic_add(momentum, slot, mass * w * v[p])
        wp.atomic_add(grid_impulse, slot, w * impulses[p] - duration[0] * volume * stress[p] * gradients[p, a])


@wp.kernel
def timed_gather_remap(
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), slots: wp.array2d(dtype=int),
    weights: wp.array2d(dtype=float), old_v: wp.array(dtype=wp.vec3), new_v: wp.array(dtype=wp.vec3),
    duration: wp.array(dtype=float), flip: float, reference_dt: float, mass: float,
    next_x: wp.array(dtype=wp.vec3), next_v: wp.array(dtype=wp.vec3), momentum: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    xp, vp = gather_at(p, x, v, slots, weights, old_v, new_v, duration, flip, reference_dt)
    next_x[p] = xp
    next_v[p] = vp
    for a in range(8):
        wp.atomic_add(momentum, slots[p, a], mass * weights[p, a] * vp)


@wp.kernel
def timed_deformation(
    F: wp.array(dtype=wp.mat33), slots: wp.array2d(dtype=int), gradients: wp.array2d(dtype=wp.vec3),
    grid_mass: wp.array(dtype=float), remapped: wp.array(dtype=wp.vec3), duration: wp.array(dtype=float),
    next_F: wp.array(dtype=wp.mat33),
):
    p = wp.tid()
    rate = wp.mat33(0.0)
    for a in range(8):
        slot = slots[p, a]
        if grid_mass[slot] > 0.0:
            rate += wp.outer(remapped[slot] / grid_mass[slot], gradients[p, a])
    next_F[p] = F[p] + duration[0] * rate


@wp.kernel
def ordered_mass_momentum(
    offsets: wp.array(dtype=int), particles: wp.array(dtype=int), corners: wp.array(dtype=int),
    weights: wp.array2d(dtype=float), v: wp.array(dtype=wp.vec3), mass: float,
    grid_mass: wp.array(dtype=float), momentum: wp.array(dtype=wp.vec3),
):
    slot = wp.tid()
    total_mass = float(0.0)
    total_momentum = wp.vec3(0.0)
    for edge in range(offsets[slot], offsets[slot + 1]):
        p, a = particles[edge], corners[edge]
        weighted_mass = mass * weights[p, a]
        total_mass += weighted_mass
        total_momentum += weighted_mass * v[p]
    grid_mass[slot] = total_mass
    momentum[slot] = total_momentum


@wp.kernel
def gather_flow(
    slots: wp.array2d(dtype=int), weights: wp.array2d(dtype=float),
    mass: wp.array(dtype=float), momentum: wp.array(dtype=wp.vec3),
    flow: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    velocity = wp.vec3(0.0)
    for a in range(8):
        slot = slots[p, a]
        if mass[slot] > 0.0:
            velocity += weights[p, a] * momentum[slot] / mass[slot]
    flow[p] = velocity


@wp.kernel
def ordered_remap(
    offsets: wp.array(dtype=int), particles: wp.array(dtype=int), corners: wp.array(dtype=int),
    weights: wp.array2d(dtype=float), v: wp.array(dtype=wp.vec3), mass: float,
    momentum: wp.array(dtype=wp.vec3),
):
    slot = wp.tid()
    value = wp.vec3(0.0)
    for edge in range(offsets[slot], offsets[slot + 1]):
        p, a = particles[edge], corners[edge]
        value += mass * weights[p, a] * v[p]
    momentum[slot] = value


@wp.kernel
def couple_fixed(
    offsets: wp.array(dtype=int),
    sources: wp.array(dtype=int),
    coefficients: wp.array(dtype=float),
    state: wp.array(dtype=wp.vec3),
    result: wp.array(dtype=wp.vec3),
):
    a = wp.tid()
    value = wp.vec3(0.0)
    for edge in range(offsets[a], offsets[a + 1]):
        value += coefficients[edge] * state[sources[edge]]
    result[a] = value


@wp.kernel
def add_grid_load(internal: wp.array(dtype=wp.vec3), external: wp.array(dtype=wp.vec3),
                  result: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    result[i] = internal[i] + external[i]


class MaterialTrajectory(ElasticTrajectory):
    """Fixed reference stencils; discrete cut codes supplied for each step.

    Fp is dimensionless and yield_stress is in Pa. Positive determinants are
    required for both F and Fp. Each step retains its own plastic state.
    """

    def __init__(self, config, device="cuda:0", requires_grad=True, *, ordered_transfers=False, precision="float32"):
        if precision not in ("float32", "float64"):
            raise ValueError("Material precision must be float32 or float64")
        self.precision = precision
        super().__init__(replace(config, yield_stress=0.0), device, requires_grad)
        self.config = config
        self.ordered_transfers = ordered_transfers
        self.yield_stress = self.array([config.yield_stress], float)
        self.nodes = self.slots.numpy()
        self.uncut = self.topology(np.zeros(self.n, dtype=np.int64))
        self.duration = self.array([config.dt], float, False)

    def array(self, values, dtype, requires_grad=None):
        if self.precision == "float64":
            from .precision import double_type
            dtype = double_type(dtype)
        return super().array(values, dtype, requires_grad)

    def zeros(self, count, dtype, requires_grad=None):
        if self.precision == "float64":
            from .precision import double_type
            dtype = double_type(dtype)
        return super().zeros(count, dtype, requires_grad)

    def launch(self, kernel, dim, **kwargs):
        if self.precision == "float64":
            from .precision import launch_double
            return launch_double(kernel, dim, **kwargs)
        return wp.launch(kernel, dim, **kwargs)

    def state(self, x=None, v=None, F=None, Fp=None, stress=None):
        base = super().state(x, v, F)
        Fp = np.tile(np.eye(3), (self.n, 1, 1)) if Fp is None else np.asarray(Fp)
        if Fp.shape != (self.n, 3, 3) or not np.isfinite(Fp).all() or np.any(np.linalg.det(Fp) <= 0):
            raise ValueError("Plastic state must be finite with positive determinants and matching shape")
        return PlasticState(base.x, base.v, base.F, self.array(Fp, wp.mat33),
                            stress=None if stress is None else self.array(stress, wp.mat33))

    def topology(self, codes):
        codes = np.asarray(codes, dtype=np.int64)
        if codes.shape != (self.n,) or np.any(codes < 0):
            raise ValueError("Cut codes must be nonnegative and match the particle count")
        keys, slots = np.unique(codes[:, None] * self.node_count + self.nodes, return_inverse=True)
        # Eq. 8 is linear: all masses depend only on reference weights and codes.
        masses = np.bincount(slots.ravel(), weights=self.weights.numpy().ravel() * self.mass)
        groups = [[] for _ in range(self.node_count)]
        for index, node in enumerate(keys % self.node_count):
            groups[node].append(index)
        offsets, sources, coefficients = [0], [], []
        for a, key in enumerate(keys):
            group = groups[key % self.node_count]
            total = sum(masses[b] for b in group)
            diagonal = 1.0
            for b in group:
                ca, cb = int(key // self.node_count), int(keys[b] // self.node_count)
                compatible = True
                while ca or cb:
                    da, db = ca % 3, cb % 3
                    compatible = compatible and (da == 0 or db == 0 or da == db)
                    ca, cb = ca // 3, cb // 3
                if b != a and compatible:
                    sources.append(b)
                    coefficients.append(masses[a] / total)
                    diagonal -= masses[b] / total
            sources.append(a)
            coefficients.append(diagonal)
            offsets.append(len(sources))
        order = np.argsort(slots.ravel(), kind="stable")
        particle_offsets = np.searchsorted(slots.ravel()[order], np.arange(len(keys) + 1))
        return FixedTopology(
            self.array(slots.reshape(self.n, 8), int, False),
            self.array(keys, wp.int64, False), self.array(offsets, int, False),
            self.array(sources, int, False), self.array(coefficients, float, False), len(keys),
            self.array(particle_offsets, int, False), self.array(order // 8, int, False),
            self.array(order % 8, int, False),
        )

    def couple(self, grid, source):
        result = self.zeros(grid.count, wp.vec3)
        self.launch(couple_fixed, grid.count,
                  inputs=[grid.offsets, grid.sources, grid.coefficients, source],
                  outputs=[result], device=self.device)
        return result

    def flow_velocity(self, velocity, topology):
        """Velocity conjugate to particle forces scattered through the coupled grid."""
        grid = topology
        mass = self.zeros(grid.count, float, False)
        momentum = self.zeros(grid.count, wp.vec3)
        self.launch(ordered_mass_momentum, grid.count,
                    inputs=[grid.particle_offsets, grid.particles, grid.corners,
                            self.weights, velocity, self.mass],
                    outputs=[mass, momentum], device=self.device)
        flow = self.zeros(self.n, wp.vec3)
        self.launch(gather_flow, self.n,
                    inputs=[grid.slots, self.weights, mass, self.couple(grid, momentum)],
                    outputs=[flow], device=self.device)
        return flow

    def correct_displacement(self, state, free, displacement, topology=None, duration=None,
                             fixed_yielded=None):
        """Complete a first-order MUSL step from a grid-induced dt * delta_v in metres.

        free is the step from state before the additional grid impulse, with the
        same topology and duration. Plasticity is updated from state's Fp.
        """
        grid = self.uncut if topology is None else topology
        duration = self.duration if duration is None else duration
        if displacement.shape != (self.n,):
            raise ValueError('Displacement must have one vector per particle')
        result = PlasticState(self.zeros(self.n, wp.vec3), self.zeros(self.n, wp.vec3),
                              self.zeros(self.n, wp.mat33), self.zeros(self.n, wp.mat33),
                              self.zeros(self.n, int, False), self.zeros(self.n, wp.mat33))
        self.launch(correct_kinematics, self.n,
                    inputs=[free.x, free.v, displacement, duration],
                    outputs=[result.x, result.v], device=self.device)
        mass = self.zeros(grid.count, float, False)
        weighted_displacement = self.zeros(grid.count, wp.vec3)
        self.launch(ordered_mass_momentum, grid.count,
                    inputs=[grid.particle_offsets, grid.particles, grid.corners,
                            self.weights, displacement, self.mass],
                    outputs=[mass, weighted_displacement], device=self.device)
        self.launch(timed_deformation, self.n,
                    inputs=[free.F, grid.slots, self.gradients, mass,
                            self.couple(grid, weighted_displacement), self.array([1.0], float, False)],
                    outputs=[result.F], device=self.device)
        if fixed_yielded is None:
            fixed_yielded = self.zeros(self.n, int, False)
            fixed_yielded.fill_(-1)
        errors = self.zeros(1, int, False)
        self.launch(project_plastic, self.n,
                    inputs=[result.F, state.Fp, self.young, self.yield_stress,
                            self.config.poisson_ratio, fixed_yielded],
                    outputs=[result.Fp, result.yielded, result.stress, errors], device=self.device)
        if int(errors.numpy()[0]):
            raise FloatingPointError('Plastic return mapping failed to converge')
        return result

    def step(self, state, gravity=(0.0, 0.0, -9.81), impulse=None, topology=None, fixed_yielded=None,
             grid_transform=None, duration=None, stage_base=None, trace=None, grid_load=None):
        """grid_load is an impulse in N s per topology slot, applied before cut-field coupling."""
        cfg = self.config
        duration = self.duration if duration is None else duration
        grid = self.uncut if topology is None else topology
        stress = state.stress
        grid_mass = self.zeros(grid.count, float, False)
        momentum = self.zeros(grid.count, wp.vec3)
        grid_impulse = self.zeros(grid.count, wp.vec3)
        old_v = self.zeros(grid.count, wp.vec3)
        new_v = self.zeros(grid.count, wp.vec3)
        remapped = self.zeros(grid.count, wp.vec3)
        result = PlasticState(self.zeros(self.n, wp.vec3), self.zeros(self.n, wp.vec3),
                              self.zeros(self.n, wp.mat33), self.zeros(self.n, wp.mat33),
                              self.zeros(self.n, int, False), self.zeros(self.n, wp.mat33))
        if fixed_yielded is None:
            fixed_yielded = self.zeros(self.n, int, False)
            fixed_yielded.fill_(-1)
        if stress is None:
            stress = self.zeros(self.n, wp.mat33)
            self.launch(compute_stress, self.n, inputs=[state.F, state.Fp, self.young, cfg.poisson_ratio],
                      outputs=[stress], device=self.device)
        if impulse is None:
            impulse = self.zeros(self.n, wp.vec3)
            self.launch(timed_body_impulse, self.n, inputs=[tuple(gravity), self.mass, duration],
                      outputs=[impulse], device=self.device)
        if self.ordered_transfers:
            self.launch(ordered_p2g, grid.count,
                      inputs=[grid.particle_offsets, grid.particles, grid.corners, self.weights, self.gradients,
                              state.v, stress, impulse, self.mass, self.volume, duration],
                      outputs=[grid_mass, momentum, grid_impulse], device=self.device)
        else:
            self.launch(timed_p2g, (self.n, 8),
                      inputs=[state.v, stress, impulse, grid.slots, self.weights, self.gradients,
                              self.mass, self.volume, duration],
                      outputs=[grid_mass, momentum, grid_impulse], device=self.device)
        if grid_load is not None:
            if grid_load.shape != (grid.count,):
                raise ValueError('Grid load must have one impulse per topology slot')
            combined = self.zeros(grid.count, wp.vec3)
            self.launch(add_grid_load, grid.count, inputs=[grid_impulse, grid_load],
                        outputs=[combined], device=self.device)
            grid_impulse = combined
        coupled_momentum = self.couple(grid, momentum)
        coupled_impulse = self.couple(grid, grid_impulse)
        self.launch(m.grid_vel, grid.count, inputs=[grid_mass, coupled_momentum, coupled_impulse],
                  outputs=[old_v, new_v], device=self.device)
        unconstrained_v = new_v
        if grid_transform is not None:
            if stage_base is None:
                new_v = grid_transform(grid, grid_mass, new_v)
            else:
                # RK stages add their force increment to the segment's base
                # velocity. The unilateral projection must constrain that sum.
                base_momentum = self.zeros(grid.count, wp.vec3)
                self.launch(ordered_remap, grid.count,
                            inputs=[grid.particle_offsets, grid.particles, grid.corners, self.weights,
                                    stage_base.v, self.mass], outputs=[base_momentum], device=self.device)
                candidate = self.zeros(grid.count, wp.vec3)
                self.launch(stage_grid_candidate, grid.count,
                            inputs=[self.couple(grid, base_momentum), grid_mass, old_v, new_v],
                            outputs=[candidate], device=self.device)
                projected = grid_transform(grid, grid_mass, candidate)
                corrected = self.zeros(grid.count, wp.vec3)
                self.launch(add_stage_projection, grid.count, inputs=[new_v, candidate, projected],
                            outputs=[corrected], device=self.device)
                new_v = corrected
        if stage_base is not None:
            self.launch(gather_stage, self.n,
                        inputs=[stage_base.x, stage_base.v, state.v, grid.slots, self.weights,
                                old_v, new_v, duration,
                                (1 - cfg.flip_fraction) / (cfg.flip_reference_dt or cfg.dt)],
                        outputs=[result.x, result.v], device=self.device)
            coupled = coupled_momentum
        elif self.ordered_transfers:
            self.launch(gather_particles, self.n,
                      inputs=[state.x, state.v, grid.slots, self.weights, old_v, new_v,
                              duration, cfg.flip_fraction, cfg.flip_reference_dt or cfg.dt],
                      outputs=[result.x, result.v], device=self.device)
            self.launch(ordered_remap, grid.count,
                      inputs=[grid.particle_offsets, grid.particles, grid.corners, self.weights, result.v, self.mass],
                      outputs=[remapped], device=self.device)
        else:
            self.launch(timed_gather_remap, self.n,
                      inputs=[state.x, state.v, grid.slots, self.weights, old_v, new_v,
                              duration, cfg.flip_fraction, cfg.flip_reference_dt or cfg.dt, self.mass],
                      outputs=[result.x, result.v, remapped], device=self.device)
        if stage_base is None:
            coupled = self.couple(grid, remapped)
        base = state if stage_base is None else stage_base
        self.launch(timed_deformation, self.n,
                  inputs=[base.F, grid.slots, self.gradients, grid_mass, coupled, duration],
                  outputs=[result.F], device=self.device)
        plastic_errors = self.zeros(1, int, False)
        self.launch(project_plastic, self.n,
                  inputs=[result.F, base.Fp, self.young, self.yield_stress, cfg.poisson_ratio, fixed_yielded],
                  outputs=[result.Fp, result.yielded, result.stress, plastic_errors], device=self.device)
        if int(plastic_errors.numpy()[0]):
            raise FloatingPointError('Plastic return mapping failed to converge')
        if trace is not None:
            trace.update(grid=grid, old_v=old_v, new_v=new_v, unconstrained_v=unconstrained_v)
        return result

    def checkpoint_vjp(self, *args, **kwargs):
        raise NotImplementedError("Material trajectories currently retain the full Tape, including Fp")
