"""Hencky plasticity and fixed cut-field transfers with independent step states."""

from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from . import material as m
from .differentiable import ElasticTrajectory, State, body_impulse, elastic_stress
from .differentiable import gather_remap, update_deformation


@wp.func
def hencky_branch(f: wp.mat33, limit: float, branch: int):
    U, s, V = wp.svd3(f)
    e = wp.vec3(wp.log(s[0]), wp.log(s[1]), wp.log(s[2]))
    mean = (e[0] + e[1] + e[2]) / 3.0
    dev = e - wp.vec3(mean)
    norm = wp.length(dev)
    if branch == 0 or (branch < 0 and norm <= limit):
        return f
    projected = e - (1.0 - limit / norm) * dev
    t = wp.vec3(wp.exp(projected[0]), wp.exp(projected[1]), wp.exp(projected[2]))
    return U * wp.diag(t) * wp.transpose(V)


@wp.func
def hencky_vjp(f: wp.mat33, limit: float, branch: int, adj: wp.mat33):
    U, s, V = wp.svd3(f)
    e = wp.vec3(wp.log(s[0]), wp.log(s[1]), wp.log(s[2]))
    mean = (e[0] + e[1] + e[2]) / 3.0
    dev = e - wp.vec3(mean)
    norm = wp.length(dev)
    if branch == 0 or (branch < 0 and norm <= limit):
        return adj, 0.0
    else:
        a = limit / norm
        projected = e - (1.0 - a) * dev
        t = wp.vec3(wp.exp(projected[0]), wp.exp(projected[1]), wp.exp(projected[2]))
        local = wp.transpose(U) * adj * V
        result = wp.mat33(0.0)
        limit_gradient = float(0.0)
        for i in range(3):
            for j in range(3):
                delta = float(0.0)
                if i == j:
                    delta = 1.0
                derivative = 1.0 / 3.0 + a * (delta - 1.0 / 3.0 - dev[i] * dev[j] / (norm * norm))
                result[j, j] += local[i, i] * t[i] * derivative / s[j]
                if i != j:
                    # The divided difference has a finite limit at repeated stretches.
                    symmetric = a * (t[i] + t[j]) / (s[i] + s[j])
                    if wp.abs(s[i] - s[j]) > 1.0e-4 * wp.max(s[i], s[j]):
                        symmetric = (t[i] - t[j]) / (s[i] - s[j])
                    skew = (t[i] + t[j]) / (s[i] + s[j])
                    result[i, j] = 0.5 * (symmetric * (local[i, j] + local[j, i])
                                          + skew * (local[i, j] - local[j, i]))
            limit_gradient += local[i, i] * t[i] * dev[i] / norm
        return U * result * wp.transpose(V), limit_gradient


@wp.func_grad(hencky_branch)
def hencky_branch_grad(f: wp.mat33, limit: float, branch: int, adj: wp.mat33):
    df, dl = hencky_vjp(f, limit, branch, adj)
    wp.adjoint[f] += df
    wp.adjoint[limit] += dl


@wp.func
def constitutive_response(F: wp.mat33, Fp: wp.mat33, mu: float, lam: float,
                          limit: float, branch: int):
    U, s, V = wp.svd3(F * wp.inverse(Fp))
    e = wp.vec3(wp.log(s[0]), wp.log(s[1]), wp.log(s[2]))
    dev = e - wp.vec3((e[0] + e[1] + e[2]) / 3.0)
    norm = wp.length(dev)
    next_Fp = Fp
    if branch == 1 or (branch < 0 and norm > limit):
        ep = e - (1.0 - limit / norm) * dev
        s = wp.vec3(wp.exp(ep[0]), wp.exp(ep[1]), wp.exp(ep[2]))
        projected = U * wp.diag(s) * wp.transpose(V)
        next_Fp = wp.inverse(projected) * F
    stress = m.corotated_stress(U, s, V, wp.inverse(next_Fp), mu, lam)
    return next_Fp, stress


@wp.func_grad(constitutive_response)
def constitutive_response_grad(F: wp.mat33, Fp: wp.mat33, mu: float, lam: float,
                               limit: float, branch: int, adj_Fp: wp.mat33, adj_stress: wp.mat33):
    old_inverse = wp.inverse(Fp)
    f = F * old_inverse
    U, s, V = wp.svd3(f)
    e = wp.vec3(wp.log(s[0]), wp.log(s[1]), wp.log(s[2]))
    dev = e - wp.vec3((e[0] + e[1] + e[2]) / 3.0)
    active = branch == 1 or (branch < 0 and wp.length(dev) > limit)
    projected = f
    next_Fp = Fp
    if active:
        projected = hencky_branch(f, limit, 1)
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
        adj_f, adj_limit = hencky_vjp(f, limit, 1, adj_projected)
        wp.adjoint[limit] += adj_limit
        wp.adjoint[F] += adj_f * wp.transpose(old_inverse)
        wp.adjoint[Fp] -= wp.transpose(old_inverse) * wp.transpose(F) * adj_f * wp.transpose(old_inverse)
    else:
        wp.adjoint[F] += adj_projected * wp.transpose(old_inverse)
        wp.adjoint[Fp] += adj_plastic - wp.transpose(old_inverse) * wp.transpose(F) * adj_projected * wp.transpose(old_inverse)


@wp.func
def hencky_project(f: wp.mat33, limit: float):
    return hencky_branch(f, limit, -1)


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
):
    p = wp.tid()
    mu = float(wp.float64(young[0]) / (wp.float64(2.0) * (wp.float64(1.0) + poisson)))
    lam = float(wp.float64(young[0]) * poisson / ((wp.float64(1.0) + poisson)
                                                * (wp.float64(1.0) - wp.float64(2.0) * poisson)))
    limit = yield_stress[0] / (2.0 * mu) * wp.sqrt(2.0 / 3.0)
    branch = fixed_yielded[p]
    if yield_stress[0] <= 0.0:
        branch = 0
    plastic, stress = constitutive_response(F[p], Fp[p], mu, lam, limit, branch)
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
    mass: float, volume: float, dt: float, grid_mass: wp.array(dtype=float),
    momentum: wp.array(dtype=wp.vec3), grid_impulse: wp.array(dtype=wp.vec3),
):
    slot = wp.tid()
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


@wp.kernel
def gather_particles(
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), slots: wp.array2d(dtype=int),
    weights: wp.array2d(dtype=float), old_v: wp.array(dtype=wp.vec3), new_v: wp.array(dtype=wp.vec3),
    dt: float, flip: float, next_x: wp.array(dtype=wp.vec3), next_v: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    pic, dv = wp.vec3(0.0), wp.vec3(0.0)
    for a in range(8):
        slot = slots[p, a]
        pic += weights[p, a] * new_v[slot]
        dv += weights[p, a] * (new_v[slot] - old_v[slot])
    next_x[p] = x[p] + dt * pic
    next_v[p] = (1.0 - flip) * pic + flip * (v[p] + dv)


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

    def step(self, state, gravity=(0.0, 0.0, -9.81), impulse=None, topology=None, fixed_yielded=None):
        cfg = self.config
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
            impulse = self.zeros(self.n, wp.vec3, False)
            self.launch(body_impulse, self.n, inputs=[wp.vec3(*gravity), self.mass, cfg.dt],
                      outputs=[impulse], device=self.device)
        if self.ordered_transfers:
            self.launch(ordered_p2g, grid.count,
                      inputs=[grid.particle_offsets, grid.particles, grid.corners, self.weights, self.gradients,
                              state.v, stress, impulse, self.mass, self.volume, cfg.dt],
                      outputs=[grid_mass, momentum, grid_impulse], device=self.device)
        else:
            self.launch(m.p2g, (self.n, 8),
                      inputs=[state.v, stress, impulse, grid.slots, self.weights, self.gradients,
                              self.mass, self.volume, cfg.dt],
                      outputs=[grid_mass, momentum, grid_impulse], device=self.device)
        coupled_momentum = self.couple(grid, momentum)
        coupled_impulse = self.couple(grid, grid_impulse)
        self.launch(m.grid_vel, grid.count, inputs=[grid_mass, coupled_momentum, coupled_impulse],
                  outputs=[old_v, new_v], device=self.device)
        if self.ordered_transfers:
            self.launch(gather_particles, self.n,
                      inputs=[state.x, state.v, grid.slots, self.weights, old_v, new_v, cfg.dt, cfg.flip_fraction],
                      outputs=[result.x, result.v], device=self.device)
            self.launch(ordered_remap, grid.count,
                      inputs=[grid.particle_offsets, grid.particles, grid.corners, self.weights, result.v, self.mass],
                      outputs=[remapped], device=self.device)
        else:
            self.launch(gather_remap, self.n,
                      inputs=[state.x, state.v, grid.slots, self.weights, old_v, new_v,
                              cfg.dt, cfg.flip_fraction, self.mass],
                      outputs=[result.x, result.v, remapped], device=self.device)
        coupled = self.couple(grid, remapped)
        self.launch(update_deformation, self.n,
                  inputs=[state.F, grid.slots, self.gradients, grid_mass, coupled, cfg.dt],
                  outputs=[result.F], device=self.device)
        self.launch(project_plastic, self.n,
                  inputs=[result.F, state.Fp, self.young, self.yield_stress, cfg.poisson_ratio, fixed_yielded],
                  outputs=[result.Fp, result.yielded, result.stress], device=self.device)
        return result

    def checkpoint_vjp(self, *args, **kwargs):
        raise NotImplementedError("Material trajectories currently retain the full Tape, including Fp")
