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


@wp.func_grad(hencky_branch)
def hencky_branch_grad(f: wp.mat33, limit: float, branch: int, adj: wp.mat33):
    U, s, V = wp.svd3(f)
    e = wp.vec3(wp.log(s[0]), wp.log(s[1]), wp.log(s[2]))
    mean = (e[0] + e[1] + e[2]) / 3.0
    dev = e - wp.vec3(mean)
    norm = wp.length(dev)
    if branch == 0 or (branch < 0 and norm <= limit):
        wp.adjoint[f] += adj
    else:
        a = limit / norm
        projected = e - (1.0 - a) * dev
        t = wp.vec3(wp.exp(projected[0]), wp.exp(projected[1]), wp.exp(projected[2]))
        local = wp.transpose(U) * adj * V
        result = wp.mat33(0.0)
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
            wp.adjoint[limit] += local[i, i] * t[i] * dev[i] / norm
        wp.adjoint[f] += U * result * wp.transpose(V)


@wp.func
def hencky_project(f: wp.mat33, limit: float):
    return hencky_branch(f, limit, -1)


@wp.kernel
def compute_stress(
    F: wp.array(dtype=wp.mat33),
    Fp: wp.array(dtype=wp.mat33),
    young: wp.array(dtype=float),
    poisson: float,
    stress: wp.array(dtype=wp.mat33),
):
    p = wp.tid()
    mu = young[0] / (2.0 * (1.0 + poisson))
    lam = young[0] * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    inverse = wp.inverse(Fp[p])
    stress[p] = elastic_stress(F[p] * inverse, mu, lam) * wp.transpose(inverse)


@wp.kernel
def project_plastic(
    F: wp.array(dtype=wp.mat33),
    Fp: wp.array(dtype=wp.mat33),
    young: wp.array(dtype=float),
    yield_stress: wp.array(dtype=float),
    poisson: float,
    fixed_yielded: wp.array(dtype=int),
    next_Fp: wp.array(dtype=wp.mat33),
    yielded: wp.array(dtype=int),
):
    p = wp.tid()
    result = Fp[p]
    yielded[p] = 0
    if yield_stress[0] > 0.0:
        f = F[p] * wp.inverse(Fp[p])
        mu = young[0] / (2.0 * (1.0 + poisson))
        limit = yield_stress[0] / (2.0 * mu) * wp.sqrt(2.0 / 3.0)
        projected = hencky_branch(f, limit, fixed_yielded[p])
        # The elastic branch returns f exactly and must leave Fp unchanged.
        if wp.ddot(projected - f, projected - f) > 0.0:
            result = wp.inverse(projected) * F[p]
            yielded[p] = 1
    next_Fp[p] = result


@dataclass
class PlasticState(State):
    Fp: wp.array
    yielded: wp.array | None = None


@dataclass
class FixedTopology:
    slots: wp.array
    keys: wp.array
    offsets: wp.array
    sources: wp.array
    coefficients: wp.array
    count: int


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

    def __init__(self, config, device="cuda:0", requires_grad=True):
        super().__init__(replace(config, yield_stress=0.0), device, requires_grad)
        self.config = config
        self.yield_stress = self.array([config.yield_stress], float)
        self.nodes = self.slots.numpy()
        self.uncut = self.topology(np.zeros(self.n, dtype=np.int64))

    def state(self, x=None, v=None, F=None, Fp=None):
        base = super().state(x, v, F)
        Fp = np.tile(np.eye(3), (self.n, 1, 1)) if Fp is None else np.asarray(Fp)
        if Fp.shape != (self.n, 3, 3) or not np.isfinite(Fp).all() or np.any(np.linalg.det(Fp) <= 0):
            raise ValueError("Plastic state must be finite with positive determinants and matching shape")
        return PlasticState(base.x, base.v, base.F, self.array(Fp, wp.mat33))

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
        return FixedTopology(
            self.array(slots.reshape(self.n, 8), int, False),
            self.array(keys, wp.int64, False), self.array(offsets, int, False),
            self.array(sources, int, False), self.array(coefficients, float, False), len(keys),
        )

    def couple(self, grid, source):
        result = self.zeros(grid.count, wp.vec3)
        wp.launch(couple_fixed, grid.count,
                  inputs=[grid.offsets, grid.sources, grid.coefficients, source],
                  outputs=[result], device=self.device)
        return result

    def step(self, state, gravity=(0.0, 0.0, -9.81), impulse=None, topology=None, fixed_yielded=None):
        cfg = self.config
        grid = self.uncut if topology is None else topology
        stress = self.zeros(self.n, wp.mat33)
        grid_mass = self.zeros(grid.count, float, False)
        momentum = self.zeros(grid.count, wp.vec3)
        grid_impulse = self.zeros(grid.count, wp.vec3)
        old_v = self.zeros(grid.count, wp.vec3)
        new_v = self.zeros(grid.count, wp.vec3)
        remapped = self.zeros(grid.count, wp.vec3)
        result = PlasticState(self.zeros(self.n, wp.vec3), self.zeros(self.n, wp.vec3),
                              self.zeros(self.n, wp.mat33), self.zeros(self.n, wp.mat33),
                              self.zeros(self.n, int, False))
        if fixed_yielded is None:
            fixed_yielded = self.zeros(self.n, int, False)
            fixed_yielded.fill_(-1)
        wp.launch(compute_stress, self.n, inputs=[state.F, state.Fp, self.young, cfg.poisson_ratio],
                  outputs=[stress], device=self.device)
        if impulse is None:
            impulse = self.zeros(self.n, wp.vec3, False)
            wp.launch(body_impulse, self.n, inputs=[wp.vec3(*gravity), self.mass, cfg.dt],
                      outputs=[impulse], device=self.device)
        wp.launch(m.p2g, (self.n, 8),
                  inputs=[state.v, stress, impulse, grid.slots, self.weights, self.gradients,
                          self.mass, self.volume, cfg.dt],
                  outputs=[grid_mass, momentum, grid_impulse], device=self.device)
        coupled_momentum = self.couple(grid, momentum)
        coupled_impulse = self.couple(grid, grid_impulse)
        wp.launch(m.grid_vel, grid.count, inputs=[grid_mass, coupled_momentum, coupled_impulse],
                  outputs=[old_v, new_v], device=self.device)
        wp.launch(gather_remap, self.n,
                  inputs=[state.x, state.v, grid.slots, self.weights, old_v, new_v,
                          cfg.dt, cfg.flip_fraction, self.mass],
                  outputs=[result.x, result.v, remapped], device=self.device)
        coupled = self.couple(grid, remapped)
        wp.launch(update_deformation, self.n,
                  inputs=[state.F, grid.slots, self.gradients, grid_mass, coupled, cfg.dt],
                  outputs=[result.F], device=self.device)
        wp.launch(project_plastic, self.n,
                  inputs=[result.F, state.Fp, self.young, self.yield_stress, cfg.poisson_ratio, fixed_yielded],
                  outputs=[result.Fp, result.yielded], device=self.device)
        return result

    def checkpoint_vjp(self, *args, **kwargs):
        raise NotImplementedError("Material trajectories currently retain the full Tape, including Fp")
