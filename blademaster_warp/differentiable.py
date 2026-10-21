"""Elastic TLMPM with immutable step states and fixed reference-grid connectivity."""

from dataclasses import dataclass
from time import perf_counter
import numpy as np
import warp as wp

from . import material as m
from .geometry import points, stencil
from .solver import Config


@wp.func
def elastic_stress(F: wp.mat33, mu: float, lam: float):
    U, sigma, V = wp.svd3(F)
    return m.corotated_stress(U, sigma, V, wp.identity(n=3, dtype=float), mu, lam)


@wp.func_grad(elastic_stress)
def elastic_stress_grad(F: wp.mat33, mu: float, lam: float, adj: wp.mat33):
    # The polar rotation derivative divides by sigma_i + sigma_j, which stays
    # regular at repeated positive stretches (including F = I).
    U, sigma, V = wp.svd3(F)
    rotation = U * wp.transpose(V)
    local = wp.transpose(U) * adj * V
    skew = wp.mat33(0.0)
    for i in range(3):
        for j in range(3):
            if i != j:
                skew[i, j] = (local[i, j] - local[j, i]) / (sigma[i] + sigma[j])
    inverse_t = wp.transpose(wp.inverse(F))
    J = wp.determinant(F)
    contraction = wp.ddot(adj, inverse_t)
    wp.adjoint[F] += (
        2.0 * mu * (adj - U * skew * wp.transpose(V))
        + lam * (2.0 * J - 1.0) * J * contraction * inverse_t
        - lam * (J - 1.0) * J * inverse_t * wp.transpose(adj) * inverse_t
    )
    wp.adjoint[mu] += 2.0 * wp.ddot(adj, F - rotation)
    wp.adjoint[lam] += (J - 1.0) * J * contraction


@wp.kernel
def compute_stress(
    F: wp.array(dtype=wp.mat33),
    young: wp.array(dtype=float),
    poisson: float,
    stress: wp.array(dtype=wp.mat33),
):
    mu = young[0] / (2.0 * (1.0 + poisson))
    lam = young[0] * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    stress[wp.tid()] = elastic_stress(F[wp.tid()], mu, lam)


@wp.kernel
def body_impulse(gravity: wp.vec3, mass: float, dt: float, impulse: wp.array(dtype=wp.vec3)):
    impulse[wp.tid()] = mass * dt * gravity


@wp.kernel
def gather_remap(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    slots: wp.array2d(dtype=int),
    weights: wp.array2d(dtype=float),
    old_v: wp.array(dtype=wp.vec3),
    new_v: wp.array(dtype=wp.vec3),
    dt: float,
    flip: float,
    mass: float,
    next_x: wp.array(dtype=wp.vec3),
    next_v: wp.array(dtype=wp.vec3),
    remapped: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    pic = wp.vec3(0.0)
    dv = wp.vec3(0.0)
    for a in range(8):
        s = slots[p, a]
        w = weights[p, a]
        pic += w * new_v[s]
        dv += w * (new_v[s] - old_v[s])
    velocity = (1.0 - flip) * pic + flip * (v[p] + dv)
    next_x[p] = x[p] + dt * pic
    next_v[p] = velocity
    for a in range(8):
        wp.atomic_add(remapped, slots[p, a], mass * weights[p, a] * velocity)


@wp.kernel
def update_deformation(
    F: wp.array(dtype=wp.mat33),
    slots: wp.array2d(dtype=int),
    gradients: wp.array2d(dtype=wp.vec3),
    grid_mass: wp.array(dtype=float),
    remapped: wp.array(dtype=wp.vec3),
    dt: float,
    next_F: wp.array(dtype=wp.mat33),
):
    p = wp.tid()
    rate = wp.mat33(0.0)
    for a in range(8):
        s = slots[p, a]
        if grid_mass[s] > 0.0:
            rate += wp.outer(remapped[s] / grid_mass[s], gradients[p, a])
    next_F[p] = F[p] + dt * rate


@dataclass
class State:
    x: wp.array
    v: wp.array
    F: wp.array


class ElasticTrajectory:
    """Uncut, purely elastic material; positions in m, velocities in m/s, E in Pa.

    Each step owns its particle and grid arrays. Reference geometry, density,
    Poisson ratio and time step are constants during differentiation.
    """

    def __init__(self, config=Config(), device="cuda:0", requires_grad=True):
        if config.yield_stress or config.dynamic_viscosity or config.maxwell_tau:
            raise ValueError("ElasticTrajectory requires zero plasticity and viscosity")
        self.config = config
        self.device = wp.get_device(device)
        self.requires_grad = requires_grad
        self.reference, origin, cells = points(
            config.size, config.grid_spacing, config.shape, config.sampling_seed
        )
        nodes, weights, gradients = stencil(self.reference, origin, config.grid_spacing, cells)
        self.n = len(self.reference)
        self.node_count = int(np.prod(cells + 1))
        self.volume = (config.grid_spacing / 2) ** 3
        self.mass = config.density * self.volume
        self.slots = self.array(nodes, int, requires_grad=False)
        self.weights = self.array(weights, float, requires_grad=False)
        self.gradients = self.array(gradients, wp.vec3, requires_grad=False)
        self.young = self.array([config.young_modulus], float)

    def array(self, values, dtype, requires_grad=None):
        return wp.array(
            np.asarray(values), dtype=dtype, device=self.device,
            requires_grad=self.requires_grad if requires_grad is None else requires_grad,
        )

    def zeros(self, count, dtype, requires_grad=None):
        return wp.zeros(
            count, dtype=dtype, device=self.device,
            requires_grad=self.requires_grad if requires_grad is None else requires_grad,
        )

    def state(self, x=None, v=None, F=None):
        x = self.reference if x is None else np.asarray(x)
        v = np.zeros((self.n, 3)) if v is None else np.asarray(v)
        F = np.tile(np.eye(3), (self.n, 1, 1)) if F is None else np.asarray(F)
        if x.shape != (self.n, 3) or v.shape != (self.n, 3) or F.shape != (self.n, 3, 3):
            raise ValueError("State shapes must match the reference particles")
        if not all(np.isfinite(a).all() for a in (x, v, F)) or np.any(np.linalg.det(F) <= 0):
            raise ValueError("State must be finite with positive deformation determinants")
        return State(self.array(x, wp.vec3), self.array(v, wp.vec3), self.array(F, wp.mat33))

    def step(self, state, gravity=(0.0, 0.0, -9.81), impulse=None):
        cfg = self.config
        stress = self.zeros(self.n, wp.mat33)
        grid_mass = self.zeros(self.node_count, float, requires_grad=False)
        momentum = self.zeros(self.node_count, wp.vec3)
        grid_impulse = self.zeros(self.node_count, wp.vec3)
        old_v = self.zeros(self.node_count, wp.vec3)
        new_v = self.zeros(self.node_count, wp.vec3)
        remapped = self.zeros(self.node_count, wp.vec3)
        result = State(self.zeros(self.n, wp.vec3), self.zeros(self.n, wp.vec3), self.zeros(self.n, wp.mat33))
        wp.launch(
            compute_stress, self.n, inputs=[state.F, self.young, cfg.poisson_ratio],
            outputs=[stress], device=self.device,
        )
        if impulse is None:
            impulse = self.zeros(self.n, wp.vec3, requires_grad=False)
            wp.launch(
                body_impulse, self.n, inputs=[wp.vec3(*gravity), self.mass, cfg.dt],
                outputs=[impulse], device=self.device,
            )
        wp.launch(
            m.p2g, (self.n, 8),
            inputs=[state.v, stress, impulse, self.slots, self.weights, self.gradients,
                    self.mass, self.volume, cfg.dt],
            outputs=[grid_mass, momentum, grid_impulse], device=self.device,
        )
        wp.launch(
            m.grid_vel, self.node_count, inputs=[grid_mass, momentum, grid_impulse],
            outputs=[old_v, new_v], device=self.device,
        )
        wp.launch(
            gather_remap, self.n,
            inputs=[state.x, state.v, self.slots, self.weights, old_v, new_v,
                    cfg.dt, cfg.effective_flip_fraction(), self.mass],
            outputs=[result.x, result.v, remapped], device=self.device,
        )
        wp.launch(
            update_deformation, self.n,
            inputs=[state.F, self.slots, self.gradients, grid_mass, remapped, cfg.dt],
            outputs=[result.F], device=self.device,
        )
        return result

    def checkpoint_vjp(
        self, initial, steps, segment_steps, seed, gravity=(0.0, 0.0, -9.81), timings=None,
    ):
        """VJP of the terminal (x, v, F), recomputing one elastic segment at a time.

        seed contains the terminal loss derivatives. Returns (initial-state VJP,
        dL/dE); this routine owns and clears the trajectory's gradient buffers.
        """
        if not self.requires_grad or steps < 1 or segment_steps < 1:
            raise ValueError("Checkpoint VJP requires gradients and positive step counts")
        def timestamp():
            wp.synchronize_device(self.device)
            return perf_counter()

        if timings is not None:
            timings.update(forward_s=0.0, recompute_s=0.0, backward_s=0.0)
            started = timestamp()
        checkpoints = [initial]
        lengths = []
        current = initial
        remaining = steps
        while remaining:
            count = min(segment_steps, remaining)
            for _ in range(count):
                current = self.step(current, gravity)
            checkpoints.append(current)
            lengths.append(count)
            remaining -= count
        if timings is not None:
            timings["forward_s"] = timestamp() - started
        gradient = seed
        young_gradient = 0.0
        for index in reversed(range(len(lengths))):
            start = checkpoints[index]
            for array in (self.young, start.x, start.v, start.F):
                array.grad.zero_()
            if timings is not None:
                started = timestamp()
            with wp.Tape() as tape:
                current = start
                for _ in range(lengths[index]):
                    current = self.step(current, gravity)
            if timings is not None:
                now = timestamp()
                timings["recompute_s"] += now - started
                started = now
            tape.backward(grads={current.x: gradient.x, current.v: gradient.v, current.F: gradient.F})
            gradient = State(
                wp.clone(start.x.grad), wp.clone(start.v.grad), wp.clone(start.F.grad)
            )
            young_gradient += float(self.young.grad.numpy()[0])
            tape.zero()
            if timings is not None:
                timings["backward_s"] += timestamp() - started
        return gradient, young_gradient
