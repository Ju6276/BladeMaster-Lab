"""D = I + eta*t*(I - P.T M_grid P/m) in reference-affine coordinates."""

from copy import copy

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
import warp as wp

from ._compensated import Expansion
from .compensated_state import CompensatedArray, as_compensated, add_gradient


@wp.kernel
def _grid_velocity(
    mapping: wp.array2d(dtype=wp.float64),
    value: wp.array(dtype=wp.vec3d),
    grid: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    result = wp.vec3d(0.0)
    for p in range(value.shape[0]):
        result += mapping[i, p] * value[p]
    grid[i] = result


@wp.kernel
def _grid_projection(
    mapping: wp.array2d(dtype=wp.float64),
    masses: wp.array(dtype=wp.float64),
    grid: wp.array(dtype=wp.vec3d),
    projected: wp.array(dtype=wp.vec3d),
):
    p = wp.tid()
    value = wp.vec3d(0.0)
    for i in range(grid.shape[0]):
        value += mapping[i, p] * masses[i] * grid[i]
    projected[p] = value


@wp.kernel
def _momentum(
    duration: wp.array(dtype=wp.float64),
    rate: wp.array(dtype=wp.float64),
    value: wp.array(dtype=wp.vec3d),
    projected: wp.array(dtype=wp.vec3d),
    original: wp.array(dtype=wp.vec3d),
    result: wp.array(dtype=wp.vec3d),
):
    p = wp.tid()
    result[p] = original[p] + duration[0] * rate[0] * (value[p] - projected[p])


class ReferenceInertia:
    """Symmetric inertia action; duration is in s and damping rate in s^-1."""

    def __init__(self, transfer, duration, rate):
        self.transfer = transfer
        transfer._check(duration, wp.float64, (1,))
        transfer._check(rate, wp.float64, (1,))
        values = np.array([duration.numpy()[0], rate.numpy()[0]])
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Inertia duration and rate must be finite and nonnegative")
        self.parameters = [duration, rate]
        self.mapping = transfer.grid_map.numpy().copy()
        self.masses = transfer.grid_mass.copy() / transfer.model.mass
        self.size = self.mapping.shape[1]
        self.values = values.copy()
        self.theta = np.longdouble(values[0]) * np.longdouble(values[1])
        self.mass_array = transfer.model.array(self.masses, float, False)

    def frozen(self):
        result = copy(self)
        result.values = np.array([a.numpy()[0] for a in self.parameters])
        if not np.isfinite(result.values).all() or np.any(result.values < 0):
            raise ValueError("Inertia duration and rate must be finite and nonnegative")
        result.theta = np.longdouble(result.values[0]) * np.longdouble(result.values[1])
        return result

    def clone(self):
        result = copy(self)
        result.parameters = [self.transfer.model.array([value], float, True) for value in self.values]
        return result

    def generator(self, value):
        if isinstance(value, Expansion):
            masses = self.masses if value.ndim == 1 else self.masses[:, None]
            return value - Expansion(self.mapping.T) @ (masses * (Expansion(self.mapping) @ value))
        value = np.asarray(value)
        vector = value.ndim == 1
        x = value[:, None] if vector else value
        mapping = self.mapping.astype(x.dtype, copy=False)
        masses = self.masses.astype(x.dtype, copy=False)
        result = x - mapping.T @ (masses[:, None] * (mapping @ x))
        return result[:, 0] if vector else result

    def apply(self, value):
        if isinstance(value, Expansion):
            theta = Expansion(self.values[0]) * self.values[1]
            return value + theta * self.generator(value)
        value = np.asarray(value)
        return value + value.dtype.type(self.theta) * self.generator(value)

    def solve_compensated(self, rhs):
        rhs = Expansion.from_value(rhs)
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.size:
            raise ValueError("Inertia right hand sides must match the transfer")
        if not np.isfinite(rhs.high).all() or not np.isfinite(rhs.low).all():
            raise ValueError("Inertia right hand sides must be finite")
        if self.theta == 0:
            return rhs
        vector = rhs.ndim == 1
        values = rhs.reshape(-1, 1) if vector else rhs
        operator = LinearOperator((self.size, self.size), matvec=self.apply, dtype=np.float64)
        solution = Expansion(np.zeros(values.shape))
        scale = np.maximum(np.max(np.abs(values.high), axis=0), np.finfo(float).tiny)
        epsilon = np.finfo(float).eps ** 2
        for _ in range(4):
            residual = values - self.apply(solution)
            active = np.max(np.abs(residual.high), axis=0) > 16 * epsilon * scale
            if not np.any(active):
                break
            correction = np.zeros(values.shape)
            for column in np.flatnonzero(active):
                correction[:, column], status = cg(
                    operator,
                    residual[:, column].numpy(),
                    rtol=8 * np.finfo(float).eps,
                    atol=0.0,
                    maxiter=max(20, 4 * self.size),
                )
                if status:
                    raise RuntimeError(f"Reference inertia CG failed: {status}")
            solution = solution + correction
        residual = values - self.apply(solution)
        if not np.isfinite(residual.high).all() or np.any(
            np.max(np.abs(residual.high), axis=0) > 64 * epsilon * scale
        ):
            raise RuntimeError("Compensated reference inertia residual exceeds roundoff")
        return solution[:, 0] if vector else solution

    def solve(self, rhs):
        rhs = np.asarray(rhs)
        if not np.issubdtype(rhs.dtype, np.floating):
            raise ValueError("Inertia solves require floating-point right hand sides")
        return self.solve_compensated(rhs).numpy(rhs.dtype)

    def add_momentum(self, original, value):
        m = self.transfer.model
        grid = wp.zeros(self.mapping.shape[0], dtype=wp.vec3d, device=m.device, requires_grad=True)
        projected = wp.zeros(self.size, dtype=wp.vec3d, device=m.device, requires_grad=True)
        result = wp.zeros(self.size, dtype=wp.vec3d, device=m.device, requires_grad=True)
        wp.launch(_grid_velocity, len(grid), [self.transfer.grid_map, value, grid], device=m.device)
        wp.launch(
            _grid_projection,
            self.size,
            [self.transfer.grid_map, self.mass_array, grid, projected],
            device=m.device,
        )
        wp.launch(
            _momentum, self.size, [*self.parameters, value, projected, original, result], device=m.device
        )
        return result

    def solve_state(self, value):
        """Apply D^-1 to a state while retaining its rounding remainder."""
        value = as_compensated(value)
        self.transfer._check(value.high, wp.vec3d, (self.size,))
        bound = self.frozen()
        advanced = bound.solve_compensated(value.cpu())
        result = CompensatedArray.from_cpu(advanced, dtype=wp.vec3d, device=value.high.device)
        tape = wp._src.context.runtime.tape
        if tape is not None:

            def backward():
                adjoint = bound.solve_compensated(result.high.grad.numpy())
                add_gradient(value.high, adjoint)
                power = (adjoint * bound.generator(advanced)).sum()
                for parameter, coefficient in zip(bound.parameters, [-bound.values[1], -bound.values[0]]):
                    add_gradient(parameter, (coefficient * power).reshape(1))

            sources = [value.high, *bound.parameters]
            tape.record_func(backward, [result.high, *[a for a in sources if a.requires_grad]])
        return result

    def solve_array(self, value):
        return self.solve_state(value).high


def apply_inertia(inertia, value):
    return value if inertia is None else inertia.apply(value)


def solve_inertia(inertia, value):
    return value if inertia is None else inertia.solve(value)


def solve_reduced(matrix, rhs):
    """Refine a small positive-definite solve with extended-precision residuals."""
    matrix, rhs = np.asarray(matrix, dtype=np.longdouble), np.asarray(rhs, dtype=np.longdouble)
    result = np.linalg.solve(matrix.astype(float), rhs.astype(float)).astype(np.longdouble)
    for _ in range(3):
        result += np.linalg.solve(matrix.astype(float), (rhs - matrix @ result).astype(float))
    return result
