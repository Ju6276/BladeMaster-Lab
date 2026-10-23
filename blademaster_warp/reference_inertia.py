"""D = I + eta*t*(I - P.T M_grid P/m) in reference-affine coordinates."""

from copy import copy

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
import warp as wp


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
        value = np.asarray(value)
        vector = value.ndim == 1
        x = value[:, None] if vector else value
        mapping = self.mapping.astype(x.dtype, copy=False)
        masses = self.masses.astype(x.dtype, copy=False)
        result = x - mapping.T @ (masses[:, None] * (mapping @ x))
        return result[:, 0] if vector else result

    def apply(self, value):
        value = np.asarray(value)
        return value + value.dtype.type(self.theta) * self.generator(value)

    def solve(self, rhs):
        rhs = np.asarray(rhs)
        vector = rhs.ndim == 1
        values = rhs[:, None] if vector else rhs
        if values.shape[0] != self.size or not np.isfinite(values).all():
            raise ValueError("Inertia right hand sides must be finite and match the transfer")
        if self.theta == 0:
            return rhs.copy()
        operator = LinearOperator((self.size, self.size), matvec=self.apply, dtype=np.float64)
        solution = np.zeros(values.shape, dtype=np.longdouble)
        precise_rhs = values.astype(np.longdouble)
        for column in range(values.shape[1]):
            scale = max(np.linalg.norm(precise_rhs[:, column]), np.finfo(float).tiny)
            for _ in range(4):
                residual = precise_rhs[:, column] - self.apply(solution[:, column])
                if np.linalg.norm(residual) <= 16 * np.finfo(np.longdouble).eps * scale:
                    break
                correction, status = cg(
                    operator,
                    residual.astype(float),
                    rtol=8 * np.finfo(float).eps,
                    atol=0.0,
                    maxiter=max(20, 4 * self.size),
                )
                if status:
                    raise RuntimeError(f"Reference inertia CG failed: {status}")
                solution[:, column] += correction
            residual = precise_rhs[:, column] - self.apply(solution[:, column])
            if np.linalg.norm(residual) > 64 * np.finfo(float).eps * scale:
                raise RuntimeError("Reference inertia residual exceeds roundoff")
        result = solution.astype(rhs.dtype)
        return result[:, 0] if vector else result

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

    def solve_array(self, value):
        """Apply D^-1 with its implicit state, duration and rate pullback."""
        self.transfer._check(value, wp.vec3d, (self.size,))
        bound = self.frozen()
        advanced = bound.solve(value.numpy().astype(np.longdouble))
        result = self.transfer.model.array(advanced, wp.vec3, True)
        tape = wp._src.context.runtime.tape
        if tape is not None:

            def backward():
                adjoint = bound.solve(result.grad.numpy().astype(np.longdouble))
                if value.requires_grad:
                    value.grad.assign(value.grad.numpy() + adjoint)
                power = np.sum(adjoint * bound.generator(advanced))
                for parameter, derivative in zip(
                    bound.parameters, [-bound.values[1] * power, -bound.values[0] * power]
                ):
                    if parameter.requires_grad:
                        parameter.grad.assign(parameter.grad.numpy() + [derivative])

            tape.record_func(backward, [a for a in [result, value, *bound.parameters] if a.requires_grad])
        return result


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
