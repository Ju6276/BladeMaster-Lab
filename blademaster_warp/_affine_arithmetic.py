"""Local compensated affine operations; cotangents belong to each high array."""

import numpy as np
import warp as wp

from ._compensated import Expansion as E
from ._compensated_linalg import divide
from ._cohesive_expansion import assign, stack
from .compensated_state import CompensatedArray, add_gradient


def transpose(a):
    return E(np.swapaxes(a.high, -1, -2), np.swapaxes(a.low, -1, -2))


def concatenate(a, b):
    return E(np.concatenate([a.high, b.high], axis=1), np.concatenate([a.low, b.low], axis=1))


def right_solve(value, root, transposed=False):
    result = E(np.zeros_like(value.high))
    for j in range(3) if transposed else range(2, -1, -1):
        rhs = value[:, :, j]
        indices = range(j) if transposed else range(j + 1, 3)
        for k in indices:
            coefficient = root[:, j, k] if transposed else root[:, k, j]
            rhs = rhs - result[:, :, k] * coefficient[:, None]
        assign(result, (slice(None), slice(None), j), divide(rhs, root[:, j, j, None]))
    return result


class StageArithmetic:
    def __init__(self, initial):
        self.values = {}
        for name, low in getattr(initial, "rounding_remainders", {}).items():
            source = getattr(initial, name)
            self.values[source] = CompensatedArray(source, low)

    def state(self, array):
        return self.values[array] if array in self.values else CompensatedArray(array)

    def remember(self, value):
        self.values[value.high] = value
        return value.high

    def value(self, array):
        return self.state(array).cpu()

    def inverse(self, inertia, source):
        return self.remember(inertia.solve_state(self.state(source)))

    def generator(self, inertia, source):
        result = CompensatedArray.from_cpu(
            inertia.generator(self.value(source)), dtype=source.dtype, device=source.device
        )
        tape = wp._src.context.runtime.tape
        if tape is not None:

            def backward():
                add_gradient(source, inertia.generator(E(result.high.grad.numpy())))

            tape.record_func(backward, [a for a in (result.high, source) if a.requires_grad])
        return self.remember(result)

    def acceleration(self, displacement, duration):
        (source, time) = (self.state(displacement), self.state(duration))
        (d, t) = (source.cpu(), time.cpu()[0])
        result = CompensatedArray.from_cpu(
            divide(d, t * t), dtype=displacement.dtype, device=displacement.device
        )
        tape = wp._src.context.runtime.tape
        if tape is not None:

            def backward():
                seed = E(result.high.grad.numpy())
                add_gradient(displacement, divide(seed, t * t))
                add_gradient(duration, divide(-2 * (seed * d).sum(), t * t * t).reshape(1))

            tape.record_func(backward, [a for a in (result.high, displacement, duration) if a.requires_grad])
        return self.remember(result)

    def increment(self, base, rate, duration, factor=1.0):
        return self.remember(self.state(base).add_scaled(self.state(rate), self.state(duration), factor))

    def advance_geometry(self, inertia, geometry, separation, frame_map, velocity):
        filtered = inertia.solve_state(self.state(velocity))
        return geometry.advance(separation, frame_map, filtered, inertia.parameters[0])


class AffineArithmetic(StageArithmetic):
    def output(self, value, source, kind=None):
        result = CompensatedArray.from_cpu(value, dtype=kind or source.dtype, device=source.device)
        return self.remember(result)

    def pack(self, transfer, velocity, gradient):
        root = E(transfer.root.numpy())
        affine = self.value(gradient) @ root
        packed = concatenate(self.value(velocity)[:, None, :], transpose(affine)).reshape(-1, 3)
        result = self.output(packed, velocity)
        tape = wp._src.context.runtime.tape
        if tape is not None:

            def backward():
                seed = E(result.grad.numpy()).reshape(-1, 4, 3)
                add_gradient(velocity, seed[:, 0])
                add_gradient(gradient, transpose(seed[:, 1:]) @ transpose(root))

            tape.record_func(backward, [a for a in (result, velocity, gradient) if a.requires_grad])
        return result

    def unpack(self, transfer, moments):
        root = E(transfer.root.numpy())
        packed = self.value(moments).reshape(-1, 4, 3)
        velocity = self.output(packed[:, 0], moments)
        gradient = self.output(right_solve(transpose(packed[:, 1:]), root), moments, wp.mat33d)
        tape = wp._src.context.runtime.tape
        if tape is not None:

            def backward():
                seed = right_solve(E(gradient.grad.numpy()), root, transposed=True)
                combined = concatenate(E(velocity.grad.numpy())[:, None, :], transpose(seed)).reshape(-1, 3)
                add_gradient(moments, combined)

            tape.record_func(backward, [a for a in (velocity, gradient, moments) if a.requires_grad])
        return (velocity, gradient)

    def kinematics(self, transfer, moments):
        flow = E(transfer.flow_map.numpy())
        maps = [E(transfer.gradient_map.numpy()[:, :, c]) for c in range(3)]
        z = self.value(moments)
        velocity = self.output(flow @ z, moments)
        rate = self.output(stack([g @ z for g in maps], axis=-1), moments, wp.mat33d)
        tape = wp._src.context.runtime.tape
        if tape is not None:

            def backward():
                result = flow.T @ E(velocity.grad.numpy())
                for c, g in enumerate(maps):
                    result = result + g.T @ E(rate.grad.numpy()[:, :, c])
                add_gradient(moments, result)

            tape.record_func(backward, [a for a in (velocity, rate, moments) if a.requires_grad])
        return (velocity, rate)

    def pullback_force(self, transfer, force, moment):
        flow = E(transfer.flow_map.numpy())
        maps = [E(transfer.gradient_map.numpy()[:, :, c]) for c in range(3)]
        value = flow.T @ self.value(force)
        for c, g in enumerate(maps):
            value = value + g.T @ self.value(moment)[:, :, c]
        result = self.output(value, force)
        tape = wp._src.context.runtime.tape
        if tape is not None:

            def backward():
                seed = E(result.grad.numpy())
                add_gradient(force, flow @ seed)
                add_gradient(moment, stack([g @ seed for g in maps], axis=-1))

            tape.record_func(backward, [a for a in (result, force, moment) if a.requires_grad])
        return result

    def shift_identity(self, source, sign):
        updated = self.output(self.value(source) + sign * np.eye(3), source)
        tape = wp._src.context.runtime.tape
        if tape is not None:

            def backward():
                add_gradient(source, updated.grad.numpy())

            tape.record_func(backward, [a for a in (updated, source) if a.requires_grad])
        return updated

    def scale(self, source, factor, inverse=False):
        original = self.value(source)
        updated = self.output(divide(original, factor) if inverse else original * factor, source)
        tape = wp._src.context.runtime.tape
        if tape is not None:

            def backward():
                seed = E(updated.grad.numpy())
                add_gradient(source, divide(seed, factor) if inverse else seed * factor)

            tape.record_func(backward, [a for a in (updated, source) if a.requires_grad])
        return updated


def apply_matrix(matrix, value):
    M, x = matrix.numpy().copy(), value.numpy().copy()
    result = wp.array(
        (M.astype(np.longdouble) @ x.astype(np.longdouble)).astype(float),
        dtype=wp.vec3d,
        device=value.device,
        requires_grad=True,
    )
    tape = wp._src.context.runtime.tape
    if tape is not None:

        def backward():
            seed = result.grad.numpy()
            if matrix.requires_grad:
                matrix.grad.assign(matrix.grad.numpy() + seed @ x.T)
            if value.requires_grad:
                value.grad.assign(value.grad.numpy() + M.T @ seed)

        tape.record_func(backward, [result, *[a for a in (matrix, value) if a.requires_grad]])
    return result
