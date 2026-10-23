"""Frictionless, perfectly inelastic interface impacts and their KKT adjoint."""

import numpy as np
from scipy.optimize import nnls
import warp as wp

from .cohesive import surface_normal, surface_normal_pullback
from .precision import launch_double


@wp.kernel
def normal_rate_map(
    gap: wp.array(dtype=wp.vec3),
    frame: wp.array(dtype=wp.mat33),
    reference_normal: wp.array(dtype=wp.vec3),
    separation_map: wp.array2d(dtype=float),
    frame_map: wp.array2d(dtype=wp.vec3),
    result: wp.array2d(dtype=wp.vec3),
):
    q, p = wp.tid()
    n = surface_normal(frame[q], reference_normal[q])
    moment = surface_normal_pullback(frame[q], reference_normal[q], gap[q])
    result[q, p] = separation_map[q, p] * n + moment * frame_map[q, p]


def normal_velocity_map(geometry, reference_normal, separation_map, frame_map):
    """Differentiate g dot n(F) with respect to particle velocity.

    The gap rate is B v and the frame rate is sum_p outer(v_p, G_p).
    The moving-normal contribution uses the same geometry as the cohesive force.
    """
    sources = [geometry.gap, geometry.frame, reference_normal, separation_map, frame_map]
    expected = [wp.vec3d, wp.mat33d, wp.vec3d, wp.float64, wp.vec3d]
    if any(a.dtype != dtype or a.device != geometry.gap.device for a, dtype in zip(sources, expected)):
        raise ValueError("Impact geometry requires float64 arrays on a common device")
    if (
        separation_map.ndim != 2
        or frame_map.shape != separation_map.shape
        or reference_normal.shape != geometry.gap.shape
        or separation_map.shape[0] != len(geometry.gap)
    ):
        raise ValueError("Impact geometry must match the interface and particle counts")
    result = wp.zeros(separation_map.shape, dtype=wp.vec3d, device=geometry.gap.device, requires_grad=True)
    launch_double(
        normal_rate_map, separation_map.shape, inputs=sources, outputs=[result], device=result.device
    )
    return result


def project_velocity(constraint, mass, velocity):
    """Minimize the mass norm of the velocity jump subject to C v >= 0.

    C contains normal-gap velocity Jacobians at closed interfaces, with axes
    (contact, particle, xyz). Masses are in kg and velocities in m/s. The caller
    locates closure events; positive-gap interfaces do not belong in C.
    """
    if (
        constraint.dtype != wp.vec3d
        or mass.dtype != wp.float64
        or velocity.dtype != wp.vec3d
        or constraint.device != velocity.device
        or mass.device != velocity.device
    ):
        raise ValueError("Impact projection requires float64 arrays on a common device")
    if (
        constraint.ndim != 2
        or velocity.ndim != 1
        or mass.shape != velocity.shape
        or constraint.shape[1] != len(velocity)
    ):
        raise ValueError("Impact columns and masses must match the particle velocity")
    C, masses, velocities = [a.numpy().copy() for a in [constraint, mass, velocity]]
    if (
        not len(masses)
        or np.any(masses <= 0.0)
        or any(not np.isfinite(a).all() for a in [C, masses, velocities])
    ):
        raise ValueError("Impact inputs must be finite with positive masses")
    if not len(C):
        return velocity
    matrix = C.reshape(len(C), -1)
    roots = np.repeat(np.sqrt(masses), 3)
    weighted = roots * velocities.ravel()
    row_scale = np.linalg.norm(matrix / roots, axis=1)
    nonzero = np.flatnonzero(row_scale > 0.0)
    if not len(nonzero):
        return velocity
    normalized = matrix[nonzero] / row_scale[nonzero, None]
    impulses, _ = nnls(normalized.T / roots[:, None], -weighted, maxiter=10 * len(nonzero))
    selected = np.flatnonzero(impulses > 0.0)
    if not len(selected):
        return velocity
    active = nonzero[selected]
    left, singular, right = np.linalg.svd(normalized[selected] / roots, full_matrices=False)
    rank = np.count_nonzero(singular > singular[0] * max(right.shape) * np.finfo(float).eps)
    left, singular, right = left[:, :rank], singular[:rank], right[:rank]
    projected = (weighted - right.T @ (right @ weighted)) / roots
    residual = matrix @ projected
    tolerance = 64 * np.finfo(float).eps * np.linalg.norm(matrix, axis=1) * np.linalg.norm(velocities)
    if np.any(residual < -tolerance):
        raise FloatingPointError("Impact projection failed unilateral velocity constraints")
    reaction = impulses[selected] / row_scale[active]
    result = wp.array(
        projected.reshape(velocities.shape), dtype=wp.vec3d, device=velocity.device, requires_grad=True
    )
    tape = wp._src.context.runtime.tape
    if tape is not None:

        def backward():
            seed = result.grad.numpy().ravel() / roots
            dual = (seed - right.T @ (right @ seed)) / roots
            multiplier = (left @ ((right @ seed) / singular)) / row_scale[active]
            if constraint.requires_grad:
                derivative = np.zeros_like(matrix)
                derivative[active] = reaction[:, None] * dual - multiplier[:, None] * projected
                constraint.grad.assign(constraint.grad.numpy() + derivative.reshape(C.shape))
            if mass.requires_grad:
                derivative = np.sum(
                    (dual * (velocities.ravel() - projected)).reshape(velocities.shape), axis=1
                )
                mass.grad.assign(mass.grad.numpy() + derivative)
            if velocity.requires_grad:
                velocity.grad.assign(velocity.grad.numpy() + (roots**2 * dual).reshape(velocities.shape))

        sources = [constraint, mass, velocity]
        tape.record_func(backward, [result, *[a for a in sources if a.requires_grad]])
    return result
