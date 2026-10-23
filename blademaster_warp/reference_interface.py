"""Material surface traces in reference coordinates.

Each quadrature point has two particle supports. Weights and their reference
gradients describe the same basis. Positions are in metres, gradients in 1/m.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp


@wp.kernel
def _weight_totals(
    weights: wp.array3d(dtype=wp.float64),
    gradients: wp.array3d(dtype=wp.vec3d),
    total: wp.array2d(dtype=wp.float64),
    derivative: wp.array2d(dtype=wp.vec3d),
):
    q, side = wp.tid()
    value = wp.float64(0.0)
    gradient = wp.vec3d(0.0)
    for p in range(weights.shape[2]):
        value += weights[q, side, p]
        gradient += gradients[q, side, p]
    total[q, side] = value
    derivative[q, side] = gradient


@wp.kernel
def _reconstruct(
    position: wp.array(dtype=wp.vec3d),
    deformation: wp.array(dtype=wp.mat33d),
    reference: wp.array(dtype=wp.vec3d),
    points: wp.array(dtype=wp.vec3d),
    weights: wp.array3d(dtype=wp.float64),
    gradients: wp.array3d(dtype=wp.vec3d),
    total: wp.array2d(dtype=wp.float64),
    derivative: wp.array2d(dtype=wp.vec3d),
    relative_displacement: wp.array2d(dtype=wp.vec3d),
    face_frame: wp.array2d(dtype=wp.mat33d),
    displacement: wp.array2d(dtype=wp.vec3d),
):
    q, side = wp.tid()
    identity = wp.identity(n=3, dtype=wp.float64)
    anchor = position[0] - reference[0] + (deformation[0] - identity) * (points[q] - reference[0])
    anchor_frame = deformation[0]
    value = wp.vec3d(0.0)
    jacobian = wp.mat33d(0.0)
    for p in range(position.shape[0]):
        w = weights[q, side, p] / total[q, side]
        dw = (gradients[q, side, p] - w * derivative[q, side]) / total[q, side]
        local = position[p] - reference[p]
        local += (deformation[p] - identity) * (points[q] - reference[p])
        relative = local - anchor
        value += w * relative
        jacobian += w * (deformation[p] - anchor_frame) + wp.outer(relative, dw)
    relative_displacement[q, side] = value
    face_frame[q, side] = anchor_frame + jacobian
    displacement[q, side] = anchor + value


@wp.kernel
def _combine(
    relative_displacement: wp.array2d(dtype=wp.vec3d),
    face_frame: wp.array2d(dtype=wp.mat33d),
    gap: wp.array(dtype=wp.vec3d),
    frame: wp.array(dtype=wp.mat33d),
):
    q = wp.tid()
    gap[q] = relative_displacement[q, 1] - relative_displacement[q, 0]
    frame[q] = wp.float64(0.5) * (face_frame[q, 1] + face_frame[q, 0])


@dataclass(frozen=True)
class ReferenceTrace:
    gap: wp.array
    frame: wp.array
    displacement: wp.array


def reconstruct(position, deformation, reference, points, weights, gradients):
    """Return the displacement jump and mean spatial Jacobian of the two traces.

    Each trace reconstructs x_p-X_p+(F_p-I)(Q-X_p), normalized by its weight sum.
    The returned frame differentiates this same interpolation with respect to Q.
    """
    inputs = [position, deformation, reference, points, weights, gradients]
    expected = [wp.vec3d, wp.mat33d, wp.vec3d, wp.vec3d, wp.float64, wp.vec3d]
    if any(a.dtype != dtype or a.device != position.device for a, dtype in zip(inputs, expected)):
        raise ValueError("Reference traces require float64 arrays on a common device")
    particles, count = len(position), len(points)
    shapes = [
        (particles,),
        (particles,),
        (particles,),
        (count,),
        (count, 2, particles),
        (count, 2, particles),
    ]
    if particles == 0 or count == 0 or any(a.shape != shape for a, shape in zip(inputs, shapes)):
        raise ValueError("Reference trace shapes must match the quadrature points and particles")
    values = [a.numpy() for a in inputs]
    if any(not np.isfinite(a).all() for a in values):
        raise ValueError("Reference trace inputs must be finite")
    if np.any(values[4] < 0) or np.any(values[4].sum(axis=2) <= 0):
        raise ValueError("Each surface side requires nonnegative weights with positive total")
    device = position.device
    gap = wp.empty(count, dtype=wp.vec3d, device=device, requires_grad=True)
    frame = wp.empty(count, dtype=wp.mat33d, device=device, requires_grad=True)
    displacement = wp.empty((count, 2), dtype=wp.vec3d, device=device, requires_grad=True)
    total = wp.empty((count, 2), dtype=wp.float64, device=device, requires_grad=True)
    derivative = wp.empty((count, 2), dtype=wp.vec3d, device=device, requires_grad=True)
    relative = wp.empty_like(displacement, requires_grad=True)
    face_frame = wp.empty((count, 2), dtype=wp.mat33d, device=device, requires_grad=True)
    # Stage normalization so backward uses the completed reductions.
    wp.launch(
        _weight_totals, (count, 2), inputs=[weights, gradients], outputs=[total, derivative], device=device
    )
    wp.launch(
        _reconstruct,
        (count, 2),
        inputs=[*inputs, total, derivative],
        outputs=[relative, face_frame, displacement],
        device=device,
    )
    wp.launch(_combine, count, inputs=[relative, face_frame], outputs=[gap, frame], device=device)
    return ReferenceTrace(gap, frame, displacement)
