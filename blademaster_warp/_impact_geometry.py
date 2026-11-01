"""Normal-gap velocity maps, including moving-face moments and remainders."""

import numpy as np
import warp as wp

from ._cohesive_expansion import E, assign, stack
from ._differential_scalar import Scalar, cross, dot
from .compensated_state import CompensatedArray, as_compensated, add_gradient


def normal_map(geometry, reference, separation, frame_map):
    sources = [
        CompensatedArray(geometry.gap, geometry.gap_remainder),
        CompensatedArray(geometry.frame, geometry.frame_remainder),
        as_compensated(reference),
        as_compensated(separation),
        as_compensated(frame_map),
    ]
    device = geometry.gap.device
    kinds = [wp.vec3d, wp.mat33d, wp.vec3d, wp.float64, wp.vec3d]
    if any(a.high.dtype != kind or a.high.device != device for a, kind in zip(sources, kinds)):
        raise ValueError("Impact geometry requires float64 arrays on a common device")
    q, F, reference, B, G = [a.cpu() for a in sources]
    if B.ndim != 2 or G.shape != (*B.shape, 3) or reference.shape != q.shape or B.shape[0] != len(q.high):
        raise ValueError("Impact geometry must match the interface and particle counts")
    result = E(np.zeros((*B.shape, 3)))
    derivatives = []
    for i in range(B.shape[0]):
        values = [*q[i], *F[i].reshape(-1), *reference[i]]
        scalars = [Scalar.independent(v, j, 15) for j, v in enumerate(values)]
        gap, normal = scalars[:3], scalars[12:]
        columns = [[scalars[3 + 3 * r + c] for r in range(3)] for c in range(3)]
        products = [
            cross(columns[1], columns[2]),
            cross(columns[2], columns[0]),
            cross(columns[0], columns[1]),
        ]
        area = [sum(normal[c] * products[c][j] for c in range(3)) for j in range(3)]
        length = dot(area, area).sqrt()
        n = [value / length for value in area]
        component = dot(gap, n)
        adj_area = [(gap[j] - component * n[j]) / length for j in range(3)]
        pairs = [
            (cross(adj_area, columns[2]), cross(columns[1], adj_area), 1, 2),
            (cross(columns[2], adj_area), cross(adj_area, columns[0]), 0, 2),
            (cross(adj_area, columns[1]), cross(columns[0], adj_area), 0, 1),
        ]
        moment = [[normal[a] * x[j] + normal[b] * y[j] for j in range(3)] for x, y, a, b in pairs]
        n_value = stack([v.value for v in n])
        n_jacobian = stack([v.derivative for v in n])
        moment_value = stack([stack([v.value for v in column]) for column in moment], axis=1)
        moment_jacobian = stack([stack([v.derivative for v in column]) for column in moment], axis=1)
        value = B[i, :, None] * n_value[None, :] + G[i] @ moment_value.T
        jacobian = B[i, :, None, None] * n_jacobian[None, :, :]
        for c in range(3):
            jacobian = jacobian + G[i, :, c, None, None] * moment_jacobian[None, :, c, :]
        assign(result, i, value)
        derivatives.append((n_value, moment_value, jacobian))
    output = CompensatedArray.from_cpu(result, dtype=wp.vec3d, device=device)
    tape = wp._src.context.runtime.tape
    if tape is not None:

        def backward():
            seeds = E(output.high.grad.numpy())
            gradients = [E(np.zeros_like(a.high.numpy())) for a in sources]
            for i, (n, moment, jacobian) in enumerate(derivatives):
                local = (seeds[i, :, :, None] * jacobian).sum(axis=0).sum(axis=0)
                assign(gradients[0], i, local[:3])
                assign(gradients[1], i, local[3:12].reshape(3, 3))
                assign(gradients[2], i, local[12:])
                assign(gradients[3], i, (seeds[i] * n[None, :]).sum(axis=1))
                assign(gradients[4], i, seeds[i] @ moment)
            for source, gradient in zip(sources, gradients):
                add_gradient(source.high, gradient)

        tape.record_func(
            backward, [output.high, *[a.high for a in sources if a.high.requires_grad and a.high.size]]
        )
    return output
