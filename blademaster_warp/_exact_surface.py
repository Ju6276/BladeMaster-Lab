"""Rational surface constructions with analytic, fixed-topology pullbacks.

Coordinates remain rational through clipping. Warp arrays hold their float64
rounding; Jacobians are evaluated before rounding construction intermediates.
"""

from fractions import Fraction

import numpy as np
import warp as wp


from .reference_coordinates import rational


def cross(a, b):
    return np.array(
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
        dtype=object,
    )


def cross_matrix(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=object)


@wp.kernel
def _vector_vjp(
    indices: wp.array(dtype=wp.vec2i),
    jacobian: wp.array(dtype=wp.mat33d),
    seed: wp.array(dtype=wp.vec3d),
    gradient: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    pair = indices[i]
    wp.atomic_add(gradient, pair[1], wp.transpose(jacobian[i]) * seed[pair[0]])


@wp.kernel
def _scalar_vjp(
    indices: wp.array(dtype=wp.vec2i),
    jacobian: wp.array(dtype=wp.vec3d),
    seed: wp.array(dtype=wp.float64),
    gradient: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    pair = indices[i]
    wp.atomic_add(gradient, pair[1], seed[pair[0]] * jacobian[i])


def _result(values, dtype, sources, derivatives):
    out = wp.array(np.asarray(values, dtype=float), dtype=dtype, device=sources[0].device, requires_grad=True)
    tape = wp._src.context.runtime.tape
    if tape is not None and len(out):
        blocks = []
        for source, entries in zip(sources, derivatives):
            if not source.requires_grad or not entries:
                continue
            indices = wp.array([(i, j) for i, j, _ in entries], dtype=wp.vec2i, device=out.device)
            kind = wp.mat33d if dtype == wp.vec3d else wp.vec3d
            jacobian = wp.array(
                np.asarray([d for _, _, d in entries], dtype=float), dtype=kind, device=out.device
            )
            blocks.append((source, indices, jacobian))

        def backward():
            for source, indices, jacobian in blocks:
                wp.launch(
                    _vector_vjp if dtype == wp.vec3d else _scalar_vjp,
                    len(indices),
                    inputs=[indices, jacobian, out.grad],
                    outputs=[source.grad],
                    device=out.device,
                )

        tape.record_func(backward, [out, *[source for source, _, _ in blocks]])
    return out


def subdivide(vertices, edges, fractions, coordinates):
    values = coordinates
    exact, derivative = [], []
    differentiate = wp._src.context.runtime.tape is not None
    identity = np.eye(3, dtype=int)
    for i, ((a, b), fraction) in enumerate(zip(edges, fractions)):
        t = Fraction(float(fraction))
        exact.append((1 - t) * values[a] + t * values[b])
        if differentiate:
            derivative.extend([(i, a, (1 - t) * identity), (i, b, t * identity)])
    exact = np.array(exact, dtype=object).reshape(-1, 3)
    return _result(exact, wp.vec3d, [vertices], [derivative]), exact


def intersections(
    vertices, exact_vertices, edges, owners, point, normal, exact_point, exact_normal, locations
):
    derivatives = [[], [], []]
    if wp._src.context.runtime.tape is not None:
        identity = np.eye(3, dtype=int)
        for i, ((ai, bi), owner) in enumerate(zip(edges, owners)):
            if ai == bi:
                derivatives[0].append((i, ai, identity))
                continue
            j = owner if len(point) != 1 else 0
            a, b = exact_vertices[[ai, bi]]
            c, n = exact_point[j], exact_normal[j]
            u, d = a - c, a - b
            denominator = np.dot(n, d)
            t = np.dot(n, u) / denominator
            offset = b - a
            common = np.outer(offset, n) / denominator
            derivatives[0].extend([(i, ai, (1 - t) * (identity + common)), (i, bi, t * (identity + common))])
            derivatives[1].append((i, j, -common))
            derivatives[2].append((i, j, np.outer(offset, u - t * d) / denominator))
    exact = np.array(locations, dtype=object).reshape(-1, 3)
    return _result(exact, wp.vec3d, [vertices, point, normal], derivatives), exact


def polygon_plane(vertices, exact_vertices, normals, face, edge):
    a, b = exact_vertices[list(edge)]
    n = rational(normals.numpy()[face])
    exact_point, exact_normal = a[None], cross(b - a, n)[None]
    point_derivatives, normal_derivatives = [[]], [[], []]
    if wp._src.context.runtime.tape is not None:
        point_derivatives[0].append((0, edge[0], np.eye(3)))
        normal_derivatives[0].extend([(0, edge[0], cross_matrix(n)), (0, edge[1], -cross_matrix(n))])
        normal_derivatives[1].append((0, face, cross_matrix(b - a)))
    point = _result(exact_point, wp.vec3d, [vertices], point_derivatives)
    normal = _result(exact_normal, wp.vec3d, [vertices, normals], normal_derivatives)
    return point, normal, exact_point, exact_normal


def quadrature(vertices, exact_vertices, triangles, owners, face_ids, orientation, reference, remainder):
    point_values, area_values, normal_values = [], [], []
    point_derivatives, area_derivatives, normal_derivatives = [[], []], [[], []], [[]]
    references, normals = rational(reference.numpy()) + remainder, rational(orientation.numpy())
    differentiate = wp._src.context.runtime.tape is not None
    identity = np.eye(3, dtype=int)
    for ids, owner, face in zip(triangles, owners, face_ids):
        a, b, c = exact_vertices[ids]
        n = normals[face]
        ab, ac = b - a, c - a
        oriented = cross(ab, ac)
        area = np.dot(oriented, n) / 6
        gradients = [cross(n, c - b) / 6, cross(ac, n) / 6, cross(n, ab) / 6]
        for j in range(3):
            i = len(point_values)
            weights = [Fraction(2, 3) if k == j else Fraction(1, 6) for k in range(3)]
            point_values.append(references[owner] + sum(w * v for w, v in zip(weights, [a, b, c])))
            area_values.append(area)
            normal_values.append(n)
            if differentiate:
                point_derivatives[0].extend((i, v, w * identity) for v, w in zip(ids, weights))
                point_derivatives[1].append((i, owner, identity))
                area_derivatives[0].extend((i, v, g) for v, g in zip(ids, gradients))
                area_derivatives[1].append((i, face, oriented / 6))
                normal_derivatives[0].append((i, face, identity))
    points = _result(
        np.array(point_values).reshape(-1, 3), wp.vec3d, [vertices, reference], point_derivatives
    )
    area = _result(area_values, wp.float64, [vertices, orientation], area_derivatives)
    normal = _result(np.array(normal_values).reshape(-1, 3), wp.vec3d, [orientation], normal_derivatives)
    return points, area, normal


@wp.kernel
def _pullback_vjp(
    inverse: wp.array(dtype=wp.mat33d),
    local: wp.array(dtype=wp.vec3d),
    rotation_derivative: wp.array2d(dtype=wp.vec3d),
    size_derivative: wp.array(dtype=wp.mat33d),
    seed: wp.array(dtype=wp.vec3d),
    gx: wp.array(dtype=wp.vec3d),
    gF: wp.array(dtype=wp.mat33d),
    gc: wp.array(dtype=wp.vec3d),
    gq: wp.array(dtype=wp.quatd),
    gh: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    p = i // 2
    dual = wp.transpose(inverse[p]) * seed[i]
    wp.atomic_add(gx, p, -dual)
    wp.atomic_add(gF, p, -wp.outer(dual, local[i]))
    wp.atomic_add(gc, 0, dual)
    wp.atomic_add(
        gq,
        0,
        wp.quatd(
            wp.dot(rotation_derivative[i, 0], dual),
            wp.dot(rotation_derivative[i, 1], dual),
            wp.dot(rotation_derivative[i, 2], dual),
            wp.dot(rotation_derivative[i, 3], dual),
        ),
    )
    wp.atomic_add(gh, 0, wp.transpose(size_derivative[i]) * dual)


def pullback(x, deformation, position, rotation, half, remainder):
    """Return offsets from the exact reference centre of the affine particle map.

    The map is anchored at the stored particle position and float64 reference
    point: X_exact + offset = X_head + F^-1 (c + R e - x).
    """
    sources = [x, deformation, position, rotation, half]
    positions, Fs, c, q, h = [rational(a.numpy()) for a in sources]
    v, w = q[0, :3], q[0, 3]
    norm2 = np.dot(q[0], q[0])
    basis = np.eye(3, dtype=int)
    R = ((w * w - np.dot(v, v)) * basis + 2 * np.outer(v, v) + 2 * w * cross_matrix(v)) / norm2
    rotation_jacobians, size_jacobians, rotated = [], [], []
    for sign in [-1, 1]:
        local = np.array([0, sign * h[0, 1], -h[0, 2]], dtype=object)
        rotated.append(R @ local)
        derivatives = [
            (
                -2 * v[j] * local
                + 2 * basis[j] * np.dot(v, local)
                + 2 * v * local[j]
                + 2 * w * cross(basis[j], local)
                - 2 * v[j] * rotated[-1]
            )
            / norm2
            for j in range(3)
        ]
        derivatives.append((2 * w * local + 2 * cross(v, local) - 2 * w * rotated[-1]) / norm2)
        rotation_jacobians.append(derivatives)
        size_jacobians.append(R @ np.diag([0, sign, -1]))
    coordinates, inverses, pulled = [], [], []
    for p, F in enumerate(Fs):
        adjugate = np.array([cross(F[1], F[2]), cross(F[2], F[0]), cross(F[0], F[1])], dtype=object).T
        inverse = adjugate / np.dot(F[0], cross(F[1], F[2]))
        inverses.append(inverse)
        for r in rotated:
            local = inverse @ (c[0] + r - positions[p])
            pulled.append(local)
            coordinates.append(local - remainder[p])
    coordinates = np.array(coordinates, dtype=object).reshape(-1, 3)
    out = wp.array(np.asarray(coordinates, dtype=float), dtype=wp.vec3d, device=x.device, requires_grad=True)
    tape = wp._src.context.runtime.tape
    if tape is not None:
        inverse = wp.array(np.asarray(inverses, dtype=float), dtype=wp.mat33d, device=x.device)
        local = wp.array(np.asarray(pulled, dtype=float), dtype=wp.vec3d, device=x.device)
        dq = wp.array(
            np.tile(np.asarray(rotation_jacobians, dtype=float), (len(x), 1, 1)),
            dtype=wp.vec3d,
            device=x.device,
        )
        dh = wp.array(
            np.tile(np.asarray(size_jacobians, dtype=float), (len(x), 1, 1)), dtype=wp.mat33d, device=x.device
        )
        gradients = [a.grad if a.requires_grad else wp.zeros_like(a, requires_grad=False) for a in sources]

        def backward():
            wp.launch(
                _pullback_vjp,
                len(out),
                inputs=[inverse, local, dq, dh, out.grad],
                outputs=gradients,
                device=x.device,
            )

        tape.record_func(backward, [out, *[a for a in sources if a.requires_grad]])
    return out, coordinates
