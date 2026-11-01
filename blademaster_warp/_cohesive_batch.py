"""Batched cohesive equations with the scalar evaluation's summation order."""

import numpy as np

from ._compensated import Expansion as E
from ._compensated_linalg import divide
from ._cohesive_expansion import assign, stack, scale_power_two


def transpose(value):
    return E(np.swapaxes(value.high, -2, -1), np.swapaxes(value.low, -2, -1))


def positive(value):
    return (value.high > 0) | ((value.high == 0) & (value.low > 0))


def norm(value):
    largest = np.max(abs(value.high), axis=-1, initial=0.0)
    exponent = np.frexp(largest)[1]
    scaled = scale_power_two(value, -exponent[..., None])
    squared = (scaled * scaled).sum(axis=-1)
    result = E(np.sqrt(squared.high))
    nonzero = result.high != 0
    if np.any(nonzero):
        selected = result[nonzero]
        assign(result, nonzero, selected + divide(squared[nonzero] - selected * selected, 2 * selected))
    result = scale_power_two(result, exponent)
    assign(result, largest == 0, 0.0)
    return result


def outer(a, b):
    return a[..., :, None] * b[..., None, :]


def cross(a, b):
    return stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                  a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                  a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)


def skew(a):
    zero = E(np.zeros(a.shape[:-1]))
    return stack([stack([zero, -a[..., 2], a[..., 1]], axis=-1),
                  stack([a[..., 2], zero, -a[..., 0]], axis=-1),
                  stack([-a[..., 1], a[..., 0], zero], axis=-1)], axis=-2)


def mv(matrix, vector):
    return (matrix @ vector[..., None])[..., 0]


def local_hessian(g, a, b):
    oriented = cross(a, b)
    length = norm(oriented)
    n = divide(oriented, length[:, None])
    c = (g * n).sum(axis=-1)
    z = divide(g - c[:, None] * n, length[:, None])
    dn = divide(E(np.eye(3)) - outer(n, n), length[:, None, None])
    haa = -divide(outer(g, n) + outer(n, g) + c[:, None, None] * np.eye(3)
                  - 3 * c[:, None, None] * outer(n, n), (length * length)[:, None, None])
    Ja, Jb = -skew(b), skew(a)
    H = E(np.zeros((len(g.high), 9, 9)))
    assign(H, (slice(None), slice(0, 3), slice(3, 6)), dn @ Ja)
    assign(H, (slice(None), slice(0, 3), slice(6, 9)), dn @ Jb)
    assign(H, (slice(None), slice(3, 6), slice(0, 3)), transpose(Ja) @ dn)
    assign(H, (slice(None), slice(6, 9), slice(0, 3)), transpose(Jb) @ dn)
    assign(H, (slice(None), slice(3, 6), slice(3, 6)), transpose(Ja) @ haa @ Ja)
    assign(H, (slice(None), slice(6, 9), slice(6, 9)), transpose(Jb) @ haa @ Jb)
    assign(H, (slice(None), slice(3, 6), slice(6, 9)), transpose(Ja) @ haa @ Jb - skew(z))
    assign(H, (slice(None), slice(6, 9), slice(3, 6)), transpose(Jb) @ haa @ Ja + skew(z))
    return H


def evaluate(equations, y):
    size, count = equations.size, equations.count
    u = equations.length * y[:size]
    dual = (equations.length * y[size:]).reshape(count, 3)
    ru = equations.reduced @ u
    rg = E(np.zeros((count, 3)))
    J = E(np.zeros((len(y.high), len(y.high))))
    assign(J, (slice(0, size), slice(0, size)), equations.reduced)
    gaps = E(np.zeros((count, 3)))
    if not count:
        return divide(ru, equations.length), J, []
    maps = stack(equations.maps)
    local = stack(equations.base) + maps @ u
    compliance = stack(equations.compliance)
    factors = stack(equations.factors)
    zero_compliance = (compliance.high == 0) & (compliance.low == 0)
    free_normal = np.ones(count, dtype=bool) if equations.fixed_contact_set is None else ~equations.fixed_contact_set
    free_loading = np.ones(count, dtype=bool) if equations.fixed_loading_set is None else ~equations.fixed_loading_set
    fast = ((equations.shared | (zero_compliance & free_normal)) & free_loading
            & np.all(local.high[:, :3] == 0, axis=1) & np.all(local.low[:, :3] == 0, axis=1)
            & np.all(dual.high == 0, axis=1) & np.all(dual.low == 0, axis=1))
    columns = np.arange(size)

    def blocks(indices, upper, lower, diagonal):
        spans = size + 3 * indices[:, None] + np.arange(3)[None, :]
        assign(J, (columns[None, :, None], spans[:, None, :]), upper)
        assign(J, (spans[:, :, None], columns[None, None, :]), lower)
        assign(J, (spans[:, :, None], spans[:, None, :]), diagonal)

    fast_indices = np.flatnonzero(fast)
    if len(fast_indices):
        blocks(fast_indices, factors[fast, None, None] * transpose(maps[fast, :3]),
               E(np.zeros((len(fast_indices), 3, size))),
               E(np.where(equations.shared[fast], 1., -1.)[:, None, None] * np.eye(3)))
    indices = np.flatnonzero(~fast)
    if not len(indices):
        return divide(E(np.r_[ru.high, rg.high.ravel()], np.r_[ru.low, rg.low.ravel()]), equations.length), J, [gaps[i] for i in range(count)]
    A = maps[indices]
    local = local[indices]
    g, a, b = local[:, :3], local[:, 3:6], local[:, 6:9]
    d = dual[indices]
    compliance, factors = compliance[indices], factors[indices]
    oriented = cross(a, b)
    measure = norm(oriented)
    n = divide(oriented, measure[:, None])
    N = divide(E(np.eye(3)) - outer(n, n), measure[:, None, None]) @ (-skew(b) @ A[:, 3:6] + skew(a) @ A[:, 6:9])
    z = g + d
    component = (z * n).sum(axis=-1)
    closed = positive(-component) & positive(compliance)
    if equations.fixed_contact_set is not None:
        closed = equations.fixed_contact_set[indices].copy()
    compression = E(np.zeros(len(indices)))
    assign(compression, closed, component[closed])
    p = z - compression[:, None] * n
    radius = norm(p)
    sigma, gc = E(equations.material[indices, 0]), E(equations.material[indices, 1])
    h = E(equations.history[indices])
    failure = divide(2 * gc, sigma)
    opening = E(radius.high.copy(), radius.low.copy())
    derivative = E(np.ones(len(indices)))
    traction_ratio = E(np.zeros(len(indices)))
    traction_derivative = E(np.zeros(len(indices)))
    elastic = positive(compliance) & positive(failure - h)
    assign(opening, elastic, 0.0)
    assign(derivative, elastic, 0.0)
    assign(traction_ratio, elastic, 1.0)
    assign(traction_derivative, elastic, 1.0)
    damaged = elastic & positive(h)
    if np.any(damaged):
        stiffness = divide(sigma[damaged] * (1 - divide(h[damaged], failure[damaged])), h[damaged])
        value = divide(1, 1 + compliance[damaged] * stiffness)
        assign(derivative, damaged, value)
        assign(opening, damaged, radius[damaged] * value)
        value = divide(compliance[damaged] * stiffness, 1 + compliance[damaged] * stiffness)
        assign(traction_ratio, damaged, value)
        assign(traction_derivative, damaged, value)
    loading = positive(opening - h) | ((h.high == 0) & positive(radius - compliance * sigma))
    if equations.fixed_loading_set is not None:
        loading = equations.fixed_loading_set[indices].copy()
    loading &= elastic
    if np.any(loading):
        c, s, f, r = compliance[loading], sigma[loading], failure[loading], radius[loading]
        value = divide(1, 1 - divide(c * s, f))
        assign(derivative, loading, value)
        assign(opening, loading, (r - c * s) * value)
        assign(traction_ratio, loading, c * s * (divide(1, r) - divide(1, f)) * value)
        assign(traction_derivative, loading, -divide(c * s, f) * value)
        released = loading & ~positive(failure - opening)
        assign(opening, released, radius[released])
        assign(derivative, released, 1.0)
        assign(traction_ratio, released, 0.0)
        assign(traction_derivative, released, 0.0)
    nonzero = radius.high != 0
    ratio = E(np.zeros(len(indices)))
    assign(ratio, nonzero, divide(opening[nonzero], radius[nonzero]))
    prox = ratio[:, None] * p
    face = mv(transpose(N), prox)
    if not equations.constrain_normal:
        prox = prox + compression[:, None] * n
        face = -mv(transpose(N), d)
    increments = factors[:, None] * (mv(transpose(A[:, :3]), d) + compression[:, None] * face)
    ru = stack([ru, *[increments[i] for i in range(len(indices))]]).sum(axis=0)
    use_gap = ((np.max(abs(g.high), axis=1) <= np.max(abs(d.high), axis=1))
               & positive(failure - h) & ~positive(opening - h))
    residual = compression[:, None] * n + traction_ratio[:, None] * p - d
    if not equations.constrain_normal:
        residual = traction_ratio[:, None] * p - d
    assign(residual, use_gap, (g - prox)[use_gap])
    shared = equations.shared[indices]
    assign(residual, shared, d[shared])
    assign(rg, indices, residual)
    radial = derivative[:, None, None] * np.eye(3)
    radial_traction = traction_ratio[:, None, None] * np.eye(3)
    if np.any(nonzero):
        direction = divide(p[nonzero], radius[nonzero, None])
        value = divide(opening[nonzero], radius[nonzero])
        assign(radial, nonzero, value[:, None, None] * np.eye(3)
               + (derivative[nonzero] - value)[:, None, None] * outer(direction, direction))
        assign(radial_traction, nonzero, radial_traction[nonzero]
               + (traction_derivative[nonzero] - traction_ratio[nonzero])[:, None, None] * outer(direction, direction))
    Dz = radial @ (E(np.eye(3)) - closed.astype(int)[:, None, None] * outer(n, n))
    Dn = -closed.astype(int)[:, None, None] * radial @ (component[:, None, None] * np.eye(3) + outer(n, z))
    Tz = radial_traction + closed.astype(int)[:, None, None] * radial @ outer(n, n)
    if not equations.constrain_normal:
        Dz = Dz + closed.astype(int)[:, None, None] * outer(n, n)
        Dn = closed.astype(int)[:, None, None] * (E(np.eye(3)) - radial) @ (component[:, None, None] * np.eye(3) + outer(n, z))
        Tz = radial_traction @ (E(np.eye(3)) - closed.astype(int)[:, None, None] * outer(n, n))
    B = A[:, :3]
    if np.any(closed):
        C, NC, BC = A[closed], N[closed], B[closed]
        if equations.constrain_normal:
            Hc = transpose(C) @ local_hessian(prox[closed], a[closed], b[closed]) @ C
            face_gradient = Hc - transpose(BC) @ NC - transpose(NC) @ BC + transpose(NC) @ (Dz[closed] @ BC + Dn[closed] @ NC)
        else:
            face_gradient = -transpose(C) @ local_hessian(d[closed], a[closed], b[closed]) @ C + transpose(BC) @ NC + transpose(NC) @ BC
        compression_gradient = mv(transpose(BC), n[closed]) + mv(transpose(NC), z[closed])
        increments = factors[closed, None, None] * (compression[closed, None, None] * face_gradient + outer(face[closed], compression_gradient))
        block = stack([equations.reduced, *[increments[i] for i in range(len(increments.high))]]).sum(axis=0)
        assign(J, (slice(0, size), slice(0, size)), block)
    face_dual = transpose(N) @ Dz if equations.constrain_normal else -transpose(N)
    upper = factors[:, None, None] * (transpose(B) + compression[:, None, None] * face_dual + closed.astype(int)[:, None, None] * outer(face, n))
    lower = Tz @ B - Dn @ N
    diagonal = -Dz
    assign(lower, shared, 0.0)
    assign(diagonal, shared, np.eye(3))
    blocks(indices, upper, lower, diagonal)
    assign(gaps, indices, prox)
    residual = divide(E(np.r_[ru.high, rg.high.ravel()], np.r_[ru.low, rg.low.ravel()]), equations.length)
    return residual, J, [gaps[i] for i in range(count)]
