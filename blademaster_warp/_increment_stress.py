"""Fixed-corotated stress from the stored deformation increment H = F - I."""

import warp as wp


@wp.func
def terms(h: wp.mat33d):
    identity = wp.identity(n=3, dtype=wp.float64)
    f = identity + h
    U, sigma, V = wp.svd3(f)
    rotation = U * wp.transpose(V)
    rotation = wp.float64(0.5) * (rotation + wp.transpose(wp.inverse(rotation)))
    strain2 = h + wp.transpose(h) + wp.transpose(h) * h
    stretch = wp.transpose(rotation) * f
    difference = rotation * strain2 * wp.inverse(stretch + identity)
    jm1 = (
        wp.trace(h)
        + wp.determinant(h)
        + h[0, 0] * h[1, 1]
        + h[0, 0] * h[2, 2]
        + h[1, 1] * h[2, 2]
        - h[0, 1] * h[1, 0]
        - h[0, 2] * h[2, 0]
        - h[1, 2] * h[2, 1]
    )
    return difference, jm1, rotation, stretch


@wp.func
def elastic(h: wp.mat33d, mu: wp.float64, lam: wp.float64):
    difference, jm1, rotation, stretch = terms(h)
    f = wp.identity(n=3, dtype=wp.float64) + h
    return wp.float64(2.0) * mu * difference + lam * jm1 * (wp.float64(1.0) + jm1) * wp.transpose(
        wp.inverse(f)
    )


@wp.func_grad(elastic)
def elastic_grad(h: wp.mat33d, mu: wp.float64, lam: wp.float64, adj: wp.mat33d):
    identity = wp.identity(n=3, dtype=wp.float64)
    f = identity + h
    difference, jm1, rotation, stretch = terms(h)
    stretch = wp.float64(0.5) * (stretch + wp.transpose(stretch))
    rhs = wp.transpose(rotation) * adj - wp.transpose(adj) * rotation
    axial = wp.inverse(wp.trace(stretch) * identity - stretch) * wp.vec3d(rhs[2, 1], rhs[0, 2], rhs[1, 0])
    skew = wp.mat33d(0.0, -axial[2], axial[1], axial[2], 0.0, -axial[0], -axial[1], axial[0], 0.0)
    inverse_t = wp.transpose(wp.inverse(f))
    J = wp.float64(1.0) + jm1
    contraction = wp.ddot(adj, inverse_t)
    wp.adjoint[h] += (
        wp.float64(2.0) * mu * (adj - rotation * skew)
        + lam * (wp.float64(2.0) * J - wp.float64(1.0)) * J * contraction * inverse_t
        - lam * jm1 * J * inverse_t * wp.transpose(adj) * inverse_t
    )
    wp.adjoint[mu] += wp.float64(2.0) * wp.ddot(adj, difference)
    wp.adjoint[lam] += jm1 * J * contraction


@wp.kernel
def compute(
    H: wp.array(dtype=wp.mat33d),
    Fp: wp.array(dtype=wp.mat33d),
    young: wp.array(dtype=wp.float64),
    poisson: wp.float64,
    stress: wp.array(dtype=wp.mat33d),
):
    p = wp.tid()
    identity = wp.identity(n=3, dtype=wp.float64)
    inverse = wp.inverse(Fp[p])
    he = (H[p] - (Fp[p] - identity)) * inverse
    mu = young[0] / (wp.float64(2.0) * (wp.float64(1.0) + poisson))
    lam = young[0] * poisson / ((wp.float64(1.0) + poisson) * (wp.float64(1.0) - wp.float64(2.0) * poisson))
    stress[p] = elastic(he, mu, lam) * wp.transpose(inverse)
