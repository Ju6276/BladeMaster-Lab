"""Fixed-rate damping in the mass-normalized reference-affine coordinates.

L = I - P.T M_grid P / m is symmetric positive semidefinite. Its nullspace
contains global affine velocity fields. The reference transfer is constant
within a topology branch; duration (s) and rate (s^-1) remain differentiable.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, expm_multiply
import warp as wp


@wp.kernel
def _kinetic_loss(
    before: wp.array(dtype=wp.vec3d),
    after: wp.array(dtype=wp.vec3d),
    mass: wp.float64,
    energy: wp.array(dtype=wp.float64),
):
    p = wp.tid()
    value = wp.float64(0.0)
    for a in range(4):
        j = 4 * p + a
        value += wp.dot(before[j] - after[j], before[j] + after[j])
    energy[p] = wp.float64(0.5) * mass * value


@dataclass(frozen=True)
class AffineDampingStep:
    moments: wp.array
    energy: wp.array


def decay(transfer, moments, duration, rate):
    """Apply exp(-rate * duration * L) without assembling the square operator.

    Energy entries are signed particle kinetic changes in J. Redistribution
    can increase one particle's energy; only their sum is dissipated energy.
    """
    m = transfer.model
    transfer._check(moments, wp.vec3d, (4 * m.n,))
    transfer._check(duration, wp.float64, (1,))
    transfer._check(rate, wp.float64, (1,))
    t, eta = float(duration.numpy()[0]), float(rate.numpy()[0])
    if not np.isfinite([t, eta]).all() or t < 0.0 or eta < 0.0:
        raise ValueError("Damping duration and rate must be finite and nonnegative")
    value = moments.numpy()
    if not np.isfinite(value).all():
        raise ValueError("Damping moments must be finite")
    mapping = transfer.grid_map.numpy().copy()
    masses = transfer.grid_mass.copy() / m.mass
    size = mapping.shape[1]

    def generator(x):
        return x - mapping.T @ (masses[:, None] * (mapping @ x))

    exponent = -t * eta
    trace = size - np.sum(masses[:, None] * mapping * mapping)

    def product(x):
        return (exponent * generator(x.reshape(size, 1))).ravel()

    operator = LinearOperator(
        (size, size),
        dtype=np.float64,
        matvec=product,
        rmatvec=product,
        matmat=lambda x: exponent * generator(x),
    )

    def apply(x):
        return x.copy() if exponent == 0.0 else expm_multiply(operator, x, traceA=exponent * trace)

    advanced = apply(value)
    result = m.array(advanced, wp.vec3)
    tape = wp._src.context.runtime.tape
    if tape is not None:

        def backward():
            seed = result.grad.numpy()
            if moments.requires_grad:
                moments.grad.assign(moments.grad.numpy() + apply(seed))
            power = np.sum(seed * generator(advanced))
            if duration.requires_grad:
                duration.grad.assign(duration.grad.numpy() + [-eta * power])
            if rate.requires_grad:
                rate.grad.assign(rate.grad.numpy() + [-t * power])

        tape.record_func(backward, [a for a in [result, moments, duration, rate] if a.requires_grad])
    energy = m.zeros(m.n, float)
    wp.launch(_kinetic_loss, m.n, [moments, result, m.mass, energy], device=m.device)
    return AffineDampingStep(result, energy)
