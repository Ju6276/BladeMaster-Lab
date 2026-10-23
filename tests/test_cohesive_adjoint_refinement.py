"""Analytic opening chains across proximal scales and stiff transverse tangents."""

import numpy as np
import pytest

from blademaster_warp.differentiable_cohesive import _implicit_pullback
from test_differentiable_cohesive import arrays


@pytest.mark.parametrize("integrated", [False, True])
@pytest.mark.parametrize("count", [1, 4])
@pytest.mark.parametrize("gap_size", [1e-12, 1e-18])
@pytest.mark.parametrize("force_compliance", [1e-10, 1e-13])
def test_opening_chain_adjoint_matches_eliminated_tangent(integrated, count, gap_size, force_compliance):
    strength, gc, area, mass, dt = 5e4, 70.0, 1e-8, 1e-4, 2e-6
    failure = 2 * gc / strength
    mobility = dt**2 / mass
    b = np.zeros((count, count + 1))
    for i in range(count):
        b[i, i : i + 2] = [-1.0, 1.0]
    gap = np.zeros((count, 3))
    gap[:, 0] = gap_size
    force = np.zeros_like(gap)
    force[:, 0] = area * strength * (1 - gap_size / failure)
    u = -mobility * b.T @ force
    values = [
        gap - b @ u,
        np.tile(np.eye(3), (count, 1, 1)),
        np.tile([1.0, 0.0, 0.0], (count, 1)),
        np.zeros(count),
        np.tile([strength, gc, 0.0], (count, 1)),
        b,
        np.zeros((count, count + 1, 3)),
        np.full(count, area),
        np.array([mass]),
        np.array([dt]),
    ]
    rng = np.random.default_rng(73)
    seed_u = rng.normal(size=u.shape) * 1e8
    seed_force = rng.normal(size=force.shape) * 1e-8
    compliance = force_compliance * (1.0 if integrated else area)
    traction = force if integrated else force / area
    gradients = _implicit_pullback(
        arrays(values),
        values,
        u,
        traction,
        compliance,
        seed_u.ravel(),
        seed_force.ravel() / force_compliance,
        integrated_force=integrated,
    )
    left, singular, right = np.linalg.svd(b, full_matrices=False)
    expected = np.empty_like(gap)
    stiffnesses = [-area * strength / failure, force[0, 0] / gap_size, force[0, 0] / gap_size]
    for axis, stiffness in enumerate(stiffnesses):
        denominator = 1 / stiffness + mobility * singular**2
        expected[:, axis] = left @ (
            (left.T @ seed_force[:, axis] - mobility * singular * (right @ seed_u[:, axis])) / denominator
        )
    np.testing.assert_allclose(gradients[0], expected, rtol=2e-8, atol=1e-7)
    assert all(np.isfinite(value).all() for value in gradients)
