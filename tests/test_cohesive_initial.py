"""Fixed-normal initial values reproduce analytic compression and cohesive loading."""

import numpy as np
import pytest

from blademaster_warp._cohesive_initial import fixed_normal_initial
from blademaster_warp._cohesive_solve import setup
from test_differentiable_cohesive import closed_pair


@pytest.mark.parametrize("integrated", [False, True])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    "gap,history", [(-1e-5, 0.0), (1e-6, 0.0), (4e-6, 0.0), (1e-5, 1e-4), (0.004, 0.003)]
)
def test_fixed_normal_initial_matches_normal_equilibrium(integrated, scale, gap, history):
    values = closed_pair(4e-6, [gap, 0.0, 0.0])
    values[3][:] = history
    compliance = setup(values, scale, integrated_force=integrated)[2]
    u, traction, selected, info = fixed_normal_initial(values, compliance, integrated_force=integrated)
    strength, gc = values[4][0, :2]
    area, mass, dt = values[7][0], values[8][0], values[9][0]
    failure = 2 * gc / strength
    mobility = 2 * dt**2 / mass
    capacity = area * strength
    if gap < 0 or (history == 0 and gap <= mobility * capacity):
        expected_gap = 0.0
        expected_force = gap / mobility
    elif history >= failure:
        expected_gap = gap
        expected_force = 0.0
    elif history > 0:
        stiffness = capacity * (1 - history / failure) / history
        expected_gap = gap / (1 + mobility * stiffness)
        assert expected_gap < history
        expected_force = stiffness * expected_gap
    else:
        expected_gap = (gap - mobility * capacity) / (1 - mobility * capacity / failure)
        expected_force = capacity * (1 - expected_gap / failure)
    force = traction if integrated else area * traction
    np.testing.assert_allclose(force, [[expected_force, 0.0, 0.0]], rtol=2e-8, atol=1e-12)
    np.testing.assert_allclose(values[0] + values[5] @ u, [[expected_gap, 0.0, 0.0]], atol=1e-14)
    np.testing.assert_allclose(u + dt**2 / mass * values[5].T @ force, 0.0, atol=1e-20)
    assert 0 < selected <= compliance
    assert max(info["initial_primal"], info["initial_dual"]) < 1e-10


def test_zero_area_initial_value_retains_free_motion():
    values = closed_pair(0.0, [-1e-4, 2e-5, 3e-5])
    compliance = setup(values, integrated_force=True)[2]
    u, force, _, info = fixed_normal_initial(values, compliance, integrated_force=True)
    np.testing.assert_array_equal(u, 0.0)
    np.testing.assert_array_equal(force, 0.0)
    assert info["initial_primal"] == info["initial_dual"] == 0.0
