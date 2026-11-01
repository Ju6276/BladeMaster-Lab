import numpy as np
import pytest
from blademaster_warp._cohesive_expansion import E
from blademaster_warp._cohesive_refinement import contact_active_candidate

class CubicEquation:
    size = 1
    def contact_set(self, state):
        return np.empty(0, dtype=bool)
    def with_contact_set(self, active):
        return self
    def __call__(self, state):
        return state * state * state - 1, (3 * state * state).reshape(1, 1), []

@pytest.mark.parametrize('initial', [100., 1000., 10000.])
def test_active_newton_can_reach_quadratic_convergence_from_a_distant_seed(initial):
    result, info = contact_active_candidate(CubicEquation(), E([initial]), 0., np.empty(0))
    np.testing.assert_allclose(result.high, [1.], rtol=0., atol=64*np.finfo(float).eps**2)
    assert abs((result * result * result - 1).high[0]) <= 64*np.finfo(float).eps**2
