import numpy as np
import pytest

from blademaster_warp._cohesive_expansion import E, assign, stack
from blademaster_warp._cohesive_refinement import contact_active_candidate


class ScaledCubic:
    size = 1

    def __init__(self, row_scale):
        self.row_scale = row_scale

    def contact_set(self, state):
        return np.zeros(1, dtype=bool)

    def with_contact_set(self, active):
        return self

    def __call__(self, state):
        x, y = state[0], state[1]
        residual = stack([self.row_scale * (x * x * x + x - 2.0), y - 1.0])
        jacobian = E(np.zeros((2, 2)))
        assign(jacobian, (0, 0), self.row_scale * (3.0 * x * x + 1.0))
        assign(jacobian, (1, 1), E(1.0))
        return residual, jacobian, []


@pytest.mark.parametrize('row_scale', [1e-40, 1.0, 1e40])
@pytest.mark.parametrize('initial_x', [0.0, 0.25])
def test_active_root_is_invariant_under_equation_row_units(row_scale, initial_x):
    mu = .01
    state, _ = contact_active_candidate(
        ScaledCubic(row_scale), E([initial_x, 1.0]), mu, np.ones(1))
    np.testing.assert_allclose(state.numpy(), [1.0, 1.0 / (1.0 - mu)], rtol=2e-15, atol=0.0)
