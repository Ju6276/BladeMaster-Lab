"""Analytic minimum-pressure reactions include zero-multiplier constraints."""

import numpy as np
import pytest

from blademaster_warp._cohesive_expansion import E, Equations
from blademaster_warp._cohesive_projection import select


@pytest.mark.parametrize("loaded", [0, 1, 2])
@pytest.mark.parametrize("scale", [2.0**-30, 1.0, 2.0**30])
@pytest.mark.parametrize("distinct_maps", [False, True])
@pytest.mark.parametrize("open_gap", [0.0, 2.0**-80, 2.0**-20])
def test_minimum_pressure_is_independent_of_zero_multiplier_seed(loaded, scale, distinct_maps, open_gap):
    coefficients = np.array([1., 2., 4.]) if distinct_maps else np.ones(3)
    area = np.array([0.5, 1., 2.])
    displacement = 0.25 * scale
    bases, maps = [], []
    for coefficient in coefficients:
        bases.append(E(np.r_[[-coefficient * displacement, 0., 0.], [0., 1., 0.], [0., 0., 1.]]))
        mapping = np.zeros((9, 3))
        mapping[:3] = coefficient * np.eye(3)
        maps.append(E(mapping))
    if open_gap:
        bases[-1] = E(bases[-1].high, np.r_[open_gap * scale, np.zeros(8)])
        loaded %= 2
    equations = Equations(
        bases, maps, E(np.eye(3)), E(2.), [E(0.25)] * 3,
        np.full(3, 2.), np.tile([2., 2., 0.], (3, 1)),
        [E(1.)] * 3, np.zeros(3, dtype=bool),
    )
    state = np.zeros(12)
    state[0] = displacement / 2.
    total = -state[0]
    state[3 + 3 * loaded] = total / coefficients[loaded]
    selected, nullity, branches = select(equations, E(state), area, True)
    active_area = area.copy()
    if open_gap:
        active_area[-1] = 0.0
    expected = total * active_area * coefficients / np.sum(active_area * coefficients**2)
    np.testing.assert_allclose(selected.numpy()[3::3], expected, rtol=2e-14, atol=0.)
    np.testing.assert_array_equal(selected.numpy()[:3], state[:3])
    np.testing.assert_array_equal(branches.contact, active_area > 0.0)
    assert nullity == np.count_nonzero(active_area) - 1
