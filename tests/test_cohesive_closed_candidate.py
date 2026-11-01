"""Analytic KKT candidates respect redundant rows, tensile strength and permanent history."""

import numpy as np
import pytest

from blademaster_warp._cohesive_expansion import E, Equations
from blademaster_warp._cohesive_refinement import (
    Balanced,
    closed_active_candidate,
    contact_active_candidate,
    regularized_response,
)


def equations(gap, count=1, shared=False, history=0.0):
    base = E(np.r_[gap, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mapping = np.zeros((9, 3))
    if not shared:
        mapping[:3] = np.eye(3)
    return Equations(
        [base] * count,
        [E(mapping)] * count,
        E(np.eye(3)),
        E(2.0),
        [E(0.25)] * count,
        np.full(count, history),
        np.tile([2.0, 2.0, 0.0], (count, 1)),
        [E(4.0 / count)] * count,
        np.full(count, shared),
    )


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("gap", [[0.2, 0.1, -0.1], [-5.0, 0.2, 0.0], [1.9, 0.0, 0.0]])
def test_closed_candidate_matches_force_balance_and_zero_opening(count, gap):
    system = equations(gap, count)
    state, openings, _ = closed_active_candidate(system)
    displacement = system.length * state[: system.size]
    traction = (4.0 * system.length * state[system.size :]).reshape(count, 3)
    np.testing.assert_allclose(displacement.high, -np.array(gap), rtol=1e-14, atol=1e-30)
    np.testing.assert_allclose(traction.high, np.tile(gap, (count, 1)), rtol=1e-14, atol=1e-30)
    for opening in openings:
        np.testing.assert_array_equal(opening.high, 0.0)
        np.testing.assert_array_equal(opening.low, 0.0)
    defect = system(state)[0]
    assert np.max(abs(defect.high)) <= 64 * np.finfo(float).eps ** 2 * np.max(abs(state.high))


@pytest.mark.parametrize("gap", [[2.1, 0.0, 0.0], [0.0, 2.1, 0.0]])
def test_strength_exceedance_rejects_closed_candidate(gap):
    assert closed_active_candidate(equations(gap)) is None


def test_shared_constraint_has_zero_displacement_and_reaction():
    state, openings, _ = closed_active_candidate(equations([0.0, 0.0, 0.0], shared=True))
    np.testing.assert_array_equal(state.high, 0.0)
    np.testing.assert_array_equal(state.low, 0.0)
    np.testing.assert_array_equal(openings[0].high, 0.0)


def test_positive_history_keeps_the_damage_branch():
    assert closed_active_candidate(equations([0.1, 0.0, 0.0], history=1e-22)) is None


@pytest.mark.parametrize("count", [1, 2, 4, 96])
@pytest.mark.parametrize("mu", [1e-2, 1e-18])
@pytest.mark.parametrize("gap", [-0.2, 0.2])
def test_contact_pivots_match_broken_interface_complementarity(count, mu, gap):
    system = equations([gap, 0.1, -0.1], count, history=2.0)
    initial = E(np.r_[[1.0, 0.0, 0.0], np.zeros(3 * count)])
    original = initial.numpy().copy()
    balanced = Balanced(system, 1.0)
    state, info = contact_active_candidate(balanced, initial, mu, np.ones(3 * count))
    expected_dual = min(gap, 0.0) / (4 + mu)
    np.testing.assert_allclose(
        (system.length * state[:3]).numpy(), [-4 * expected_dual, 0.0, 0.0], rtol=1e-14, atol=1e-30
    )
    np.testing.assert_allclose(
        (system.length * state[3:]).numpy().reshape(count, 3),
        np.tile([expected_dual, 0.0, 0.0], (count, 1)),
        rtol=1e-13,
        atol=1e-30,
    )
    residual = regularized_response(balanced, state, mu, np.ones(3 * count))[0]
    assert np.max(abs(residual.high)) <= 64 * np.finfo(float).eps ** 2 * np.max(abs(state.high))
    assert info["active_set_pivots"] == (1 if gap < 0 else 0)
    np.testing.assert_array_equal(initial.numpy(), original)
    assert system.fixed_contact_set is None
