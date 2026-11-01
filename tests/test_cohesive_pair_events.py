"""Swept pair queries include between-sample crossings and preserve guard gradients."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_pair_events import candidates, guard
from blademaster_warp.compensated_state import CompensatedArray


def array(values, kind=wp.vec3d):
    return wp.array(values, dtype=kind, device="cuda:0", requires_grad=True)


@pytest.mark.parametrize(
    "positions,directions",
    [
        ([-2, 0, 2], [-1, 1]),
        ([2, 0, 2], [-1, 1]),
        ([2, -2, 2], [-1, 1, -1, 1]),
        ([2, 1, 2], []),
        ([2, 1.1, 5], [-1, 1]),
        ([0.5, 1, 1.5], [1]),
    ],
)
def test_quadratic_sweep_detects_entry_exit_reentry_and_tangency(positions, directions):
    states = [array([[0, 0, 0], [x, 0, 0]]) for x in positions]
    events = candidates(*states, np.array([[0, 0, 0], [0.1, 0, 0]]), np.array([1, 2]), 1.0, 0.1)
    assert [event.direction for event in events] == directions
    assert all(event.pair == (0, 1) and 0 <= event.lower <= event.upper <= 1 for event in events)


def test_sweep_keeps_reference_neighbour_exclusion_and_detects_far_same_field():
    states = [array([[0, 0, 0], [x, 0, 0]]) for x in [2, 0.5, 2]]
    codes = np.array([0, 0])
    assert not candidates(*states, np.array([[0, 0, 0], [0.1, 0, 0]]), codes, 1.0, 0.1)
    assert len(candidates(*states, np.array([[0, 0, 0], [0.3, 0, 0]]), codes, 1.0, 0.1)) == 2


def test_sub_ulp_sweep_and_guard_keep_distance_remainders():
    high = array([[0, 0, 0], [1, 0, 0]])
    states = [
        CompensatedArray(high, np.array([[0, 0, 0], [offset, 0, 0]])) for offset in [2**-80, 0, -(2**-80)]
    ]
    events = candidates(*states, np.array([[0, 0, 0], [0.1, 0, 0]]), np.array([1, 2]), 1.0, 0.1)
    assert len(events) == 1 and events[0].direction == -1
    assert events[0].lower <= 0.5 <= events[0].upper
    diameter = array([1.0], wp.float64)
    assert guard(states[0], (0, 1), diameter).high.numpy()[0] == 2**-80
    assert guard(states[2], (0, 1), diameter).high.numpy()[0] == -(2**-80)


def test_guard_position_and_diameter_derivatives():
    positions = array([[0.2, -0.3, 0.1], [-0.1, 0.2, 0.3]])
    diameter = array([0.7], wp.float64)
    with wp.Tape() as tape:
        value = guard(positions, (0, 1), diameter)
    tape.backward(grads={value.high: array([1.0], wp.float64)})
    rng = np.random.default_rng(5104)
    for source in [positions, diameter]:
        baseline = source.numpy().copy()
        direction = rng.normal(size=baseline.shape)
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            values = []
            for sign in [-1, 1]:
                source.assign(baseline + sign * epsilon * direction)
                values.append(guard(positions, (0, 1), diameter).high.numpy()[0])
            np.testing.assert_allclose(ad, (values[1] - values[0]) / (2 * epsilon), rtol=2e-7)
        source.assign(baseline)
