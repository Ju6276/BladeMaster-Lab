"""Actual material-flow contact events retain history and subdivide the same duration."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_event_stages import advance_contact_events
from blademaster_warp.cohesive_pair_events import guard
from blademaster_warp.compensated_state import CompensatedArray
from test_spatial_pair_stages import setup


def advance(mode, *, staggered=False, rebound=False):
    model, state, history, parameters, diameter, _, data, _, _ = setup(mode)
    codes = np.where(model.reference[:, 0] < 0, 1, 2)
    if staggered:
        x = state.x.numpy()
        side = np.where(codes == 1, -1, 1)
        x[:, 0] += side * (model.reference[:, 1] - model.reference[:, 1].min()) * 1e-5
        state.x.assign(x)
    if rebound:
        data["duration"].assign([4e-6])
        side = np.where(codes == 1, -1, 1)
        load = model.array(side[:, None] * [2e5, 0, 0] * model.mass, wp.vec3)
        data["force"] = lambda s, z, t: load
    result = advance_contact_events(
        **data, pair_history=history, codes=codes, parameters=parameters, diameter=diameter
    )
    return model, state, history, parameters, diameter, data, codes, result


def validate(model, initial, data, result):
    durations = [segment.duration.numpy()[0] for segment in result.segments]
    assert all(value > 0 for value in durations)
    np.testing.assert_allclose(sum(durations), data["duration"].numpy()[0], rtol=2e-15, atol=0)
    times = [event.time.numpy()[0] for event in result.events]
    assert all(a < b for a, b in zip(times[:-1], times[1:]))
    assert all(0 < value <= data["duration"].numpy()[0] for value in times)
    np.testing.assert_allclose(
        model.mass * np.sum(result.material.state.v.numpy() - initial.v.numpy(), axis=0), 0, atol=1e-18
    )


@pytest.mark.parametrize("mode", ["entry", "exit"])
def test_single_contact_event_keeps_incoming_history_and_releases_once(mode):
    model, state, history, parameters, diameter, data, codes, result = advance(mode)
    validate(model, state, data, result)
    assert len(result.events) == 1
    assert len(result.events[0].changes) == 16
    assert len(result.segments) == 2
    if mode == "entry":
        assert all(active for _, _, active in result.events[0].changes)
        assert result.released_energy.numpy()[0] == 0
        assert len(history.pairs) == 0 and len(result.pair_history.pairs) == 16
    else:
        assert not any(active for _, _, active in result.events[0].changes)
        assert result.released_energy.numpy()[0] > 0
        assert len(history.pairs) == 16 and len(result.pair_history.pairs) == 0
        before = result.segments[0].step
        assert len(before.history.pairs) == 16
        np.testing.assert_allclose(
            result.released_energy.numpy()[0], np.sum(before.final_storage.numpy()[:, 1]), rtol=2e-14
        )
    end = result.material.state
    continued = advance_contact_events(
        **dict(data, state=end, geometry=result.material.geometry, history=result.material.history),
        pair_history=result.pair_history,
        modes=result.modes,
        codes=codes,
        parameters=parameters,
        diameter=diameter,
    )
    assert not continued.events
    assert continued.released_energy.numpy()[0] == 0


def test_staggered_contacts_are_regenerated_after_each_actual_event():
    model, state, _, _, diameter, data, _, result = advance("entry", staggered=True)
    validate(model, state, data, result)
    assert len(result.events) >= 4
    assert sum(len(event.changes) for event in result.events) == 16
    assert all(active for event in result.events for _, _, active in event.changes)
    for event in result.events:
        before = next(
            segment.step.material.state
            for segment in result.segments
            if segment.start.numpy()[0] + segment.duration.numpy()[0] == event.time.numpy()[0]
        )
        position = CompensatedArray(before.x, before.rounding_remainders.get("x"))
        for p, q, _ in event.changes:
            assert abs(guard(position, (p, q), diameter).high.numpy()[0]) <= 1e-15


def test_entry_and_exit_inside_one_step_are_both_integrated():
    model, state, _, _, _, data, _, result = advance("entry", rebound=True)
    validate(model, state, data, result)
    assert len(result.events) >= 2
    assert any(active for event in result.events for _, _, active in event.changes)
    assert any(not active for event in result.events for _, _, active in event.changes)
    assert not len(result.pair_history.pairs)
    assert result.released_energy.numpy()[0] > 0
