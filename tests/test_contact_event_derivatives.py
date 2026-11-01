"""Natural differences rebuild distinct contact times and all segment histories."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_event_stages import advance_contact_events
from test_spatial_pair_stages import setup


@pytest.mark.parametrize("mode", ["entry", "exit"])
def test_distinct_contact_events_include_state_time_geometry_and_energy_derivatives(mode):
    model, state, history, parameters, diameter, _, data, _, _ = setup(mode)
    codes = np.where(model.reference[:, 0] < 0, 1, 2)
    side = np.where(codes == 1, -1, 1)
    offset = model.reference[:, 1:] - model.reference[:, 1:].min(axis=0)
    x = state.x.numpy()
    x[:, 0] += side * 1e-6 * (offset[:, 0] + np.sqrt(2) * offset[:, 1])
    state.x.assign(x)

    def advance():
        return advance_contact_events(
            **data, pair_history=history, codes=codes, parameters=parameters, diameter=diameter
        )

    def outputs(result):
        return [
            result.material.state.x,
            result.material.state.v,
            result.material.state.F,
            result.released_energy,
            result.segments[-1].step.final_storage,
        ]

    with wp.Tape() as tape:
        result = advance()
    assert len(result.events) == 16
    assert all(len(event.changes) == 1 for event in result.events)
    rng = np.random.default_rng(5110)
    values = outputs(result)
    scales = [0.01, 1.0, 1.0, 1e-9, 1e-7]
    seeds = [rng.normal(size=value.numpy().shape) / scale for value, scale in zip(values, scales)]
    tape.backward(
        grads={
            value: model.array(seed, value.dtype, False) for value, seed in zip(values, seeds) if value.size
        }
    )
    sources = [state.x, state.v, data["duration"], diameter, parameters]
    amplitudes = [1e-6, 0.02, 2e-7, 1e-6, np.array([[100, 50, 0.01, 0.01, 0.01]])]
    initial = [source.numpy().copy() for source in sources]
    directions = [rng.normal(size=value.shape) * scale for value, scale in zip(initial, amplitudes)]
    ad = sum(np.sum(source.grad.numpy() * direction) for source, direction in zip(sources, directions))
    for epsilon in [1e-4, 1e-5]:
        trials = []
        for sign in [-1, 1]:
            for source, value, direction in zip(sources, initial, directions):
                source.assign(value + sign * epsilon * direction)
            trials.append([value.numpy() for value in outputs(advance())])
        for source, value in zip(sources, initial):
            source.assign(value)
        fd = sum(np.sum((b - a) * seed) for a, b, seed in zip(*trials, seeds)) / (2 * epsilon)
        np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=2e-8)
