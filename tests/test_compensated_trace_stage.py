"""Composed trace maps through affine stages, restarts and their reverse pass."""

from types import SimpleNamespace

import numpy as np
import warp as wp

from blademaster_warp.cohesive_geometry import InterfaceGeometry
from blademaster_warp.cohesive_stages import AffineState
from blademaster_warp.compensated_state import CompensatedArray
from test_cohesive_grid_update import patch
from test_cohesive_stages import setup, rollout, targets


def prepared():
    m, state, geometry, history, data = setup("opened")
    _, _, _, _, raw = patch(gc=70.0)
    trace = SimpleNamespace(separation_map=raw["separation_map"], frame_map=raw["frame_map"])
    return m, state, geometry, history, data, trace


def test_composed_trace_stage_restart_preserves_all_state_and_remainders():
    m, state, geometry, history, data, trace = prepared()
    B, G = data["transfer"].compose_trace(trace)
    data = dict(data, separation_map=B, frame_map=G)
    first = rollout(state, geometry, history, data, steps=2)
    restored = AffineState.restore(first.state.snapshot(), device=m.device)
    restored_geometry = InterfaceGeometry.restore(first.geometry.snapshot(), device=m.device)
    restored_history = m.array(first.history.numpy(), float)
    rebuilt = dict(
        data,
        separation_map=CompensatedArray.restore(B.snapshot(), dtype=wp.float64, device=m.device),
        frame_map=CompensatedArray.restore(G.snapshot(), dtype=wp.vec3d, device=m.device),
    )
    direct = rollout(first.state, first.geometry, first.history, data, steps=1)
    resumed = rollout(restored, restored_geometry, restored_history, rebuilt, steps=1)
    for actual, expected in zip(targets(resumed), targets(direct)):
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    for name in direct.state.rounding_remainders:
        np.testing.assert_array_equal(
            resumed.state.rounding_remainders[name], direct.state.rounding_remainders[name]
        )
    np.testing.assert_array_equal(resumed.geometry.gap_remainder, direct.geometry.gap_remainder)
    np.testing.assert_array_equal(resumed.geometry.frame_remainder, direct.geometry.frame_remainder)


def test_composed_trace_stage_reverse_matches_recomputed_map():
    m, state, geometry, history, data, trace = prepared()

    def run():
        B, G = data["transfer"].compose_trace(trace)
        return rollout(state, geometry, history, dict(data, separation_map=B, frame_map=G), steps=2)

    with wp.Tape() as tape:
        result = run()
    rng = np.random.default_rng(372)
    seed_v = rng.normal(size=result.state.v.numpy().shape)
    seed_f = rng.normal(size=result.interface.force.numpy().shape)
    tape.backward(
        grads={
            result.state.v: m.array(seed_v, wp.vec3, False),
            result.interface.force: m.array(seed_f, wp.vec3, False),
        }
    )
    source = trace.separation_map
    base = source.numpy().copy()
    direction = base * rng.normal(size=base.shape) * 0.01
    ad = float(np.sum(source.grad.numpy() * direction))
    for epsilon in [1e-4, 1e-5]:
        trials = []
        for sign in [-1.0, 1.0]:
            source.assign(base + sign * epsilon * direction)
            updated = run()
            trials.append(
                float(
                    np.sum(seed_v * updated.state.v.numpy())
                    + np.sum(seed_f * updated.interface.force.numpy())
                )
            )
        fd = (trials[1] - trials[0]) / (2 * epsilon)
        np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-8)
    source.assign(base)
