"""Checkpointing through the birth of indexed particle-contact history."""

import numpy as np
import warp as wp

from blademaster_warp.cohesive_pairing import PairHistory
from blademaster_warp.cohesive_spatial_contact import advance_spatial_pairs
from blademaster_warp.cohesive_stages import AffineState
from test_spatial_pair_stages import setup


def test_spatial_pair_birth_checkpoint_matches_complete_backward():
    model, state, initial_history, parameters, diameter, _, data, _, _ = setup("entry")
    codes = np.where(model.reference[:, 0] < 0, 1, 2)

    def advance(s, h):
        return advance_spatial_pairs(
            **dict(data, state=s), pair_history=h, codes=codes, parameters=parameters, diameter=diameter
        )

    def arrays(s):
        return [getattr(s, name) for name in ["x", "v", "affine", "F", "Fp", "stress"]]

    def targets(result):
        return [
            *arrays(result.material.state),
            result.history.values.high,
            result.initial_storage,
            result.final_storage,
        ]

    sources = [state.x, state.v, state.affine, state.F, state.Fp]
    shared = [data["duration"], parameters, diameter, model.young]
    with wp.Tape() as complete:
        first = advance(state, initial_history)
        last = advance(first.material.state, first.history)
    assert len(initial_history.pairs) == 0 and len(first.history.pairs) == 16
    state_snapshot, history_snapshot = first.material.state.snapshot(), first.history.snapshot()
    rng = np.random.default_rng(4096)
    scales = [0.01, 1, 100, 1, 1, 2e4, 0.001, 1e-7, 1e-7]
    seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(targets(last), scales)]
    complete.backward(grads={a: model.array(seed, a.dtype, False) for a, seed in zip(targets(last), seeds)})
    expected = [a.grad.numpy().copy() for a in sources + shared]
    complete.zero()

    resumed_state = AffineState.restore(state_snapshot, device=model.device)
    resumed_history = PairHistory.restore(history_snapshot, device=model.device)
    with wp.Tape() as tail:
        resumed = advance(resumed_state, resumed_history)
    for a, b in zip(targets(resumed), targets(last)):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    np.testing.assert_array_equal(resumed.history.pairs, last.history.pairs)
    tail.backward(grads={a: model.array(seed, a.dtype, False) for a, seed in zip(targets(resumed), seeds)})
    boundary = [a.grad.numpy().copy() for a in arrays(resumed_state) + [resumed_history.values.high]]
    tail_shared = [a.grad.numpy().copy() for a in shared]
    tail.zero()

    with wp.Tape() as prefix:
        prefix_result = advance(state, initial_history)
    prefix_targets = arrays(prefix_result.material.state) + [prefix_result.history.values.high]
    prefix.backward(grads={a: model.array(seed, a.dtype, False) for a, seed in zip(prefix_targets, boundary)})
    actual = [a.grad.numpy().copy() for a in sources]
    actual += [a.grad.numpy() + seed for a, seed in zip(shared, tail_shared)]
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
