"""Closed and weakly damaged affine stages retain natural time and state derivatives."""

import numpy as np
import pytest
import warp as wp

from test_cohesive_stages import setup, rollout, targets


@pytest.mark.parametrize("history_size", [0.0, 1e-15])
def test_closed_and_weak_history_stage_derivatives(history_size):
    m, state, geometry, history, data = setup("rigid", 2e-6)
    if history_size:
        transfer = data["transfer"]
        mapping = data["separation_map"].cpu().numpy()
        _, singular, right = np.linalg.svd(mapping, full_matrices=False)
        rank = np.count_nonzero(singular > singular[0] * max(mapping.shape) * np.finfo(float).eps)
        velocity = transfer.pack(state.v, state.affine).numpy()
        velocity += np.random.default_rng(218).normal(scale=0.03, size=velocity.shape)
        velocity -= right[:rank].T @ (right[:rank] @ velocity)
        state.v, state.affine = transfer.unpack(m.array(velocity, wp.vec3))
        data["damping_rate"] = m.array([2000.0], float)
        history.fill_(history_size)

    def run():
        result = rollout(state, geometry, history, data, steps=1)
        if not history_size:
            np.testing.assert_array_equal(result.geometry.gap.numpy(), 0.0)
            np.testing.assert_array_equal(result.history.numpy(), 0.0)
        return result

    with wp.Tape() as tape:
        result = run()
    if history_size:
        arrays = [result.interface.force]
        seeds = [np.random.default_rng(927).normal(size=arrays[0].numpy().shape) / 1e-3]
    else:
        arrays = targets(result)
        rng = np.random.default_rng(941)
        seeds = [
            rng.normal(size=a.numpy().shape) / scale
            for a, scale in zip(arrays, [0.01, 1.0, 100.0, 1.0, 1.0, 2e4, 0.001, 1.0, 0.001])
        ]
    tape.backward(grads={a: m.array(seed, a.dtype, False) for a, seed in zip(arrays, seeds)})
    directions = {"duration": [(data["duration"], np.array([2e-7]))]}
    if history_size:
        directions["history"] = [(history, np.full(len(history), 0.1 * history_size))]
    else:
        angular = np.array([[0.0, -3.0, 1.0], [3.0, 0.0, -2.0], [-1.0, 2.0, 0.0]])
        directions["rotation"] = [
            (state.v, m.reference @ angular.T),
            (state.affine, np.tile(angular, (m.n, 1, 1))),
        ]
    for name, pairs in directions.items():
        derivative = sum(float(np.sum(source.grad.numpy() * direction)) for source, direction in pairs)
        initial = [source.numpy().copy() for source, _ in pairs]
        for epsilon in ([0.01, 0.001, 0.0001] if history_size else [1e-4, 1e-5]):
            values = []
            for sign in [-1.0, 1.0]:
                for (source, direction), original in zip(pairs, initial):
                    source.assign(original + sign * epsilon * direction)
                updated = run()
                values.append(
                    [a.numpy() for a in ([updated.interface.force] if history_size else targets(updated))]
                )
            difference = sum(float(np.sum((b - a) * seed)) for a, b, seed in zip(*values, seeds)) / (
                2 * epsilon
            )
            np.testing.assert_allclose(
                derivative,
                difference,
                rtol=2e-5,
                atol=1e-8 if history_size else 2e-8,
                err_msg=f"{name}/{epsilon}",
            )
        for (source, _), original in zip(pairs, initial):
            source.assign(original)
