"""Interface virtual work and all-input finite differences with cancelling seeds."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp._cohesive_momentum import momentum_residual

DTYPES = [wp.float64, wp.vec3d, wp.float64, wp.vec3d, wp.mat33d, wp.float64, wp.float64, wp.vec3d]


@pytest.mark.parametrize("integrated", [False, True])
def test_momentum_all_input_pullbacks(integrated):
    rng = np.random.default_rng(991)
    base = [
        rng.normal(size=(4, 7)),
        rng.normal(size=(4, 7, 3)),
        rng.uniform(0.01, 0.02, 4),
        rng.normal(size=(4, 3)) * 100,
        rng.normal(size=(4, 3, 3)),
        np.array([0.2]),
        np.array([0.01]),
        rng.normal(size=(7, 3)) * 0.1,
    ]
    sources = [wp.array(value, dtype=kind, requires_grad=True) for value, kind in zip(base, DTYPES)]

    def evaluate():
        return momentum_residual(*sources, integrated=integrated)

    with wp.Tape() as tape:
        result = evaluate()
    seed = rng.normal(size=(7, 3))
    tape.backward(grads={result: wp.array(seed, dtype=wp.vec3d)})
    for index, source in enumerate(sources):
        saved = source.numpy().copy()
        direction = rng.normal(size=saved.shape) * np.maximum(abs(saved), 0.001)
        ad = float(np.sum(direction * source.grad.numpy()))
        for epsilon in [1e-4, 1e-5]:
            outputs = []
            for sign in [-1, 1]:
                source.assign(saved + sign * epsilon * direction)
                outputs.append(evaluate().numpy())
            source.assign(saved)
            fd = float(np.sum(seed * (outputs[1] - outputs[0])) / (2 * epsilon))
            error = abs(ad - fd) / max(abs(ad), abs(fd), 1e-12)
            assert error <= 2e-5, (integrated, index, epsilon, ad, fd, error)


def test_force_pullback_retains_the_remainder_of_opposing_momentum_seeds():
    sources = [
        np.ones((1, 3)),
        np.zeros((1, 3, 3)),
        np.ones(1),
        np.zeros((1, 3)),
        np.zeros((1, 3, 3)),
        np.ones(1),
        np.ones(1),
        np.zeros((3, 3)),
    ]
    sources = [wp.array(value, dtype=kind, requires_grad=True) for value, kind in zip(sources, DTYPES)]
    with wp.Tape() as tape:
        result = momentum_residual(*sources, integrated=True)
    seed = np.zeros((3, 3))
    seed[:, 0] = [2**80, 1.0, -(2**80)]
    tape.backward(grads={result: wp.array(seed, dtype=wp.vec3d)})
    np.testing.assert_array_equal(sources[3].grad.numpy(), [[1.0, 0.0, 0.0]])
