"""Closed constraints retain exact multiplier nullspaces and history tangents."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_cohesive import _evaluate
from blademaster_warp._cohesive_solve import setup
from test_differentiable_cohesive import arrays, closed_pair


@pytest.mark.parametrize("integrated", [False, True])
@pytest.mark.parametrize("compliance", [2.377e-9, 1.822786161493124e-13])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_closed_residual_has_exact_zero_multiplier_block_and_history_derivative(integrated, compliance, sign):
    area = 1.3061e-8
    strength = 5e4
    capacity = strength * (area if integrated else 1.0)
    q = compliance * capacity * np.array([[0.02, -0.03, 0.01]])
    traction = capacity * np.array([[sign * 0.27, 0.11, 0.22]])
    normal = np.array([[-1.0, 0.01, -0.03]])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    values = [
        q,
        np.eye(3)[None],
        normal,
        np.zeros(1),
        np.array([[strength, 70.0, 0.0]]),
        np.array([[1.0, -1.0]]),
        np.zeros((1, 2, 3)),
        np.array([area]),
        np.ones(1),
        np.array([2e-6]),
    ]
    inputs = arrays(values)
    u = wp.zeros(2, dtype=wp.vec3d, requires_grad=True)
    force = wp.array(traction, dtype=wp.vec3d, requires_grad=True)
    with wp.Tape() as tape:
        _, _, residual = _evaluate(inputs, u, force, compliance, integrated_force=integrated)
    np.testing.assert_array_equal(residual.numpy(), q)
    trial = q + compliance * traction
    positive = trial - np.minimum(np.sum(trial * normal, axis=1), 0)[:, None] * normal
    for axis in range(3):
        tape.zero()
        seed = np.eye(3)[axis : axis + 1]
        tape.backward(grads={residual: wp.array(seed, dtype=wp.vec3d)})
        np.testing.assert_array_equal(force.grad.numpy(), 0.0)
        np.testing.assert_array_equal(inputs[0].grad.numpy(), seed)
        np.testing.assert_allclose(
            inputs[3].grad.numpy(), -positive[:, axis] / (compliance * capacity), rtol=2e-14, atol=2e-15
        )


@pytest.mark.parametrize("integrated", [False, True])
@pytest.mark.parametrize("ratio", [1e-18, 1e-9, 1e-3, 1e3])
def test_unloading_residual_retains_weak_force_and_gap_tangents(integrated, ratio):
    values = closed_pair(1.3061e-8, [0.0, 0.0, 0.0])
    strength, gc = values[4][0, :2]
    capacity = strength * (values[7][0] if integrated else 1.0)
    failure = 2 * gc / strength
    compliance = 1e-7 / capacity
    old = ratio * compliance * capacity
    force = capacity * np.array([[0.2, 0.08, 0.04]])
    values[0] = old / (capacity * (1 - old / failure)) * force
    values[3][:] = old
    base_compliance = setup(values, integrated_force=integrated)[2]
    at, basis, compliance, length = setup(values, compliance / base_compliance, integrated_force=integrated)
    size = basis.shape[1]
    state = np.r_[np.zeros(size), compliance * force.ravel() / length]
    _, jacobian, _ = at(state)
    denominator = old + compliance * capacity * (1 - old / failure)
    gain = old / denominator
    complement = compliance * capacity * (1 - old / failure) / denominator
    np.testing.assert_allclose(jacobian[size:, size:], -gain * np.eye(3), rtol=2e-14, atol=0.0)

    inputs = arrays(values)
    u = wp.zeros(2, dtype=wp.vec3d, requires_grad=True)
    traction = wp.array(force, dtype=wp.vec3d, requires_grad=True)
    with wp.Tape() as tape:
        _, _, residual = _evaluate(inputs, u, traction, compliance, integrated_force=integrated)
    for axis in range(3):
        tape.zero()
        seed = np.eye(3)[axis : axis + 1]
        tape.backward(grads={residual: wp.array(seed, dtype=wp.vec3d)})
        np.testing.assert_allclose(traction.grad.numpy(), -compliance * gain * seed, rtol=2e-12, atol=0.0)
        np.testing.assert_allclose(inputs[0].grad.numpy(), complement * seed, rtol=2e-12, atol=0.0)
        expected_history = -(values[0] + compliance * force)[0, axis] * compliance * capacity / denominator**2
        np.testing.assert_allclose(inputs[3].grad.numpy(), [expected_history], rtol=2e-12, atol=0.0)
