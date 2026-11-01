import numpy as np
import pytest
import warp as wp
from blademaster_warp.natural_task import _position_loss, _force_loss


@pytest.mark.parametrize("count", [1, 7, 19])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_weighted_position_objective_matches_analytic_gradient(count, device):
    rng = np.random.default_rng(18)
    xn, tn = rng.normal(size=(2, count, 3))
    wn = rng.uniform(0.1, 1.0, size=count)
    scale = 0.003
    x = wp.array(xn, dtype=wp.vec3d, device=device, requires_grad=True)
    target = wp.array(tn, dtype=wp.vec3d, device=device, requires_grad=True)
    weights = wp.array(wn, dtype=wp.float64, device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(_position_loss, 1, inputs=[x, target, weights, scale], outputs=[loss], device=device)
    tape.backward(loss=loss)
    distances = np.sum((xn - tn) ** 2, axis=1)
    value = np.dot(wn, distances) / (wn.sum() * scale**2)
    dx = 2 * wn[:, None] * (xn - tn) / (wn.sum() * scale**2)
    dw = (distances / scale**2 - value) / wn.sum()
    np.testing.assert_allclose(loss.numpy(), [value], rtol=1e-12)
    np.testing.assert_allclose(x.grad.numpy(), dx, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(target.grad.numpy(), -dx, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(weights.grad.numpy(), dw, rtol=1e-12, atol=1e-8)


@pytest.mark.parametrize("count", [1, 7, 19])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_force_objective_matches_analytic_force_and_duration_gradient(count, device):
    rng = np.random.default_rng(81)
    fn = rng.normal(size=(count + 4, 3))
    force = wp.array(fn, dtype=wp.vec3d, device=device, requires_grad=True)
    duration = wp.array([0.0002], dtype=wp.float64, device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(_force_loss, 1, inputs=[force, 2, count, duration, 0.7], outputs=[loss], device=device)
    tape.backward(loss=loss)
    net = fn[2 : count + 2].sum(axis=0)
    derivative = np.zeros_like(fn)
    derivative[2 : count + 2] = 2 * 0.7 * 0.0002 * net
    np.testing.assert_allclose(loss.numpy(), [0.7 * 0.0002 * np.dot(net, net)], rtol=1e-12)
    np.testing.assert_allclose(force.grad.numpy(), derivative, rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(duration.grad.numpy(), [0.7 * np.dot(net, net)], rtol=1e-12)
