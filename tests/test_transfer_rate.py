"""PIC damping rate, transfer invariants and velocity adjoints."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable import ElasticTrajectory
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.material import g2p
from blademaster_warp.solver import Config, Solver


@pytest.mark.parametrize("options", [
    {"flip_reference_dt": -1.0},
    {"flip_reference_dt": float("nan")},
    {"dt": 0.001, "flip_fraction": 0.5, "flip_reference_dt": 0.0001},
])
def test_invalid_transfer_rate(options):
    with pytest.raises(ValueError):
        Config(**options)


@pytest.mark.parametrize("kind", ["elastic", "material", "ordered"])
@pytest.mark.parametrize("divisor", [1, 2, 4])
def test_transfer_momentum_energy_and_velocity_adjoint(kind, divisor):
    cfg = Config(size=(0.012, 0.012, 0.012), grid_spacing=0.006,
                 dt=0.0002 / divisor, flip_fraction=0.8,
                 flip_reference_dt=0.0002, young_modulus=1.0, blade_grid_constraint=False)
    if kind == "elastic":
        model = ElasticTrajectory(cfg)
        precision, tolerance = "float32", 2e-7
    else:
        model = MaterialTrajectory(cfg, precision="float64", ordered_transfers=kind == "ordered")
        precision, tolerance = "float64", 2e-13
    velocity = np.random.default_rng(42).normal(0, 0.1, (model.n, 3))
    initial = model.state(v=velocity)

    # Equal particle masses: P = W diag(1 / sum_p W) W^T is the PIC map.
    nodes, weights = model.slots.numpy(), model.weights.numpy()
    W = np.zeros((model.n, model.node_count))
    for p in range(model.n):
        np.add.at(W[p], nodes[p], weights[p])
    grid_mass = W.sum(axis=0)
    inverse_mass = np.divide(1, grid_mass, out=np.zeros_like(grid_mass), where=grid_mass > 0)
    P = (W * inverse_mass) @ W.T
    rate = 1000.0
    transfer = np.eye(model.n) - cfg.dt * rate * (np.eye(model.n) - P)
    expected = transfer @ velocity
    with wp.Tape() as tape:
        result = model.step(initial, gravity=(0, 0, 0))
    seed = model.array(expected, wp.vec3, requires_grad=False)
    tape.backward(grads={result.v: seed})
    np.testing.assert_allclose(result.v.numpy(), expected, atol=tolerance, rtol=0)
    np.testing.assert_allclose(result.x.numpy(), model.reference + cfg.dt * (P @ velocity),
                               atol=tolerance, rtol=0)
    np.testing.assert_allclose(initial.v.grad.numpy(), transfer.T @ expected, atol=tolerance, rtol=0)
    np.testing.assert_allclose(result.v.numpy().sum(axis=0), velocity.sum(axis=0), atol=tolerance * model.n)
    assert np.sum(result.v.numpy().astype(float) ** 2) <= np.sum(velocity ** 2)

    original = Solver(cfg, precision=precision)
    original.v.assign(velocity)
    original.material_step()
    np.testing.assert_allclose(original.v.numpy(), expected, atol=tolerance, rtol=0)
    if divisor == 1:
        legacy = Solver(replace(cfg, flip_reference_dt=0), precision=precision)
        legacy.v.assign(velocity)
        legacy.material_step()
        np.testing.assert_allclose(legacy.v.numpy(), original.v.numpy(), atol=tolerance, rtol=0)


def test_unresolved_velocity_mode_converges_to_fixed_decay_rate():
    cfg = Config(dt=0.0002, flip_fraction=0.8, flip_reference_dt=0.0002)
    velocity = np.array([[1., 0., 0.], [-1., 0., 0.]], dtype=np.float32)
    slots = wp.array(np.tile(np.arange(8), (2, 1)), dtype=int, device="cuda:0")
    weights = wp.full((2, 8), 1 / 8, dtype=float, device="cuda:0")
    grid = wp.zeros(8, dtype=wp.vec3, device="cuda:0")
    errors = []
    for divisor in (1, 2, 4, 8):
        current = replace(cfg, dt=cfg.dt / divisor)
        x = wp.zeros(2, dtype=wp.vec3, device="cuda:0")
        v = wp.array(velocity, dtype=wp.vec3, device="cuda:0")
        for _ in range(4 * divisor):
            wp.launch(g2p, 2, [x, v, slots, weights, grid, grid,
                               current.dt, current.effective_flip_fraction()], device="cuda:0")
        # Opposite velocities cancel on the grid, leaving dv/dt = -1000 v.
        errors.append(abs(float(v.numpy()[0, 0]) - np.exp(-0.8)))
        np.testing.assert_array_equal(x.numpy(), 0)
        np.testing.assert_array_equal(v.numpy().sum(axis=0), 0)
    assert all(b < 0.52 * a for a, b in zip(errors, errors[1:])), errors
