"""Analytic motion, energy and restart through the natural cohesive entry point."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_natural import NaturalCohesive, NaturalHistory
from blademaster_warp.cohesive_rigid_contact import predict_rigid_contacts
from blademaster_warp.cohesive_stages import AffineState
from blademaster_warp.differentiable_contact import Tool
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.energy import energy_snapshot
from blademaster_warp.reference_affine import build_transfer
from blademaster_warp.solver import Config


def setup(gravity, velocity):
    cfg = Config(
        size=(0.012, 0.012, 0.012), grid_spacing=0.006, dt=0.0002,
        young_modulus=20000.0, yield_stress=0.0, flip_fraction=1.0,
        external_tool=True, blade_grid_constraint=False, label_board_layer=False,
        blade_half_size=(0.0005, 0.0077, 0.006), blade_bevel_height=0.002,
    )
    model = MaterialTrajectory(cfg, precision="float64", ordered_transfers=True)
    state = AffineState.from_material(
        model.state(v=np.tile(velocity, (model.n, 1)), stress=np.zeros((model.n, 3, 3))),
        model.zeros(model.n, wp.mat33),
    )
    parameters = model.array([[1000.0, 500.0, 0.2, 0.3, 0.2]], float)
    half = model.array([cfg.blade_half_size], wp.vec3)
    bevel = model.array([cfg.blade_bevel_height], float)
    offset = model.array([cfg.grid_spacing / 4], float)
    point_map = model.array(np.eye(model.n), float, False)
    tool = Tool(
        model.array([[0.0002, 0.0001, 0.05]], wp.vec3), model.zeros(1, wp.vec3),
        model.array([[0.0, 0.0, 0.0, 1.0]], wp.quat), model.zeros(1, wp.vec3),
    )

    def provider(transfer, tool, x, moments, tau, elapsed, history):
        pose = driver.pose(tool, elapsed)
        return predict_rigid_contacts(
            x, moments, tau, history, point_map, transfer.flow_map,
            parameters=parameters, position=pose.position, rotation=pose.rotation,
            tool_velocity=tool.velocity, omega=tool.omega, half=half, bevel=bevel, offset=offset,
        )

    driver = NaturalCohesive(
        model, material=model.array(np.tile([50000.0, 70.0, 0.0], (model.n, 1)), wp.vec3),
        contact_provider=provider, pair_parameters=parameters,
        diameter=model.array([cfg.grid_spacing / 2], float), gravity=gravity,
    )
    return model, driver, state, driver.history(model.zeros(model.n, wp.vec3)), tool


def advance(driver, state, history, tool, steps, duration=None):
    for _ in range(steps):
        result = driver.step(state, history, tool, duration=duration, cutting=False)
        assert not result.geometry_events and not result.contact_events
        state, history, tool = result.state, result.history, result.tool
    return result


def assert_snapshot_equal(first, second):
    if isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_snapshot_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_snapshot_equal(a, b)
    else:
        assert first == second


@pytest.mark.parametrize("gravity,velocity", [
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, -9.81), (0.1, 0.2, 0.3)),
])
def test_natural_uncut_analytic_motion_energy_and_time_gradient(gravity, velocity):
    model, driver, initial, history, tool = setup(gravity, velocity)
    steps = 3
    step_duration = model.array([model.config.dt], float)
    transfer = build_transfer(model, model.uncut)
    before = energy_snapshot(model, initial, gravity, affine_transfer=transfer)
    with wp.Tape() as tape:
        result = advance(driver, initial, history, tool, steps, step_duration)
    duration = steps * model.config.dt
    g, v = np.asarray(gravity), np.asarray(velocity)
    np.testing.assert_allclose(
        result.state.x.numpy(), initial.x.numpy() + duration * v + 0.5 * duration**2 * g,
        rtol=0, atol=2e-17,
    )
    np.testing.assert_allclose(result.state.v.numpy(), np.tile(v + duration * g, (model.n, 1)),
                               rtol=0, atol=2e-15)
    for name in ["F", "Fp", "affine", "stress"]:
        np.testing.assert_allclose(getattr(result.state, name).numpy(), getattr(initial, name).numpy(),
                                   rtol=0, atol=2e-12)
    after = energy_snapshot(model, result.state, gravity, affine_transfer=transfer)
    np.testing.assert_allclose(sum(after.values()), sum(before.values()), rtol=1e-12, atol=1e-18)
    assert_snapshot_equal(history.snapshot(), result.history.snapshot())
    tape.backward(grads={result.state.x: model.array(np.ones((model.n, 3)), wp.vec3, False)})
    expected = model.n * steps * np.sum(v + duration * g)
    np.testing.assert_allclose(step_duration.grad.numpy()[0], expected, rtol=2e-12, atol=1e-12)
    np.testing.assert_allclose(initial.v.grad.numpy(), duration, rtol=2e-12, atol=1e-15)


def test_natural_history_and_material_restart():
    model, driver, initial, history, tool = setup((0.0, 0.0, -9.81), (0.1, 0.2, 0.3))
    first = advance(driver, initial, history, tool, 1)
    direct = advance(driver, first.state, first.history, first.tool, 2)
    restored = AffineState.restore(first.state.snapshot(), device=model.device)
    retained = NaturalHistory.restore(first.history.snapshot(), device=model.device)
    tool = Tool(*[
        model.array(getattr(first.tool, name).numpy(), kind)
        for name, kind in [("position", wp.vec3), ("velocity", wp.vec3), ("rotation", wp.quat), ("omega", wp.vec3)]
    ])
    resumed = advance(driver, restored, retained, tool, 2)
    assert_snapshot_equal(direct.state.snapshot(), resumed.state.snapshot())
    assert_snapshot_equal(direct.history.snapshot(), resumed.history.snapshot())
