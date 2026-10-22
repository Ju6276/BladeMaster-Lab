"""Contact switches and gradients through integrated tool velocity controls."""

import numpy as np
import pytest
import warp as wp

from scripts.diff_contact_scan import evaluate
from scripts.diff_optimize import PushTask, prescribed_motion
from scripts.diff_validate import configuration


def test_contact_onset_has_distinct_one_sided_derivatives():
    cfg = configuration(0.006)
    left, left_mode, _, _ = evaluate(cfg, -1e-6, 0.0, 0.0)
    value, mode, gradient, _ = evaluate(cfg, 0.0, 0.0, 0.0, True)
    right, right_mode, _, _ = evaluate(cfg, 1e-6, 0.0, 0.0)
    assert (left_mode, mode, right_mode) == (1, 0, 0)
    assert left > 0 and value == right == 0
    assert gradient[0] == 0
    assert (value - left) / 1e-6 < -1


def test_static_to_dynamic_friction_retains_force_jump():
    cfg = configuration(0.006)
    sn, gk = cfg.contact_scales()
    critical = cfg.friction_static * sn * 0.0003 / ((gk + 2 * cfg.tangent_damping * np.sqrt(gk)) * cfg.dt)
    _, stick, _, stick_force = evaluate(cfg, -0.0003, critical * (1 - 1e-4), 0.0)
    _, slip, _, slip_force = evaluate(cfg, -0.0003, critical * (1 + 1e-4), 0.0)
    assert (stick, slip) == (1, 2)
    np.testing.assert_allclose(stick_force[0], slip_force[0], rtol=1e-6)
    np.testing.assert_allclose(slip_force[1] / stick_force[1],
                               cfg.friction_dynamic / cfg.friction_static / (1 - 1e-4), rtol=1e-5)


@pytest.mark.parametrize("speed", [0.01, 2.0])
def test_contact_branch_gradient_matches_difference(speed):
    cfg = configuration(0.006)
    _, mode, gradient, _ = evaluate(cfg, -0.0003, speed, 0.0, True)
    for index, epsilon in ((0, 1e-6), (1, 1e-3)):
        values = []
        for sign in (-1, 1):
            parameters = [-0.0003, speed]
            parameters[index] += sign * epsilon
            value, shifted_mode, _, _ = evaluate(cfg, *parameters, 0.0)
            assert shifted_mode == mode
            values.append(value)
        np.testing.assert_allclose(gradient[index], (values[1] - values[0]) / (2 * epsilon),
                                   rtol=2e-3, atol=1e-5)


def test_tool_controls_differentiate_both_integrated_position_and_velocity():
    controls = wp.array([0.22, 0.28, 0.25, 0.2], dtype=float, device="cuda:0", requires_grad=True)
    position = wp.zeros(1, dtype=wp.vec3, device="cuda:0", requires_grad=True)
    velocity = wp.zeros(1, dtype=wp.vec3, device="cuda:0", requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(prescribed_motion, 1, [controls, 11, 8, 0.0002, 0.2, wp.vec3(0.0)],
                  [position, velocity], device="cuda:0")
    tape.backward(grads={position: wp.array([[3, 0, 0]], dtype=wp.vec3, device="cuda:0"),
                         velocity: wp.array([[2, 0, 0]], dtype=wp.vec3, device="cuda:0")})
    expected = np.array([3 * 8 * 0.0002 * 0.2, 3 * 3 * 0.0002 * 0.2 + 2 * 0.2, 0, 0])
    np.testing.assert_allclose(controls.grad.numpy(), expected, rtol=1e-6, atol=1e-8)


def test_longer_control_perturbation_changes_contact_branches():
    task = PushTask(steps=128)
    controls = np.array([0.22, 0.28, 0.25, 0.2])
    checks = task.gradient_check(controls, epsilons=(0.003,))
    assert any(row["changed_modes_low"] > 0 and row["changed_modes_high"] > 0 for row in checks)
    derivative, original = task.evaluate(controls, True)
    assert task.evaluate(controls - 0.02 * derivative)[1]["loss"] < original["loss"]


def test_optimized_task_uses_original_forward_physics():
    task = PushTask(steps=128)
    replay = task.replay([0.4, 0.5, 0.6, 0.3])
    assert replay["position_max_abs_m"] < 1e-6
    assert not any(replay["solver_errors"])
