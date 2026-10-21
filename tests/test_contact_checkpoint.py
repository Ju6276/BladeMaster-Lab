"""Segment adjoints include friction memory, running costs and tool controls."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable import ElasticTrajectory
from blademaster_warp.differentiable_contact import PrescribedContact, Tool
from scripts.diff_optimize import PushTask, prescribed_motion
from scripts.diff_validate import configuration


@wp.kernel
def objective_kernel(v: wp.array(dtype=wp.vec3), history: wp.array2d(dtype=wp.vec3),
                     reaction: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):
    p = wp.tid()
    wp.atomic_add(loss, 0, wp.length_sq(v[p]) + 1.0e6 * wp.length_sq(history[0, p]))
    if p == 0:
        wp.atomic_add(loss, 0, 0.01 * wp.length_sq(reaction[0]))


@pytest.mark.parametrize("segment", [1, 7, 24])
def test_contact_checkpoint_transfers_all_state_and_parameter_adjoints(segment):
    model = ElasticTrajectory(replace(configuration(0.006), external_tool=True))
    contact = PrescribedContact(model)
    x = model.reference.copy()
    x[:, 2] -= 0.0003
    initial = model.state(x=x, v=np.tile([0.01, 0.002, 0.0], (model.n, 1)))
    history = model.array(np.full((2, model.n, 3), 1e-6), wp.vec3)
    controls = model.array([0.3, 0.2, 0.4], float)
    rotation = model.array([[0, 0, 0, 1]], wp.quat, False)
    omega = model.zeros(1, wp.vec3, False)
    origin = wp.vec3(float(x[:, 0].min() - 0.006 / 4 - model.config.blade_half_size[0] + 0.0002),
                     0.0, model.config.size[2] / 2)

    def tool_at_step(step):
        position, velocity = model.zeros(1, wp.vec3), model.zeros(1, wp.vec3)
        wp.launch(prescribed_motion, 1, [controls, step, 8, model.config.dt, 0.2, origin],
                  [position, velocity], device=model.device)
        return Tool(position, velocity, rotation, omega)

    def objective(step, state, current_history, reaction, loss):
        wp.launch(objective_kernel, model.n, [state.v, current_history, reaction], [loss], device=model.device)

    loss = model.zeros(1, float)
    with wp.Tape() as tape:
        state, current_history = initial, history
        for step in range(24):
            state, current_history, reaction = contact.step(state, current_history, tool_at_step(step))
            objective(step, state, current_history, reaction, loss)
    tape.backward(loss)
    arrays = [initial.x, initial.v, initial.F, history, controls, model.young]
    expected = [array.grad.numpy().copy() for array in arrays]
    expected_loss = float(loss.numpy()[0])
    expected_x = state.x.numpy().copy()
    tape.zero()
    measured = {}
    actual = contact.checkpoint_vjp(initial, history, 24, segment, tool_at_step, objective, [controls], measured)
    results = [actual.initial_gradient.x.numpy(), actual.initial_gradient.v.numpy(),
               actual.initial_gradient.F.numpy(), actual.history_gradient.numpy(),
               actual.parameter_gradients[controls], actual.parameter_gradients[model.young]]
    assert np.linalg.norm(expected[3]) > 0
    for reference, result in zip(expected, results):
        assert np.linalg.norm(reference - result) / max(np.linalg.norm(reference), 1e-10) < 1e-3
    np.testing.assert_allclose(actual.loss.numpy()[0], expected_loss, rtol=1e-5)
    np.testing.assert_allclose(actual.terminal.x.numpy(), expected_x, atol=1e-7, rtol=0)
    assert all(value > 0 for value in measured.values())


@pytest.mark.parametrize("force_limit", [2.0, 0.05])
def test_push_checkpoint_matches_terminal_and_running_objective(force_limit):
    task = PushTask(steps=128, controls=8, force_limit=force_limit)
    controls = np.resize([0.22, 0.28, 0.25, 0.2], task.controls)
    full_gradient, full = task.evaluate(controls, True)
    task.segment_steps = 23
    gradient, result = task.evaluate(controls, True)
    np.testing.assert_allclose(gradient, full_gradient, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(result["loss"], full["loss"], rtol=1e-5)
    assert result["timings"]["recompute_s"] > 0
    if force_limit < 1.0:
        assert result["peak_force_n"] > force_limit


def test_contact_checkpoint_rejects_untracked_parameters():
    model = ElasticTrajectory(replace(configuration(0.006), external_tool=True))
    contact = PrescribedContact(model)
    with pytest.raises(ValueError, match="gradient buffers"):
        contact.checkpoint_vjp(model.state(), contact.history(), 1, 1, None, None,
                               [model.array([1.0], float, False)])
