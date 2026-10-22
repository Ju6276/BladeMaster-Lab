"""Constitutive friction: continuity, constitutive work and trajectory VJPs."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_cutting import branch_penalty
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.differentiable_natural import CutInstruction, NaturalCutContact
from blademaster_warp.solver import Config
from scripts.diff_cut_validate import terminal_loss


@wp.kernel
def probe(parameters: wp.array(dtype=float), dt: float, damping: float, midpoint: int,
          impulse: wp.array(dtype=wp.vec3), history: wp.array(dtype=wp.vec3)):
    ratio = dt / 0.0002
    j, h = branch_penalty(parameters[0], wp.vec3(parameters[1], parameters[2], 0.0),
                           wp.vec3(0.0, 0.0, 1.0), 3.0e-5,
                           wp.vec3(parameters[3], parameters[4], 0.0), dt,
                           0.1 * ratio * ratio, parameters[5], parameters[6],
                           0.05 * ratio * ratio, damping, -1, midpoint, 0)
    impulse[0] = j
    history[0] = h


def model():
    return MaterialTrajectory(Config(size=(.012, .012, .012), grid_spacing=.006, dt=.0002,
                                      external_tool=True, blade_grid_constraint=False, young_modulus=20000,
                                      yield_stress=200, friction_static=.05, friction_dynamic=.05),
                              precision='float64', ordered_transfers=True)


def evaluate(m, parameters, *, dt=.0002, damping=.1, midpoint=0):
    impulse, history = m.zeros(1, wp.vec3), m.zeros(1, wp.vec3)
    m.launch(probe, 1, inputs=[parameters, dt, damping, midpoint], outputs=[impulse, history], device=m.device)
    return impulse, history


@pytest.mark.parametrize('dt', [.0002, .0001])
def test_equal_coulomb_coefficients_have_continuous_history(dt):
    m = model()
    mass, k, normal_force, speed = 3e-5, .05 * 3e-5 / .0002**2, .1 * 3e-5 / .0002**2 * .001, .1
    damping = .2 * np.sqrt(mass * k)
    threshold_history = (.2 * normal_force - damping * speed) / k
    for epsilon in (1e-8, 1e-10):
        for midpoint in (0, 1):
            theta = .5 if midpoint else 1.
            boundary = threshold_history - theta * dt * speed
            states = []
            for sign in (-1, 1):
                p = m.array([-.001, speed, 0., boundary + sign * epsilon, 0., .2, .2], float)
                states.append(evaluate(m, p, dt=dt, midpoint=midpoint)[1].numpy()[0, 0])
            jump = abs(states[1] - states[0])
            assert jump <= 2 * epsilon



@pytest.mark.parametrize('sliding', [False, True])
def test_midpoint_force_and_history_vjp(sliding, monkeypatch):
    monkeypatch.setattr(wp.config, 'verify_autograd_array_access', True)
    m = model()
    base = np.array([-.001, .1, .02, .0005 if sliding else .0001, .00004, .2, .2])
    p = m.array(base, float)
    seeds = [np.array([[.3, -.2, .5]]), np.array([[.7, .2, -.1]])]
    with wp.Tape() as tape:
        arrays = evaluate(m, p)
    tape.backward(grads={a: m.array(s, wp.vec3, False) for a, s in zip(arrays, seeds)})
    direction = np.array([.01, .3, .2, .03, -.01, .2, .2])
    ad = float(p.grad.numpy() @ direction)
    assert abs(ad) > 1e-5
    for epsilon in (1e-7, 3e-8):
        values = []
        for sign in (-1, 1):
            outputs = evaluate(m, m.array(base + sign * epsilon * direction, float, False))
            values.append(sum(np.sum(a.numpy() * s) for a, s in zip(outputs, seeds)))
        np.testing.assert_allclose(ad, (values[1] - values[0]) / (2 * epsilon), rtol=1e-6, atol=1e-9)


def test_zero_coulomb_radius_has_zero_tangential_force_derivative():
    m = model()
    base = np.array([-.001, 0., 0., 0., 0., 0., 0.])
    parameters = m.array(base, float)
    with wp.Tape() as tape:
        impulse, _ = evaluate(m, parameters)
    tape.backward(grads={impulse: m.array([[1., 1., 0.]], wp.vec3, False)})
    np.testing.assert_array_equal(parameters.grad.numpy()[1:5], 0)
    for sign in (-1, 1):
        values = base.copy()
        values[1:5] += sign * 1e-5
        force, _ = evaluate(m, m.array(values, float, False))
        np.testing.assert_array_equal(force.numpy()[0, :2], 0)


@pytest.mark.parametrize('sliding', [False, True])
def test_undamped_midpoint_friction_work_balance(sliding):
    m = model()
    dt, k = .0002, .05 * 3e-5 / .0002**2
    old = np.array([.00039 if sliding else .0001, 0., 0.])
    v = np.array([.1, .03, 0.])
    p = m.array([-.001, *v[:2], *old[:2], .2, .2], float, False)
    impulse, history = evaluate(m, p, dt=dt, damping=0.)
    updated = history.numpy()[0]
    delta = updated - old
    plastic = old + dt * v - updated
    stored = .5 * k * (updated @ updated - old @ old)
    slip_work = k * (updated @ plastic)
    integration_loss = .5 * k * (delta @ delta)
    input_work = -float(impulse.numpy()[0] @ v)
    np.testing.assert_allclose(input_work, stored + slip_work + integration_loss, atol=1e-18)
    assert slip_work >= -1e-18 and integration_loss >= 0
    if sliding:
        assert slip_work > 0


def test_midpoint_board_history_checkpoint_and_friction_gradient():
    m = model()
    c = NaturalCutContact(m, friction_update='midpoint')
    initial = m.state(x=m.reference - [0, 0, .0003], v=np.tile([.2, .03, -.01], (m.n, 1)))
    history = c.history()
    tool = c.tool([1, 1, 1])
    instruction = CutInstruction(cutting=False)
    steps = 8

    def objective(step, state, h, force, loss):
        if step == steps - 1:
            m.launch(terminal_loss, m.n, inputs=[state.v, state.Fp], outputs=[loss], device=m.device)

    def rollout():
        state, h, loss = initial, history, m.zeros(1, float)
        for step in range(steps):
            state, h, force = c.step(state, h, tool, instruction)
            objective(step, state, h, force, loss)
        return loss, h

    with wp.Tape() as tape:
        loss, h = rollout()
    tape.backward(loss)
    gradient = c.friction.grad.numpy().copy()
    initial_gradient = initial.x.grad.numpy().copy()
    assert abs(gradient.sum()) > 1e-5
    assert np.any(h.rigid_modes.numpy()[0] % 10 == 2)
    tape.zero()
    timings = {}
    result = c.checkpoint_vjp(initial, history, [instruction] * steps, 3, lambda _: tool, objective,
                              timings=timings)
    np.testing.assert_allclose(result.parameter_gradients[c.friction], gradient, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(result.initial_gradient.x.numpy(), initial_gradient, rtol=1e-10, atol=1e-9)
    assert max(timings['recompute_max_abs'].values()) == 0
    for epsilon in (1e-6, 3e-7):
        values = []
        for sign in (-1, 1):
            c.friction.assign([.05 + sign * epsilon] * 2)
            values.append(float(rollout()[0].numpy()[0]))
        np.testing.assert_allclose(gradient.sum(), (values[1] - values[0]) / (2 * epsilon), rtol=1e-5, atol=1e-7)
