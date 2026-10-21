"""Geometry event times and particle-state derivatives through split TLMPM flow."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_geometry import GeometryEventContact
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.differentiable_natural import CutInstruction
from blademaster_warp.solver import Config
from scripts.diff_cut_validate import reaction_loss, terminal_loss


def fixture(angle=0., omega=0., spacing=.006, shape='block', young=20000, seed=7):
    cfg = Config(size=(.012, .012, .012), grid_spacing=spacing, dt=.0002,
                 sampling_seed=seed, shape=shape, young_modulus=young, yield_stress=.01 * young,
                 external_tool=True, blade_grid_constraint=False, blade_bevel_height=.005,
                 contact_reference_dt=.0002, flip_reference_dt=.0002)
    m = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    c = GeometryEventContact(m)
    x = m.reference + [0, 0, .03]
    velocity = np.random.default_rng(3).normal(scale=.03, size=x.shape)
    initial, history = m.state(x=x, v=velocity), c.history()
    normal = np.array([np.sin(angle), 0, np.cos(angle)])
    z = (max(x @ normal) + cfg.blade_half_size[2] + .00005 - .0004 * normal[0]) / normal[2]
    tool = c.tool([.0004, 0, z], [0, 0, -1], [0, np.sin(angle / 2), 0, np.cos(angle / 2)], [0, omega, 0])
    return m, c, initial, history, tool


@pytest.mark.parametrize('angle,omega', [(0., 0.), (.2, .5)])
@pytest.mark.parametrize('parameter', ['position', 'velocity', 'x'])
@pytest.mark.parametrize('spacing,shape,young,seed', [(.006, 'block', 20000, 7), (.004, 'block', 400000, 7),
                                                     (.006, 'cylinder', 20000, 7), (.006, 'block', 20000, 0)])
def test_event_time_and_state_derivatives(angle, omega, parameter, spacing, shape, young, seed, monkeypatch):
    monkeypatch.setattr(wp.config, 'verify_autograd_array_access', True)
    m, c, initial, history, tool = fixture(angle, omega, spacing, shape, young, seed)

    def evaluate():
        state, h, force = c.step(initial, history, tool)
        loss = m.zeros(1, float)
        m.launch(terminal_loss, m.n, inputs=[state.v, state.Fp], outputs=[loss], device=m.device)
        m.launch(reaction_loss, 1, inputs=[force, m.config.dt], outputs=[loss], device=m.device)
        return loss, state, h

    with wp.Tape() as tape:
        loss, state, updated = evaluate()
    assert len(updated.geometry_events) > 0
    if seed == 0 and angle == 0:
        assert set(updated.codes.numpy()) == {0, 1, 2}
    particle, event_time = updated.geometry_events[0]
    assert 0 < float(event_time.numpy()[0]) < m.config.dt
    source = initial.x if parameter == 'x' else getattr(tool, parameter)
    base = source.numpy().copy()
    direction = np.zeros_like(base)
    direction[particle if parameter == 'x' else 0, 2] = 1.
    tape.backward(loss)
    loss_ad = float(np.sum(source.grad.numpy() * direction))
    tape.zero()
    tape.backward(grads={event_time: m.array([1.], float, False)})
    time_ad = float(np.sum(source.grad.numpy() * direction))
    assert abs(loss_ad) > 1e-7 and abs(time_ad) > 1e-7
    event_ids = [p for p, _ in updated.geometry_events]
    for epsilon in ((1e-6, 3e-7) if parameter == 'velocity' else (1e-8, 3e-9)):
        losses, times = [], []
        for sign in (-1, 1):
            source.assign(base + sign * epsilon * direction)
            shifted, _, h = evaluate()
            assert [p for p, _ in h.geometry_events] == event_ids
            losses.append(float(shifted.numpy()[0]))
            times.append(float(h.geometry_events[0][1].numpy()[0]))
        np.testing.assert_allclose(loss_ad, (losses[1] - losses[0]) / (2 * epsilon), rtol=2e-4, atol=1e-6)
        np.testing.assert_allclose(time_ad, (times[1] - times[0]) / (2 * epsilon), rtol=2e-5, atol=1e-8)


def test_geometry_event_checkpoint():
    m, c, initial, history, tool = fixture(.2, .5)
    steps = 3

    def knife(step):
        return c.pose(tool, m.array([step * m.config.dt], float, False))

    def objective(step, state, h, force, loss):
        m.launch(reaction_loss, 1, inputs=[force, m.config.dt], outputs=[loss], device=m.device)
        if step == steps - 1:
            m.launch(terminal_loss, m.n, inputs=[state.v, state.Fp], outputs=[loss], device=m.device)

    with wp.Tape() as tape:
        state, h, loss = initial, history, m.zeros(1, float)
        for step in range(steps):
            state, h, force = c.step(state, h, knife(step))
            objective(step, state, h, force, loss)
    tape.backward(loss)
    expected = tool.position.grad.numpy().copy()
    expected_initial = initial.x.grad.numpy().copy()
    tape.zero()
    timings = {}
    result = c.checkpoint_vjp(initial, history, [CutInstruction()] * steps, 2, knife, objective,
                              parameters=(tool.position,), timings=timings)
    np.testing.assert_allclose(result.parameter_gradients[tool.position], expected, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(result.initial_gradient.x.numpy(), expected_initial, rtol=1e-9, atol=1e-10)
    assert max(timings['recompute_max_abs'].values()) == 0


def test_free_fall_event_time_matches_quadratic_flow():
    m, c, initial, history, tool = fixture()
    x = initial.x.numpy() + [.02, 0, 0]
    initial = m.state(x=x, v=np.tile([0, 0, .02], (m.n, 1)))
    particle = int(np.argmax(x[:, 2]))
    gap = tool.position.numpy()[0, 2] - m.config.blade_half_size[2] - x[particle, 2]
    b, a = -1.02, 9.81
    expected = 2 * gap / (-b + np.sqrt(b * b - 4 * a * gap))
    with wp.Tape() as tape:
        _, updated, _ = c.step(initial, history, tool)
    actual_particle, time = updated.geometry_events[0]
    assert actual_particle == particle
    np.testing.assert_allclose(time.numpy()[0], expected, rtol=1e-10, atol=1e-14)
    tape.backward(grads={time: m.array([1.], float, False)})
    np.testing.assert_allclose(tool.position.grad.numpy()[0, 2], -1 / (b + 2 * a * expected),
                               rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize('parameter', ['young', 'knife_z'])
def test_cut_fields_plastic_state_and_history_vjp(parameter):
    m, c, initial, history, tool = fixture(seed=0)
    initial = m.state(x=initial.x.numpy(), v=initial.v.numpy(),
                      F=np.tile(np.diag([1.03, .99, 1.01]), (m.n, 1, 1)))

    def evaluate():
        state, h, force = c.step(initial, history, tool)
        assert set(h.codes.numpy()) == {0, 1, 2}
        assert np.any(state.yielded.numpy())
        return [state.x, state.v, state.F, state.Fp, state.stress, h.rigid, h.damage, force]

    with wp.Tape() as tape:
        arrays = evaluate()
    rng = np.random.default_rng(91)
    seeds = [rng.normal(size=a.numpy().shape) / max(1., np.linalg.norm(a.numpy())) for a in arrays]
    tape.backward(grads={a: m.array(s, a.dtype, False) for a, s in zip(arrays, seeds)})
    source = m.young if parameter == 'young' else tool.position
    base = source.numpy().copy()
    direction = np.zeros_like(base)
    direction.flat[0 if parameter == 'young' else 2] = 1
    ad = float(np.sum(source.grad.numpy() * direction))
    assert abs(ad) > 1e-8
    epsilons = (.01, .003) if parameter == 'young' else (1e-8, 3e-9)
    for epsilon in epsilons:
        values = []
        for sign in (-1, 1):
            source.assign(base + sign * epsilon * direction)
            values.append(sum(np.sum(a.numpy() * s) for a, s in zip(evaluate(), seeds)))
        np.testing.assert_allclose(ad, (values[1] - values[0]) / (2 * epsilon), rtol=3e-4, atol=1e-7)
