"""History reset, old cut digits and adjoints across successive knife passes."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_geometry import GeometryEventContact
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.differentiable_natural import CutInstruction
from blademaster_warp.solver import Config
from scripts.diff_cut_validate import reaction_loss, terminal_loss
from scripts.diff_multicut_validate import CutPath, multicut_path
from scripts.diff_natural_validate import time_edges


def model(shape='block'):
    cfg = Config(size=(.012, .012, .012), grid_spacing=.006, dt=.0002, shape=shape,
                 sampling_seed=7, young_modulus=20000, yield_stress=200,
                 external_tool=True, blade_grid_constraint=True, blade_bevel_height=.005,
                 contact_reference_dt=.0002, flip_reference_dt=.0002)
    m = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    return m, GeometryEventContact(m, friction_update='midpoint', integration='midpoint')


def test_new_cut_resets_damage_adjoint_and_preserves_material_and_old_digits():
    m, contact = model()
    codes = np.where(m.reference[:, 0] < 0, 1, 2)
    Fp = np.tile(np.diag([1.02, .99, 1.01]), (m.n, 1, 1))
    state = m.state(x=m.reference + [0, 0, .1], F=Fp, Fp=Fp)
    history = contact.history(codes=codes, damage=np.full(m.n, .7))
    tool = contact.tool([1, 1, 1])
    for index in (0, 1):
        with wp.Tape() as tape:
            result, updated, _ = contact.step(state, history, tool, CutInstruction(False, index))
        tape.backward(grads={updated.damage: m.array(np.ones(m.n), float, False)})
        np.testing.assert_array_equal(history.damage.grad.numpy(), np.full(m.n, 1. if index == 0 else 0.))
        np.testing.assert_array_equal(updated.damage.numpy(), np.full(m.n, .7 if index == 0 else 0.))
        np.testing.assert_array_equal(updated.codes.numpy(), codes)
        np.testing.assert_allclose(result.Fp.numpy(), Fp, rtol=0, atol=1e-13)
        tape.zero()
    np.testing.assert_array_equal(history.damage.numpy(), np.full(m.n, .7))


@pytest.mark.parametrize('corrupt_call', [1, 2])
def test_checkpoint_rejects_nonfinite_saved_or_recomputed_state(monkeypatch, corrupt_call):
    m, contact = model()
    step = contact.step
    calls = 0

    def corrupt(*args):
        nonlocal calls
        state, h, force = step(*args)
        calls += 1
        if calls == corrupt_call:
            state.stress.assign(np.full((m.n, 3, 3), np.nan))
        return state, h, force

    monkeypatch.setattr(contact, 'step', corrupt)

    def objective(step, state, history, force, loss):
        m.launch(terminal_loss, m.n, inputs=[state.v, state.Fp], outputs=[loss], device=m.device)

    with pytest.raises(FloatingPointError, match='Nonfinite checkpoint stress'):
        contact.checkpoint_vjp(m.state(), contact.history(), [CutInstruction(False)], 1,
                              lambda _: contact.tool([1, 1, 1]), objective, timings={})


@pytest.mark.parametrize('shape', ['block', 'cylinder'])
@pytest.mark.parametrize('second_yaw', [0., np.pi / 3])
def test_first_cut_gradient_reaches_second_cut_and_checkpoints(shape, second_yaw, monkeypatch):
    monkeypatch.setattr(wp.config, 'verify_autograd_array_access', True)
    m, contact = model(shape)
    x = m.reference + [0, 0, .03]
    initial, history = m.state(x=x), contact.history()
    top = float(x[:, 2].max() + m.config.blade_half_size[2] + .00005)
    times = np.array([0, 4, 8, 12, 18, 24]) * m.config.dt
    positions = np.array([[.0004, 0, top], [.0004, 0, top - .0008],
                          [.0004, 0, top], [.0004, 0, top],
                          [.0004, 0, top - .0012], [.0004, 0, top]])
    basis = np.zeros((len(times), 2, 3))
    basis[:3, 0, 0] = .001
    basis[1, 1, 2] = -.001
    path = CutPath('two-pass-window', times, positions, np.array([0, 0, 0, second_yaw, second_yaw, second_yaw]),
                   np.array([True, False, False, True, False]), np.array([0, 0, 0, 1, 1]), basis, times[3])
    q = m.array([0., 0.], float)
    tool_at = path.bind(m, q)
    steps = 24
    instructions = [path.instruction(i * m.config.dt, m.config.dt) for i in range(steps)]

    def objective(step, state, h, force, loss):
        if step >= 12:
            m.launch(reaction_loss, 1, inputs=[force, m.config.dt], outputs=[loss], device=m.device)
        if step == steps - 1:
            m.launch(terminal_loss, m.n, inputs=[state.v, state.Fp], outputs=[loss], device=m.device)

    def rollout():
        state, h, loss = initial, history, m.zeros(1, float)
        first_codes = None
        for step, instruction in enumerate(instructions):
            old_damage = h.damage.numpy()
            state, h, force = contact.step(state, h, tool_at(step * m.config.dt), instruction)
            if step == 11:
                first_codes = h.codes.numpy()
                assert np.count_nonzero(first_codes) > 0
            if step >= 12:
                np.testing.assert_array_equal(h.codes.numpy() % 3, first_codes)
            if not instruction.cutting:
                np.testing.assert_array_equal(h.damage.numpy(), old_damage)
            objective(step, state, h, force, loss)
        assert np.count_nonzero(h.codes.numpy() // 3) > 0
        return loss, state, h

    with wp.Tape() as tape:
        loss, terminal, last = rollout()
    tape.backward(loss)
    expected = q.grad.numpy().copy()
    expected_x = initial.x.grad.numpy().copy()
    assert np.min(np.abs(expected)) > 1e-8
    tape.zero()
    for length in (4, 7):
        timings = {}
        result = contact.checkpoint_vjp(initial, history, instructions, length,
            lambda step: tool_at(step * m.config.dt), objective, parameters=(q,), timings=timings,
            checkpoint_device='cpu')
        np.testing.assert_allclose(result.parameter_gradients[q], expected, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(result.initial_gradient.x.numpy(), expected_x, rtol=1e-9, atol=1e-9)
        np.testing.assert_array_equal(result.history.codes.numpy(), last.codes.numpy())
        assert result.history.cut_index == 1
        assert max(timings['recompute_max_abs'].values()) == 0
    for control in range(2):
        direction = np.eye(2)[control]
        for epsilon in (1e-5, 1e-6):
            values = []
            for sign in (-1, 1):
                q.assign(sign * epsilon * direction)
                shifted, _, _ = rollout()
                values.append(float(shifted.numpy()[0]))
            np.testing.assert_allclose(expected[control], (values[1] - values[0]) / (2 * epsilon),
                                       rtol=.02, atol=1e-8)
        q.zero_()


@pytest.mark.parametrize('name', ['resume', 'two-cuts', 'crossing', 'three-cuts-push'])
def test_multicut_schedule_knots_and_first_cut_control_support(name):
    cfg = Config(dt=.00017)
    path = multicut_path(cfg, name)
    steps = int(np.ceil(path.times[-1] / cfg.dt))
    edges = time_edges(steps, cfg.dt, .25, 0., 'straight', path.times)
    for time in path.times:
        assert time in edges
    for a, b in zip(edges[:-1], edges[1:]):
        assert not np.any((path.times > a) & (path.times < b))
    if name != 'resume':
        np.testing.assert_array_equal(path.basis[path.times >= path.objective_start], 0.)
    assert np.any(path.cutting)


def test_cut_path_rejects_invalid_time_and_control_shapes():
    path = multicut_path(Config(), 'crossing')
    with pytest.raises(ValueError):
        replace(path, times=path.times[::-1])
    with pytest.raises(ValueError):
        replace(path, basis=path.basis[:, :1])
