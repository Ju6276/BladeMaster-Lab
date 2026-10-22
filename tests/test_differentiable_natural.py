"""Natural cut labels, predictive contact sets, history and trajectory derivatives."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.differentiable_cutting import pair_snapshot
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.differentiable_natural import CutInstruction, NaturalCutContact
from blademaster_warp.solver import Config, Solver
from scripts.diff_cut_validate import differentiable_tool, reaction_loss, terminal_loss


def configuration(**kwargs):
    cfg = Config(size=(.012, .012, .012), grid_spacing=.006, dt=.0001,
                 young_modulus=20000, yield_stress=200, external_tool=True,
                 blade_grid_constraint=False, blade_bevel_height=.005,
                 contact_reference_dt=.0002, flip_reference_dt=.0002)
    return replace(cfg, **kwargs)


@pytest.mark.parametrize('shape', ['block', 'cylinder'])
@pytest.mark.parametrize('angle', [0., .2])
def test_natural_sides_match_original_with_moving_knife(shape, angle):
    cfg = configuration(shape=shape)
    solver = Solver(cfg, precision='float64')
    port = ExternalToolPort(solver)
    m = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    c = NaturalCutContact(m, release='boundary')
    state, history = m.state(), c.history()
    for step in range(32):
        theta = angle + .1 * step * cfg.dt
        rotation = [0, np.sin(theta / 2), 0, np.cos(theta / 2)]
        position = [.0004, .01 * step * cfg.dt, .0312 - .8 * step * cfg.dt]
        velocity, omega = [0, .01, -.8], [0, .1, 0]
        expected = port.advance(ToolState(position, rotation, velocity, omega), cfg.dt, graph=False)
        state, history, force = c.step(state, history, c.tool(position, velocity, rotation, omega))
        np.testing.assert_array_equal(history.codes.numpy(), solver.codes.numpy())
        keys, pairs = pair_snapshot(solver)
        np.testing.assert_array_equal(history.pair_keys, keys)
        np.testing.assert_allclose(history.pairs.numpy(), pairs, atol=1e-13)
        for name in ('x', 'v', 'F', 'Fp', 'stress'):
            np.testing.assert_allclose(getattr(state, name).numpy(), getattr(solver, name).numpy(),
                                       rtol=2e-9, atol=2e-9, err_msg=name)
        np.testing.assert_allclose(force.numpy()[0], expected.force_n, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(history.rigid.numpy(), solver.histories.numpy(), atol=1e-13)
        np.testing.assert_allclose(history.damage.numpy(), solver.damage.numpy(), atol=1e-13)
    assert np.count_nonzero(history.codes.numpy()) > 0


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_natural_pair_search_matches_all_pairs_and_history(precision):
    cfg = configuration(pair_search='all')
    solver = Solver(cfg, precision=precision)
    port = ExternalToolPort(solver)
    x = solver.reference.copy()
    x[:, 0] *= .8
    codes = np.where(x[:, 0] < 0, 1, 2)
    v = np.zeros_like(x)
    v[:, 1] = np.where(x[:, 0] < 0, .15, -.15)
    solver.x.assign(x)
    solver.v.assign(v)
    solver.codes.assign(codes)
    m = MaterialTrajectory(cfg, precision=precision, ordered_transfers=True)
    c = NaturalCutContact(m, release='boundary')
    state, history = m.state(x=x, v=v), c.history(codes=codes)
    tool = c.tool([1, 1, 1])
    count = 0
    tolerance = 2e-7 if precision == 'float32' else 1e-13
    for step in range(5):
        port.advance(ToolState([1, 1, 1], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0]),
                     cfg.dt, cutting=False, graph=False)
        state, history, _ = c.step(state, history, tool, CutInstruction(cutting=False))
        keys, pairs = pair_snapshot(solver)
        np.testing.assert_array_equal(history.pair_keys, keys)
        np.testing.assert_allclose(history.pairs.numpy(), pairs, atol=tolerance, rtol=2e-5)
        np.testing.assert_allclose(state.v.numpy(), solver.v.numpy(), atol=tolerance, rtol=2e-5)
        count += len(keys)
    assert count > 0
    assert np.max(np.abs(history.pairs.numpy())) > 0


def test_predictive_pair_search_catches_approaching_separated_particles():
    m = MaterialTrajectory(configuration(), precision='float64', ordered_transfers=True)
    c = NaturalCutContact(m)
    x = m.reference + [1, 1, 1]
    x[0], x[1] = [-.002, 0, .1], [.002, 0, .1]
    v = np.zeros_like(x)
    v[0, 0], v[1, 0] = 10., -10.
    codes = np.zeros(m.n, dtype=np.int64)
    codes[0], codes[1] = 1, 2
    keys = c.contact_keys(m.state(x=x, v=v), c.history(codes=codes).codes)
    assert 1 in keys
    v *= -1
    assert 1 not in c.contact_keys(m.state(x=x, v=v), c.history(codes=codes).codes)


def test_knife_offset_changes_side_labels_from_the_same_intact_state():
    m = MaterialTrajectory(configuration(), precision='float64', ordered_transfers=True)
    c = NaturalCutContact(m)
    initial, history = m.state(), c.history()
    labels = []
    for offset in (0., .002):
        _, updated, _ = c.step(initial, history, c.tool([offset, 0, .028], [0, 0, -.12]))
        labels.append(updated.codes.numpy())
    assert np.count_nonzero(labels[0] != labels[1]) > 0
    assert np.count_nonzero(history.codes.numpy()) == 0


def test_pair_contact_reentry_resets_history():
    m = MaterialTrajectory(configuration(), precision='float64', ordered_transfers=True)
    c = NaturalCutContact(m)
    x = m.reference + [1, 1, 1]
    x[0], x[1] = [-.001, 0, .1], [.001, 0, .1]
    codes = np.zeros(m.n, dtype=np.int64)
    codes[0], codes[1] = 1, 2
    state = m.state(x=x)
    old = c.history(codes=codes, pair_keys=[1], pairs=[[0, 1e-5, 0]])
    tool, instruction = c.tool([2, 2, 2]), CutInstruction(cutting=False)
    separated = x.copy()
    separated[1, 0] = .01
    _, empty, _ = c.step(m.state(x=separated), old, tool, instruction)
    assert len(empty.pair_keys) == 0
    _, returned, _ = c.step(state, empty, tool, instruction)
    _, fresh, _ = c.step(state, c.history(codes=codes), tool, instruction)
    np.testing.assert_array_equal(returned.pair_keys, fresh.pair_keys)
    np.testing.assert_array_equal(returned.pairs.numpy(), fresh.pairs.numpy())


def test_cut_pause_and_new_cut_preserve_codes_and_reset_only_damage():
    m = MaterialTrajectory(configuration(), precision='float64', ordered_transfers=True)
    c = NaturalCutContact(m)
    codes = np.where(m.reference[:, 0] < 0, 1, 2)
    state = m.state(x=m.reference + [0, 0, .1])
    old = c.history(codes=codes, damage=np.full(m.n, .7))
    tool = c.tool([1, 1, 1])
    _, paused, _ = c.step(state, old, tool, CutInstruction(cutting=False))
    np.testing.assert_array_equal(paused.damage.numpy(), old.damage.numpy())
    _, next_cut, _ = c.step(state, paused, tool, CutInstruction(cutting=False, cut_index=1))
    np.testing.assert_array_equal(next_cut.damage.numpy(), np.zeros(m.n))
    np.testing.assert_array_equal(next_cut.codes.numpy(), codes)
    np.testing.assert_array_equal(old.damage.numpy(), np.full(m.n, .7))
    for index in (0, -1, True, 1.5, c.max_cut_index + 1):
        with pytest.raises(ValueError, match='Cut indices'):
            c.step(state, next_cut, tool, CutInstruction(cut_index=index))


@pytest.mark.parametrize('segment_steps', [3, 7])
@pytest.mark.parametrize('release', ['boundary', 'integrated'])
@pytest.mark.parametrize('checkpoint_device', [None, 'cpu'])
def test_natural_rollout_gradient_and_checkpoint(segment_steps, release, checkpoint_device, monkeypatch):
    monkeypatch.setattr(wp.config, 'verify_autograd_array_access', True)
    m = MaterialTrajectory(configuration(), precision='float64', ordered_transfers=True)
    c = NaturalCutContact(m, release=release)
    initial, history = m.state(), c.history()
    q = m.array([.4, 6., .3], float)
    steps = 20

    def knife(step):
        return differentiable_tool(m, q, [0, 0, .0312], step * m.config.dt)

    def objective(step, state, h, force, loss):
        m.launch(reaction_loss, 1, inputs=[force, .01], outputs=[loss], device=m.device)
        if step == steps - 1:
            m.launch(terminal_loss, m.n, inputs=[state.v, state.Fp], outputs=[loss], device=m.device)

    def rollout():
        state, h = initial, history
        loss = m.zeros(1, float)
        codes = []
        for step in range(steps):
            state, h, force = c.step(state, h, knife(step))
            objective(step, state, h, force, loss)
            codes.append(h.codes.numpy())
        return loss, np.asarray(codes)

    with wp.Tape() as tape:
        loss, codes = rollout()
    tape.backward(loss)
    gradient = q.grad.numpy().copy()
    initial_gradient = initial.x.grad.numpy().copy()
    assert np.any(codes != 0)
    assert np.linalg.norm(gradient) > 1e-5
    tape.zero()
    timings = {}
    result = c.checkpoint_vjp(initial, history, [CutInstruction()] * steps, segment_steps,
                              knife, objective, parameters=(q,), timings=timings,
                              checkpoint_device=checkpoint_device)
    np.testing.assert_allclose(result.parameter_gradients[q], gradient, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(result.initial_gradient.x.numpy(), initial_gradient, rtol=1e-9, atol=1e-9)
    assert max(timings['recompute_max_abs'].values()) == 0
    direction = np.array([.2, .3, .1])
    ad = gradient @ direction
    for epsilon in (1e-5, 1e-6):
        values = []
        for sign in (-1, 1):
            q.assign(np.array([.4, 6., .3]) + sign * epsilon * direction)
            shifted, shifted_codes = rollout()
            np.testing.assert_array_equal(shifted_codes, codes)
            values.append(float(shifted.numpy()[0]))
        np.testing.assert_allclose(ad, (values[1] - values[0]) / (2 * epsilon), rtol=2e-5, atol=1e-7)
