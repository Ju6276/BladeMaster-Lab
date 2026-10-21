"""Work and momentum balance, natural release gradients, and checkpoint propagation."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_cutting import EventCutContact, FixedCutContact
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.solver import Config
from scripts.diff_cut_validate import differentiable_tool, reaction_loss, terminal_loss


def fixture():
    cfg = Config(size=(.012, .012, .012), grid_spacing=.006, dt=.0002,
                 external_tool=True, blade_grid_constraint=False, cutting_energy=1.)
    m = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    c = EventCutContact(m)
    positions = m.reference + [.0016, 0, .009]
    initial = m.state(x=positions)
    damage = np.zeros(m.n)
    active = (np.abs(positions[:, 0]) < .0005) & (positions[:, 2] > .019)
    damage[active] = .2
    history = c.history(damage=damage)
    branch = c.freeze(np.zeros(m.n, dtype=np.int64), [], [], np.zeros(m.n, dtype=int))
    return m, c, initial, history, branch, active


def test_release_impulse_work_and_momentum_balance():
    m, c, initial, h, branch, active = fixture()
    tool = c.tool([0, 0, .04], [0, 0, -.12])
    fixed = FixedCutContact(m)
    old_state, old_h, old_force = fixed.step(initial, h, tool, branch)
    state, updated, force = c.step(initial, h, tool, branch)
    assert np.count_nonzero(active) == 4
    assert np.all(updated.damage.numpy()[active] == 1)
    fraction = (1 - h.damage.numpy()[active]) / (old_h.trial_damage.numpy()[active] - h.damage.numpy()[active])
    assert np.all((fraction > 0) & (fraction < 1))
    np.testing.assert_allclose(fraction, fraction[0], atol=1e-15)
    np.testing.assert_allclose(force.numpy(), old_force.numpy() * fraction[0], rtol=1e-13, atol=1e-14)
    delta_momentum = m.mass * np.sum(state.v.numpy() - old_state.v.numpy(), axis=0)
    np.testing.assert_allclose(delta_momentum, m.config.dt * (old_force.numpy()[0] - force.numpy()[0]), atol=1e-18)
    threshold = m.config.cutting_energy * m.volume**(2 / 3) * (1 - m.config.sharpness) / m.config.sharpness
    np.testing.assert_allclose(updated.damage_work.numpy(), threshold * (updated.damage.numpy() - h.damage.numpy()),
                               atol=1e-20)
    assert old_h.damage_work.numpy().sum() > updated.damage_work.numpy().sum()
    # Evolving damage controls the next step even when supplied flags remain zero.
    _, following, following_force = c.step(initial, updated, tool, branch)
    np.testing.assert_allclose(following_force.numpy(), 0, atol=1e-14)
    np.testing.assert_allclose(following.damage_work.numpy(), 0, atol=1e-20)


@pytest.mark.parametrize('segment_steps', [1, 3])
@pytest.mark.parametrize('parameter', ['fracture_energy', 'knife_speed'])
def test_natural_release_state_gradient_and_checkpoint(segment_steps, parameter):
    m, c, initial, history, branch, active = fixture()
    q = m.array([0., 1., 0.], float)
    steps = 5

    def tool(step):
        return differentiable_tool(m, q, [0, 0, .04], step * m.config.dt)

    def objective(step, state, h, force, loss):
        m.launch(reaction_loss, 1, inputs=[force, 1 / steps], outputs=[loss], device=m.device)
        if step == steps - 1:
            m.launch(terminal_loss, m.n, inputs=[state.v, state.Fp], outputs=[loss], device=m.device)

    def rollout():
        state, h = initial, history
        loss = m.zeros(1, float)
        for step in range(steps):
            state, h, force = c.step(state, h, tool(step), branch)
            objective(step, state, h, force, loss)
        return loss, state

    with wp.Tape() as tape:
        loss, state = rollout()
    tape.backward(loss)
    array, index = (c.fracture_energy, 0) if parameter == 'fracture_energy' else (q, 1)
    expected = array.grad.numpy().copy()
    expected_x_gradient = initial.x.grad.numpy().copy()
    assert abs(expected[index]) > 1e-5
    expected_loss = loss.numpy().copy()
    tape.zero()
    timings = {}
    result = c.checkpoint_vjp(initial, history, [branch] * steps, segment_steps, tool, objective,
                              parameters=(q,), timings=timings)
    np.testing.assert_allclose(result.loss.numpy(), expected_loss, atol=1e-14)
    np.testing.assert_allclose(result.parameter_gradients[array], expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(result.initial_gradient.x.numpy(), expected_x_gradient, rtol=1e-11, atol=1e-11)
    assert max(timings['recompute_max_abs'].values()) == 0
    base = array.numpy().copy()
    for epsilon in (1e-4, 3e-5):
        values = []
        for sign in (-1, 1):
            trial = base.copy()
            trial[index] += sign * epsilon
            array.assign(trial)
            values.append(float(rollout()[0].numpy()[0]))
        np.testing.assert_allclose(expected[index], (values[1] - values[0]) / (2 * epsilon), rtol=2e-5, atol=1e-9)


def test_release_rejects_normal_switching_hold_rule():
    cfg = Config(external_tool=True, blade_grid_constraint=False, cut_hold=True)
    with pytest.raises(ValueError, match='cut_hold=False'):
        EventCutContact(MaterialTrajectory(cfg))


@pytest.mark.parametrize('energy', [1e5, 1e6])
def test_no_release_matches_recorded_integrator(energy):
    m, c, initial, history, branch, active = fixture()
    c.fracture_energy.assign([energy])
    fixed = FixedCutContact(m)
    fixed.fracture_energy.assign([energy])
    tool = c.tool([0, 0, .04], [0, 0, -.12])
    a, ha, fa = c.step(initial, history, tool, branch)
    b, hb, fb = fixed.step(initial, history, tool, branch)
    for name in ('x', 'v', 'F', 'Fp', 'stress'):
        np.testing.assert_array_equal(getattr(a, name).numpy(), getattr(b, name).numpy())
    np.testing.assert_array_equal(fa.numpy(), fb.numpy())
    np.testing.assert_array_equal(ha.damage.numpy(), hb.damage.numpy())
