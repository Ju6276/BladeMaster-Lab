"""Accepted-segment work, analytic dynamics and observer independence."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_geometry import GeometryEventContact
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.differentiable_natural import CutInstruction
from blademaster_warp.energy_ledger import EnergyLedger
from blademaster_warp.solver import Config
from test_differentiable_geometry import fixture


def model(dt, **kwargs):
    cfg = Config(size=(.012, .012, .012), grid_spacing=.006, dt=dt, young_modulus=20000,
                 yield_stress=0., external_tool=True, blade_grid_constraint=False,
                 contact_reference_dt=.0002, flip_reference_dt=.0002, **kwargs)
    return MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)


@pytest.mark.parametrize('integration', ['midpoint', 'symplectic'])
def test_freefall_ledger_matches_discrete_energy_identity(integration):
    m = model(.0001)
    c = GeometryEventContact(m, integration=integration)
    ledger = EnergyLedger(c)
    state, history = m.state(x=m.reference + [0, 0, .1]), c.history()
    tool = c.tool([1, 1, 1])
    steps = 8
    for _ in range(steps):
        state, history, _ = c.step(state, history, tool, CutInstruction(cutting=False))
    report = ledger.report()
    expected = 0. if integration == 'midpoint' else -.5 * m.n * m.mass * 9.81**2 * steps * m.config.dt**2
    assert report['segments'] == steps
    assert report['duration_s'] == pytest.approx(steps * m.config.dt)
    np.testing.assert_allclose(report['totals']['balance_residual_j'], expected, atol=1e-16)
    for name in ('knife_work_j', 'plastic_j', 'friction_viscous_j', 'friction_sliding_j',
                 'history_transport_j', 'normal_transfer_work_j', 'friction_transfer_work_j', 'release_transfer_work_j',
                 'release_correction_work_j'):
        np.testing.assert_allclose(report['totals'][name], 0., atol=1e-16)


def run_board(dt, friction, steps):
    m = model(dt, friction_static=friction, friction_dynamic=friction)
    c = GeometryEventContact(m)
    ledger = EnergyLedger(c)
    kn = m.config.contact_scales()[0] * m.mass / dt**2
    x = m.reference.copy()
    x[:, 2] = m.config.grid_spacing / 4 - m.mass * 9.81 / kn
    state, history = m.state(x=x, v=np.tile([.01, 0, 0], (m.n, 1))), c.history()
    tool = c.tool([1, 1, 1])
    for _ in range(steps):
        state, history, _ = c.step(state, history, tool, CutInstruction(cutting=False))
    return ledger.report()


def test_board_energy_residual_converges_and_dissipation_is_nonnegative():
    residuals = []
    for steps in (8, 16, 32):
        report = run_board(.0004 / steps, 10., steps)
        totals = report['totals']
        assert totals['friction_viscous_j'] > 0
        for name in ('friction_sliding_j', 'friction_integration_j', 'history_transport_j', 'pic_j'):
            assert totals[name] >= -1e-17
        assert totals['friction_identity_max_abs_j'] < 1e-18
        assert abs(totals['release_correction_work_j']) < 1e-17
        residuals.append(abs(totals['balance_residual_j']))
    assert residuals[0] / residuals[1] > 3.5
    assert residuals[1] / residuals[2] > 3.5


def test_zero_friction_full_step_creates_no_tangential_energy():
    report = run_board(.00005, 0., 8)
    for name in ('friction_viscous_j', 'friction_sliding_j', 'tangent_change_j', 'history_transport_j'):
        np.testing.assert_allclose(report['totals'][name], 0., atol=1e-18)
    assert abs(report['totals']['balance_residual_j']) < 1e-16


@pytest.mark.parametrize('integration', ['midpoint', 'symplectic'])
def test_event_energy_counts_only_accepted_segments_and_preserves_dynamics(integration):
    m, c, initial, history, tool = fixture(.2, .5, friction_update='midpoint', integration=integration)
    steps = 4

    def run():
        state, h = initial, history
        work, events = 0., 0
        for step in range(steps):
            pose = c.pose(tool, m.array([step * m.config.dt], float, False))
            state, h, _ = c.step(state, h, pose)
            work += float(h.tool_work.numpy()[0])
            events += len(h.geometry_events)
        return state, h, work, events

    with wp.Tape() as baseline_tape:
        baseline, baseline_h, work, events = run()
    seed = m.array(np.ones((m.n, 3)), wp.vec3, False)
    baseline_tape.backward(grads={baseline.x: seed})
    baseline_gradient = initial.x.grad.numpy().copy()
    baseline_tape.zero()
    assert events > 0
    ledger = EnergyLedger(c)
    with wp.Tape() as tape:
        measured, measured_h, measured_work, measured_events = run()
    assert measured_events == events
    for name in ('x', 'v', 'F', 'Fp', 'stress'):
        np.testing.assert_array_equal(getattr(baseline, name).numpy(), getattr(measured, name).numpy())
    for name in ('rigid', 'pairs', 'damage', 'codes'):
        np.testing.assert_array_equal(getattr(baseline_h, name).numpy(), getattr(measured_h, name).numpy())
    report = ledger.report()
    assert report['segments'] > steps
    np.testing.assert_allclose(report['duration_s'], steps * m.config.dt, atol=1e-18)
    np.testing.assert_allclose(report['totals']['knife_work_j'], work, atol=1e-18)
    assert measured_work == work
    tape.backward(grads={measured.x: m.array(np.ones((m.n, 3)), wp.vec3, False)})
    assert np.isfinite(initial.x.grad.numpy()).all()
    np.testing.assert_allclose(initial.x.grad.numpy(), baseline_gradient, atol=1e-13, rtol=1e-13)


def test_ledger_global_balance_includes_discrete_event_energy():
    m, c, state, history, tool = fixture(.2, .5, friction_update='midpoint', integration='midpoint')
    ledger = EnergyLedger(c)
    for step in range(4):
        pose = c.pose(tool, m.array([step * m.config.dt], float, False))
        state, history, _ = c.step(state, history, pose)
    report = ledger.report()
    terms = report['totals']
    energy_change = sum(report['final'][name] - report['initial'][name]
                        for name in ('kinetic_j', 'elastic_j', 'contact_normal_j', 'contact_tangent_j'))
    dissipation = sum(terms[name] for name in ('plastic_j', 'friction_viscous_j', 'friction_sliding_j',
                                              'friction_integration_j', 'history_transport_j', 'pic_j'))
    expected = energy_change + dissipation - terms['knife_work_j'] - terms['gravity_work_j']
    np.testing.assert_allclose(terms['global_balance_residual_j'], expected, atol=1e-17)


def test_ledger_separates_plastic_dissipation_from_pic_transfer():
    cfg = Config(size=(.012, .012, .012), grid_spacing=.006, dt=1e-6,
                 young_modulus=20000, yield_stress=200, external_tool=True,
                 blade_grid_constraint=False, contact_reference_dt=.0002, flip_reference_dt=.0002)
    m = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    c = GeometryEventContact(m)
    ledger = EnergyLedger(c)
    F = np.tile(np.diag([1.1, .95, .95]), (m.n, 1, 1))
    velocity = np.random.default_rng(61).normal(scale=.01, size=(m.n, 3))
    initial = m.state(x=m.reference @ F[0].T + [0, 0, .1], v=velocity, F=F)
    c.step(initial, c.history(), c.tool([1, 1, 1]), CutInstruction(cutting=False))
    terms = ledger.report()['totals']
    assert terms['plastic_j'] > 1e-6
    assert terms['pic_j'] > 1e-12
    assert terms['knife_work_j'] == 0.


def test_checkpoint_recomputation_does_not_count_energy_twice():
    from scripts.diff_cut_validate import terminal_loss

    m, c, initial, history, tool = fixture(.2, .5, friction_update='midpoint', integration='midpoint')
    ledger = EnergyLedger(c)
    steps = 4

    def objective(step, state, history, force, loss):
        if step == steps - 1:
            m.launch(terminal_loss, m.n, inputs=[state.v, state.Fp], outputs=[loss], device=m.device)

    c.checkpoint_vjp(initial, history, [CutInstruction()] * steps, 2,
                     lambda step: c.pose(tool, m.array([step * m.config.dt], float, False)), objective)
    assert c.energy_ledger is ledger
    np.testing.assert_allclose(ledger.report()['duration_s'], steps * m.config.dt, atol=1e-18)
    assert ledger.report()['segments'] > steps
