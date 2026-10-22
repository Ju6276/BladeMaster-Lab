"""Tool work from actual contact impulses, including rotation and event segments."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_cutting import FixedCutContact
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.solver import Config
from test_differentiable_geometry import fixture


@pytest.mark.parametrize('stage', [False, True])
@pytest.mark.parametrize('angle', [0., .3])
@pytest.mark.parametrize('motion', ['stationary', 'translation', 'rotation'])
def test_planar_contact_work_matches_analytic_wrench(stage, angle, motion):
    cfg = Config(size=(.012, .012, .012), grid_spacing=.006, dt=.0002,
                 external_tool=True, friction_static=0., friction_dynamic=0.,
                 tangent_damping=0., yield_stress=0, contact_reference_dt=.0002)
    m = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    c = FixedCutContact(m, friction_update='projected')
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    origin = np.array([0., 0., .1])
    local = np.array([cfg.blade_half_size[0] + .0001, .002, .003])
    lever = rotation @ np.array([cfg.blade_half_size[0], local[1], local[2]])
    normal = rotation[:, 0]
    velocity = np.array([.02, 0, 0]) if motion == 'translation' else np.zeros(3)
    omega = np.array([0, 2., 0]) if motion == 'rotation' else np.zeros(3)
    tool = c.tool(origin, velocity, [0, np.sin(angle / 2), 0, np.cos(angle / 2)], omega)
    state = m.state(x=np.tile(origin + rotation @ local, (m.n, 1)))
    history = c.history()
    branch = c.freeze(np.zeros(m.n, np.int64), [], [], np.zeros(m.n), cutting=False)
    _, updated, force = c.step(state, history, tool, branch,
                               stage_base=state if stage else None,
                               stage_history=history if stage else None)
    tool_point_velocity = velocity + np.cross(omega, lever)
    gap = .0001 - cfg.grid_spacing / 4
    if not stage:
        gap += cfg.dt * min(-tool_point_velocity @ normal, 0.)
    impulse = -cfg.contact_scales()[0] * m.mass / cfg.dt * gap * normal
    expected = m.n * impulse @ tool_point_velocity
    wrench_work = (m.n * impulse @ velocity + m.n * np.cross(lever, impulse) @ omega)
    np.testing.assert_allclose(updated.tool_work.numpy()[0], expected, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(expected, wrench_work, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(force.numpy()[0], -m.n * impulse / cfg.dt, rtol=1e-12, atol=1e-12)
    if motion == 'rotation':
        assert expected > 1e-8
        assert force.numpy()[0] @ velocity == 0


@pytest.mark.parametrize('omega', [0., 2.])
def test_grid_constraint_work_includes_angular_impulse(omega, monkeypatch):
    from blademaster_warp import differentiable_constraint

    cfg = Config(size=(.012, .012, .012), grid_spacing=.006, dt=.0002,
                 external_tool=True, blade_grid_constraint=True, yield_stress=0,
                 friction_static=0., friction_dynamic=0., tangent_damping=0.)
    m = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    c = FixedCutContact(m, friction_update='projected')
    state = m.state(x=np.tile([cfg.blade_half_size[0] + 1e-5, .002, .103], (m.n, 1)),
                    v=np.tile([-2., 0, 0], (m.n, 1)))
    history = c.history()
    tool = c.tool([0, 0, .1], [.02, 0, 0], omega=[0, omega, 0])
    branch = c.freeze(np.zeros(m.n, np.int64), [], [], np.zeros(m.n), cutting=False)
    captured = []
    solve = differentiable_constraint.solve

    def record(*args, **kwargs):
        result = solve(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(differentiable_constraint, 'solve', record)
    _, updated, force = c.step(state, history, tool, branch, stage_base=state, stage_history=history)
    assert len(captured) == 1 and captured[0].batches > 0
    constraint = captured[0]
    linear = constraint.linear_impulse.numpy()[0]
    angular = constraint.angular_impulse.numpy()[0]
    lever = np.array([cfg.blade_half_size[0], .002, .003])
    np.testing.assert_allclose(angular, np.cross(lever, linear), rtol=1e-12, atol=1e-18)
    overlap = cfg.grid_spacing / 4 - 1e-5
    penalty = np.array([m.n * cfg.contact_scales()[0] * m.mass / cfg.dt * overlap, 0, 0])
    expected = ((penalty + linear) @ tool.velocity.numpy()[0]
                + (np.cross(lever, penalty) + angular) @ tool.omega.numpy()[0])
    np.testing.assert_allclose(updated.tool_work.numpy()[0], expected, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(force.numpy()[0], -(penalty + linear) / cfg.dt, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('integration', ['symplectic', 'midpoint'])
def test_work_across_natural_events_and_checkpoint(integration, monkeypatch):
    from blademaster_warp.differentiable_natural import CutInstruction

    monkeypatch.setattr(wp.config, 'verify_autograd_array_access', True)
    m, c, initial, history, tool = fixture(friction_update='projected', integration=integration)
    steps = 3

    def knife(step):
        return c.pose(tool, m.array([step * m.config.dt], float, False))

    def objective(step, state, h, force, loss):
        m.launch(add_work, 1, inputs=[h.tool_work], outputs=[loss], device=m.device)

    def run():
        state, h, loss = initial, history, m.zeros(1, float)
        force_work = 0.
        events = 0
        for step in range(steps):
            state, h, force = c.step(state, h, knife(step))
            force_work -= m.config.dt * force.numpy()[0] @ tool.velocity.numpy()[0]
            events += len(h.geometry_events)
            objective(step, state, h, force, loss)
        assert events > 0
        np.testing.assert_allclose(loss.numpy()[0], force_work, rtol=1e-12, atol=1e-16)
        return loss

    with wp.Tape() as tape:
        loss = run()
    assert loss.numpy()[0] > 0
    tape.backward(loss)
    expected = tool.position.grad.numpy().copy()
    tape.zero()
    result = c.checkpoint_vjp(initial, history, [CutInstruction()] * steps, 2, knife, objective,
                              parameters=(tool.position,), checkpoint_device='cpu')
    np.testing.assert_allclose(result.loss.numpy(), loss.numpy(), rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(result.parameter_gradients[tool.position], expected, rtol=1e-9, atol=1e-10)
    base = tool.position.numpy().copy()
    assert abs(expected[0, 2]) > 1e-6
    for epsilon in (1e-8, 3e-9):
        values = []
        for sign in (-1, 1):
            tool.position.assign(base + [[0, 0, sign * epsilon]])
            values.append(float(run().numpy()[0]))
        np.testing.assert_allclose((values[1] - values[0]) / (2 * epsilon), expected[0, 2],
                                   rtol=2e-4, atol=1e-6)
    tool.position.assign(base)


@pytest.mark.parametrize('integration', ['symplectic', 'midpoint'])
def test_rotating_event_work_gradient(integration, monkeypatch):
    monkeypatch.setattr(wp.config, 'verify_autograd_array_access', True)
    m, c, initial, history, tool = fixture(.2, .5, friction_update='projected', integration=integration)

    def run():
        state, h, loss = initial, history, m.zeros(1, float)
        events = 0
        for step in range(12):
            at_step = c.pose(tool, m.array([step * m.config.dt], float, False))
            state, h, _ = c.step(state, h, at_step)
            events += len(h.geometry_events)
            m.launch(add_work, 1, inputs=[h.tool_work], outputs=[loss], device=m.device)
        assert events > 0 and loss.numpy()[0] > 0
        return loss

    with wp.Tape() as tape:
        loss = run()
    tape.backward(loss)
    ad = tool.omega.grad.numpy()[0, 1]
    assert abs(ad) > 1e-9
    base = tool.omega.numpy().copy()
    for epsilon in (1e-5, 3e-6):
        values = []
        for sign in (-1, 1):
            tool.omega.assign(base + [[0, sign * epsilon, 0]])
            values.append(run().numpy()[0])
        np.testing.assert_allclose((values[1] - values[0]) / (2 * epsilon), ad,
                                   rtol=2e-4, atol=1e-10)
    tool.omega.assign(base)


@wp.kernel
def add_work(work: wp.array(dtype=float), total: wp.array(dtype=float)):
    wp.atomic_add(total, 0, work[0])
