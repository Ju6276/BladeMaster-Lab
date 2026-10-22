"""Natural contact activation, moving constraints and history resets."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_constraint import solve
from blademaster_warp.differentiable_geometry import GeometryEventContact
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.differentiable_midpoint import project_history
from blademaster_warp.solver import Config


def model(spacing=.006, dt=.0001, **kwargs):
    cfg = Config(size=(.012, .012, .012), grid_spacing=spacing, dt=dt,
                 young_modulus=20000, yield_stress=0, external_tool=True,
                 blade_grid_constraint=False, contact_reference_dt=.0002,
                 flip_reference_dt=.0002, **kwargs)
    return MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)


@pytest.mark.parametrize('spacing', [.006, .004])
@pytest.mark.parametrize('rotation', [False, True])
def test_moving_grid_face_matches_relative_velocity_and_gradient(spacing, rotation):
    m = model(spacing, .0005)
    c = GeometryEventContact(m)
    grid = m.topology(np.zeros(m.n, dtype=np.int64))
    mass = np.bincount(grid.slots.numpy().ravel(), weights=m.mass * m.weights.numpy().ravel())
    x = np.tile([m.config.blade_half_size[0] + 1e-5, .001, .002], (m.n, 1))
    angular = np.array([0., 50., 0.]) if rotation else np.zeros(3)
    translation = np.zeros(3) if rotation else np.array([.1, 0., 0.])
    point_velocity = translation + np.cross(angular, [m.config.blade_half_size[0], .001, .002])
    tool = c.tool([0, 0, 0], translation, omega=angular)
    particle_x = m.array(x, wp.vec3)
    grid_mass = m.array(mass, float, False)
    velocity = m.zeros(grid.count, wp.vec3)
    with wp.Tape() as tape:
        result = solve(m, grid, grid_mass, velocity, particle_x, tool, moving_tool=True)
    seed = m.array([[1., 0., 0.]], wp.vec3, False)
    tape.backward(grads={result.linear_impulse: seed})
    expected = point_velocity[0] - 1e-5 / m.config.dt
    assert result.batches > 0 and result.residual <= 1e-6
    np.testing.assert_allclose(result.velocity.numpy()[:, 0], expected, atol=2e-6)
    derivative = tool.omega.grad.numpy()[0, 1] if rotation else tool.velocity.grad.numpy()[0, 0]
    expected_derivative = m.n * m.mass * (.002 if rotation else 1.)
    np.testing.assert_allclose(derivative, expected_derivative, rtol=2e-5)
    tape.zero()
    shift = np.array([.25, -.07, .04])
    shifted = solve(m, grid, grid_mass, m.array(np.tile(shift, (grid.count, 1)), wp.vec3), particle_x,
                    c.tool([0, 0, 0], translation + shift, omega=angular), moving_tool=True)
    np.testing.assert_allclose(shifted.velocity.numpy() - shift, result.velocity.numpy(), atol=1e-12)
    np.testing.assert_allclose(shifted.linear_impulse.numpy(), result.linear_impulse.numpy(), atol=1e-14)


@pytest.mark.parametrize('spacing', [.006, .004])
def test_moving_grid_face_releases_and_reestablishes_entry_side(spacing):
    m = model(spacing, .0005)
    c = GeometryEventContact(m)
    grid = m.topology(np.zeros(m.n, dtype=np.int64))
    mass = m.array(np.bincount(grid.slots.numpy().ravel(),
                              weights=m.mass * m.weights.numpy().ravel()), float, False)
    velocity = m.zeros(grid.count, wp.vec3)
    old_side = m.array(np.ones(m.n), float, False)
    x = np.tile([-.00051, m.config.blade_half_size[1] + .001, .002], (m.n, 1))
    outside = solve(m, grid, mass, velocity, m.array(x, wp.vec3), c.tool([0, 0, 0]),
                    old_side, moving_tool=True)
    np.testing.assert_array_equal(outside.side.numpy(), 0.)
    x[:, 1] = 0.
    returned = solve(m, grid, mass, velocity, m.array(x, wp.vec3), c.tool([0, 0, 0], [-.1, 0, 0]),
                     outside.side, moving_tool=True)
    np.testing.assert_array_equal(returned.side.numpy(), -1.)
    assert returned.batches > 0
    assert returned.linear_impulse.numpy()[0, 0] < 0


@pytest.mark.parametrize('spacing', [.006, .004])
def test_midpoint_pair_search_matches_exhaustive_stage_contacts(spacing):
    m = model(spacing)
    c = GeometryEventContact(m)
    x = m.reference.copy() + [0, 0, .1]
    x[:, 0] *= 1.001
    codes = np.where(x[:, 0] < 0, 1, 2)
    left = np.flatnonzero(x[:, 0] < 0)
    right = np.flatnonzero(x[:, 0] > 0)
    distances = np.linalg.norm(x[left, None] - x[right], axis=2)
    a, b = np.unravel_index(np.argmin(distances), distances.shape)
    p, r = int(left[a]), int(right[b])
    v = np.zeros_like(x)
    v[:, 0] = np.where(codes == 1, .2, -.2)
    v[p, 0], v[r, 0] = -.02, .02
    initial = m.state(x=x, v=v)
    history = c.history(codes=codes)
    tool = c.tool([1, 1, 1])
    key = min(p, r) * m.n + max(p, r)
    assert key not in c.contact_keys(initial, history.codes)
    seed = np.random.default_rng(15).normal(size=x.shape)
    with wp.Tape() as tape:
        state, updated, _ = c.advance_segment(initial, history, tool, m.duration, False)
    assert key in updated.pair_keys
    tape.backward(grads={state.v: m.array(seed, wp.vec3, False)})
    gradient = initial.v.grad.numpy().copy()
    tape.zero()
    oracle = GeometryEventContact(m)
    all_pairs = [(a, b) for a in range(m.n) for b in range(a + 1, m.n)
                 if codes[a] != codes[b] or np.linalg.norm(m.reference[a] - m.reference[b]) >= 2 * spacing]
    keys = np.array([a * m.n + b for a, b in all_pairs], dtype=np.int64)
    oracle.contact_keys = lambda state, labels, dt=None: keys
    with wp.Tape() as reference_tape:
        expected, expected_h, _ = oracle.advance_segment(initial, history, tool, m.duration, False)
    reference_tape.backward(grads={expected.v: m.array(seed, wp.vec3, False)})
    for name in ('x', 'v', 'F', 'Fp'):
        np.testing.assert_allclose(getattr(state, name).numpy(), getattr(expected, name).numpy(), atol=1e-12)
    np.testing.assert_allclose(gradient, initial.v.grad.numpy(), rtol=1e-11, atol=1e-11)
    rows = {int(k): i for i, k in enumerate(keys)}
    np.testing.assert_allclose(updated.pairs.numpy(),
                               expected_h.pairs.numpy()[[rows[int(k)] for k in updated.pair_keys]], atol=1e-14)


@pytest.mark.parametrize('surface', ['board', 'blade'])
def test_contact_history_has_zero_rate_before_contact(surface):
    m = model(tangent_damping=0., friction_static=10., friction_dynamic=10.)
    c = GeometryEventContact(m)
    x, v = m.reference.copy(), np.zeros((m.n, 3))
    if surface == 'board':
        x[:, 2] = m.config.grid_spacing / 4 + 1e-5
        normal_axis, tangent_axis, channel = 2, 0, 0
        tool = c.tool([1, 1, 1])
    else:
        x[:, 2] += .1
        x[:, 0] = m.config.blade_half_size[0] + m.config.grid_spacing / 4 + 1e-5
        normal_axis, tangent_axis, channel = 0, 1, 1
        tool = c.tool([0, 0, .106])
    v[:, normal_axis], v[:, tangent_axis] = -.5, .1
    state, history, _ = c.advance_segment(m.state(x=x, v=v), c.history(), tool, m.duration, False)
    np.testing.assert_allclose(state.v.numpy()[:, tangent_axis], .1, atol=1e-13)
    np.testing.assert_allclose(history.rigid.numpy()[channel, :, tangent_axis], .1 * m.config.dt,
                               atol=1e-13)


@pytest.mark.parametrize('knife_speed', [0., .1])
def test_midpoint_grid_projection_constrains_the_final_stage_velocity(knife_speed):
    from dataclasses import replace

    base = model(dt=1e-5)
    m = MaterialTrajectory(replace(base.config, blade_grid_constraint=True),
                           precision='float64', ordered_transfers=True)
    c = GeometryEventContact(m)
    x = m.reference.copy() + [0, 0, .1]
    x[:, 0] = m.config.blade_half_size[0] + 1e-7
    v = np.zeros_like(x)
    v[:, 0] = -.5
    tool = c.tool([0, 0, .106], [knife_speed, 0, 0])
    initial = m.state(x=x, v=v)
    with wp.Tape() as tape:
        state, history, force = c.advance_segment(initial, c.history(), tool, m.duration, False)
    assert history.grid_batches > 0
    np.testing.assert_allclose(state.v.numpy()[:, 0], knife_speed, atol=2e-6)
    particle_impulse = m.mass * np.sum(state.v.numpy()[:, 0] - v[:, 0])
    np.testing.assert_allclose(particle_impulse, -force.numpy()[0, 0] * m.config.dt, atol=1e-12)
    seed = np.zeros_like(v)
    seed[:, 0] = 1 / m.n
    tape.backward(grads={state.v: m.array(seed, wp.vec3, False)})
    np.testing.assert_allclose(tool.velocity.grad.numpy()[0, 0], 1., atol=2e-5)
    np.testing.assert_allclose(np.sum(initial.v.grad.numpy()[:, 0]), 0., atol=2e-5)


@wp.kernel
def zero_cone_probe(parameters: wp.array(dtype=float), damping: float, output: wp.array(dtype=wp.vec3)):
    output[0] = project_history(-.0002, wp.vec3(parameters[0], 0., 0.), wp.vec3(0., 0., 1.),
                                 wp.vec3(parameters[1], 0., 0.), 75., 37.5, damping, 0., 0.)


@pytest.mark.parametrize('damping', [0., .1])
def test_zero_friction_history_cancels_trial_force_and_its_derivative(damping):
    m = model()
    parameters = m.array([0., 0.], float)
    output = m.zeros(1, wp.vec3)
    with wp.Tape() as tape:
        m.launch(zero_cone_probe, 1, inputs=[parameters, damping], outputs=[output], device=m.device)
    tape.backward(grads={output: m.array([[1., 0., 0.]], wp.vec3, False)})
    np.testing.assert_allclose(parameters.grad.numpy(), [-damping / 37.5, 0.], atol=1e-15)
    for sign in (-1, 1):
        values = sign * np.array([1e-7, 2e-7])
        result = m.zeros(1, wp.vec3)
        m.launch(zero_cone_probe, 1, inputs=[m.array(values, float), damping], outputs=[result], device=m.device)
        np.testing.assert_allclose(37.5 * result.numpy()[0, 0] + damping * values[0], 0., atol=1e-15)


@pytest.mark.parametrize('spacing', [.006, .004])
def test_natural_blade_face_to_edge_trajectory_derivative(spacing):
    m = model(spacing)
    c = GeometryEventContact(m)
    x = m.reference.copy() + [1, 1, 1]
    x[0] = [m.config.blade_half_size[0] + .75 * spacing / 4,
            m.config.blade_half_size[1] - 3.7e-5, .1]
    v = np.zeros_like(x)
    v[:, 1] = .2
    tool = c.tool([0, 0, .1])
    initial = m.state(x=x, v=v)
    seed = np.zeros_like(v)
    seed[0] = [1., .4, .2]

    def rollout():
        state, history = initial, c.history()
        features = []
        for _ in range(5):
            state, history, _ = c.advance_segment(state, history, tool, m.duration, False)
            features.append(int(history.rigid_modes.numpy()[2, 0]))
        return state, features

    with wp.Tape() as tape:
        terminal, features = rollout()
    assert len(set(features)) >= 2
    tape.backward(grads={terminal.v: m.array(seed, wp.vec3, False)})
    gradient = initial.x.grad.numpy()[0, 1]
    assert abs(gradient) > 1e-3
    tape.zero()
    for epsilon in (1e-8, 3e-9):
        values = []
        for sign in (-1, 1):
            shifted = x.copy()
            shifted[0, 1] += sign * epsilon
            initial.x.assign(shifted)
            state, _ = rollout()
            values.append(float(np.sum(state.v.numpy() * seed)))
        np.testing.assert_allclose(gradient, (values[1] - values[0]) / (2 * epsilon), rtol=2e-5, atol=1e-7)
