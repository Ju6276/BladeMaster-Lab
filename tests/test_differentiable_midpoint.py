"""Analytic sticking/sliding contact dynamics and midpoint sensitivity convergence."""

import numpy as np
import pytest
from scipy.linalg import expm
import warp as wp

from blademaster_warp.differentiable_geometry import GeometryEventContact
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.solver import Config


def oscillator(dt, *, sliding=False):
    cfg = Config(size=(.012, .012, .012), grid_spacing=.006, dt=dt,
                 external_tool=True, blade_grid_constraint=False, yield_stress=0,
                 young_modulus=20000, contact_reference_dt=.0002, flip_reference_dt=.0002,
                 tangent_damping=0. if sliding else .1,
                 friction_static=.2, friction_dynamic=.2)
    m = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    c = GeometryEventContact(m)
    sn, gk = cfg.contact_scales()
    kn, kt = sn * m.mass / dt**2, gk * m.mass / dt**2
    damping = 2 * cfg.tangent_damping * np.sqrt(m.mass * kt)
    x = m.reference.copy()
    x[:, 2] = cfg.grid_spacing / 4 - m.mass * 9.81 / kn
    velocity = np.zeros_like(x)
    velocity[:, 0] = .1 if sliding else 1e-5
    rigid = np.zeros((2, m.n, 3))
    if sliding:
        rigid[0, :, 0] = cfg.friction_dynamic * m.mass * 9.81 / kt
    state = m.state(x=x, v=velocity)
    history = c.history(rigid=rigid)
    tool = c.tool([0, 0, 1])
    return m, c, state, history, tool, kt, damping


def test_sticking_midpoint_state_and_gradient_converge_to_linear_ode():
    errors, gradient_errors = [], []
    total = .0004
    for steps in (8, 16, 32):
        dt = total / steps
        m, c, initial, history, tool, kt, damping = oscillator(dt)
        exact = expm(np.array([[-damping / m.mass, -kt / m.mass], [1., 0.]]) * total)
        with wp.Tape() as tape:
            state = initial
            for _ in range(steps):
                state, history, _ = c.advance_segment(state, history, tool, m.duration, False)
        seed = np.zeros_like(initial.v.numpy())
        seed[:, 0] = 1 / m.n
        tape.backward(grads={state.v: m.array(seed, wp.vec3, False)})
        expected = exact @ np.array([1e-5, 0.])
        observed = np.array([np.mean(state.v.numpy()[:, 0]), np.mean(history.rigid.numpy()[0, :, 0])])
        errors.append(np.max(np.abs(observed - expected)))
        ad = float(np.sum(initial.v.grad.numpy()[:, 0]))
        gradient_errors.append(abs(ad - exact[0, 0]))
        assert np.max(np.abs(state.v.numpy()[:, 2])) < 1e-12
        assert np.max(np.abs(state.Fp.numpy() - np.eye(3))) < 1e-12
    for sequence in (errors, gradient_errors):
        assert sequence[0] / sequence[1] > 3.8
        assert sequence[1] / sequence[2] > 3.8
    assert gradient_errors[-1] < 2e-5


@pytest.mark.parametrize('steps', [2, 8])
def test_sliding_midpoint_matches_constant_coulomb_acceleration(steps):
    total = .0004
    m, c, initial, history, tool, _, _ = oscillator(total / steps, sliding=True)
    state = initial
    for _ in range(steps):
        state, history, _ = c.advance_segment(state, history, tool, m.duration, False)
    acceleration = -.2 * 9.81
    np.testing.assert_allclose(state.v.numpy()[:, 0], .1 + acceleration * total, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(state.x.numpy()[:, 0] - initial.x.numpy()[:, 0],
                               .1 * total + .5 * acceleration * total**2, rtol=1e-10, atol=1e-14)


def test_midpoint_requires_midpoint_friction():
    m, _, _, _, _, _, _ = oscillator(.00005)
    with pytest.raises(ValueError, match='midpoint friction'):
        GeometryEventContact(m, friction_update='implicit')
