"""Compare contact-history propagation with its two-state linear dynamics."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_cutting import branch_penalty
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.presets import banana_config, banana_dt, tangent_history_dt


@wp.kernel
def sticking_step(state: wp.array(dtype=wp.vec2), dt: float, gk: float, gc: float,
                  result: wp.array(dtype=wp.vec2)):
    v = wp.vec3(state[0][0], 0.0, 0.0)
    history = wp.vec3(dt * state[0][1], 0.0, 0.0)
    mass = float(3.0e-5)
    impulse, updated = branch_penalty(-0.001, v, wp.vec3(0.0, 0.0, 1.0), mass,
                                      history, dt, 0.1, 0.2, 0.2, gk, gc, 1)
    result[0] = wp.vec2(v[0] + impulse[0] / mass, updated[0] / dt)


@pytest.mark.parametrize('damping', [0.1, 0.5, 1.0, 2.0])
def test_sticking_history_step_and_adjoint(damping):
    cfg = replace(banana_config(), tangent_damping=damping)
    maximum = banana_dt(cfg, cfg.grid_spacing)
    cfg = replace(cfg, dt=tangent_history_dt(cfg, maximum))
    model = MaterialTrajectory(cfg, precision='float64')
    _, stiffness = cfg.contact_scales()
    linear = np.array([[1 - 2 * damping * np.sqrt(stiffness), -stiffness], [1, 1]])
    assert np.max(np.abs(np.linalg.eigvals(linear))) < 1
    initial = np.array([[0.003, -0.002]])
    seed = np.array([[0.3, -0.2]])
    source = model.array(initial, wp.vec2)
    current = source
    steps = 32
    with wp.Tape() as tape:
        for _ in range(steps):
            following = model.zeros(1, wp.vec2)
            model.launch(sticking_step, 1, inputs=[current, cfg.dt, stiffness, damping],
                         outputs=[following], device=model.device)
            current = following
    tape.backward(grads={current: model.array(seed, wp.vec2, False)})
    propagation = np.linalg.matrix_power(linear, steps)
    np.testing.assert_allclose(current.numpy()[0], propagation @ initial[0], atol=1e-14, rtol=1e-11)
    np.testing.assert_allclose(source.grad.numpy()[0], propagation.T @ seed[0], atol=1e-13, rtol=1e-11)


def test_sticking_history_requires_damping():
    with pytest.raises(ValueError, match='positive tangential damping'):
        tangent_history_dt(replace(banana_config(), tangent_damping=0), 6.25e-5)


def test_step_reduction_preserves_contact_stiffness():
    cfg = banana_config()
    maximum = banana_dt(cfg, cfg.grid_spacing)
    step = tangent_history_dt(cfg, maximum)
    assert step < maximum
    assert replace(cfg, dt=step).contact_scales()[1] / step**2 == pytest.approx(
        replace(cfg, dt=maximum).contact_scales()[1] / maximum**2)
    with pytest.raises(ValueError, match='contact_reference_dt'):
        tangent_history_dt(replace(cfg, contact_reference_dt=0), maximum)
