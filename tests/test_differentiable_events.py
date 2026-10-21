"""Event time gradients for moving edge planes and actual cutting-work thresholds."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_cutting import FixedCutContact
from blademaster_warp.differentiable_events import edge_crossings, threshold_times
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.solver import Config


def model():
    return MaterialTrajectory(
        Config(size=(.012, .012, .012), grid_spacing=.006, dt=.0002,
               external_tool=True, blade_grid_constraint=False, cut_hold=True,
               blade_bevel_height=.005, cutting_energy=1.),
        precision='float64', ordered_transfers=True,
    )


def test_threshold_event_requires_a_new_crossing():
    m = model()
    before = m.array([.8, .2, 1., .5], float)
    after = m.array([1.2, .3, 1.1, .5], float)
    times = m.zeros(4, float)
    with wp.Tape() as tape:
        m.launch(threshold_times, 4, inputs=[before, after, .2, .001], outputs=[times], device=m.device)
    np.testing.assert_allclose(times.numpy(), [.2005, -1, -1, -1], atol=1e-15)
    tape.backward(grads={times: m.array(np.ones(4), float, False)})
    np.testing.assert_allclose(before.grad.numpy(), [-.00125, 0, 0, 0], atol=1e-15)
    np.testing.assert_allclose(after.grad.numpy(), [-.00125, 0, 0, 0], atol=1e-15)


@pytest.mark.parametrize('angle', [0., .35])
@pytest.mark.parametrize('parameter', ['particle', 'edge'])
def test_moving_edge_time_and_location_gradient(angle, parameter):
    m = model()
    normal = np.array([np.sin(angle), 0, np.cos(angle)])
    x0 = np.array([[.0002, .001, .018]])
    x1 = x0 + 1e-5 * normal
    edge0 = x0 + 6e-6 * normal
    edge1 = edge0 - 12e-6 * normal + [.0003, 0, 0]
    # Keep the translation tangent to the tilted plane.
    edge1 -= .0003 * normal[0] * normal
    values = [x0, x1, edge0, edge1]
    arrays = [m.array(value, wp.vec3) for value in values]
    time_seed = m.array([1.], float, False)
    location_seed = m.array([[.3, .2, .1]], wp.vec3, False)

    def evaluate(inputs):
        times, positions = m.zeros(1, float), m.zeros(1, wp.vec3)
        m.launch(edge_crossings, 1, inputs=[*inputs, wp.vec3d(*normal), .01, .0002],
                 outputs=[times, positions], device=m.device)
        return times, positions

    with wp.Tape() as tape:
        times, positions = evaluate(arrays)
    tape.backward(grads={times: time_seed, positions: location_seed})
    np.testing.assert_allclose(times.numpy()[0], .01 + .0002 * 6 / 22, atol=1e-15)
    index = 1 if parameter == 'particle' else 3
    direction = np.array([[.2, -.1, .7]])
    ad = float(np.sum(arrays[index].grad.numpy() * direction))
    for epsilon in (1e-9, 3e-10):
        loss = []
        for sign in (-1, 1):
            inputs = list(arrays)
            inputs[index] = m.array(values[index] + sign * epsilon * direction, wp.vec3)
            t, p = evaluate(inputs)
            loss.append(float(t.numpy()[0] + p.numpy()[0] @ [.3, .2, .1]))
        np.testing.assert_allclose(ad, (loss[1] - loss[0]) / (2 * epsilon), rtol=2e-7, atol=1e-8)


@pytest.mark.parametrize('parameter', ['fracture_energy', 'initial_damage', 'knife_position'])
def test_cutting_contact_release_time_gradient(parameter):
    m = model()
    c = FixedCutContact(m)
    positions = m.reference + [.2, .2, .2]
    positions[0] = [.0001, 0, .0195]
    state = m.state(x=positions)
    damage = np.zeros(m.n)
    damage[0] = .2
    history = c.history(damage=damage)
    branch = c.freeze(np.zeros(m.n, dtype=np.int64), [], [], np.zeros(m.n, dtype=int))
    tool = c.tool([0, 0, .04], [0, 0, -.12])

    def evaluate(h, knife):
        _, updated, _ = c.step(state, h, knife, branch)
        times = m.zeros(m.n, float)
        m.launch(threshold_times, m.n, inputs=[h.damage, updated.trial_damage, 0., m.config.dt],
                 outputs=[times], device=m.device)
        return times, updated

    with wp.Tape() as tape:
        times, updated = evaluate(history, tool)
    seed = np.zeros(m.n)
    seed[0] = 1
    tape.backward(grads={times: m.array(seed, float, False)})
    assert updated.damage.numpy()[0] == 1
    assert updated.trial_damage.numpy()[0] > 1
    assert 0 < times.numpy()[0] < m.config.dt
    if parameter == 'fracture_energy':
        ad = c.fracture_energy.grad.numpy()[0]
        np.testing.assert_allclose(ad, times.numpy()[0] / m.config.cutting_energy, rtol=1e-12)
        epsilons = (1e-3, 3e-4)
    elif parameter == 'initial_damage':
        ad = history.damage.grad.numpy()[0]
        epsilons = (1e-4, 3e-5)
    else:
        ad = tool.position.grad.numpy()[0, 2]
        epsilons = (1e-7, 3e-8)
    assert abs(ad) > 1e-6
    for epsilon in epsilons:
        values = []
        for sign in (-1, 1):
            h, knife = history, tool
            if parameter == 'fracture_energy':
                c.fracture_energy.assign([1 + sign * epsilon])
            elif parameter == 'initial_damage':
                other = damage.copy()
                other[0] += sign * epsilon
                h = c.history(damage=other)
            else:
                knife = c.tool([0, 0, .04 + sign * epsilon], [0, 0, -.12])
            times, _ = evaluate(h, knife)
            values.append(times.numpy()[0])
        np.testing.assert_allclose(ad, (values[1] - values[0]) / (2 * epsilon), rtol=2e-6, atol=1e-9)


@wp.kernel
def sum_damage(damage: wp.array(dtype=float), loss: wp.array(dtype=float)):
    wp.atomic_add(loss, 0, damage[wp.tid()])


@pytest.mark.parametrize('segment_steps', [1, 3])
def test_fracture_energy_checkpoint_gradient(segment_steps):
    m = model()
    c = FixedCutContact(m)
    c.fracture_energy.assign([1e6])
    positions = m.reference + [.2, .2, .2]
    positions[0] = [.0001, 0, .0195]
    initial = m.state(x=positions)
    history = c.history()
    branch = c.freeze(np.zeros(m.n, dtype=np.int64), [], [], np.zeros(m.n, dtype=int))
    tool = c.tool([0, 0, .04], [0, 0, -.12])

    def objective(step, state, h, force, loss):
        if step == 3:
            m.launch(sum_damage, m.n, inputs=[h.damage], outputs=[loss], device=m.device)

    def rollout():
        state, h = initial, history
        loss = m.zeros(1, float)
        for step in range(4):
            state, h, force = c.step(state, h, tool, branch)
            objective(step, state, h, force, loss)
        return loss

    with wp.Tape() as tape:
        loss = rollout()
    tape.backward(loss)
    expected = float(c.fracture_energy.grad.numpy()[0])
    assert abs(expected) > 1e-12
    tape.zero()
    timings = {}
    result = c.checkpoint_vjp(initial, history, [branch] * 4, segment_steps,
                              lambda step: tool, objective, timings=timings)
    np.testing.assert_allclose(result.parameter_gradients[c.fracture_energy][0], expected, rtol=1e-12)
    assert max(timings['recompute_max_abs'].values()) == 0
    for epsilon in (100., 30.):
        values = []
        for sign in (-1, 1):
            c.fracture_energy.assign([1e6 + sign * epsilon])
            values.append(float(rollout().numpy()[0]))
        np.testing.assert_allclose(expected, (values[1] - values[0]) / (2 * epsilon), rtol=2e-7, atol=1e-15)
