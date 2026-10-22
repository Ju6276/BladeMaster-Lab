"""Contact work, stored energy, damping and sliding from independent identities."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.contact import friction as original_friction
from blademaster_warp.differentiable_cutting import branch_penalty
from blademaster_warp.energy import contact_storage, friction_increment_energy
from blademaster_warp.friction import tangential_return
from blademaster_warp.precision import launch_double


@wp.kernel
def update(history: wp.array(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3),
           normal: wp.array(dtype=wp.vec3), load: wp.array(dtype=float),
           stiffness: float, damping: float, dt: float, mus: float, mud: float, theta: float,
           result: wp.array(dtype=wp.vec3), force: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    f, h = tangential_return(normal[i], velocity[i], history[i], load[i],
                             stiffness, damping, dt, mus, mud, theta, -1)
    result[i] = h
    force[i] = f


def run(history, velocity, normal, load, k, c, dt, mus, mud, theta):
    h = np.asarray(history).reshape(-1, 3)
    arrays = [wp.array(a, dtype=wp.vec3d, device='cuda:0') for a in (h, velocity, normal)]
    load = wp.array(np.broadcast_to(load, (len(h),)).copy(), dtype=wp.float64, device='cuda:0')
    result = wp.zeros_like(arrays[0])
    force = wp.zeros_like(arrays[0])
    launch_double(update, len(h), inputs=[*arrays, load, k, c, dt, mus, mud, theta],
                  outputs=[result, force], device='cuda:0')
    return result.numpy(), force.numpy()


@pytest.mark.parametrize('theta', [.5, 1.])
@pytest.mark.parametrize('damping', [0., .01, .4])
def test_random_friction_increments_balance_energy(theta, damping):
    rng = np.random.default_rng(318)
    normals = rng.normal(size=(256, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    def tangent(a):
        return a - np.sum(a * normals, axis=1)[:, None] * normals

    history = tangent(rng.normal(scale=1e-4, size=(256, 3)))
    velocity = tangent(rng.normal(scale=.5, size=(256, 3)))
    load = rng.uniform(0., .3, 256)
    k, dt = 500., .0001
    updated, force = run(history, velocity, normals, load, k, damping, dt, .3, .2, theta)
    terms = friction_increment_energy(history, updated, dt * velocity, force, k, damping, dt, theta)
    residual = terms['input_work_j'] - terms['storage_change_j']
    for name in ('viscous_j', 'sliding_j', 'integration_j'):
        assert np.min(terms[name]) >= -1e-18
        residual -= terms[name]
    np.testing.assert_allclose(residual, 0., atol=2e-18)
    np.testing.assert_allclose(np.sum(updated * normals, axis=1), 0., atol=1e-18)


@pytest.mark.parametrize('theta', [.5, 1.])
@pytest.mark.parametrize('dt', [1e-4, 5e-5, 2.5e-5])
def test_zero_friction_creates_no_elastic_history(theta, dt):
    updated, force = run([[0., 0, 0]], [[.5, 0, 0]], [[0., 0, 1.]], .1,
                         500., 2 * .1 * np.sqrt(.01 * 500), dt, 0., 0., theta)
    np.testing.assert_allclose(force, 0., atol=1e-18)
    np.testing.assert_allclose(updated, 0., atol=1e-18)


@pytest.mark.parametrize('theta,order', [(.5, 2), (1., 1)])
def test_sliding_relaxation_matches_analytic_kelvin_voigt(theta, order):
    k, c, speed, load, mu = 500., .4, .5, .1, .2
    equilibrium = mu * load / k
    duration = .0016
    exact = equilibrium * (1 - np.exp(-k * duration / c))
    errors = []
    for dt in (1e-4, 5e-5, 2.5e-5):
        h = np.zeros((1, 3))
        for _ in range(round(duration / dt)):
            h, force = run(h, [[speed, 0, 0]], [[0., 0, 1.]], load, k, c, dt, mu, mu, theta)
            np.testing.assert_allclose(force[0], [-mu * load, 0, 0], atol=1e-16)
        errors.append(abs(h[0, 0] - exact))
    assert errors[0] / errors[1] > .95 * 2**order
    assert errors[1] / errors[2] > .95 * 2**order


def test_contact_spring_energy_and_virtual_work():
    gap = np.array([-.002, .003])
    history = np.array([[.001, -.002, 0], [0, 0, 0]])
    kn, kt = 1000., 500.
    energy = contact_storage(gap, history, kn, kt)
    np.testing.assert_allclose(energy['normal_j'], [.002, 0.])
    np.testing.assert_allclose(energy['tangent_j'], [.00125, 0.])
    direction = np.array([[.2, -.1, 0], [0, 0, 0]])
    for epsilon in (1e-6, 1e-7):
        positive = contact_storage(gap, history + epsilon * direction, kn, kt)['tangent_j']
        negative = contact_storage(gap, history - epsilon * direction, kn, kt)['tangent_j']
        np.testing.assert_allclose((positive - negative) / (2 * epsilon),
                                   kt * np.sum(history * direction, axis=-1), rtol=1e-10)


@wp.kernel
def parameter_update(velocity: wp.array(dtype=wp.vec3), history: wp.array(dtype=wp.vec3),
                     parameters: wp.array(dtype=float), result: wp.array(dtype=wp.vec3),
                     force: wp.array(dtype=wp.vec3)):
    f, h = tangential_return(wp.vec3(0., 0., 1.), velocity[0], history[0],
                             parameters[0], parameters[1], parameters[2], parameters[3],
                             parameters[4], parameters[5], 0.5, -1)
    result[0] = h
    force[0] = f


@pytest.mark.parametrize('speed', [.001, .5])
def test_friction_state_and_parameter_vjp(speed):
    velocity = wp.array([[speed, .3*speed, 0]], dtype=wp.vec3d, device='cuda:0', requires_grad=True)
    history = wp.array([[.00001, -.00001, 0]], dtype=wp.vec3d, device='cuda:0', requires_grad=True)
    parameters = wp.array([.1, 500., .4, .0001, .3, .2], dtype=wp.float64, device='cuda:0', requires_grad=True)
    seeds = [np.array([[1., .3, -.2]]), np.array([[-.2, .7, .1]])]

    def evaluate():
        result = wp.zeros(1, dtype=wp.vec3d, device='cuda:0', requires_grad=True)
        force = wp.zeros_like(result, requires_grad=True)
        launch_double(parameter_update, 1, inputs=[velocity, history, parameters],
                      outputs=[result, force], device='cuda:0')
        return result, force

    with wp.Tape() as tape:
        outputs = evaluate()
    tape.backward(grads={a: wp.array(s, dtype=wp.vec3d, device='cuda:0') for a, s in zip(outputs, seeds)})
    for source in (velocity, history, parameters):
        base = source.numpy()
        direction = np.random.default_rng(16).normal(size=base.shape) * np.maximum(np.abs(base), 1e-5)
        ad = float(np.sum(source.grad.numpy() * direction))
        for epsilon in (1e-5, 1e-6):
            losses = []
            for sign in (-1, 1):
                source.assign(base + sign * epsilon * direction)
                losses.append(sum(np.sum(a.numpy() * s) for a, s in zip(evaluate(), seeds)))
            np.testing.assert_allclose(ad, (losses[1]-losses[0])/(2*epsilon), rtol=1e-6, atol=1e-10)
        source.assign(base)


@wp.kernel
def solver_contact_update(history: wp.array(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3),
                          dt: float, mu: float, stage: int, original: int,
                          updated: wp.array(dtype=wp.vec3), force: wp.array(dtype=wp.vec3)):
    from_history = history[0]
    normal = wp.vec3(0., 0., 1.)
    mass = float(.01)
    kn = float(1000.)
    kt = float(500.)
    impulse = wp.vec3(0.)
    result = wp.vec3(0.)
    if original != 0:
        impulse, result = original_friction(normal, velocity[0], .1 * dt, from_history,
                                            mass, dt, mu, mu, kt * dt * dt / mass, .1)
    else:
        impulse, result = branch_penalty(-.0001, velocity[0], normal, mass, from_history,
                                          dt, kn * dt * dt / mass, mu, mu,
                                          kt * dt * dt / mass, .1, -1, stage, stage)
    updated[0] = result
    force[0] = (impulse - wp.dot(impulse, normal) * normal) / dt


@pytest.mark.parametrize('stage,original', [(0, 1), (0, 0), (1, 0)])
@pytest.mark.parametrize('mu', [0., .2, .5])
def test_solver_contact_work_has_no_created_energy(stage, original, mu):
    h0 = np.zeros((1, 3))
    velocity = np.array([[.5, 0., 0.]])
    damping = 2 * .1 * np.sqrt(.01 * 500)
    for dt in (1e-4, 5e-5, 2.5e-5):
        arrays = [wp.array(a, dtype=wp.vec3d, device='cuda:0') for a in (h0, velocity)]
        updated, force = [wp.zeros_like(arrays[0]) for _ in range(2)]
        launch_double(solver_contact_update, 1, inputs=[*arrays, dt, mu, stage, original],
                      outputs=[updated, force], device='cuda:0')
        terms = friction_increment_energy(h0, updated.numpy(), dt * velocity, force.numpy(),
                                           500., damping, dt, .5 if stage else 1.)
        dissipation = terms['viscous_j'] + terms['sliding_j'] + terms['integration_j']
        assert np.min(dissipation) >= 0
        np.testing.assert_allclose(terms['input_work_j'], terms['storage_change_j'] + dissipation, atol=1e-20)
        if mu == 0:
            np.testing.assert_allclose(updated.numpy(), 0., atol=1e-20)
            np.testing.assert_array_equal(force.numpy(), 0.)


def test_contact_storage_broadcasts_stiffness_with_gap():
    result = contact_storage(-.001, np.zeros((2, 3)), [100., 200.], 500.)
    np.testing.assert_allclose(result['normal_j'], [5e-5, 1e-4])
