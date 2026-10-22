"""Fracture work, irreversible memory, rotating surfaces and their derivatives."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive import CohesiveParameters, cohesive_response
from blademaster_warp.precision import launch_double


def array(value, dtype, grad=False):
    return wp.array(np.asarray(value), dtype=dtype, requires_grad=grad)


def inputs(gap, history, parameters, deformation=None, normal=None):
    gap = np.asarray(gap).reshape(-1, 3)
    n = len(gap)
    deformation = np.tile(np.eye(3), (n, 1, 1)) if deformation is None else deformation
    normal = np.tile([1., 0., 0.], (n, 1)) if normal is None else normal
    return [array(gap, wp.vec3d, True), array(deformation, wp.mat33d, True),
            array(normal, wp.vec3d, True), array(np.broadcast_to(history, (n,)), wp.float64, True),
            array(np.broadcast_to(parameters, (n, 3)), wp.vec3d, True)]


def response(values):
    outputs = [wp.zeros(len(values[0]), dtype=dtype, requires_grad=True)
               for dtype in [wp.vec3d, wp.mat33d, *[wp.float64] * 4]]
    launch_double(cohesive_response, len(values[0]), inputs=values, outputs=outputs)
    return outputs


@pytest.mark.parametrize('parameters', [(0., 1., 1.), (1., -1., 1.), (1., 1., np.inf), (1., 2., 1.)])
def test_invalid_cohesive_parameters(parameters):
    with pytest.raises(ValueError):
        CohesiveParameters(*parameters)


@pytest.mark.parametrize('gc', [7., 70., 700., 1400.])
def test_loading_unloading_fracture_work_and_compression(gc):
    parameters = CohesiveParameters(5e8, 5e4, gc)
    onset, failure = parameters.strength / parameters.stiffness, 2 * gc / parameters.strength
    partial = .5 * (onset + failure)
    opening = np.array([0., onset, partial, onset / 4, partial, failure, 1.2 * failure, 0., -3e-5, 0.])
    history = np.maximum.accumulate(np.maximum(opening, 0.))
    gap = np.column_stack([opening, np.zeros((len(opening), 2))])
    traction, _, maximum, damage, storage, fracture = [a.numpy() for a in response(inputs(gap, history, parameters.as_tuple()))]
    work = np.cumsum(.5 * (traction[1:, 0] + traction[:-1, 0]) * np.diff(opening))
    np.testing.assert_allclose(work, storage[1:] - storage[0] + fracture[1:], atol=gc * 3e-14)
    np.testing.assert_allclose(work[-1], gc, rtol=3e-14)
    np.testing.assert_array_equal(maximum, history)
    assert np.all(np.diff(damage) >= 0.) and np.all(np.diff(fracture) >= 0.)
    assert damage[2] == damage[3] == damage[4]
    np.testing.assert_allclose(traction[-2], [-15000., 0., 0.], rtol=1e-14)


def test_rotation_and_surface_work_conjugacy():
    rng = np.random.default_rng(973)
    count = 64
    F = np.eye(3) + rng.normal(scale=.12, size=(count, 3, 3))
    normal = rng.normal(size=(count, 3))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    gap = rng.normal(scale=.001, size=(count, 3))
    previous = rng.uniform(0., .003, count)
    values = inputs(gap, previous, [5e8, 5e4, 70.], F, normal)
    with wp.Tape() as tape:
        outputs = response(values)
    one = array(np.ones(count), wp.float64)
    tape.backward(grads={outputs[4]: one, outputs[5]: one})
    np.testing.assert_allclose(values[0].grad.numpy(), outputs[0].numpy(), rtol=1e-11, atol=1e-8)
    np.testing.assert_allclose(values[1].grad.numpy(), outputs[1].numpy(), rtol=1e-11, atol=1e-9)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = response(inputs(gap @ q.T, previous, [5e8, 5e4, 70.], q @ F, normal))
    np.testing.assert_allclose(rotated[0].numpy(), outputs[0].numpy() @ q.T, rtol=1e-11, atol=1e-8)
    np.testing.assert_allclose(rotated[1].numpy(), q @ outputs[1].numpy(), rtol=1e-11, atol=1e-9)
    for a, b in zip(rotated[2:], outputs[2:]):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-11, atol=1e-11)
    omega = np.array([[0., -.3, .2], [.3, 0., -.1], [-.2, .1, 0.]])
    power = np.sum(outputs[0].numpy() * (gap @ omega.T), axis=1)
    power += np.sum(outputs[1].numpy() * (omega @ F), axis=(1, 2))
    np.testing.assert_allclose(power, 0., atol=1e-10)


@pytest.mark.parametrize('maximum', [.0002, .0003, .001])
def test_loading_boundary_has_analytic_one_sided_tangents(maximum):
    stiffness, strength, gc = 5e8, 5e4, 70.
    onset, failure = strength / stiffness, 2 * gc / strength
    values = inputs([[maximum, 0., 0.]], maximum, [stiffness, strength, gc])
    with wp.Tape() as tape:
        outputs = response(values)
    tape.backward(grads={outputs[0]: array([[1., 0., 0.]], wp.vec3d)})
    ad = values[0].grad.numpy()[0, 0]
    left = stiffness * (1 - outputs[3].numpy()[0])
    right = -strength / (failure - onset)
    assert min(abs(ad - left), abs(ad - right)) < 1e-6
    f0 = outputs[0].numpy()[0, 0]
    for epsilon in [1e-8, 1e-9]:
        values[0].assign([[maximum - epsilon, 0., 0.]])
        minus = response(values)[0].numpy()[0, 0]
        values[0].assign([[maximum + epsilon, 0., 0.]])
        plus = response(values)[0].numpy()[0, 0]
        np.testing.assert_allclose([(f0 - minus) / epsilon, (plus - f0) / epsilon], [left, right], rtol=1e-8)


def test_state_and_material_gradients():
    rng = np.random.default_rng(83)
    count = 48
    F = np.eye(3) + rng.normal(scale=.1, size=(count, 3, 3))
    normal = rng.normal(size=(count, 3))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    values = inputs(rng.normal(scale=.001, size=(count, 3)), rng.uniform(0., .003, count),
                    [5e8, 5e4, 70.], F, normal)
    seeds = [rng.normal(size=(count, 3)) / 5e4, rng.normal(size=(count, 3, 3)) / 70.,
             rng.normal(size=count) / .0028, rng.normal(size=count),
             rng.normal(size=count) / 70., rng.normal(size=count) / 70.]
    with wp.Tape() as tape:
        outputs = response(values)
    tape.backward(grads={a: array(seed, a.dtype) for a, seed in zip(outputs, seeds)})
    for source in values:
        base = source.numpy()
        direction = rng.normal(size=base.shape) * np.maximum(np.abs(base), 1e-4)
        ad = float(np.sum(source.grad.numpy() * direction))
        for epsilon in [1e-5, 1e-6]:
            states = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                states.append([a.numpy() for a in response(values)])
            fd = sum(float(np.sum((b - a) * seed)) for a, b, seed in zip(states[0], states[1], seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=1e-6, atol=1e-8)
        source.assign(base)


def test_intact_zero_opening_has_elastic_tangent():
    values = inputs([[0., 0., 0.]], 0., [5e8, 5e4, 70.])
    seed = np.array([[.3, -.2, .5]])
    with wp.Tape() as tape:
        outputs = response(values)
    tape.backward(grads={outputs[0]: array(seed, wp.vec3d)})
    for output in outputs:
        np.testing.assert_array_equal(output.numpy(), 0.)
    np.testing.assert_allclose(values[0].grad.numpy(), 5e8 * seed, rtol=1e-14)


def test_surface_frame_depends_only_on_tangent_columns():
    gap = [[-3e-5, .0002, .0001]]
    parameters = [5e8, 5e4, 70.]
    full = response(inputs(gap, .0003, parameters))
    surface = response(inputs(gap, .0003, parameters, deformation=[np.diag([0., 1., 1.])]))
    for a, b in zip(full, surface):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-13, atol=1e-12)
