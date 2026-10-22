"""Zero-opening loads, fracture work, constrained stationarity and derivatives."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive import ExtrinsicCohesiveParameters, cohesive_proximal_response
from blademaster_warp.precision import launch_double


def array(value, vector=False, grad=False):
    return wp.array(np.asarray(value), dtype=wp.vec3d if vector else wp.float64, requires_grad=grad)


def response(q, history, parameters, normal=None):
    q = np.asarray(q).reshape(-1, 3)
    count = len(q)
    normal = np.tile([1., 0., 0.], (count, 1)) if normal is None else normal
    values = [q, normal, np.broadcast_to(history, (count,)), np.broadcast_to(parameters, (count, 3))]
    inputs = [array(v, i != 2, True) for i, v in enumerate(values)]
    return inputs, evaluate(inputs)


def evaluate(inputs):
    outputs = [wp.zeros(len(inputs[0]), dtype=wp.vec3d if i < 2 else wp.float64,
                        requires_grad=True) for i in range(5)]
    launch_double(cohesive_proximal_response, len(inputs[0]), inputs=inputs, outputs=outputs)
    return outputs


@pytest.mark.parametrize('strength,gc', [(0., 7.), (1., -1.), (np.inf, 7.), (1., np.nan)])
def test_invalid_material(strength, gc):
    with pytest.raises(ValueError):
        ExtrinsicCohesiveParameters(strength, gc)


@pytest.mark.parametrize('compliance', [0., -1., np.inf, 5.6e-8])
def test_invalid_proximal_compliance(compliance):
    with pytest.raises(ValueError):
        ExtrinsicCohesiveParameters(5e4, 70.).proximal_parameters(compliance)


@pytest.mark.parametrize('gc', [7., 70., 700., 1400.])
def test_loaded_birth_and_exact_fracture_work(gc):
    strength = 5e4
    failure = 2 * gc / strength
    compliance = .1 * failure / strength
    parameters = ExtrinsicCohesiveParameters(strength, gc).proximal_parameters(compliance)
    preload = np.array([[.7, .2, -.1], [-3., .3, .4], [0., 0., 0.]]) * strength
    _, outputs = response(compliance * preload, 0., parameters)
    gap, traction, maximum, storage, fracture = [a.numpy() for a in outputs]
    np.testing.assert_array_equal(gap, 0.)
    np.testing.assert_allclose(traction, preload, rtol=1e-14, atol=1e-10)
    np.testing.assert_array_equal(maximum, 0.)
    np.testing.assert_array_equal(storage, 0.)
    np.testing.assert_array_equal(fracture, 0.)
    x = np.array([0., .3, .6, .1, .6, 1., 1.2, .3, 0.]) * failure
    history = np.maximum.accumulate(x)
    t = np.array([strength if high == 0 else strength * (1 - high / failure) * r / high
                  if high < failure else 0. for r, high in zip(x, history)])
    q = np.column_stack([x + compliance * t, np.zeros((len(x), 2))])
    _, result = response(q, history, parameters)
    gap, traction, maximum, storage, fracture = [a.numpy() for a in result]
    np.testing.assert_allclose(gap[:, 0], x, atol=failure * 1e-14)
    work = .5 * (t[1:] + t[:-1]) * np.diff(x)
    np.testing.assert_allclose(work, np.diff(storage) + np.diff(fracture), atol=gc * 1e-14)
    np.testing.assert_allclose(work.sum(), gc, rtol=1e-14)
    np.testing.assert_allclose(maximum, history, atol=failure * 1e-14)


def test_kkt_rotation_and_natural_gradients():
    rng = np.random.default_rng(15071)
    count = 64
    strength = np.full(count, 5e4)
    gc = rng.choice([7., 70., 700., 1400.], count)
    failure = 2 * gc / strength
    compliance = rng.uniform(.02, .7, count) * failure / strength
    parameters = np.column_stack([strength, gc, compliance])
    q = rng.normal(size=(count, 3)) * failure[:, None] * .7
    normal = rng.normal(size=(count, 3))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    previous = rng.uniform(.05, 1.3, count) * failure
    previous[::3] = 0.
    inputs, _ = response(q, previous, parameters, normal)
    with wp.Tape() as tape:
        outputs = evaluate(inputs)
    gap, traction, maximum, storage, fracture = [a.numpy() for a in outputs]
    np.testing.assert_allclose(q - gap, compliance[:, None] * traction, atol=1e-14)
    assert np.min(np.sum(gap * normal, axis=1)) > -1e-14
    assert np.all(maximum >= previous) and np.all(storage >= 0.) and np.all(fracture >= 0.)
    scales = [failure[:, None], strength[:, None], failure, gc, gc]
    seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(outputs, scales)]
    tape.backward(grads={a: array(s, i < 2) for i, (a, s) in enumerate(zip(outputs, seeds))})
    for index, source in enumerate(inputs):
        base = source.numpy()
        direction = rng.normal(size=base.shape)
        if index == 0:
            direction *= failure[:, None]
        elif index in (2, 3):
            direction *= base
        ad = float(np.sum(source.grad.numpy() * direction))
        for epsilon in [1e-6, 1e-7]:
            trials = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                trials.append([a.numpy() for a in evaluate(inputs)])
            fd = sum(np.sum((b - a) * s) for a, b, s in zip(*trials, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=1e-6, atol=1e-8)
        source.assign(base)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    _, rotated = response(q @ rotation.T, previous, parameters, normal @ rotation.T)
    for i, (a, b, scale) in enumerate(zip(outputs, rotated, scales)):
        expected = a.numpy() @ rotation.T if i < 2 else a.numpy()
        np.testing.assert_allclose((b.numpy() - expected) / scale, 0., atol=1e-13)


def test_activation_traction_is_continuous_with_correct_one_sided_tangents():
    strength, gc = 5e4, 70.
    failure = 2 * gc / strength
    compliance = .3 * failure / strength
    parameters = ExtrinsicCohesiveParameters(strength, gc).proximal_parameters(compliance)
    threshold = compliance * strength
    _, base = response([[threshold, 0., 0.]], 0., parameters)
    for epsilon in [1e-8, 1e-9]:
        _, minus = response([[threshold - epsilon, 0., 0.]], 0., parameters)
        _, plus = response([[threshold + epsilon, 0., 0.]], 0., parameters)
        np.testing.assert_allclose((base[0].numpy() - minus[0].numpy()) / epsilon, 0., atol=1e-12)
        np.testing.assert_allclose((plus[0].numpy() - base[0].numpy())[0, 0] / epsilon, 1 / .7, rtol=1e-8)
        np.testing.assert_allclose((base[1].numpy() - minus[1].numpy())[0, 0] / epsilon, 1 / compliance, rtol=1e-8)
        np.testing.assert_allclose((plus[1].numpy() - base[1].numpy())[0, 0] / epsilon,
                                   -strength / failure / .7, rtol=1e-8)


@pytest.mark.parametrize('history_fraction', [.25, 1., 1.2])
@pytest.mark.parametrize('normal_gap', [-1e-4, 0.])
def test_closed_damaged_interface_retains_unloading_tangent(history_fraction, normal_gap):
    strength, gc = 5e4, 70.
    failure = 2 * gc / strength
    compliance = .1 * failure / strength
    parameters = ExtrinsicCohesiveParameters(strength, gc).proximal_parameters(compliance)
    history = history_fraction * failure
    stiffness = strength * (1 - history / failure) / history if history < failure else 0.
    gain = 1 / (1 + compliance * stiffness)
    inputs, _ = response([[normal_gap, 0., 0.]], history, parameters)
    for output_index, expected in [(0, gain), (1, stiffness * gain)]:
        with wp.Tape() as tape:
            outputs = evaluate(inputs)
        tape.backward(grads={outputs[output_index]: array([[0., 1., 0.]], True)})
        ad = inputs[0].grad.numpy()[0, 1]
        np.testing.assert_allclose(ad, expected, rtol=1e-12, atol=1e-12)
        tape.zero()
        for epsilon in [1e-8, 1e-9]:
            _, minus = response([[normal_gap, -epsilon, 0.]], history, parameters)
            _, plus = response([[normal_gap, epsilon, 0.]], history, parameters)
            fd = (plus[output_index].numpy()[0, 1] - minus[output_index].numpy()[0, 1]) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=1e-12, atol=1e-12)


def test_virgin_loaded_interface_history_has_right_sided_derivative():
    strength, gc = 5e4, 70.
    failure = 2 * gc / strength
    compliance = .1 * failure / strength
    parameters = ExtrinsicCohesiveParameters(strength, gc).proximal_parameters(compliance)
    q = compliance * strength * np.array([[-.3, .2, -.1]])
    positive = q.copy()
    positive[:, 0] = 0.
    expected = [positive / (compliance * strength), -positive / (compliance**2 * strength),
                np.ones(1), np.array([np.sum(positive**2) / (2 * compliance**2 * strength)]),
                np.array([.5 * strength])]
    inputs, _ = response(q, 0., parameters)
    for index in range(5):
        seed = np.array([[.3, .7, -.2]]) if index < 2 else np.ones(1)
        with wp.Tape() as tape:
            outputs = evaluate(inputs)
        base = outputs[index].numpy()
        tape.backward(grads={outputs[index]: array(seed, index < 2)})
        ad = inputs[2].grad.numpy()[0]
        np.testing.assert_allclose(ad, np.sum(expected[index] * seed), rtol=1e-12, atol=1e-12)
        tape.zero()
        for epsilon in [1e-7, 1e-8]:
            _, shifted = response(q, failure * epsilon, parameters)
            fd = np.sum((shifted[index].numpy() - base) * seed) / (failure * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-6, atol=1e-8)
