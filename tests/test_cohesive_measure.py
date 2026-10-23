"""Area-weighted proximal forces, empty supports and one-sided area derivatives."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive import cohesive_measure_proximal_response
from blademaster_warp.precision import launch_double
from test_cohesive_proximal import array, response


def inputs(q, history, area, parameters=(5e4, 70.0, 0.001), normal=(1.0, 0.0, 0.0)):
    q = np.asarray(q).reshape(-1, 3)
    count = len(q)
    values = [
        q,
        np.broadcast_to(normal, (count, 3)),
        np.broadcast_to(history, (count,)),
        np.broadcast_to(parameters, (count, 3)),
        np.broadcast_to(area, (count,)),
    ]
    return [array(value, index in [0, 1, 3], True) for index, value in enumerate(values)]


def evaluate(sources):
    output = [
        wp.zeros(len(sources[0]), dtype=wp.vec3d if i < 2 else wp.float64, requires_grad=True)
        for i in range(5)
    ]
    launch_double(cohesive_measure_proximal_response, len(sources[0]), inputs=sources, outputs=output)
    return output


def test_positive_measure_matches_pressure_form_and_stationarity():
    rng = np.random.default_rng(71)
    count = 64
    area = rng.uniform(1e-9, 1e-5, count)
    gc = rng.choice([7.0, 70.0, 700.0, 1400.0], count)
    failure = 2 * gc / 5e4
    compliance = rng.uniform(0.02, 0.7, count) * failure / (area * 5e4)
    parameters = np.column_stack([np.full(count, 5e4), gc, compliance])
    q = rng.normal(size=(count, 3)) * failure[:, None] * 0.7
    normal = rng.normal(size=(count, 3))
    previous = rng.uniform(0.05, 1.3, count) * failure
    previous[::3] = 0.0
    source = inputs(q, previous, area, parameters, normal)
    actual = [a.numpy() for a in evaluate(source)]
    density_parameters = parameters.copy()
    density_parameters[:, 2] *= area
    expected = [a.numpy() for a in response(q, previous, density_parameters, normal)[1]]
    expected[1] *= area[:, None]
    scales = [failure[:, None], (area * 5e4)[:, None], failure, gc, gc]
    for a, b, scale in zip(actual, expected, scales):
        np.testing.assert_allclose((a - b) / scale, 0.0, atol=3e-14)
    np.testing.assert_allclose(q - actual[0], compliance[:, None] * actual[1], atol=1e-14)


@pytest.mark.parametrize("history", [0.0, 2e-4, 0.003])
@pytest.mark.parametrize("q", [[1e-4, 4e-5, -2e-5], [-1e-4, 4e-5, -2e-5], [0.0, 0.0, 0.0]])
def test_zero_measure_preserves_free_motion_and_has_zero_force(history, q):
    source = inputs(q, history, 0.0)
    with wp.Tape() as tape:
        result = evaluate(source)
    np.testing.assert_allclose(result[0].numpy(), [q], atol=1e-20)
    np.testing.assert_array_equal(result[1].numpy(), 0.0)
    assert all(np.isfinite(a.numpy()).all() for a in result)
    seed = np.array([[0.2, -0.5, 0.7]])
    tape.backward(grads={result[0]: array(seed, True)})
    np.testing.assert_allclose(source[0].grad.numpy(), seed, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(source[1].grad.numpy(), 0.0, atol=1e-16)


@pytest.mark.parametrize("history", [0.0, 2e-4, 0.003])
def test_zero_area_opening_keeps_analytic_force_and_gap_area_tangents(history):
    q = np.array([1e-4, 4e-5, -2e-5])
    source = inputs(q, history, 0.0)
    radius = np.linalg.norm(q)
    high = max(history, radius)
    stiffness = 5e4 * (1 - high / 0.0028) / high if high < 0.0028 else 0.0
    pressure = stiffness * q
    for target, expected in [(0, -0.001 * pressure), (1, pressure)]:
        with wp.Tape() as tape:
            result = evaluate(source)
        for axis in range(3):
            seed = np.eye(3)[axis : axis + 1]
            tape.backward(grads={result[target]: array(seed, True)})
            np.testing.assert_allclose(source[4].grad.numpy()[0], expected[axis], rtol=2e-13, atol=1e-10)
            tape.zero()
        for area in [4e-11, 4e-12]:
            source[4].fill_(area)
            value = evaluate(source)[target].numpy()[0]
            fd = (value - result[target].numpy()[0]) / area
            np.testing.assert_allclose(fd, expected, rtol=2e-5, atol=1e-8)
        source[4].zero_()


def test_compression_contact_exists_only_on_positive_measure():
    q = [-1e-4, 0.0, 0.0]
    for area in [4e-6, 4e-10, 4e-14]:
        result = evaluate(inputs(q, 0.0, area))
        np.testing.assert_array_equal(result[0].numpy(), 0.0)
        np.testing.assert_allclose(result[1].numpy(), [[-0.1, 0.0, 0.0]], rtol=1e-14)
    result = evaluate(inputs(q, 0.0, 0.0))
    np.testing.assert_array_equal(result[0].numpy(), [q])
    np.testing.assert_array_equal(result[1].numpy(), 0.0)


@wp.kernel
def integrated_energy(
    area: wp.array(dtype=wp.float64),
    storage: wp.array(dtype=wp.float64),
    fracture: wp.array(dtype=wp.float64),
    energy: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    energy[i] = area[i] * (storage[i] + fracture[i])


@pytest.mark.parametrize("history", [0.0, 2e-4, 0.003])
def test_zero_measure_keeps_the_one_sided_energy_shape_derivative(history):
    q = np.array([1e-4, 4e-5, -2e-5])
    source = inputs(q, history, 0.0)
    with wp.Tape() as tape:
        result = evaluate(source)
        energy = wp.zeros(1, dtype=wp.float64, requires_grad=True)
        wp.launch(integrated_energy, 1, inputs=[source[4], *result[3:]], outputs=[energy])
    tape.backward(grads={energy: array([1.0])})
    radius = np.linalg.norm(q)
    high = max(history, radius)
    stiffness = 5e4 * (1 - high / 0.0028) / high if high < 0.0028 else 0.0
    expected = 0.5 * stiffness * radius**2 + 0.5 * 5e4 * min(high, 0.0028)
    np.testing.assert_array_equal(energy.numpy(), 0.0)
    np.testing.assert_allclose(source[4].grad.numpy()[0], expected, rtol=1e-14)
    for index in [0, 1, 2, 3]:
        np.testing.assert_array_equal(source[index].grad.numpy(), 0.0)
    for area in [4e-11, 4e-12]:
        source[4].fill_(area)
        value = evaluate(source)
        fd = value[3].numpy()[0] + value[4].numpy()[0]
        np.testing.assert_allclose(fd, expected, rtol=2e-5)


def test_positive_measure_all_input_pullbacks_match_finite_differences():
    rng = np.random.default_rng(514)
    count = 32
    area = rng.uniform(1e-7, 1e-5, count)
    gc = rng.choice([7.0, 70.0, 700.0, 1400.0], count)
    failure = 2 * gc / 5e4
    compliance = rng.uniform(0.02, 0.7, count) * failure / (area * 5e4)
    parameters = np.column_stack([np.full(count, 5e4), gc, compliance])
    q = rng.normal(size=(count, 3)) * failure[:, None] * 0.7
    history = rng.uniform(0.05, 1.3, count) * failure
    history[::3] = 0.0
    source = inputs(q, history, area, parameters, rng.normal(size=(count, 3)))
    with wp.Tape() as tape:
        result = evaluate(source)
    scales = [failure[:, None], (area * 5e4)[:, None], failure, gc, gc]
    seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(result, scales)]
    tape.backward(grads={a: array(s, i < 2) for i, (a, s) in enumerate(zip(result, seeds))})
    for index, parameter in enumerate(source):
        base = parameter.numpy().copy()
        direction = rng.normal(size=base.shape)
        if index == 0:
            direction *= failure[:, None]
        elif index in [2, 3, 4]:
            direction *= base
        ad = np.sum(parameter.grad.numpy() * direction)
        for epsilon in [1e-6, 1e-7]:
            trials = []
            for sign in [-1, 1]:
                parameter.assign(base + sign * epsilon * direction)
                trials.append([a.numpy() for a in evaluate(source)])
            fd = sum(np.sum((b - a) * s) for a, b, s in zip(*trials, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=1e-6, atol=1e-8)
        parameter.assign(base)
