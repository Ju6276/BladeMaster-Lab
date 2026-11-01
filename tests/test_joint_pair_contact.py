"""Pair normal derivatives, relative-map precision and joint contact momentum."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from blademaster_warp._cohesive_expansion import E
from blademaster_warp._pair_contact_law import response
from blademaster_warp.cohesive_contact import solve
from blademaster_warp.cohesive_pair_contact import PairContacts, predict_pair_contacts
from blademaster_warp.compensated_state import CompensatedArray
from test_mapped_contact import fixture


@pytest.mark.parametrize("speed", [0.001, 1.4])
def test_pair_law_all_inputs_and_particle_exchange(speed):
    values = np.r_[
        [0.0008, -0.0006, 0.0025],
        [speed, 0.1 * speed, -0.1],
        [1e-5, -1e-5, 2e-6],
        [1000, 500, 0.2, 0.3, 0.2],
        0.0032,
        2e-4,
    ]

    def evaluate(x):
        return response(dict(history=x[6:9], parameters=x[9:14], diameter=x[14]), x[:3], x[3:6], x[15])

    result = evaluate(values)
    assert result.mode == ("stick" if speed < 1 else "slide")
    exchanged = values.copy()
    exchanged[:9] *= -1
    opposite = evaluate(exchanged)
    np.testing.assert_array_equal(opposite.force.high, -result.force.high)
    np.testing.assert_array_equal(opposite.history.high, -result.history.high)
    rng = np.random.default_rng(2759)
    direction = rng.normal(size=16) * np.maximum(abs(values), 1e-5)
    for epsilon in [1e-4, 1e-5]:
        a, b = evaluate(values - epsilon * direction), evaluate(values + epsilon * direction)
        for name in ["force", "history"]:
            ad = (getattr(result, name + "_jacobian") @ direction).numpy()
            fd = (getattr(b, name).numpy() - getattr(a, name).numpy()) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-8)


def test_pair_opening_retains_sub_ulp_distance():
    for sign in [-1, 1]:
        point = E([1, 0, 0], [sign * 2.0**-80, 0, 0])
        result = response(
            dict(history=[0, 0, 0], parameters=[1, 1, 0, 0.5, 0.2], diameter=1), point, [0, 0, 0], 0.1
        )
        np.testing.assert_allclose(result.gap.numpy(), sign * 2.0**-80, rtol=1e-15, atol=0)
        np.testing.assert_allclose(
            result.force.numpy(), [2.0**-80 if sign < 0 else 0, 0, 0], rtol=1e-15, atol=0
        )


def test_two_particle_normal_contact_matches_reduced_mass_solution():
    inputs, geometry, _, array = fixture(False, 0)
    row = dict(
        point=[0, 0, 0.0028],
        velocity=[0, 0, 0],
        history=[0, 0, 0],
        parameters=[1000, 500, 0, 0, 0],
        diameter=0.003,
    )
    pairs = PairContacts.from_rows([[1, -1]], [row], "cuda:0")
    root, u, _, force, history = solve(inputs, geometry, None, pairs)
    dt, mass, kn = inputs[9].numpy()[0], inputs[8].numpy()[0], 1000.0
    gap = (0.0028 - 0.003) / (1 + 2 * dt * dt * kn / mass)
    exact_force = -kn * gap
    exact_displacement = dt * dt / mass * exact_force * np.array([[0, 0, 1], [0, 0, -1]])
    np.testing.assert_allclose(u.numpy(), exact_displacement, rtol=2e-13, atol=1e-18)
    np.testing.assert_allclose(force.high.numpy(), [[0, 0, exact_force]], rtol=2e-13, atol=1e-16)
    np.testing.assert_array_equal(np.sum(u.numpy(), axis=0), 0)
    np.testing.assert_array_equal(history.high.numpy(), 0)


def test_pair_prediction_retains_position_map_remainder_and_transpose():
    _, _, _, array = fixture(False, 0)
    tiny = 2.0**-80
    point_map = CompensatedArray(array(np.eye(2)), np.diag([tiny, -tiny]))
    moment_map = array(np.eye(2))
    position = array([[1, 0, 0], [1, 0, 0]], wp.vec3d)
    moments = array(np.zeros((2, 3)), wp.vec3d)
    history = array(np.zeros((1, 3)), wp.vec3d)
    with wp.Tape() as tape:
        result = predict_pair_contacts(
            position,
            moments,
            array([0.1]),
            history,
            point_map,
            moment_map,
            np.array([[0, 1]]),
            array([[1000, 500, 0.2, 0.3, 0.2]]),
            array([0.003]),
        )
    np.testing.assert_array_equal(result.point.high.numpy(), [[2 * tiny, 0, 0]])
    tape.backward(grads={result.point.high: array([[1, 0, 0]], wp.vec3d, False)})
    np.testing.assert_array_equal(point_map.high.grad.numpy(), [[1, 1], [-1, -1]])
    np.testing.assert_array_equal(position.grad.numpy(), [[1, 0, 0], [-1, 0, 0]])


@pytest.mark.parametrize("interface", [False, True])
@pytest.mark.parametrize("speed", [0.001, 1.4])
def test_joint_pair_gradient_all_inputs(interface, speed):
    inputs, geometry, _, array = fixture(interface, speed)
    count = int(interface)
    inputs[5] = array(np.tile([-1, 1, 0], (count, 1)))
    inputs[6] = array(np.zeros((count, 3, 3)), wp.vec3d)
    rows = [
        dict(
            point=p,
            velocity=[speed, 0.1 * speed, -0.1],
            history=[1e-5, -1e-5, 2e-6],
            parameters=[1000, 500, 0.2, 0.3, 0.2],
            diameter=0.0032,
        )
        for p in [[0.0008, -0.0006, 0.0025], [-0.0004, 0.0024, 0.0003]]
    ]
    pairs = PairContacts.from_rows([[1, -1, 0], [0, 1, -1]], rows, "cuda:0")
    pairs = replace(
        pairs, parameters=array(pairs.parameters.numpy()[:1]), diameter=array(pairs.diameter.numpy()[:1])
    )

    def run():
        root, u, force, contact, history = solve(inputs, geometry, None, pairs)
        return [u, force, contact.high, history.high]

    with wp.Tape() as tape:
        outputs = run()
    rng = np.random.default_rng(2760)
    seeds = [rng.normal(size=a.numpy().shape) * scale for a, scale in zip(outputs, [1e4, 1, 1, 1e4])]
    tape.backward(grads={a: array(s, a.dtype, False) for a, s in zip(outputs, seeds) if a.size})
    for source, scale in [
        (pairs.mapping.high, 0.1),
        (pairs.point.high, 1e-4),
        (pairs.velocity.high, 0.1),
        (pairs.history.high, 1e-5),
        (pairs.parameters, [100, 50, 0.02, 0.01, 0.01]),
        (pairs.diameter, 3e-4),
        (inputs[8], 0.001),
        (inputs[9], 2e-5),
    ]:
        baseline = source.numpy().copy()
        direction = rng.normal(size=baseline.shape) * scale
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            values = []
            for sign in [-1, 1]:
                source.assign(baseline + sign * epsilon * direction)
                values.append([a.numpy() for a in run()])
            fd = sum(np.sum((b - a) * s) for a, b, s in zip(*values, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-8)
        source.assign(baseline)


def test_empty_pair_set_has_no_contact_force_or_history():
    inputs, geometry, _, _ = fixture(False, 0)
    pairs = PairContacts.from_rows(np.empty((0, 2)), [], "cuda:0")
    _, u, _, force, history = solve(inputs, geometry, None, pairs)
    np.testing.assert_array_equal(u.numpy(), 0)
    assert force.high.shape == history.high.shape == (0,)
