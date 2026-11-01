"""Contact return, implicit gradients, and uncut-material contact."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp._cohesive_expansion import E
from blademaster_warp._contact_law import response
from blademaster_warp.cohesive_contact import MappedContacts, solve
from blademaster_warp.cohesive_geometry import InterfaceGeometry
from test_friction_energy import run as warp_return


@pytest.mark.parametrize("speed", [0.001, 0.4])
@pytest.mark.parametrize("damping", [0.0, 0.2])
def test_compensated_contact_matches_original_return_and_all_input_differences(speed, damping):
    rng = np.random.default_rng(2741)
    n = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    values = np.r_[
        -2e-4 * n,
        [speed, -0.2 * speed, 0.3 * speed],
        [1e-5, -2e-5, 3e-6],
        n,
        [1000.0, 500.0, damping, 0.3, 0.2],
        2e-4,
    ]

    def evaluate(x):
        return response(x[:3], x[3:6], x[6:9], x[9:12], x[12:17], x[17])

    base = evaluate(values)
    h, f = warp_return(
        values[6:9], values[3:6].reshape(1, 3), n.reshape(1, 3), 0.2, 500.0, damping, 2e-4, 0.3, 0.2, 1.0
    )
    np.testing.assert_allclose(base.history.high, h[0], rtol=2e-12, atol=1e-18)
    np.testing.assert_allclose(base.force.high, f[0] + 0.2 * n, rtol=2e-12, atol=1e-16)
    direction = rng.normal(size=18) * np.maximum(abs(values), 1e-5)
    for epsilon in [1e-5, 1e-6]:
        plus, minus = evaluate(values + epsilon * direction), evaluate(values - epsilon * direction)
        for name in ["force", "history"]:
            exact = getattr(base, name + "_jacobian") @ E(direction)
            fd = (getattr(plus, name).high - getattr(minus, name).high) / (2 * epsilon)
            np.testing.assert_allclose(exact.high, fd, rtol=2e-7, atol=1e-10)


def test_contact_switch_uses_history_remainder():
    modes = []
    for low in [-(2.0**-60), 2.0**-60]:
        local = response(
            [0, 0, -0.5], [0, 0, 0], E([0.25, 0, 0], [low, 0, 0]), [0, 0, 1], [1, 1, 0, 0.5, 0.25], 0.25
        )
        modes.append(local.mode)
        np.testing.assert_allclose(local.force.high[0], -0.25 if low < 0 else -0.125, rtol=1e-15)
    assert modes == ["stick", "slide"]


def fixture(interface, speed):
    def array(value, kind=wp.float64, grad=True):
        return wp.array(np.asarray(value), dtype=kind, device="cuda:0", requires_grad=grad)

    count = int(interface)
    n = 2
    geometry = InterfaceGeometry(
        array(np.tile([2e-4, 0, 0], (count, 1)), wp.vec3d),
        array(np.tile(np.eye(3), (count, 1, 1)), wp.mat33d),
    )
    inputs = [
        geometry.gap,
        geometry.frame,
        array(np.tile([1.0, 0, 0], (count, 1)), wp.vec3d),
        array(np.full(count, 5e-4)),
        array(np.tile([1000.0, 1.0, 0.0], (count, 1)), wp.vec3d),
        array(np.tile([-1.0, 1.0], (count, 1))),
        array(np.zeros((count, n, 3)), wp.vec3d),
        array(np.full(count, 4e-4)),
        array([0.01]),
        array([2e-4]),
    ]
    rows = [
        dict(
            point=[0, 0, -2e-4],
            velocity=[speed, 0.1 * speed, 0],
            history=[1e-5, -1e-5, 0],
            normal=[0, 0, 1],
            parameters=[1000, 500, 0.2, 0.3, 0.2],
        )
    ]
    contacts = MappedContacts.from_rows(np.array([[1.0, 0.0]]), rows, "cuda:0")
    return inputs, geometry, contacts, array


@pytest.mark.parametrize("interface", [False, True])
@pytest.mark.parametrize("speed", [0.001, 0.4])
def test_joint_contact_gradient_with_and_without_cut(interface, speed):
    inputs, geometry, contacts, array = fixture(interface, speed)

    def run():
        root, u, force, contact, history = solve(inputs, geometry, None, contacts)
        return [u, contact.high, history.high]

    with wp.Tape() as tape:
        output = run()
    rng = np.random.default_rng(2742)
    seeds = [rng.normal(size=a.numpy().shape) * s for a, s in zip(output, [1e4, 1, 1e4])]
    tape.backward(grads={a: array(s, a.dtype, False) for a, s in zip(output, seeds)})
    for source, scale in [
        (contacts.arrays[1], 1e-4),
        (contacts.arrays[2], 0.1),
        (contacts.arrays[3], 1e-5),
        (contacts.parameters, np.array([100.0, 50.0, 0.02, 0.01, 0.01])),
        (inputs[8], 0.001),
        (inputs[9], 2e-5),
        (contacts.arrays[0], 0.1),
    ]:
        baseline = source.numpy().copy()
        direction = rng.normal(size=baseline.shape) * scale
        ad = np.sum(source.grad.numpy() * direction)
        for eps in [1e-4, 1e-5]:
            trials = []
            for sign in [-1.0, 1.0]:
                source.assign(baseline + sign * eps * direction)
                trials.append([a.numpy() for a in run()])
            fd = sum(np.sum((b - a) * s) for a, b, s in zip(*trials, seeds)) / (2 * eps)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-8)
        source.assign(baseline)


@pytest.mark.parametrize("interface", [False, True])
def test_empty_contact_set_preserves_interface_solution_and_gradient(interface):
    from blademaster_warp.cohesive_geometry import solve_geometry

    inputs, geometry, _, array = fixture(interface, 0.001)
    contacts = MappedContacts.from_rows(np.empty((0, 2)), [], "cuda:0")
    with wp.Tape() as tape:
        root, u, force, contact, history = solve(inputs, geometry, None, contacts)
    assert contact.high.shape == history.high.shape == (0,)
    if not interface:
        np.testing.assert_array_equal(u.numpy(), 0.0)
        return
    seed = np.array([[0.3, -0.2, 0.4], [-0.1, 0.5, 0.2]])
    tape.backward(grads={u: array(seed, wp.vec3d, False)})
    joint = inputs[0].grad.numpy().copy()
    tape.zero()
    with wp.Tape() as original:
        result, _ = solve_geometry(geometry, *inputs[2:], integrated_force=True)
    original.backward(grads={result.displacement: array(seed, wp.vec3d, False)})
    np.testing.assert_allclose(u.numpy(), result.displacement.numpy(), rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(force.numpy(), result.force.numpy(), rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(joint, inputs[0].grad.numpy(), rtol=1e-11, atol=1e-14)


def test_zero_contact_mapping_retains_its_first_order_motion():
    inputs, geometry, contacts, array = fixture(False, 0.001)
    contacts.mapping.high.zero_()
    with wp.Tape() as tape:
        root, u, _, force, _ = solve(inputs, geometry, None, contacts)
    np.testing.assert_array_equal(u.numpy(), 0.0)
    seed = np.array([[0.3, -0.2, 0.4], [-0.1, 0.5, 0.2]])
    tape.backward(grads={u: array(seed, wp.vec3d, False)})
    expected = inputs[9].numpy()[0] ** 2 / inputs[8].numpy()[0] * (force.high.numpy() @ seed.T)
    np.testing.assert_allclose(contacts.mapping.high.grad.numpy(), expected, rtol=2e-12, atol=1e-16)
