"""Independent normal contact, adhesive force selection and joint pullbacks."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_contact import MappedContacts, solve
from blademaster_warp.cohesive_geometry import InterfaceGeometry
from blademaster_warp.differentiable_cohesive import _evaluate


def array(value, dtype=wp.float64, grad=True):
    return wp.array(np.asarray(value), dtype=dtype, device="cuda:0", requires_grad=grad)


def fixture(gc=70.0, fraction=0.4, tangent=0.0002, gap=-0.0003, areas=None, strengths=None):
    areas = np.array([1e-4]) if areas is None else np.asarray(areas)
    count = len(areas)
    strengths = np.full(count, 5e4) if strengths is None else np.asarray(strengths)
    values = [
        np.tile([gap, tangent, 0], (count, 1)),
        np.tile(np.eye(3), (count, 1, 1)),
        np.tile([1, 0, 0], (count, 1)),
        fraction * 2 * gc / strengths,
        np.column_stack([strengths, np.full(count, gc), np.zeros(count)]),
        np.ones((count, 1)),
        np.zeros((count, 1, 3)),
        areas,
        np.array([0.01]),
        np.array([0.001]),
    ]
    kinds = [
        wp.vec3d,
        wp.mat33d,
        wp.vec3d,
        wp.float64,
        wp.vec3d,
        wp.float64,
        wp.vec3d,
        wp.float64,
        wp.float64,
        wp.float64,
    ]
    inputs = [array(v, kind) for v, kind in zip(values, kinds)]
    geometry = InterfaceGeometry(inputs[0], inputs[1])
    contacts = MappedContacts.from_rows(
        [[1.0]],
        [
            dict(
                point=[-0.0003, 0, 0],
                velocity=[0, 0, 0],
                history=[0, 0, 0],
                normal=[1, 0, 0],
                parameters=[1000, 500, 0, 0, 0],
            )
        ],
        "cuda:0",
    )
    return inputs, geometry, contacts


def evaluate(inputs, geometry, contacts):
    root, u, selected, force, history = solve(inputs, geometry, None, contacts, constrain_normal=False)
    raw, _, _ = _evaluate(
        inputs,
        u,
        selected,
        root.compliance,
        integrated_force=True,
        remainders=(geometry.gap_remainder, geometry.frame_remainder, root.displacement.low),
        constrain_normal=False,
    )
    return [u, selected, force.high, history.high, raw[0], raw[2], raw[3], raw[4], raw[6]]


@pytest.mark.parametrize("gc", [7.0, 70.0, 700.0, 1400.0])
@pytest.mark.parametrize("fraction", [0.0, 0.4, 1.2])
@pytest.mark.parametrize("tangent", [0.0, 0.0002])
def test_independent_normal_force_is_counted_once(gc, fraction, tangent):
    inputs, geometry, contacts = fixture(gc, fraction, tangent)
    result = evaluate(inputs, geometry, contacts)
    u, adhesive, contact = [a.numpy() for a in result[:3]]
    expected = 0.3 / 1.1
    np.testing.assert_allclose(adhesive[0, 0], 0, atol=1e-24)
    np.testing.assert_allclose(contact[0, 0], expected, rtol=1e-13)
    np.testing.assert_allclose(u[0, 0], expected / 1e4, rtol=1e-13)


@pytest.mark.parametrize("gap,fraction", [(-0.0003, 0.0), (-0.0003, 0.4), (0.001, 0.0), (-0.0003, 1.2)])
def test_separated_joint_natural_gradients(gap, fraction):
    inputs, geometry, contacts = fixture(fraction=fraction, gap=gap)
    inputs[6].assign([[[0.1, -0.2, 0.3]]])
    rng = np.random.default_rng(402)
    with wp.Tape() as tape:
        result = evaluate(inputs, geometry, contacts)
    scales = [1e3, 1, 1, 1e3, 1e3, 1e3, 0.01, 0.01, 1e3]
    seeds = [rng.normal(size=a.numpy().shape) * s for a, s in zip(result, scales)]
    tape.backward(grads={a: array(s, a.dtype, False) for a, s in zip(result, seeds)})
    groups = [
        (inputs, [1e-4, 0.1, 0.1, 1e-4, [1000, 7, 0], 0.1, 0.1, 1e-5, 0.001, 1e-4]),
        (
            [
                contacts.mapping.high,
                contacts.point.high,
                contacts.velocity.high,
                contacts.history.high,
                contacts.parameters,
            ],
            [0.1, 1e-4, 0.01, 1e-5, [100, 50, 0, 0, 0]],
        ),
    ]
    for sources, magnitudes in groups:
        baseline = [a.numpy().copy() for a in sources]
        directions = [rng.normal(size=v.shape) * scale for v, scale in zip(baseline, magnitudes)]
        if sources is inputs and fraction == 0:
            directions[3] *= 0
        ad = sum(np.sum(a.grad.numpy() * d) for a, d in zip(sources, directions))
        for epsilon in [1e-4, 1e-5]:
            values = []
            for sign in [-1, 1]:
                for a, base, direction in zip(sources, baseline, directions):
                    a.assign(base + sign * epsilon * direction)
                values.append([a.numpy() for a in evaluate(inputs, geometry, contacts)])
            fd = sum(np.sum((b - a) * s) for a, b, s in zip(*values, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=2e-8)
        for a, base in zip(sources, baseline):
            a.assign(base)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("strengths", [[5e4, 5e4], [1e4, 5e4]])
def test_redundant_adhesive_force_selection_and_area_pullback(compressed, strengths):
    direction = np.array([0, 1, 0] if compressed else [1, 0, 0])
    inputs, geometry, contacts = fixture(
        fraction=0,
        tangent=0.0002 if compressed else 0,
        gap=-0.0003 if compressed else 0.0002,
        areas=[3e-5, 7e-5],
        strengths=strengths,
    )
    contacts = MappedContacts.from_rows(np.zeros((0, 1)), [], "cuda:0")
    with wp.Tape() as tape:
        result = evaluate(inputs, geometry, contacts)
    expected = [0.6, 1.4] if strengths[0] == strengths[1] else [0.3, 1.7]
    np.testing.assert_allclose(result[1].numpy() @ direction, expected, rtol=2e-11, atol=1e-12)
    seed = np.array([2, -1])[:, None] * direction
    tape.backward(grads={result[1]: array(seed, wp.vec3d, False)})
    source = inputs[7]
    baseline = source.numpy().copy()
    delta = np.array([1e-6, -1e-6])
    ad = np.sum(source.grad.numpy() * delta)
    for epsilon in [1e-4, 1e-5]:
        values = []
        for sign in [-1, 1]:
            source.assign(baseline + sign * epsilon * delta)
            values.append(evaluate(inputs, geometry, contacts)[1].numpy())
        fd = np.sum((values[1] - values[0]) * seed) / (2 * epsilon)
        np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=2e-8)
    source.assign(baseline)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("angle", [0.0, 0.3, 1.2])
def test_strength_limited_redundant_reactions_rotate_and_differentiate(compressed, angle):
    from scipy.spatial.transform import Rotation

    axis = np.array([1.0, 2.0, -1.0]) / np.sqrt(6.0)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = Rotation.from_rotvec(angle * axis).as_matrix()
    inputs, geometry, _ = fixture(
        fraction=0,
        tangent=0.0002 if compressed else 0,
        gap=-0.0003 if compressed else 0.0002,
        areas=[3e-5, 7e-5],
        strengths=[1e4, 5e4],
    )
    base_gap = inputs[0].numpy().copy()
    inputs[0].assign(base_gap @ rotation.T)
    inputs[1].assign(np.tile(rotation, (2, 1, 1)))
    contacts = MappedContacts.from_rows(np.zeros((0, 1)), [], "cuda:0")
    direction = rotation[:, 1 if compressed else 0]
    with wp.Tape() as tape:
        selected = evaluate(inputs, geometry, contacts)[1]
    expected = np.array([0.3, 1.7])[:, None] * direction
    np.testing.assert_allclose(selected.numpy(), expected, rtol=2e-11, atol=2e-12)
    seed = np.random.default_rng(4092).normal(size=(2, 3))
    tape.backward(grads={selected: array(seed, wp.vec3d, False)})
    derivative_rotation = cross @ rotation
    rotation_ad = np.sum(inputs[0].grad.numpy() * (base_gap @ derivative_rotation.T))
    rotation_ad += np.sum(inputs[1].grad.numpy() * derivative_rotation)
    np.testing.assert_allclose(rotation_ad, np.sum(seed * (expected @ cross.T)), rtol=2e-5, atol=2e-8)
    for epsilon in [1e-4, 1e-5]:
        values = []
        for sign in [-1, 1]:
            moved = Rotation.from_rotvec((angle + sign * epsilon) * axis).as_matrix()
            inputs[0].assign(base_gap @ moved.T)
            inputs[1].assign(np.tile(moved, (2, 1, 1)))
            values.append(evaluate(inputs, geometry, contacts)[1].numpy())
        fd = np.sum(seed * (values[1] - values[0])) / (2 * epsilon)
        np.testing.assert_allclose(rotation_ad, fd, rtol=2e-5, atol=2e-8)
    inputs[0].assign(base_gap @ rotation.T)
    inputs[1].assign(np.tile(rotation, (2, 1, 1)))
    sources = [inputs[7], inputs[4], inputs[9]]
    deltas = [np.array([1e-6, -1e-6]), np.array([[1000, 7, 0], [-1000, -7, 0]]), np.array([1e-4])]
    baseline = [a.numpy().copy() for a in sources]
    ad = sum(np.sum(a.grad.numpy() * d) for a, d in zip(sources, deltas))
    weak_change = deltas[0][0] * 1e4 + 3e-5 * deltas[1][0, 0]
    total_change = -2 * 2.0 / baseline[2][0] * deltas[2][0]
    analytic = np.array([weak_change, total_change - weak_change])[:, None] * direction
    np.testing.assert_allclose(ad, np.sum(seed * analytic), rtol=2e-5, atol=2e-8)
    for epsilon in [1e-4, 1e-5]:
        values = []
        for sign in [-1, 1]:
            for source, value, delta in zip(sources, baseline, deltas):
                source.assign(value + sign * epsilon * delta)
            values.append(evaluate(inputs, geometry, contacts)[1].numpy())
        fd = np.sum(seed * (values[1] - values[0])) / (2 * epsilon)
        np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=2e-8)
    for source, value in zip(sources, baseline):
        source.assign(value)


@pytest.mark.parametrize("exponent", [60, 80])
def test_distinct_sub_ulp_opening_does_not_inherit_a_neighbor_closure(exponent):
    inputs, geometry, _ = fixture(
        fraction=0, tangent=1e-4, gap=2e-4, areas=[3e-5, 7e-5], strengths=[1e4, 5e4]
    )
    difference = 2.0**-exponent
    geometry.gap_remainder[0, 1] = difference
    contacts = MappedContacts.from_rows(np.zeros((0, 1)), [], "cuda:0")
    root, _, selected, *_ = solve(inputs, geometry, None, contacts, constrain_normal=False)
    weak = 0.3 * (1 - difference / (2 * 70.0 / 1e4))
    np.testing.assert_allclose(selected.numpy(), [[0, weak, 0], [2, 1 - weak, 0]], rtol=2e-10, atol=2e-11)
    assert not root.branches.closed[0]
    assert root.branches.closed[1]


def test_compressed_reaction_with_only_normal_motion_has_zero_adhesion():
    from blademaster_warp._cohesive_expansion import E
    from blademaster_warp._cohesive_projection import select_constrained

    mapping = E(np.array([[0.0, 0.0, 1.0]]))
    original = E(np.zeros(3))
    initial = E(np.array([0.0, 0.0, 1.0]))
    normal = E(np.array([0.0, 0.0, 1.0]))
    entries = [(0, slice(0, 3), slice(0, 3), True, normal, E(1.0), E(1.0), 2)]
    reaction, _ = select_constrained(mapping, original, initial, entries, E(1.0))
    np.testing.assert_array_equal(reaction.numpy(), np.zeros(3))
    np.testing.assert_array_equal((mapping @ reaction).numpy(), (mapping @ original).numpy())
