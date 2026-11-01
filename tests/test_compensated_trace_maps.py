"""Composition cancellation, full map pullbacks and interface kinematics."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
import warp as wp

from blademaster_warp._compensated import Expansion as E
from blademaster_warp.cohesive_geometry import InterfaceGeometry
from blademaster_warp.compensated_state import CompensatedArray
from blademaster_warp.reference_affine import compose_maps
from blademaster_warp.cohesive_geometry import solve_geometry
from test_differentiable_cohesive import arrays, closed_pair


def array(value, kind=wp.float64):
    return wp.array(value, dtype=kind, requires_grad=True)


def test_composition_retains_cancelled_product_and_input_derivatives():
    a = 2.0**60
    separation = array([[a, 1.0, -a]])
    frame = array([[[a, 0, 0], [1.0, 2.0, 3.0], [-a, 0, 0]]], wp.vec3d)
    transfer = array([[1.0], [1.0], [1.0]])
    with wp.Tape() as tape:
        B, G = compose_maps(separation, frame, transfer)
    np.testing.assert_array_equal(B.high.numpy(), [[1.0]])
    np.testing.assert_array_equal(G.high.numpy(), [[[1.0, 2.0, 3.0]]])
    tape.backward(grads={B.high: array([[2.0]]), G.high: array([[[3.0, 5.0, 7.0]]], wp.vec3d)})
    np.testing.assert_array_equal(separation.grad.numpy(), [[2.0, 2.0, 2.0]])
    np.testing.assert_array_equal(frame.grad.numpy(), np.tile([3.0, 5.0, 7.0], (1, 3, 1)))
    np.testing.assert_array_equal(transfer.grad.numpy(), [[5 * a], [36.0], [-5 * a]])


def test_composed_map_total_value_and_geometry_pullback():
    rng = np.random.default_rng(371)
    b = rng.normal(size=(3, 4))
    g = rng.normal(size=(3, 4, 3))
    p = rng.normal(size=(4, 5))
    inputs = [array(b), array(g, wp.vec3d), array(p)]
    velocity = array(rng.normal(size=(5, 3)), wp.vec3d)
    duration = array([0.003])
    geometry = InterfaceGeometry(
        array(np.zeros((3, 3)), wp.vec3d), array(np.tile(np.eye(3), (3, 1, 1)), wp.mat33d)
    )
    with wp.Tape() as tape:
        B, G = compose_maps(*inputs)
        advanced = geometry.advance(B, G, velocity, duration)
    with localcontext() as context:
        context.prec = 90
        for i in range(3):
            for j in range(5):
                expected = sum(
                    Decimal.from_float(float(b[i, k])) * Decimal.from_float(float(p[k, j])) for k in range(4)
                )
                actual = Decimal.from_float(float(B.high.numpy()[i, j])) + Decimal.from_float(
                    float(B.low[i, j])
                )
                assert abs(actual - expected) <= Decimal("1e-29")
    seed = rng.normal(size=(3, 3))
    tape.backward(grads={advanced.gap: array(seed, wp.vec3d)})
    expected_v = duration.numpy()[0] * p.T @ b.T @ seed
    np.testing.assert_allclose(velocity.grad.numpy(), expected_v, rtol=2e-13, atol=1e-16)
    np.testing.assert_allclose(
        inputs[0].grad.numpy(), duration.numpy()[0] * seed @ velocity.numpy().T @ p.T, rtol=2e-13, atol=1e-16
    )
    np.testing.assert_allclose(
        inputs[2].grad.numpy(), duration.numpy()[0] * b.T @ seed @ velocity.numpy().T, rtol=2e-13, atol=1e-16
    )
    np.testing.assert_allclose(
        duration.grad.numpy()[0], np.sum(seed * (b @ p @ velocity.numpy())), rtol=2e-13
    )


def test_empty_trace_composition_has_zero_input_gradient():
    source = array(np.zeros((0, 4)))
    frame = array(np.zeros((0, 4, 3)), wp.vec3d)
    transfer = array(np.ones((4, 5)))
    geometry = InterfaceGeometry(array(np.zeros((0, 3)), wp.vec3d), array(np.zeros((0, 3, 3)), wp.mat33d))
    velocity = array(np.ones((5, 3)), wp.vec3d)
    duration = array([0.01])
    with wp.Tape() as tape:
        B, G = compose_maps(source, frame, transfer)
        moved = geometry.advance(B, G, velocity, duration)
        extended = geometry.extrapolate(moved, 0.5)
    assert B.high.shape == G.high.shape == (0, 5)
    tape.backward(grads={extended.gap: array(np.zeros((0, 3)), wp.vec3d)})
    np.testing.assert_array_equal(velocity.grad.numpy(), 0.0)
    np.testing.assert_array_equal(duration.grad.numpy(), 0.0)
    np.testing.assert_array_equal(transfer.grad.numpy(), 0.0)


def test_interface_root_keeps_map_remainder_in_displacement():
    values = closed_pair(4e-6, [1e-6, 0.0, 0.0])
    values[5][:] = 1.0
    source = arrays(values)
    low = np.array([[2.0**-54, -(2.0**-54)]])
    mapping = CompensatedArray(source[5], low)
    result, _ = solve_geometry(
        InterfaceGeometry(*source[:2]), *source[2:5], mapping, source[6], *source[7:], integrated_force=True
    )
    displacement = result.displacement_state.cpu()
    with localcontext() as context:
        context.prec = 90
        b = [Decimal(1) + Decimal.from_float(value) for value in low[0]]
        q = Decimal.from_float(float(values[0][0, 0]))
        squared = sum(value * value for value in b)
        for i in range(2):
            expected = -q * b[i] / squared
            actual = Decimal.from_float(float(displacement.high[i, 0])) + Decimal.from_float(
                float(displacement.low[i, 0])
            )
            assert abs(actual - expected) < Decimal("1e-33")
            assert abs(Decimal.from_float(float(displacement.high[i, 0])) - expected) > Decimal("1e-25")


def test_composed_map_to_equilibrium_gradient_matches_rebuilt_inputs():
    values = closed_pair(4e-6, [1e-6, 0.0, 0.0])
    source = arrays(values)
    huge = 2.0**60
    q = array([[huge, 1.0, -huge]])
    g = array(np.zeros((1, 3, 3)), wp.vec3d)
    p = array(np.tile([-1.0, 1.0], (3, 1)))

    def run():
        B, G = compose_maps(q, g, p)
        return solve_geometry(
            InterfaceGeometry(*source[:2]), *source[2:5], B, G, *source[7:], integrated_force=True
        )[0]

    with wp.Tape() as tape:
        result = run()
    seed_u = np.array([[0.4, -0.1, 0.2], [-0.6, 0.2, 0.1]])
    seed_t = np.array([[0.2, -0.3, 0.1]])
    tape.backward(grads={result.displacement: array(seed_u, wp.vec3d), result.force: array(seed_t, wp.vec3d)})
    directions = [
        (q, np.array([[0.0, 0.1, 0.0]])),
        (p, np.array([[0.0, 0.0], [0.04, -0.03], [0.0, 0.0]])),
        (source[0], np.array([[1e-7, 2e-8, -1e-8]])),
    ]
    for target, direction in directions:
        base = target.numpy().copy()
        ad = float(np.sum(target.grad.numpy() * direction))
        for epsilon in [1e-4, 1e-5]:
            trials = []
            for sign in [-1.0, 1.0]:
                target.assign(base + sign * epsilon * direction)
                moved = run()
                trials.append(
                    float(np.sum(moved.displacement.numpy() * seed_u) + np.sum(moved.force.numpy() * seed_t))
                )
            fd = (trials[1] - trials[0]) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-10)
        target.assign(base)


@pytest.mark.parametrize("scale", [1.0, 1e-8])
@pytest.mark.parametrize("integrated_force", [False, True])
def test_closed_constraint_adjoint_uses_total_mapping(scale, integrated_force):
    expected = 1.0 / scale
    for in_remainder in [False, True]:
        values = closed_pair(4e-6, [0.0, 0.0, 0.0])
        mapping = scale * values[5]
        values[5] = np.zeros_like(mapping) if in_remainder else mapping
        sources = arrays(values)
        paired = CompensatedArray(
            sources[5], mapping if in_remainder else np.zeros_like(mapping)
        )
        with wp.Tape() as tape:
            result, _ = solve_geometry(
                InterfaceGeometry(*sources[:2]), *sources[2:5], paired, sources[6], *sources[7:],
                integrated_force=integrated_force,
            )
        tape.backward(grads={result.displacement: array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], wp.vec3d)})
        np.testing.assert_array_equal(result.displacement.numpy(), 0.0)
        np.testing.assert_allclose(sources[0].grad.numpy(), [[expected, 0.0, 0.0]], rtol=2e-13, atol=0.0)


@pytest.mark.parametrize("representation", ["normalized", "remainder", "sub_ulp"])
def test_redundant_closed_force_area_gradient_keeps_mapping_remainder(representation):
    values = closed_pair(4e-6, [1e-6, 0.0, 0.0])
    values = [np.repeat(a, 2, axis=0) if i < 8 else a for i, a in enumerate(values)]
    values[7] = 4e-6 * np.array([0.6, 0.4])
    mapping = values[5].copy()
    low = np.zeros_like(mapping)
    if representation == "remainder":
        values[5][:] = 0.0
        low = mapping
    elif representation == "sub_ulp":
        low = 2.0**-54 * mapping
    sources = arrays(values)
    with wp.Tape() as tape:
        result, _ = solve_geometry(
            InterfaceGeometry(*sources[:2]), *sources[2:5], CompensatedArray(sources[5], low),
            sources[6], *sources[7:], integrated_force=True,
        )
    tape.backward(grads={result.force: array([[0.2, 0.0, 0.0], [-0.3, 0.0, 0.0]], wp.vec3d)})
    np.testing.assert_allclose(result.force.numpy()[:, 0], [0.075, 0.05], rtol=2e-13, atol=0.0)
    np.testing.assert_allclose(sources[7].grad.numpy(), [6250.0, -9375.0], rtol=2e-13, atol=0.0)


@pytest.mark.parametrize("theta", [None, 0.0, 2.0])
@pytest.mark.parametrize("exponent", [-35, -60])
@pytest.mark.parametrize("integrated_force", [False, True])
def test_closed_adjoint_retains_independent_constraint_in_map_remainder(exponent, integrated_force, theta):
    delta = 2.0**exponent
    values = closed_pair(4e-6, [0.0, 0.0, 0.0])
    values = [np.repeat(a, 2, axis=0) if i < 8 else a for i, a in enumerate(values)]
    values[5] = np.ones((2, 2))
    values[7][:] = 2e-6
    source = arrays(values)
    from blademaster_warp.reference_inertia import ReferenceInertia
    from test_reference_inertia import transfer

    inertia = None
    if theta is not None:
        tr = transfer(2, source[0].device)
        inertia = ReferenceInertia(tr, source[9], tr.model.array([theta / values[9][0]], float))
    mapping = CompensatedArray(source[5], np.array([[0.0, 0.0], [0.0, delta]]))

    def run():
        return solve_geometry(
            InterfaceGeometry(*source[:2]), *source[2:5], mapping, source[6], *source[7:],
            integrated_force=integrated_force, inertia=inertia,
        )[0]

    with wp.Tape() as tape:
        result = run()
    seed = np.array([[0.0, 0.0, 0.0], [delta, 0.0, 0.0]])
    tape.backward(grads={result.displacement: array(seed, wp.vec3d)})
    np.testing.assert_allclose(
        source[0].grad.numpy(), [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], rtol=2e-13, atol=0.0
    )
    direction = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    for epsilon in [1e-10, 1e-11]:
        step = epsilon * delta**2
        losses = []
        for sign in [-1.0, 1.0]:
            source[0].assign(sign * step * direction)
            moved = run()
            losses.append(float(np.sum(moved.displacement.numpy() * seed)))
        np.testing.assert_allclose((losses[1] - losses[0]) / (2 * step), 2.0, rtol=2e-13, atol=0.0)
