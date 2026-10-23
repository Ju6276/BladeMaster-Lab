"""Exact cancellation, independent decimal solves, and transported state gradients."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
import warp as wp

from blademaster_warp._compensated import Expansion
from blademaster_warp.compensated_state import CompensatedArray
from blademaster_warp.reference_inertia import ReferenceInertia
from blademaster_warp.cohesive_geometry import InterfaceGeometry
from test_reference_inertia import transfer


def exact(value):
    return Decimal.from_float(float(value))


def decimal_value(value, index):
    return exact(value.high[index]) + exact(value.low[index])


@pytest.mark.parametrize("exponent", [-1020, -500, 0, 500, 1020])
def test_products_preserve_rounding_remainders_without_splitter_overflow(exponent):
    left = np.ldexp(np.array([1.1, -1.7, 1.0 + 2**-52]), exponent)
    right = np.ldexp(np.array([1.3, 1.1, 1.0 - 2**-52]), -exponent)
    result = Expansion(left) * right
    with localcontext() as context:
        context.prec = 100
        for i in range(len(left)):
            assert decimal_value(result, i) == exact(left[i]) * exact(right[i])


@pytest.mark.parametrize("kind", ["dot", "matrix", "batch"])
def test_matrix_products_match_decimal_after_severe_cancellation(kind):
    a = np.array([[2.0**80, 1.0, -(2.0**80)], [2.0**70, 3.0, -(2.0**70)]])
    b = np.array([[1.0, -2.0], [0.25, 0.125], [1.0, -2.0]])
    if kind == "dot":
        a, b = a[0], b[:, 0]
    elif kind == "batch":
        a = np.stack([a, 0.5 * a])
    result = Expansion(a) @ Expansion(b)
    expected = np.array(0.25) if kind == "dot" else np.array([[0.25, 0.125], [0.75, 0.375]])
    if kind == "batch":
        expected = np.stack([expected, 0.5 * expected])
    np.testing.assert_array_equal(result.numpy(), expected)
    np.testing.assert_array_equal(result.low, np.zeros_like(expected))


def decimal_solve(matrix, rhs):
    n = len(matrix)
    a = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(a[row][column]))
        a[column], a[pivot] = a[pivot], a[column]
        divisor = a[column][column]
        a[column] = [v / divisor for v in a[column]]
        for row in range(n):
            if row != column:
                factor = a[row][column]
                a[row] = [x - factor * y for x, y in zip(a[row], a[column])]
    return [row[-1] for row in a]


@pytest.mark.parametrize("theta", [0.0, 0.04, 20.0])
def test_inertia_preserves_sub_ulp_input_against_decimal_equilibrium(theta):
    tr = transfer(8, wp.get_device())
    m = tr.model
    duration = m.array([2e-6], float)
    rate = m.array([theta / 2e-6], float)
    inertia = ReferenceInertia(tr, duration, rate)
    rng = np.random.default_rng(219)
    high = rng.normal(size=(8, 3))
    low = rng.normal(size=(8, 3)) * 2**-70
    initial = CompensatedArray(m.array(high, wp.vec3), low)
    with wp.Tape() as tape:
        result = inertia.solve_state(initial)
    seed = rng.normal(size=(8, 3))
    tape.backward(grads={result.high: m.array(seed, wp.vec3, False)})
    P = inertia.mapping
    with localcontext() as context:
        context.prec = 90
        coefficient = exact(duration.numpy()[0]) * exact(rate.numpy()[0])
        D = [
            [
                Decimal(i == j)
                + coefficient
                * (Decimal(i == j) - sum(exact(P[k, i]) * exact(P[k, j]) for k in range(len(P))))
                for j in range(8)
            ]
            for i in range(8)
        ]
        velocity, adjoint = [], []
        for c in range(3):
            solved = decimal_solve(D, [exact(high[i, c]) + exact(low[i, c]) for i in range(8)])
            dual = decimal_solve(D, [exact(seed[i, c]) for i in range(8)])
            velocity.append(solved)
            adjoint.append(dual)
            for i, value in enumerate(solved):
                actual = exact(result.high.numpy()[i, c]) + exact(result.low[i, c])
                assert abs(actual - value) < Decimal("2e-28") * max(Decimal(1), abs(value))
        L = [
            [Decimal(i == j) - sum(exact(P[k, i]) * exact(P[k, j]) for k in range(len(P))) for j in range(8)]
            for i in range(8)
        ]
        power = sum(
            adjoint[c][i] * sum(L[i][j] * velocity[c][j] for j in range(8))
            for c in range(3)
            for i in range(8)
        )
        np.testing.assert_allclose(
            duration.grad.numpy(), [-float(exact(rate.numpy()[0]) * power)], rtol=2e-13, atol=1e-12
        )
        np.testing.assert_allclose(
            rate.grad.numpy(), [-float(exact(duration.numpy()[0]) * power)], rtol=2e-13, atol=1e-15
        )
        np.testing.assert_allclose(
            initial.high.grad.numpy(), np.array(adjoint, dtype=float).T, rtol=2e-13, atol=1e-13
        )


def test_state_increment_restart_and_geometry_keep_a_small_displacement():
    device = wp.get_device()
    tr = transfer(3, device)
    m = tr.model
    base = CompensatedArray(m.array(np.ones((3, 3)), wp.vec3))
    rate = m.array(np.full((3, 3), 2**-80), wp.vec3)
    duration = m.array([0.5], float)
    geometry = InterfaceGeometry(m.array(np.zeros((1, 3)), wp.vec3), m.array(np.eye(3)[None], wp.mat33d))
    B = m.array([[1.0, 0.0, 0.0]], float)
    G = m.array(np.zeros((1, 3, 3)), wp.vec3)
    with wp.Tape() as tape:
        moved = base.add_scaled(rate, duration)
        difference = moved.add_scaled(base, m.array([1.0], float), -1.0)
        updated = geometry.advance(B, G, difference, duration)
    np.testing.assert_array_equal(difference.high.numpy(), np.full((3, 3), 2**-81))
    np.testing.assert_array_equal(updated.gap.numpy(), np.full((1, 3), 2**-82))
    restored = CompensatedArray.restore(moved.snapshot(), dtype=wp.vec3d, device=device)
    np.testing.assert_array_equal(restored.high.numpy(), moved.high.numpy())
    np.testing.assert_array_equal(restored.low, moved.low)
    tape.backward(grads={updated.gap: m.array([[1.0, -2.0, 3.0]], wp.vec3, False)})
    np.testing.assert_array_equal(duration.grad.numpy(), [2**-79])
    expected = np.zeros((3, 3))
    expected[0] = np.array([1.0, -2.0, 3.0]) * 0.25
    np.testing.assert_array_equal(rate.grad.numpy(), expected)
