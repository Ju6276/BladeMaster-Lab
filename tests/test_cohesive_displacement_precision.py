"""Closed-form interface motion and its sub-ULP state increments."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_geometry import InterfaceGeometry, solve_geometry
from blademaster_warp.compensated_state import CompensatedArray
from blademaster_warp.differentiable_cohesive import solve_force
from blademaster_warp._cohesive_linear import minimum_norm
from test_differentiable_cohesive import arrays, closed_pair


def test_minimum_norm_keeps_row_space_for_inconsistent_redundant_equations():
    matrix = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 1.0]])
    solution = minimum_norm(matrix, np.array([1.0, 3.0, -2.0]))
    with localcontext() as context:
        context.prec = 90
        for actual, numerator in zip(solution, [64, -47, 17]):
            high = float(actual)
            low = float(actual - np.longdouble(high))
            total = Decimal(high) + Decimal(low)
            assert abs(total - Decimal(numerator) / 15) < Decimal(2) ** -57
    np.testing.assert_allclose(np.array([-1, -1, 1]) @ solution, 0.0, atol=2.0**-58)
    np.testing.assert_allclose(matrix.T @ (matrix @ solution - [1, 3, -2]), 0.0, atol=2.0**-53)


@pytest.mark.parametrize("rows", [0, 4])
def test_minimum_norm_empty_or_zero_map(rows):
    solution = minimum_norm(np.zeros((rows, 3)), np.ones((rows, 2)))
    np.testing.assert_array_equal(solution, np.zeros((3, 2)))


@pytest.mark.parametrize("integrated", [False, True])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_sub_ulp_displacement_survives_state_update_and_total_gap_pullback(integrated, sign):
    values = closed_pair(4e-6, [sign * 2.0**-20, 0.0, 0.0])
    source = arrays(values)
    remainder = np.array([[sign * 2.0**-80, 0.0, 0.0]])
    geometry = InterfaceGeometry(*source[:2], gap_remainder=remainder)
    expected_high = np.array([[sign * 2.0**-21, 0.0, 0.0], [-sign * 2.0**-21, 0.0, 0.0]])
    expected_low = np.array([[sign * 2.0**-81, 0.0, 0.0], [-sign * 2.0**-81, 0.0, 0.0]])
    base = CompensatedArray(wp.array(-expected_high, dtype=wp.vec3d))
    duration = wp.array([1.0], dtype=wp.float64, requires_grad=True)
    with wp.Tape() as tape:
        response, _ = solve_geometry(geometry, *source[2:], integrated_force=integrated)
        total = base.add_scaled(response.displacement_state, duration)
    np.testing.assert_array_equal(response.displacement.numpy(), expected_high)
    np.testing.assert_array_equal(response.displacement_remainder, expected_low)
    np.testing.assert_array_equal(total.cpu().numpy(), expected_low)
    restored = CompensatedArray.restore(response.displacement_state.snapshot(), dtype=wp.vec3d)
    np.testing.assert_array_equal(restored.low, expected_low)

    seed = np.array([[0.3, -0.2, 0.7], [-0.6, 0.4, 0.9]])
    tape.backward(grads={total.high: wp.array(seed, dtype=wp.vec3d)})
    expected_gradient = 0.5 * (seed[:1] - seed[1:])
    np.testing.assert_allclose(source[0].grad.numpy(), expected_gradient, rtol=1e-13, atol=1e-15)
    for fraction in [0.25, 0.5]:
        delta = fraction * 2.0**-80
        outputs = []
        for direction in [-1.0, 1.0]:
            trial_remainder = remainder.copy()
            trial_remainder[0, 0] += direction * delta
            trial_geometry = InterfaceGeometry(*source[:2], gap_remainder=trial_remainder)
            trial, _ = solve_geometry(trial_geometry, *source[2:], integrated_force=integrated)
            outputs.append(base.add_scaled(trial.displacement_state, duration).cpu().numpy())
        finite_difference = np.sum(seed * (outputs[1] - outputs[0])) / (2 * delta)
        np.testing.assert_allclose(finite_difference, expected_gradient[0, 0], rtol=1e-13)


def test_integrated_force_entry_preserves_nonbinary_displacement_and_its_derivative():
    values = closed_pair(4e-6, [2.0**-20, 0.0, 0.0])
    values[5] = np.array([[-1.0, -1.0, 2.0]])
    values[6] = np.zeros((1, 3, 3))
    source = arrays(values)
    with wp.Tape() as tape:
        response = solve_force(*source)
    state = response.displacement_state
    with localcontext() as context:
        context.prec = 90
        opening = Decimal(2) ** -20
        expected = [opening / 6, opening / 6, -opening / 3]
        bound = Decimal(2) ** -60 * opening
        for high, low, reference in zip(state.high.numpy()[:, 0], state.low[:, 0], expected):
            actual = Decimal(float(high)) + Decimal(float(low))
            assert abs(actual - reference) < bound
            assert abs(Decimal(float(high)) - reference) > bound
    assert np.any(state.low != 0.0)
    base = CompensatedArray(wp.zeros(3, dtype=wp.vec3d))
    with wp.Tape() as downstream:
        total = base.add_scaled(state, wp.array([2.0], dtype=wp.float64))
    seed = np.array([[0.3, 0.0, 0.0], [-0.6, 0.0, 0.0], [0.7, 0.0, 0.0]])
    downstream.backward(grads={total.high: wp.array(seed, dtype=wp.vec3d)})
    tape.backward()
    expected_gradient = 2.0 * (seed[0, 0] + seed[1, 0] - 2.0 * seed[2, 0]) / 6.0
    np.testing.assert_allclose(source[0].grad.numpy()[0, 0], expected_gradient, rtol=1e-13)
