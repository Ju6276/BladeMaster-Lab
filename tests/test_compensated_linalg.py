"""Decimal oracles for retained weak modes, rank deficiency and scaled solves."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from blademaster_warp._compensated import Expansion
from blademaster_warp._compensated_linalg import Factorization, minimum_norm


def decimal_value(value, index):
    return Decimal.from_float(float(value.high[index])) + Decimal.from_float(float(value.low[index]))


@pytest.mark.parametrize("small", [2.0**-60, 2.0**-80, 1e-22])
def test_solve_retains_a_pivot_stored_below_the_matrix_ulp(small):
    matrix = Expansion(np.ones((2, 2)), [[0.0, 0.0], [0.0, small]])
    rhs = Expansion([[2.0, 3.0], [2.0, 3.0]], [[0.0, 0.0], [3 * small, -2 * small]])
    result = Factorization(matrix).solve(rhs)
    with localcontext() as context:
        context.prec = 100
        determinant = decimal_value(matrix, (1, 1)) - 1
        for column in range(2):
            second = (decimal_value(rhs, (1, column)) - decimal_value(rhs, (0, column))) / determinant
            first = decimal_value(rhs, (0, column)) - second
            for row, expected in enumerate([first, second]):
                assert abs(decimal_value(result, (row, column)) - expected) < Decimal("1e-28")


@pytest.mark.parametrize("exponent", [-1000, -900, 0, 900])
def test_refinement_keeps_a_widely_scaled_right_hand_side(exponent):
    matrix = np.array([[1.0, 2.0, 3.0], [2.0, -1.0, 4.0], [-1.0, 3.0, 2.0]])
    scale = np.ldexp(1.0, exponent)
    rhs = Expansion(np.array([1.0, -2.0, 3.0]) * scale)
    result = Factorization(matrix).solve(rhs)
    with localcontext() as context:
        context.prec = 100
        unit = Decimal.from_float(scale)
        # Solve the integer equations independently by elimination.
        coefficients = [[Decimal(int(x)) for x in row] for row in matrix]
        values = [Decimal.from_float(float(x)) for x in rhs.high]
        for j in range(3):
            pivot = max(range(j, 3), key=lambda i: abs(coefficients[i][j]))
            coefficients[j], coefficients[pivot] = coefficients[pivot], coefficients[j]
            values[j], values[pivot] = values[pivot], values[j]
            divisor = coefficients[j][j]
            coefficients[j] = [x / divisor for x in coefficients[j]]
            values[j] /= divisor
            for i in range(3):
                if i != j:
                    factor = coefficients[i][j]
                    coefficients[i] = [a - factor * b for a, b in zip(coefficients[i], coefficients[j])]
                    values[i] -= factor * values[j]
        tolerance = max(abs(unit) * Decimal("1e-28"), Decimal.from_float(np.nextafter(0.0, 1.0)) * 16)
        for i, expected in enumerate(values):
            assert abs(decimal_value(result, i) - expected) < tolerance


def test_rank_deficient_least_squares_matches_exact_minimum_norm():
    matrix = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 1.0]])
    result = minimum_norm(matrix, [1.0, 3.0, -2.0])
    with localcontext() as context:
        context.prec = 90
        for i, numerator in enumerate([64, -47, 17]):
            assert abs(decimal_value(result, i) - Decimal(numerator) / 15) < Decimal("1e-28")
    weak = minimum_norm(np.diag([1.0, 1e-14, 0.0]), [2.0, 3e-14, 0.0])
    np.testing.assert_allclose(weak.numpy(), [2.0, 3.0, 0.0], rtol=1e-15, atol=1e-30)


@pytest.mark.parametrize("rows, columns", [(0, 3), (3, 0), (3, 4)])
def test_zero_rank_solution_has_the_requested_shape(rows, columns):
    result = minimum_norm(np.zeros((rows, columns)), np.ones((rows, 2)))
    np.testing.assert_array_equal(result.numpy(), np.zeros((columns, 2)))
