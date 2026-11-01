"""Decimal oracles for retained weak modes, rank deficiency and scaled solves."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from blademaster_warp._compensated import Expansion
from blademaster_warp._compensated_linalg import Factorization, divide, equilibrated_solve, minimum_norm, solve
from blademaster_warp._compensated_sparse import try_solve


def decimal_value(value, index):
    return Decimal.from_float(float(value.high[index])) + Decimal.from_float(float(value.low[index]))


@pytest.mark.parametrize("exponent", [-900, 0, 900])
@pytest.mark.parametrize("broadcast", [False, True])
def test_division_preserves_remainders_across_scales_and_broadcasts(exponent, broadcast):
    numerator = Expansion(
        np.ldexp([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]], exponent),
        np.ldexp([[1.0, -1.0, 2.0], [-3.0, 4.0, -5.0]], exponent - 55),
    )
    base = np.array([[3.0], [7.0]]) if broadcast else np.array([[3.0, 5.0, 7.0], [11.0, 13.0, 17.0]])
    denominator = Expansion(np.ldexp(base, exponent), np.ldexp(np.ones_like(base), exponent - 55))
    result = divide(numerator, denominator)
    with localcontext() as context:
        context.prec = 100
        for row, column in np.ndindex(numerator.shape):
            expected = decimal_value(numerator, (row, column)) / decimal_value(
                denominator, (row, 0 if broadcast else column)
            )
            assert abs(decimal_value(result, (row, column)) - expected) < Decimal("1e-31")


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("small", [2.0**-60, 2.0**-80, 1e-22])
def test_solve_retains_a_pivot_stored_below_the_matrix_ulp(small, sparse):
    matrix = Expansion(np.ones((2, 2)), [[0.0, 0.0], [0.0, small]])
    rhs = Expansion([[2.0, 3.0], [2.0, 3.0]], [[0.0, 0.0], [3 * small, -2 * small]])
    result = solve(matrix, rhs) if sparse else Factorization(matrix).solve(rhs)
    with localcontext() as context:
        context.prec = 100
        determinant = decimal_value(matrix, (1, 1)) - 1
        for column in range(2):
            second = (decimal_value(rhs, (1, column)) - decimal_value(rhs, (0, column))) / determinant
            first = decimal_value(rhs, (0, column)) - second
            for row, expected in enumerate([first, second]):
                assert abs(decimal_value(result, (row, column)) - expected) < Decimal("1e-28")


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("exponent", [-1000, -900, 0, 900])
def test_refinement_keeps_a_widely_scaled_right_hand_side(exponent, sparse):
    matrix = np.array([[1.0, 2.0, 3.0], [2.0, -1.0, 4.0], [-1.0, 3.0, 2.0]])
    scale = np.ldexp(1.0, exponent)
    rhs = Expansion(np.array([1.0, -2.0, 3.0]) * scale)
    result = solve(matrix, rhs) if sparse else Factorization(matrix).solve(rhs)
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


def test_sparse_refinement_resolves_remainder_only_load():
    matrix = Expansion([[2.0, 1.0], [1.0, 3.0]])
    rhs = Expansion([1.0, 1.0], [2.0**-60, -(2.0**-62)])
    result = try_solve(matrix, rhs)
    assert result is not None
    with localcontext() as context:
        context.prec = 100
        first, second = [decimal_value(rhs, i) for i in range(2)]
        expected = [(3 * first - second) / 5, (2 * second - first) / 5]
        for i in range(2):
            assert abs(decimal_value(result, i) - expected[i]) < Decimal("1e-30")


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


@pytest.mark.parametrize("rows, columns", [(3, 0), (0, 2), (0, 0)])
def test_empty_right_hand_sides_preserve_shape(rows, columns):
    rhs = Expansion(np.empty((rows, columns)))
    result = Factorization(np.eye(rows)).solve(rhs)
    np.testing.assert_array_equal(result.high, rhs.high)
    np.testing.assert_array_equal(result.low, rhs.low)



@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("size", [3, 9])
def test_reaction_block_solution_matches_coupled_system(rank, size):
    rng = np.random.default_rng(51)
    count = 2
    c = rng.normal(size=(3 * count, size))
    if rank == 0:
        c[3:] = c[:3]
    d = np.zeros((3 * count, 3 * count))
    for i in range(count):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        block = rotation[:, :rank] @ np.diag(np.geomspace(0.1, 10.0, rank)) @ rotation[:, :rank].T
        d[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = block
    a = np.block([[np.eye(size), c.T], [c, -d]])
    expected = rng.normal(size=len(a))
    rhs = a @ expected
    actual = minimum_norm(a, rhs).numpy()
    np.testing.assert_allclose(a @ actual, rhs, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual[:size], expected[:size], rtol=2e-12, atol=2e-12)



@pytest.mark.parametrize("weak", [1e-6, 1e-10, 1e-14])
def test_local_weak_reaction_derivative_is_retained(weak):
    matrix = np.block([[np.eye(3), np.eye(3)], [np.eye(3), -np.diag([0.0, weak, 1.0])]])
    expected = np.arange(1.0, 7.0)
    rhs = matrix @ expected
    actual = minimum_norm(matrix, rhs).numpy()
    np.testing.assert_allclose(matrix @ actual, rhs, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)



@pytest.mark.parametrize("areas", [(3e-5, 7e-5), (0.15, 0.7), (1.3e-10, 2.1e-9)])
@pytest.mark.parametrize("exponent", [-900, 0, 900])
def test_equilibrated_kkt_preserves_zero_constrained_increment(areas, exponent):
    a, b = np.sqrt(areas)
    matrix = np.array([[1, 0, a, -1], [0, 1, b, 0], [-a, -b, 0, 0], [1, 0, 0, 0]])
    unit = np.ldexp(1.0, exponent)
    rhs = Expansion(np.ldexp(np.array([2 * a, -b, 0, 0]), exponent))
    result = equilibrated_solve(matrix, rhs)
    with localcontext() as context:
        context.prec = 100
        scale = Decimal.from_float(unit)
        expected = [Decimal(0), Decimal(0), -scale, -3 * Decimal.from_float(a) * scale]
        for i, value in enumerate(expected):
            assert abs(decimal_value(result, i) - value) < abs(scale) * Decimal("1e-28")



@pytest.mark.parametrize("permutation", [0, 1, 2])
def test_singleton_constraints_survive_independent_row_column_permutations(permutation):
    matrix = np.zeros((5, 5))
    matrix[:2, :2] = np.diag([1.1, 1.0])
    matrix[2:, :2] = [[1.3571428558, -7.15e-15], [7.4e-5, 1.3571428575], [-1.1e-4, -6.05e-10]]
    matrix[2:, 2:] = [[1, 2, 3], [2, -1, 4], [-1, 3, 2]]
    rng = np.random.default_rng(permutation)
    rows, columns = rng.permutation(5), rng.permutation(5)
    rhs = np.array([0.0, 0.0, 1.0, -2.0, 3.0])[rows]
    result = Factorization(matrix[np.ix_(rows, columns)]).solve(rhs)
    with localcontext() as context:
        context.prec = 100
        expected = [Decimal(0), Decimal(0), -Decimal(3) / 5, Decimal(4) / 5, Decimal(0)]
        for i, column in enumerate(columns):
            assert abs(decimal_value(result, i) - expected[column]) < Decimal("1e-28")


@pytest.mark.parametrize("exponent", [80, 110, 150])
@pytest.mark.parametrize("matrix_rhs", [False, True])
def test_sparse_refinement_preserves_load_below_backward_error_bound(exponent, matrix_rhs):
    high = np.eye(3) if matrix_rhs else np.ones(3)
    low = -np.ldexp(high, -exponent)
    rhs = Expansion(high, low)
    result = try_solve(Expansion(np.eye(3)), rhs)
    assert result is not None
    np.testing.assert_array_equal(result.high, rhs.high)
    np.testing.assert_array_equal(result.low, rhs.low)
