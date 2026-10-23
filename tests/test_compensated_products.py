"""Decimal matrix-product oracles across cancellation, scales and broadcasting."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from blademaster_warp._compensated import Expansion


def decimal_array(value):
    result = np.empty(value.shape, dtype=object)
    for index in np.ndindex(value.shape):
        result[index] = Decimal.from_float(float(value.high[index])) + Decimal.from_float(
            float(value.low[index])
        )
    return result


@pytest.mark.parametrize("exponent", [-400, 0, 400])
@pytest.mark.parametrize(
    "left_shape,right_shape",
    [((7,), (7,)), ((3, 7), (7,)), ((7,), (7, 4)), ((3, 7), (7, 4)), ((2, 1, 3, 7), (1, 5, 7, 4))],
)
def test_matrix_product_matches_decimal_oracle(left_shape, right_shape, exponent):
    rng = np.random.default_rng(9613)
    left, right = [
        Expansion(
            np.ldexp(rng.normal(size=shape), exponent),
            np.ldexp(rng.normal(size=shape), exponent - 54),
        )
        for shape in (left_shape, right_shape)
    ]
    result = left @ right
    with localcontext() as context:
        context.prec = 100
        a, b = decimal_array(left), decimal_array(right)
        expected = np.asarray(a @ b, dtype=object)
        scale = np.asarray(abs(a) @ abs(b), dtype=object)
        actual = decimal_array(result)
        assert actual.shape == expected.shape
        for index in np.ndindex(actual.shape):
            assert abs(actual[index] - expected[index]) <= Decimal("1e-28") * scale[index]


@pytest.mark.parametrize("exponent", [-1000, 0, 900])
def test_opposing_terms_preserve_small_product(exponent):
    scale = np.ldexp(1.0, exponent)
    left = Expansion([[2.0**80, 1.0, -(2.0**80)]])
    right = Expansion(np.full((3, 1), scale))
    result = left @ right
    np.testing.assert_array_equal(result.high, [[scale]])
    np.testing.assert_array_equal(result.low, 0.0)


@pytest.mark.parametrize(
    "left_shape,right_shape,expected_shape",
    [((0,), (0,), ()), ((3, 0), (0, 4), (3, 4)), ((0, 3, 7), (1, 7, 4), (0, 3, 4))],
)
def test_empty_matrix_products(left_shape, right_shape, expected_shape):
    result = Expansion(np.zeros(left_shape)) @ Expansion(np.zeros(right_shape))
    assert result.shape == expected_shape
    np.testing.assert_array_equal(result.high, np.zeros(expected_shape))
    np.testing.assert_array_equal(result.low, np.zeros(expected_shape))


@pytest.mark.parametrize("left,right", [(np.array(1.0), np.ones(3)), (np.ones((2, 3)), np.ones((4, 2)))])
def test_invalid_matrix_product_shapes(left, right):
    with pytest.raises(ValueError):
        Expansion(left) @ Expansion(right)
