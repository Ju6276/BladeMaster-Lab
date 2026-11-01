"""Decimal oracles for products and reductions across cancellation and scales."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from blademaster_warp._compensated import Expansion
from blademaster_warp._compensated_kernels import array_product


@pytest.mark.parametrize(
    "left,right",
    [
        (1.0 + 2.0**-27, 1.0 - 2.0**-27),
        (np.ldexp(1.1, 900), np.ldexp(1.3, -900)),
        (np.ldexp(1.1, -1020), np.ldexp(1.3, 1020)),
        (np.ldexp(1.0 + 2.0**-52, -1022), 0.5),
        (np.ldexp(1.0 + 2.0**-52, -1022), 1.0 + 2.0**-27),
        (np.nextafter(0.0, 1.0), 0.5),
        (np.nextafter(0.0, 1.0), 1.0 + 2.0**-52),
        (np.finfo(float).max, 1.0 - 2.0**-27),
    ],
)
def test_product_remainder_matches_decimal_at_float_boundaries(left, right):
    high, low = array_product(left, right)
    with localcontext() as context:
        context.prec = 1600
        exact = Decimal.from_float(float(left)) * Decimal.from_float(float(right))
        expected_high = float(exact)
        expected_low = float(exact - Decimal.from_float(expected_high))
    assert high == expected_high
    assert low == expected_low


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


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("exponent", [-900, 0, 900])
def test_reduction_matches_decimal_with_cancellation(axis, exponent):
    rng = np.random.default_rng(217)
    high = np.ldexp(rng.normal(size=(7, 6, 5)), exponent)
    high[1::2] = -high[:-1:2]
    low = np.ldexp(rng.normal(size=high.shape), exponent - 56)
    value = Expansion(high, low)
    # Transposition and slicing exercise reductions over noncontiguous storage.
    value = Expansion(value.high.transpose(2, 0, 1)[::2], value.low.transpose(2, 0, 1)[::2])
    result = value.sum(axis)
    with localcontext() as context:
        context.prec = 100
        source = decimal_array(value)
        expected = np.asarray(source.sum(axis=axis), dtype=object)
        scale = np.asarray(abs(source).sum(axis=axis), dtype=object)
        actual = decimal_array(result)
        assert actual.shape == expected.shape
        for index in np.ndindex(actual.shape):
            assert abs(actual[index] - expected[index]) <= Decimal("1e-28") * scale[index]


@pytest.mark.parametrize(
    "shape,axis,expected_shape",
    [((0,), None, ()), ((0, 3), 0, (3,)), ((3, 0), 1, (3,)), ((0, 3), 1, (0,)), ((), None, ())],
)
def test_empty_and_scalar_reductions(shape, axis, expected_shape):
    value = np.full(shape, 2.0)
    result = Expansion(value).sum(axis)
    assert result.shape == expected_shape
    np.testing.assert_array_equal(result.high, value.sum(axis=axis))
    np.testing.assert_array_equal(result.low, 0.0)
