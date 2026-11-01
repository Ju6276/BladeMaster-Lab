"""Contact modes change at odd roots, including close and sub-ulp crossings."""

from fractions import Fraction

import numpy as np
import pytest

from blademaster_warp._contact_events import crossing_brackets, evaluate


def polynomial(roots):
    result = [Fraction(1)]
    for root in roots:
        values = [Fraction(0)] * (len(result) + 1)
        for i, value in enumerate(result):
            values[i] -= root * value
            values[i + 1] += value
        result = values
    return result


@pytest.mark.parametrize(
    "roots",
    [
        [Fraction(1, 3)],
        [Fraction(1, 4), Fraction(3, 4)],
        [Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8)],
        [Fraction(1, 2) - Fraction(1, 2**80), Fraction(1, 2) + Fraction(1, 2**80)],
        [Fraction(1, 4)] * 2 + [Fraction(3, 4)],
        [Fraction(1, 4)] * 3 + [Fraction(3, 4)],
        [Fraction(0)] * 2 + [Fraction(3, 4)],
        [Fraction(1, 4)] + [Fraction(1)] * 2,
        [Fraction(0), Fraction(1, 2), Fraction(1)],
    ],
)
def test_exact_brackets_include_every_odd_root(roots):
    coefficients = polynomial(roots)
    expected = sorted(root for root in set(roots) if 0 < root <= 1 and roots.count(root) % 2)
    brackets = crossing_brackets(coefficients)
    assert len(brackets) == len(expected)
    for (lo, hi, direction), root in zip(brackets, expected):
        assert lo <= root <= hi
        expected_direction = 1 if sum(value > root for value in roots) % 2 == 0 else -1
        assert direction == expected_direction
        if lo != hi:
            assert evaluate(coefficients, lo) * evaluate(coefficients, hi) < 0


@pytest.mark.parametrize("coefficients", [[0], [1], [-1], [1, -4, 4], [1, 0, 1], [0, 0, 0, 0, 1]])
def test_constant_tangent_and_outside_guards_have_no_crossings(coefficients):
    assert crossing_brackets(coefficients) == []


def test_irrational_pair_entry_and_exit_are_separate():
    coefficients = [Fraction(1, 8), -1, 1]
    roots = np.roots([1, -1, 0.125])
    brackets = crossing_brackets(coefficients)
    assert len(brackets) == 2
    for (lo, hi, direction), root in zip(brackets, sorted(roots)):
        assert float(lo) < root < float(hi)
        assert evaluate(coefficients, lo) * evaluate(coefficients, hi) < 0
    assert [direction for _, _, direction in brackets] == [-1, 1]


def test_scaling_does_not_change_brackets():
    base = polynomial([Fraction(1, 3), Fraction(2, 3)])
    expected = crossing_brackets(base)
    for scale in [Fraction(1, 2**1000), Fraction(2**1000)]:
        assert crossing_brackets([scale * value for value in base]) == expected
