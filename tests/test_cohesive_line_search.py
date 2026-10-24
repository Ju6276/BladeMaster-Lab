"""Smooth roots and a cusp whose merit admits weak, but no strong, Wolfe steps."""

import numpy as np
import pytest

from blademaster_warp._compensated import Expansion
from blademaster_warp._compensated_linalg import solve
from blademaster_warp._cohesive_globalization import line_search, newton_step


def scalar(value):
    return float(value.numpy())


def test_nonsmooth_merit_crosses_derivative_jump():
    def at(x):
        coordinate = scalar(x[0])
        value = np.sqrt(1.0 + abs(coordinate))
        derivative = (1.0 if coordinate >= 0.0 else -1.0) / (2.0 * value)
        return Expansion([value]), Expansion([[derivative]]), None

    initial = Expansion([1.0])
    residual, jacobian, _ = at(initial)
    direction = solve(jacobian, -residual)
    result, trace = line_search(at, initial, direction, residual, jacobian, 0.0)
    value, derivative, _ = at(result)
    step = trace[-1]["alpha"]
    slope0 = scalar((residual * (jacobian @ direction)).sum())
    slope = scalar((value * (derivative @ direction)).sum())
    merit = 0.5 * scalar((residual * residual).sum())
    actual = 0.5 * scalar((value * value).sum())
    assert scalar(result[0]) < 0.0
    assert actual <= merit + 1e-4 * step * slope0
    assert slope >= 0.9 * slope0
    assert abs(slope) > 0.9 * abs(slope0)


@pytest.mark.parametrize("scale", [2.0**-300, 1.0, 2.0**300])
@pytest.mark.parametrize("method", [line_search, newton_step])
def test_linear_equilibrium_accepts_exact_newton_step(scale, method):
    matrix = Expansion(scale * np.array([[2.0, -1.0], [1.0, 3.0]]))
    target = Expansion([0.25, -0.125])

    def at(x):
        return matrix @ (x - target), matrix, None

    initial = Expansion([1.0, 2.0])
    residual, jacobian, _ = at(initial)
    direction = solve(jacobian, -residual)
    result, trace = method(at, initial, direction, residual, jacobian, 0.0)
    np.testing.assert_allclose(result.high, target.high, rtol=0.0, atol=1e-30)
    assert trace[-1]["alpha"] == 1.0


def test_watchdog_reduces_nonlinear_residual_after_overshoot():
    def at(x):
        return x * x * x - 1.0, (3.0 * x * x).reshape(1, 1), None

    initial = Expansion([0.1])
    residual, jacobian, _ = at(initial)
    direction = solve(jacobian, -residual)
    trial = at(initial + direction)[0]
    assert abs(scalar(trial[0])) > abs(scalar(residual[0]))
    result, trace = newton_step(at, initial, direction, residual, jacobian, 0.0)
    assert abs(scalar(at(result)[0][0])) < abs(scalar(residual[0]))
    assert all(np.isfinite(row["residual"]) for row in trace)


def test_line_search_rejects_ascent_direction():
    def at(x):
        return x, Expansion([[1.0]]), None

    initial = Expansion([1.0])
    residual, jacobian, _ = at(initial)
    with pytest.raises(RuntimeError, match="descent"):
        line_search(at, initial, Expansion([1.0]), residual, jacobian, 0.0)


@pytest.mark.parametrize("method", [line_search, newton_step])
def test_domain_invalid_trial_is_reduced(method):
    def at(x):
        coordinate = scalar(x[0])
        with np.errstate(invalid="ignore"):
            value = np.log(coordinate)
        return Expansion([value]), Expansion([[1.0 / coordinate]]), None

    initial = Expansion([4.0])
    residual, jacobian, _ = at(initial)
    direction = solve(jacobian, -residual)
    assert scalar((initial + direction)[0]) < 0.0
    result, _ = method(at, initial, direction, residual, jacobian, 0.0)
    assert scalar(result[0]) > 0.0
    assert abs(scalar(at(result)[0][0])) < abs(scalar(residual[0]))
