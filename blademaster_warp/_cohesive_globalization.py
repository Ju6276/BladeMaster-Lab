"""Scaled dogleg trust regions for the semismooth equilibrium residual."""

import numpy as np
from scipy.linalg.lapack import dgeequ

from ._compensated import Expansion as E
from ._compensated_linalg import divide
from ._cohesive_expansion import norm, root, positive, scale_power_two


def dot(a, b):
    return (a * b).sum()


class TrustRegion:
    def __init__(self, at, jacobian):
        rows, columns, _, _, _, status = dgeequ(jacobian.high)
        if status:
            raise RuntimeError(f"Residual equilibration failed: {status}")
        self.at = at
        self.rows = E(rows)
        self.columns = E(columns)
        self.radius = None

    def step(self, y, direction, residual, jacobian):
        value = self.rows * residual
        exponent = int(np.frexp(np.max(abs(value.high)))[1])
        value = scale_power_two(value, -exponent)
        matrix = self.rows[:, None] * jacobian * self.columns[None, :]
        gradient = matrix.T @ value
        newton = divide(direction, self.columns)
        jg = matrix @ gradient
        ratio = divide(norm(gradient), norm(jg))
        cauchy = scale_power_two(-ratio * ratio * gradient, exponent)
        newton_length = float(norm(newton).numpy())
        cauchy_length = float(norm(cauchy).numpy())
        radius = newton_length if self.radius is None else self.radius
        merit = 0.5 * dot(value, value)
        newton_slope = dot(value, scale_power_two(self.rows * (jacobian @ direction), -exponent))
        trace = []
        for iteration in range(80):
            if newton_length <= radius:
                correction = newton
            elif cauchy_length >= radius:
                correction = divide(radius, norm(cauchy)) * cauchy
            else:
                difference = newton - cauchy
                length_exponent = int(np.frexp(max(newton_length, cauchy_length, radius))[1])
                scaled_difference = scale_power_two(difference, -length_exponent)
                scaled_cauchy = scale_power_two(cauchy, -length_exponent)
                scaled_radius = np.ldexp(radius, -length_exponent)
                a = dot(scaled_difference, scaled_difference)
                b = dot(scaled_cauchy, scaled_difference)
                c = dot(scaled_cauchy, scaled_cauchy) - scaled_radius * scaled_radius
                discriminant = root(b * b - a * c)
                fraction = divide(-c, b + discriminant) if b.high >= 0 else divide(-b + discriminant, a)
                correction = cauchy + fraction * difference
            candidate = y + self.columns * correction
            trial_residual, trial_jacobian, _ = self.at(candidate)
            updated = scale_power_two(self.rows * trial_residual, -exponent)
            length = float(norm(correction).numpy())
            if not np.isfinite(updated.high).all() or not np.isfinite(updated.low).all():
                radius = 0.25 * length
                trace.append(dict(iteration=iteration, ratio=0.0, radius=radius, residual=float("inf")))
                continue
            linear = value + scale_power_two(matrix @ correction, -exponent)
            predicted = merit - 0.5 * dot(linear, linear)
            actual = merit - 0.5 * dot(updated, updated)
            ratio = float(divide(actual, predicted).numpy()) if positive(predicted) else 0.0
            boundary = length >= 0.95 * radius
            if ratio < 0.25:
                radius = 0.25 * length
            elif ratio > 0.75 and boundary:
                radius *= 2.0
            trace.append(
                dict(
                    iteration=iteration, ratio=ratio, radius=radius, residual=float(np.max(abs(updated.high)))
                )
            )
            if ratio >= 0.001 and positive(actual):
                self.radius = radius
                tangent = scale_power_two(self.rows * (trial_jacobian @ direction), -exponent)
                if positive(0.9 * newton_slope - dot(updated, tangent)):
                    candidate, extra = self.newton_search(y, direction, residual, jacobian)
                    return candidate, trace + extra
                return candidate, trace
        candidate, extra = self.newton_search(y, direction, residual, jacobian)
        return candidate, trace + extra

    def newton_search(self, y, direction, residual, jacobian):
        """Weak Wolfe search retains weak Newton modes when the dogleg stalls."""
        value = self.rows * residual
        exponent = int(np.frexp(np.max(abs(value.high)))[1])
        value = scale_power_two(value, -exponent)
        tangent = scale_power_two(self.rows * (jacobian @ direction), -exponent)
        slope0 = dot(value, tangent)
        if not positive(-slope0):
            raise RuntimeError("The Newton direction is not a merit descent direction")
        merit0 = 0.5 * dot(value, value)
        lower, upper, alpha = E(0.0), E(1.0), E(1.0)
        trace = []
        for iteration in range(128):
            candidate = y + alpha * direction
            residual, derivative, _ = self.at(candidate)
            finite = all(
                np.isfinite(part).all()
                for part in [residual.high, residual.low, derivative.high, derivative.low]
            )
            if not finite:
                upper = alpha
                alpha = 0.5 * (lower + upper)
                continue
            value = scale_power_two(self.rows * residual, -exponent)
            tangent = scale_power_two(self.rows * (derivative @ direction), -exponent)
            merit, slope = 0.5 * dot(value, value), dot(value, tangent)
            curvature = dot(tangent, tangent)
            sufficient = not positive(merit - merit0 - 0.0001 * alpha * slope0)
            curved = not positive(0.9 * slope0 - slope)
            trace.append(
                dict(
                    mode="newton_search",
                    iteration=iteration,
                    alpha=float(alpha.numpy()),
                    alpha_low=float(alpha.low),
                    residual=float(np.max(abs(value.high))),
                )
            )
            if sufficient and curved:
                self.radius = float(norm(divide(alpha * direction, self.columns)).numpy())
                return candidate, trace
            if positive(slope) or not sufficient:
                upper = alpha
            else:
                lower = alpha
            proposed = alpha - divide(slope, curvature) if positive(curvature) else lower
            if not positive(proposed - lower) or not positive(upper - proposed):
                proposed = 0.5 * (lower + upper)
            if np.array_equal(proposed.high, alpha.high) and np.array_equal(proposed.low, alpha.low):
                raise RuntimeError(f"Compensated Newton search stagnated: {trace[-4:]}")
            alpha = proposed
        raise RuntimeError(f"Compensated Newton search iteration limit: {trace[-4:]}")
