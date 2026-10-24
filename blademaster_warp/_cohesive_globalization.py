"""Armijo and weak Wolfe merit search across nonsmooth branch changes."""

import numpy as np

from ._compensated import Expansion as E
from ._compensated_linalg import divide, solve


def positive(value):
    return bool(value.high > 0 or (value.high == 0 and value.low > 0))


def line_search(at, y, direction, residual, jacobian, tolerance):
    slope0 = (residual * (jacobian @ direction)).sum()
    if not positive(-slope0):
        raise RuntimeError("The Newton direction is not a merit descent direction")
    merit0 = (residual * residual).sum() * 0.5
    lower, upper, alpha = E(0.0), E(1.0), E(1.0)
    trace = []
    for iteration in range(128):
        candidate = y + alpha * direction
        value, derivative, _ = at(candidate)
        finite = all(
            np.isfinite(component).all()
            for component in (value.high, value.low, derivative.high, derivative.low)
        )
        if not finite:
            trace.append(dict(iteration=iteration, alpha=float(alpha.numpy()), residual=float("inf")))
            upper = alpha
            alpha = 0.5 * (lower + upper)
            continue
        tangent = derivative @ direction
        merit = (value * value).sum() * 0.5
        slope = (value * tangent).sum()
        curvature = (tangent * tangent).sum()
        sufficient = not positive(merit - merit0 - 0.0001 * alpha * slope0)
        curved = not positive(0.9 * slope0 - slope)
        row = dict(
            iteration=iteration,
            alpha=float(alpha.numpy()),
            alpha_low=float(alpha.low),
            residual=float(np.max(abs(value.high))),
            slope=float(slope.numpy()),
        )
        trace.append(row)
        if (sufficient and curved) or np.max(abs(value.high)) <= tolerance:
            return candidate, trace
        if positive(slope) or not sufficient:
            upper = alpha
        else:
            lower = alpha
        proposed = alpha - divide(slope, curvature) if positive(curvature) else lower
        if not positive(proposed - lower) or not positive(upper - proposed):
            proposed = 0.5 * (lower + upper)
        if np.array_equal(proposed.high, alpha.high) and np.array_equal(proposed.low, alpha.low):
            raise RuntimeError(f"Compensated merit search stagnated: {trace[-4:]}")
        alpha = proposed
    raise RuntimeError(f"Compensated merit search iteration limit: {trace[-4:]}")


def newton_step(at, y, direction, residual, jacobian, tolerance, trial_limit=8):
    slope = (residual * (jacobian @ direction)).sum()
    merit = 0.5 * (residual * residual).sum()
    target = merit + 0.0001 * slope
    trial = y
    step = direction
    trace = []
    for iteration in range(trial_limit):
        trial = trial + step
        if not np.isfinite(trial.high).all() or not np.isfinite(trial.low).all():
            break
        value, derivative, _ = at(trial)
        if not np.isfinite(value.high).all() or not np.isfinite(derivative.high).all():
            break
        row = dict(iteration=iteration, alpha=1.0, residual=float(np.max(abs(value.high))), mode="watchdog")
        trace.append(row)
        if not positive(0.5 * (value * value).sum() - target) or np.max(abs(value.high)) <= tolerance:
            return trial, trace
        if iteration + 1 < trial_limit:
            try:
                step = solve(derivative, -value)
            except (np.linalg.LinAlgError, RuntimeError):
                break
    result, fallback = line_search(at, y, direction, residual, jacobian, tolerance)
    for row in fallback:
        row["mode"] = "rollback_line_search"
    return result, trace + fallback
