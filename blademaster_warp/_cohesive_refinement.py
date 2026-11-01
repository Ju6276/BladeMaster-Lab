"""Scaled semismooth equilibrium with vanishing dual regularization."""

from copy import copy

import numpy as np
from scipy.linalg.lapack import dgeequ

from ._compensated_linalg import column_basis, divide, solve, minimum_norm
from ._cohesive_globalization import TrustRegion
from ._cohesive_expansion import E, assign, stack


class Balanced:
    def __init__(self, equations, factor):
        self.equations = equations
        self.factor = factor
        self.size = equations.size

    def contact_set(self, state):
        return self.equations.contact_set(state)

    def with_contact_set(self, active):
        equations = copy(self.equations)
        equations.fixed_contact_set = active.copy()
        return Balanced(equations, self.factor)

    def __call__(self, state):
        residual, jacobian, gaps = self.equations(state)
        assign(residual, slice(0, self.size), self.factor * residual[: self.size])
        assign(jacobian, (slice(0, self.size), slice(None)), self.factor * jacobian[: self.size])
        return residual, jacobian, gaps


def _cached_response(equations):
    """Reuse the last state for one solve with fixed equation parameters."""
    previous = None

    def clone(response):
        residual, jacobian, gaps = response
        return (
            E(residual.high.copy(), residual.low.copy()),
            E(jacobian.high.copy(), jacobian.low.copy()),
            [E(gap.high.copy(), gap.low.copy()) for gap in gaps],
        )

    def evaluate(state):
        nonlocal previous
        if previous is not None:
            high, low, response = previous
            if np.array_equal(state.high, high) and np.array_equal(state.low, low):
                return clone(response)
        response = equations(state)
        previous = state.high.copy(), state.low.copy(), clone(response)
        return response

    return evaluate


def regularized_response(equations, state, mu, weights, evaluate=None):
    size = equations.size
    residual, jacobian, gaps = (equations if evaluate is None else evaluate)(state)
    assign(residual, slice(size, None), residual[size:] - mu * weights * state[size:])
    for i, weight in enumerate(weights):
        j = size + i
        assign(jacobian, (j, j), jacobian[j, j] - mu * weight)
    return residual, jacobian, gaps


def _scaled_relative_residual(residual, jacobian, state):
    rows, columns, _, _, _, status = dgeequ(jacobian.high)
    if status:
        raise RuntimeError(f"Residual equilibration failed: {status}")
    scale = max(np.max(abs(state.high / columns)), np.finfo(float).tiny)
    return float(np.max(abs(rows * residual.high)) / scale)


def contact_active_candidate(equations, state, mu, weights):
    active = equations.contact_set(state)
    visited = set()
    updates = 0
    evaluations = 0
    tolerance = 64 * np.finfo(float).eps**2
    for pivot in range(80):
        key = active.tobytes()
        if key in visited:
            raise RuntimeError("Contact active-set candidate cycled")
        visited.add(key)
        fixed = equations.with_contact_set(active)

        def at(value):
            return regularized_response(fixed, value, mu, weights)

        for iteration in range(80):
            residual, jacobian, _ = at(state)
            error = _scaled_relative_residual(residual, jacobian, state)
            if not np.isfinite(error):
                raise RuntimeError("Contact active-set candidate left the finite domain")
            if error <= tolerance:
                break
            correction = solve(jacobian, -residual)
            state = state + correction
            updates += 1
        else:
            raise RuntimeError(f"Contact active-set Newton iteration limit: {error}")
        residual, jacobian, _ = regularized_response(equations, state, mu, weights)
        if _scaled_relative_residual(residual, jacobian, state) <= tolerance:
            return state, dict(active_set_pivots=pivot, active_set_iterations=updates,
                               active_set_evaluations=evaluations)
        changed = np.flatnonzero(active != equations.contact_set(state))
        if not len(changed):
            raise RuntimeError("Contact active-set candidate failed the natural residual check")
        active[changed] = ~active[changed]
    raise RuntimeError("Contact active-set pivot limit")


def closed_active_candidate(equations):
    if not equations.constrain_normal or not equations.maps or np.any(equations.history != 0.0):
        return None
    mapping = stack([value[:3] for value in equations.maps]).reshape(3 * len(equations.maps), equations.size)
    gap = stack([value[:3] for value in equations.base]).reshape(-1)
    basis = column_basis(mapping.T)
    if basis.shape[1]:
        coordinates = minimum_norm(mapping @ basis, gap)
        modes = solve(equations.reduced, basis)
        displacement = -modes @ solve(basis.T @ modes, coordinates)
    else:
        displacement = E(np.zeros(equations.size))
    columns = [factor * value[:3].T for factor, value in zip(equations.factors, equations.maps)]
    reaction_map = E(
        np.concatenate([value.high for value in columns], axis=1),
        np.concatenate([value.low for value in columns], axis=1),
    )
    dual = minimum_norm(reaction_map, -equations.reduced @ displacement)
    state = divide(
        E(np.r_[displacement.high, dual.high], np.r_[displacement.low, dual.low]), equations.length
    )
    residual, _, gaps = equations(state)
    scale = max(np.max(abs(state.high)), np.finfo(float).tiny)
    error = float(np.max(abs(residual.high)))
    closed = all(np.all(value.high == 0.0) and np.all(value.low == 0.0) for value in gaps)
    if not np.isfinite(error) or error > 64 * np.finfo(float).eps ** 2 * scale or not closed:
        return None
    trace = [
        dict(
            iterations=1,
            residual=error,
            relative_residual=error / scale,
            coordinates="original",
            proximal_factor=1.0,
        )
    ]
    return state, gaps, trace


def regularized(equations, y, weights):
    size = equations.size
    trace = []
    evaluate = _cached_response(equations)
    for exponent in range(2, 35, 2):
        mu = 10.0**-exponent

        def at(state):
            return regularized_response(equations, state, mu, weights, evaluate)

        line_evaluations = 0
        search = None
        active_info = {}
        try:
            candidate, active_info = contact_active_candidate(equations, y, mu, weights)
        except (RuntimeError, np.linalg.LinAlgError) as error:
            active_info = dict(active_set_start_error=str(error))
        else:
            y = candidate
        for iteration in range(80):
            residual, jacobian, _ = at(y)
            rows, columns, _, _, _, status = dgeequ(jacobian.high)
            if status:
                raise RuntimeError(f"Residual equilibration failed: {status}")
            scale = max(np.max(abs(y.high / columns)), np.finfo(float).tiny)
            tolerance = 64 * np.finfo(float).eps ** 2 * scale
            error = np.max(abs(rows * residual.high))
            regularization_scale = np.max(abs(rows[size:] * weights * y[size:].high), initial=0.0)
            if error <= max(tolerance, 0.01 * mu * regularization_scale):
                break
            if search is None:
                # One merit function is retained throughout each nonlinear solve.
                search = TrustRegion(at, jacobian)
            correction = solve(jacobian, -residual)
            try:
                y, line = search.step(y, correction, residual, jacobian)
            except RuntimeError as search_failure:
                try:
                    y, active_info = contact_active_candidate(equations, y, mu, weights)
                except RuntimeError as active_failure:
                    raise RuntimeError(
                        f"Equilibrium search failed: {search_failure}; {active_failure}"
                    ) from active_failure
                break
            line_evaluations += len(line)
        else:
            y, active_info = contact_active_candidate(equations, y, mu, weights)
        original, original_jacobian, gaps = evaluate(y)
        error = np.max(abs(original.high))
        scale = max(np.max(abs(y.high)), np.finfo(float).tiny)
        scaled_error = 0.0 if error == 0.0 else _scaled_relative_residual(original, original_jacobian, y)
        row = dict(
            mu=mu,
            iterations=iteration + active_info.get("active_set_iterations", 0),
            residual=float(error),
            relative_residual=float(error / scale),
            scaled_relative_residual=scaled_error,
            line_evaluations=line_evaluations,
            **active_info,
        )
        trace.append(row)
        if error <= 64 * np.finfo(float).eps ** 2 * scale and scaled_error <= 64 * np.finfo(float).eps ** 2:
            return y, gaps, trace
    raise RuntimeError(
        f"Original equilibrium residual exceeds roundoff: scaled={scaled_error}, unscaled={error/scale}"
    )


def refine(equations, y, weights):
    candidate = closed_active_candidate(equations)
    if candidate is not None:
        return candidate
    size = equations.size
    free = max([np.max(abs(base[:3].high)) for base in equations.base] + [0.0])
    seed_displacement = equations.length * y[:size]
    motion = max(
        [
            np.max(abs((base + mapping @ seed_displacement)[:3].high))
            for base, mapping in zip(equations.base, equations.maps)
        ]
        + [0.0]
    )
    if motion == 0.0:
        motion = free
    motion = max(motion, float(np.max(equations.history, initial=0.0)))
    proximal = max(
        [float((c * m[0]).numpy()) for c, m in zip(equations.compliance, equations.material)] + [0.0]
    )
    factor = min(1.0, motion / proximal) if motion > 0.0 and proximal > 0.0 else 1.0
    if factor == 0.0:
        raise FloatingPointError("The proximal coordinate scale underflowed")
    scaled = copy(equations)
    scaled.compliance = [factor * c for c in equations.compliance]
    scaled.factors = [divide(f, factor) for f in equations.factors]

    state = E(y.high.copy(), y.low.copy())
    assign(state, slice(size, None), factor * state[size:])
    state, _, trace = regularized(Balanced(scaled, factor), state, weights)
    assign(state, slice(size, None), divide(state[size:], factor))
    for correction in range(13):
        residual, jacobian, gaps = equations(state)
        scale = max(np.max(abs(state.high)), np.finfo(float).tiny)
        relative = float(np.max(abs(residual.high)) / scale)
        if relative <= 64 * np.finfo(float).eps ** 2:
            break
        if correction == 12:
            raise RuntimeError(f"Original proximal equilibrium residual: {relative}")
        for i, weight in enumerate(weights):
            k = size + i
            coefficient = 1e-34 * weight
            assign(residual, k, residual[k] - coefficient * state[k])
            assign(jacobian, (k, k), jacobian[k, k] - coefficient)
        try:
            state = state + solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            state, gaps, original_trace = regularized(Balanced(equations, 1.0), state, weights)
            trace.extend(original_trace)
    polished = 0
    for _ in range(4):
        residual, jacobian, gaps = equations(state)
        error = float(np.max(abs(residual.high)))
        if error == 0.0:
            break
        fixed = copy(equations)
        fixed.fixed_contact_set = equations.contact_set(state)
        fixed.fixed_loading_set = equations.loading_set(state)
        candidate = state
        for inner in range(16):
            fixed_residual, fixed_jacobian, _ = fixed(candidate)
            try:
                step = solve(fixed_jacobian, -fixed_residual)
            except (np.linalg.LinAlgError, RuntimeError):
                step = minimum_norm(fixed_jacobian, -fixed_residual)
            candidate = candidate + step
            trial, _, trial_gaps = equations(candidate)
            trial_error = float(np.max(abs(trial.high)))
            if not np.isfinite(trial_error) or trial_error < error:
                break
        if not np.isfinite(trial_error) or trial_error >= error:
            break
        state, gaps = candidate, trial_gaps
        polished += 1
    residual, _, gaps = equations(state)
    relative = float(np.max(abs(residual.high)) / max(np.max(abs(state.high)), np.finfo(float).tiny))
    trace.append(
        dict(
            mu=1e-34,
            iterations=correction + polished,
            relative_residual=relative,
            residual=float(np.max(abs(residual.high))),
            coordinates="original",
            proximal_factor=factor,
        )
    )
    return state, gaps, trace
