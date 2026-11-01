"""Mass-metric unilateral projection using a compensated NNLS active set."""

import numpy as np

from ._cohesive_expansion import E, assign, divide, norm, root, stack
from ._compensated_linalg import minimum_norm


def nonnegative_least_squares(matrix, target):
    matrix, target = map(E.from_value, (matrix, target))
    count = matrix.shape[1]
    value = E(np.zeros(count))
    passive = np.zeros(count, dtype=bool)
    # The stopping bound measures arithmetic error in the normal equations.
    roundoff = 64 * np.finfo(float).eps ** 2
    for iteration in range(max(1, 10 * count * count)):
        residual = target - matrix @ value
        dual = matrix.T @ residual
        scale = abs(matrix.high).T @ (abs(target.high) + abs(matrix.high) @ abs(value.high))
        violated = (~passive) & (dual.high > roundoff * scale)
        if not np.any(violated):
            return value, passive
        index = np.argmax(np.where(violated, dual.high, -np.inf))
        passive[index] = True
        for inner in range(max(1, 10 * count)):
            indices = np.flatnonzero(passive)
            trial = E(np.zeros(count))
            assign(trial, indices, minimum_norm(matrix[:, indices], target))
            negative = indices[trial.high[indices] <= 0]
            if not len(negative):
                value = trial
                break
            ratios = divide(value[negative], value[negative] - trial[negative])
            leaving = negative[np.argmin(ratios.high)]
            alpha = ratios[np.argmin(ratios.high)]
            value = value + alpha * (trial - value)
            assign(value, leaving, E(0.0))
            passive[leaving] = False
        else:
            raise RuntimeError("Impact NNLS inner active set did not converge")
    raise RuntimeError("Impact NNLS active set did not converge")


def project(constraint, mass, velocity):
    constraint, mass, velocity = map(E.from_value, (constraint, mass, velocity))
    matrix = constraint.reshape(constraint.shape[0], -1)
    roots = stack([root(value) for value in mass for _ in range(3)])
    weighted = roots * velocity.reshape(-1)
    rows = divide(matrix, roots[None, :])
    scales = stack([norm(row) for row in rows])
    nonzero = np.flatnonzero(scales.high > 0)
    if not len(nonzero):
        return velocity, np.empty(0, dtype=int), E(np.empty(0))
    normalized = divide(rows[nonzero], scales[nonzero, None])
    impulses, passive = nonnegative_least_squares(normalized.T, -weighted)
    active = nonzero[passive]
    if not len(active):
        return velocity, active, E(np.empty(0))
    projected = divide(weighted + normalized.T @ impulses, roots)
    reaction = divide(impulses[passive], scales[active])
    residual = matrix @ projected
    bound = (
        64
        * np.finfo(float).eps ** 2
        * (abs(matrix.high) @ (abs(velocity.high).reshape(-1) + abs(projected.high)))
    )
    if np.any(residual.high < -bound) or np.any(abs(residual.high[active]) > bound[active]):
        raise FloatingPointError("Impact projection failed unilateral velocity constraints")
    return projected.reshape(velocity.shape), active, reaction


def pullback(constraint, mass, velocity, projected, active, reaction, seed):
    matrix = constraint.reshape(constraint.shape[0], -1)[active]
    roots = stack([root(value) for value in mass for _ in range(3)])
    weighted = divide(matrix, roots[None, :])
    multiplier = minimum_norm(weighted.T, divide(seed.reshape(-1), roots))
    dual = divide(seed.reshape(-1) - matrix.T @ multiplier, roots * roots)
    geometry = reaction[:, None] * dual[None, :] - multiplier[:, None] * projected.reshape(-1)[None, :]
    masses = (dual.reshape(velocity.shape) * (velocity - projected)).sum(axis=1)
    velocities = (roots * roots * dual).reshape(velocity.shape)
    return geometry, masses, velocities
