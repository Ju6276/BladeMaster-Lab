"""Fixed-normal ADMM initial values for moving cohesive equilibrium."""

import numpy as np

from .reference_inertia import solve_inertia


def fixed_normal_initial(values, compliance, *, integrated_force=False, max_iterations=500000, inertia=None):
    q, frame, reference, history, material, mapping, _, area, mass, dt = values
    normal = (
        reference[:, :1] * np.cross(frame[:, :, 1], frame[:, :, 2])
        + reference[:, 1:2] * np.cross(frame[:, :, 2], frame[:, :, 0])
        + reference[:, 2:3] * np.cross(frame[:, :, 0], frame[:, :, 1])
    ).astype(np.longdouble)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    root = (np.ones_like(area) if integrated_force else np.sqrt(area))[:, None]
    left, singular, right = np.linalg.svd(root * mapping, full_matrices=False)
    rank = np.count_nonzero(singular > singular[0] * max(mapping.shape) * np.finfo(float).eps)
    left = left[:, :rank]
    if inertia is not None:
        modes = right[:rank].T
        metric = modes.T @ solve_inertia(inertia, modes)
        factor = np.linalg.cholesky(0.5 * (metric + metric.T))
        rotation, singular, _ = np.linalg.svd(singular[:rank, None] * factor, full_matrices=False)
        left = left @ rotation
    left = left.astype(np.longdouble)
    coefficient = np.longdouble(mass[0]) / np.longdouble(dt[0]) ** 2
    eigen = singular[:rank].astype(np.longdouble) ** 2 / coefficient
    weighted_q = root * q.astype(np.longdouble)
    strength = material[:, 0].astype(np.longdouble)
    failure = 2 * material[:, 1] / strength
    old = history.astype(np.longdouble)
    gap = np.zeros_like(weighted_q)
    multiplier = np.zeros_like(weighted_q)
    maximum_compliance = compliance
    tiny = np.finfo(float).tiny
    for iteration in range(max_iterations):
        difference = weighted_q - root * gap + multiplier
        raw = weighted_q - left @ ((eigen / (eigen + compliance))[:, None] * (left.T @ difference))
        trial = (raw + multiplier) / root
        local_compliance = compliance * area if integrated_force else np.full(len(q), compliance)
        component = np.sum(trial * normal, axis=1)
        compression = np.where((component < 0) & (local_compliance > 0), component, 0)
        positive = trial - compression[:, None] * normal
        radius = np.linalg.norm(positive, axis=1)
        elastic = (old < failure) & (local_compliance > 0)
        denominator = old + local_compliance * strength * (1 - old / failure)
        gain = np.ones_like(radius)
        gain[elastic] = old[elastic] / denominator[elastic]
        opened = radius * gain
        loading = elastic & ((opened > old) | ((old == 0) & (radius > local_compliance * strength)))
        opened[loading] = (radius[loading] - local_compliance[loading] * strength[loading]) / (
            1 - local_compliance[loading] * strength[loading] / failure[loading]
        )
        opened = np.where(opened >= failure, radius, opened)
        previous = gap
        ratio = np.divide(opened, radius, out=np.zeros_like(opened), where=radius > 0)
        gap = ratio[:, None] * positive
        multiplier += raw - root * gap
        primal = np.linalg.norm(raw - root * gap) / max(
            np.linalg.norm(raw), np.linalg.norm(root * gap), np.linalg.norm(weighted_q), tiny
        )
        dual = np.linalg.norm(root * (gap - previous)) / max(np.linalg.norm(multiplier), tiny)
        if max(primal, dual) < 1e-10:
            break
        if iteration % 10 == 9:
            updated = compliance
            if primal > 10 * dual:
                updated /= 2
            elif dual > 10 * primal:
                updated = min(2 * updated, maximum_compliance)
            multiplier *= updated / compliance
            compliance = updated
    traction = multiplier / (compliance * root)
    measure = np.ones_like(area) if integrated_force else area
    displacement = -solve_inertia(inertia, mapping.T @ (measure[:, None] * traction)) / coefficient
    info = dict(initial_iterations=iteration + 1, initial_primal=float(primal), initial_dual=float(dual))
    return displacement.astype(float), traction.astype(float), compliance, info
