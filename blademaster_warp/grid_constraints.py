"""Mass-weighted grid velocity constraints and their fixed-rank KKT adjoint."""

import numpy as np
import warp as wp


def project_velocity(mapping, mass, velocity, active=None):
    """Project grid velocities onto Q v = 0 without changing free directions.

    Q is dimensionless, nodal mass is in kg, and velocity is in m/s.
    Active rows select intact constraints; their topology is fixed during a VJP.
    """
    if (
        mapping.dtype != wp.float64
        or mass.dtype != wp.float64
        or velocity.dtype != wp.vec3d
        or mapping.device != velocity.device
        or mass.device != velocity.device
    ):
        raise ValueError("Grid constraints require float64 arrays on a common device")
    if mapping.ndim != 2 or mass.shape != velocity.shape or mapping.shape[1] != len(velocity):
        raise ValueError("Constraint columns and nodal masses must match the grid velocity")
    q, masses, velocities = [a.numpy() for a in [mapping, mass, velocity]]
    if (
        any(not np.isfinite(a).all() for a in [q, masses, velocities])
        or not len(masses)
        or np.any(masses <= 0)
    ):
        raise ValueError("Grid constraint inputs must be finite with positive nodal masses")
    mask = np.ones(len(q), dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if mask.shape != (len(q),):
        raise ValueError("Active rows must match the interface constraints")
    indices = np.flatnonzero(mask)
    if not len(indices):
        return velocity
    q = q[indices]
    root = np.sqrt(masses)
    left, singular, right = np.linalg.svd(q / root, full_matrices=False)
    rank = np.count_nonzero(singular > singular[0] * max(q.shape) * np.finfo(float).eps)
    left, singular, right = left[:, :rank], singular[:rank], right[:rank]
    weighted = root[:, None] * velocities
    reaction = left @ ((right @ weighted) / singular[:, None])
    projected = (weighted - right.T @ (right @ weighted)) / root[:, None]
    result = wp.array(projected, dtype=wp.vec3d, device=velocity.device, requires_grad=True)
    tape = wp._src.context.runtime.tape
    if tape is not None:

        def backward():
            seed = result.grad.numpy() / root[:, None]
            dual = (seed - right.T @ (right @ seed)) / root[:, None]
            constraint_dual = left @ ((right @ seed) / singular[:, None])
            if mapping.requires_grad:
                gradient = mapping.grad.numpy()
                gradient[indices] -= reaction @ dual.T + constraint_dual @ projected.T
                mapping.grad.assign(gradient)
            if mass.requires_grad:
                gradient = np.sum(dual * (velocities - projected), axis=1)
                mass.grad.assign(mass.grad.numpy() + gradient)
            if velocity.requires_grad:
                velocity.grad.assign(velocity.grad.numpy() + masses[:, None] * dual)

        tape.record_func(backward, [result, *[a for a in [mapping, mass, velocity] if a.requires_grad]])
    return result
