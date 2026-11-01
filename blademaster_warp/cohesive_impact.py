"""Frictionless, perfectly inelastic interface impacts and their KKT adjoint."""

from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from ._cohesive_expansion import E, assign
from .compensated_state import CompensatedArray, as_compensated, add_gradient
from ._impact_geometry import normal_map
from ._impact_projection import project, pullback


def normal_velocity_map(geometry, reference_normal, separation_map, frame_map, *, compensated=False):
    """Differentiate g dot n(F) with respect to particle or affine-moment velocity."""
    result = normal_map(geometry, reference_normal, separation_map, frame_map)
    return result if compensated else result.high


def project_velocity(constraint, mass, velocity):
    """Minimize the mass norm of the velocity jump subject to C v >= 0.

    C has axes (closed contact, velocity vector, xyz). Masses are kg and
    velocities m/s. The caller locates closure events before applying impulses.
    Compensated velocity inputs retain the rounding remainder in the output.
    """
    paired = isinstance(velocity, CompensatedArray)
    sources = [as_compensated(a) for a in (constraint, mass, velocity)]
    c, m, v = [a.high for a in sources]
    if (
        c.dtype != wp.vec3d
        or m.dtype != wp.float64
        or v.dtype != wp.vec3d
        or c.device != v.device
        or m.device != v.device
    ):
        raise ValueError("Impact projection requires float64 arrays on a common device")
    if c.ndim != 2 or v.ndim != 1 or m.shape != v.shape or c.shape[1] != len(v):
        raise ValueError("Impact columns and masses must match the particle velocity")
    C, masses, velocities = [a.cpu() for a in sources]
    if (
        not len(m)
        or np.any(masses.high <= 0)
        or any(not np.isfinite(a.high).all() for a in (C, masses, velocities))
    ):
        raise ValueError("Impact inputs must be finite with positive masses")
    if not len(c):
        return velocity
    projected, active, reaction = project(C, masses, velocities)
    if not len(active):
        return velocity
    result = CompensatedArray.from_cpu(projected, dtype=wp.vec3d, device=v.device)
    tape = wp._src.context.runtime.tape
    if tape is not None:

        def backward():
            dc, dm, dv = pullback(
                C, masses, velocities, projected, active, reaction, E(result.high.grad.numpy())
            )
            matrix = E(np.zeros_like(C.high)).reshape(len(c), -1)
            assign(matrix, active, dc)
            add_gradient(c, matrix.reshape(C.shape))
            add_gradient(m, dm)
            add_gradient(v, dv)

        tape.record_func(backward, [result.high, *[a for a in (c, m, v) if a.requires_grad and a.size]])
    return result if paired else result.high


@dataclass(frozen=True)
class AffineImpact:
    state: object
    dissipation: CompensatedArray
    constraint: CompensatedArray


def project_affine_state(transfer, state, geometry, reference_normal, separation_map, frame_map):
    """Apply a closed-interface impulse to all translational and affine moments.

    Positions, deformation, plasticity and cohesive history stay at the event
    position. The mass metric includes the subparticle affine kinetic energy.
    """
    from ._affine_arithmetic import AffineArithmetic

    arithmetic = AffineArithmetic(state)
    before = arithmetic.pack(transfer, state.v, state.affine)
    constraint = normal_velocity_map(geometry, reference_normal, separation_map, frame_map, compensated=True)
    mass = transfer.model.array(np.full(len(before), transfer.model.mass), float, False)
    after = project_velocity(constraint, mass, arithmetic.state(before))
    after_head = arithmetic.remember(after)
    velocity, affine = arithmetic.unpack(transfer, after_head)
    updated = replace(state, v=velocity, affine=affine).capture_remainders(arithmetic)
    jump = after.cpu() - arithmetic.value(before)
    loss = 0.5 * transfer.model.mass * (jump * jump).sum()
    dissipated = CompensatedArray.from_cpu(loss.reshape(1), dtype=wp.float64, device=before.device)
    tape = wp._src.context.runtime.tape
    if tape is not None:

        def backward():
            gradient = transfer.model.mass * E(dissipated.high.grad.numpy()[0]) * jump
            add_gradient(after_head, gradient)
            add_gradient(before, -gradient)

        tape.record_func(backward, [dissipated.high, before, after_head])
    return AffineImpact(updated, dissipated, constraint)
