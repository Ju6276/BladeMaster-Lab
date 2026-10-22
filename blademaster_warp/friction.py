"""Tangential Kelvin–Voigt element in series with a Coulomb slider."""

import warp as wp


@wp.func
def tangential_return(normal: wp.vec3, relative: wp.vec3, history: wp.vec3,
                      normal_force: float, stiffness: float, damping: float,
                      duration: float, static: float, dynamic: float, theta: float, mode: int):
    """SI units; unit normal, k>0, c>=0, dt>0, 0<=dynamic<=static, 1/2<=theta<=1."""
    velocity = relative - wp.dot(relative, normal) * normal
    elastic = history - wp.dot(history, normal) * normal
    trial = -stiffness * (elastic + theta * duration * velocity) - damping * velocity
    force = trial
    slip = wp.vec3(0.0)
    magnitude = wp.length(trial)
    if mode == 2 or (mode < 0 and (static * normal_force == 0.0 or magnitude > static * normal_force)):
        force = wp.vec3(0.0)
        if magnitude > 0.0:
            force = trial * (dynamic * normal_force / magnitude)
        slip = duration * (force - trial) / (damping + theta * duration * stiffness)
    return force, elastic + duration * velocity - slip
