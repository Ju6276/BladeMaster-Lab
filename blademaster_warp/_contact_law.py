"""Compensated backward-Euler penalty contact and Kelvin–Voigt/Coulomb return.

The Jacobian columns are point, velocity, elastic history, unit normal,
(normal stiffness, tangential stiffness, damping, static friction, dynamic
friction), and duration. Positions and history are in metres, forces in N.
"""

from dataclasses import dataclass

import numpy as np

from ._cohesive_expansion import E, divide, norm, positive


@dataclass(frozen=True)
class ContactResponse:
    force: E
    history: E
    force_jacobian: E
    history_jacobian: E
    gap: E
    mode: str


def response(point, velocity, history, normal, parameters, duration):
    p, v, h, n, parameters, dt = map(E.from_value, (point, velocity, history, normal, parameters, duration))
    identity = E(np.eye(18))
    jp, jv, jh, jn = identity[:3], identity[3:6], identity[6:9], identity[9:12]
    jkn, jkt, jc, _, jmud, jdt = [identity[i] for i in range(12, 18)]
    kn, kt, c, mus, mud = [parameters[i] for i in range(5)]
    gap = (n * p).sum()
    if not positive(-gap):
        return ContactResponse(
            E(np.zeros(3)), E(np.zeros(3)), E(np.zeros((3, 18))), E(np.zeros((3, 18))), gap, "open"
        )

    def projected(value, derivative):
        component = (value * n).sum()
        jacobian = (n[:, None] * derivative + value[:, None] * jn).sum(axis=0)
        return value - component * n, derivative - n[:, None] * jacobian - component * jn

    vt, jvt = projected(v, jv)
    hb, jhb = projected(h, jh)
    load = -kn * gap
    jload = -gap * jkn - kn * (n[:, None] * jp + p[:, None] * jn).sum(axis=0)
    updated = hb + dt * vt
    jupdated = jhb + vt[:, None] * jdt + dt * jvt
    trial = -kt * updated - c * vt
    jtrial = -updated[:, None] * jkt - kt * jupdated - vt[:, None] * jc - c * jvt
    radius = norm(trial)
    tangent, jtangent, mode = trial, jtrial, "stick"
    bound = mus * load
    if not positive(bound) or positive(radius - bound):
        tangent, jtangent = E(np.zeros(3)), E(np.zeros((3, 18)))
        if positive(radius):
            direction = divide(trial, radius)
            jdirection = divide(
                jtrial - direction[:, None] * (direction[:, None] * jtrial).sum(axis=0), radius
            )
            tangent = mud * load * direction
            jtangent = direction[:, None] * (load * jmud + mud * jload) + mud * load * jdirection
        denominator = c + dt * kt
        jdenominator = jc + kt * jdt + dt * jkt
        numerator = c * hb - dt * tangent
        jnumerator = hb[:, None] * jc + c * jhb - tangent[:, None] * jdt - dt * jtangent
        updated = divide(numerator, denominator)
        jupdated = divide(jnumerator - updated[:, None] * jdenominator, denominator)
        mode = "slide"
    force = load * n + tangent
    jforce = n[:, None] * jload + load * jn + jtangent
    return ContactResponse(force, updated, jforce, jupdated, gap, mode)
