"""Particle-centre contact, with the changing pair normal in the Jacobian.

Columns: relative position, velocity, history, kn/kt/c/mus/mud,
diameter and duration. Lengths are m; force is N on the first particle.
"""

import numpy as np

from ._cohesive_expansion import E, assign, divide, norm, positive
from ._contact_law import ContactResponse, response as plane_response


def response(row, point, velocity, duration):
    point, diameter = E.from_value(point), E.from_value(row["diameter"]).reshape(())
    distance = norm(point)
    if not positive(distance):
        raise ValueError("Coincident particle centres do not define a contact normal")
    normal = divide(point, distance)
    normal_jacobian = divide(E(np.eye(3)) - normal[:, None] * normal[None, :], distance)
    offset = point - diameter * normal
    local = plane_response(offset, velocity, row["history"], normal, row["parameters"], duration)
    chain = E(np.zeros((18, 16)))
    assign(chain, (slice(0, 3), slice(0, 3)), E(np.eye(3)) - diameter * normal_jacobian)
    assign(chain, (slice(0, 3), 14), -normal)
    assign(chain, (slice(3, 6), slice(3, 6)), E(np.eye(3)))
    assign(chain, (slice(6, 9), slice(6, 9)), E(np.eye(3)))
    assign(chain, (slice(9, 12), slice(0, 3)), normal_jacobian)
    assign(chain, (slice(12, 17), slice(9, 14)), E(np.eye(5)))
    assign(chain, (17, 15), E(1.0))
    return ContactResponse(
        local.force,
        local.history,
        local.force_jacobian @ chain,
        local.history_jacobian @ chain,
        local.gap,
        local.mode,
    )
