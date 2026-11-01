"""Finite-blade contact Jacobian in world point, velocity, tool and history coordinates.

Columns 0:24 match _blade_contact_geometry; 24:27 are elastic history,
27:32 are contact parameters, and 32 is duration.
"""

import numpy as np

from ._blade_contact_geometry import query
from ._cohesive_expansion import E, assign
from ._contact_law import ContactResponse, response as plane_response


def response(row, point, velocity, duration):
    geometry = query(
        point,
        velocity,
        row["position"],
        row["rotation"],
        row["tool_velocity"],
        row["omega"],
        row["half"],
        row["bevel"],
        row["offset"],
    )
    local = plane_response(
        geometry.offset, geometry.relative, row["history"], geometry.normal, row["parameters"], duration
    )
    chain = E(np.zeros((18, 33)))
    assign(chain, (slice(0, 3), slice(0, 24)), geometry.jacobian[:3])
    assign(chain, (slice(3, 6), slice(0, 24)), geometry.jacobian[6:9])
    assign(chain, (slice(9, 12), slice(0, 24)), geometry.jacobian[3:6])
    assign(chain, (slice(6, 9), slice(24, 27)), E(np.eye(3)))
    assign(chain, (slice(12, 17), slice(27, 32)), E(np.eye(5)))
    assign(chain, (17, 32), E(1.0))
    return ContactResponse(
        local.force,
        local.history,
        local.force_jacobian @ chain,
        local.history_jacobian @ chain,
        local.gap,
        local.mode,
    )
