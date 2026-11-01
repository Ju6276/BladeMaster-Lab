"""Finite blade closest-point geometry and moving rigid-surface derivatives.

Input columns: particle position/velocity, tool position, xyzw quaternion,
tool velocity/omega, blade half-extents, bevel height and contact offset.
Lengths are in m, velocities in m/s, and omega in rad/s.
"""

from dataclasses import dataclass

import numpy as np

from ._cohesive_expansion import E, stack
from ._blade_contact_kernels import query_kernel




@dataclass(frozen=True)
class BladeGeometry:
    offset: E
    normal: E
    relative: E
    closest: E
    gap: E
    jacobian: E

    @property
    def values(self):
        return stack([*self.offset, *self.normal, *self.relative, *self.closest, self.gap])


def query(point, velocity, position, rotation, tool_velocity, omega, half, bevel, offset):
    raw = [
        E.from_value(v).reshape(-1)
        for v in [point, velocity, position, rotation, tool_velocity, omega, half, bevel, offset]
    ]
    sizes = [3, 3, 3, 4, 3, 3, 3, 1, 1]
    if any(a.shape != (n,) for a, n in zip(raw, sizes)):
        raise ValueError("Blade geometry inputs require vec3, xyzw quaternion and scalar dimensions")
    if any(not np.isfinite(a.high).all() or not np.isfinite(a.low).all() for a in raw):
        raise ValueError("Blade geometry inputs must be finite")
    if not np.isclose(np.linalg.norm(raw[3].numpy()), 1.0, rtol=1e-12, atol=0.0):
        raise ValueError("Blade rotation must be a unit quaternion")
    if np.any(raw[6].numpy() <= 0.0) or not 0.0 <= float(raw[7].numpy()[0]) <= 2.0 * raw[6].numpy()[2]:
        raise ValueError("Blade half-extents must be positive and bevel must lie within its height")
    if float(raw[8].numpy()[0]) < 0.0:
        raise ValueError("Blade contact offset must be nonnegative")
    high, low = query_kernel(
        np.concatenate([a.high for a in raw]), np.concatenate([a.low for a in raw])
    )
    parts = [E(high[start:start + 3, 0], low[start:start + 3, 0]) for start in [0, 3, 6, 9]]
    return BladeGeometry(*parts, E(high[12, 0], low[12, 0]), E(high[:, 1:], low[:, 1:]))
