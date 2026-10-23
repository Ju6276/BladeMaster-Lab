"""Reference sampling coordinates retained before floating-point rounding."""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .geometry import _sample


def rational(values):
    values = np.asarray(values)
    return np.array([Fraction(float(v)) for v in values.flat], dtype=object).reshape(values.shape)


@dataclass(frozen=True)
class ReferenceCoordinates:
    head: np.ndarray
    exact: np.ndarray

    def __post_init__(self):
        head, exact = np.asarray(self.head, dtype=float), np.asarray(self.exact, dtype=object)
        if head.ndim != 2 or head.shape[1] != 3 or exact.shape != head.shape:
            raise ValueError("Reference coordinates must have shape (particles, 3)")
        if not np.isfinite(head).all() or not np.isfinite(np.asarray(exact, dtype=float)).all():
            raise ValueError("Reference coordinates must be finite")
        for name, values in [("head", head), ("exact", exact)]:
            values = values.copy()
            values.flags.writeable = False
            object.__setattr__(self, name, values)

    @property
    def remainder(self):
        return self.exact - rational(self.head)

    @classmethod
    def sampled(cls, size, spacing, shape, seed=0):
        head, origin, _, indices, offsets, reflected = _sample(size, spacing, shape, seed, metadata=True)
        step = Fraction(float(spacing)) / 2
        exact = rational(origin) + (indices.astype(object) + Fraction(1, 2)) * step + rational(offsets)
        exact[reflected, 2] = step - exact[reflected, 2]
        return cls(head, exact)

    @classmethod
    def from_values(cls, values):
        return cls(np.asarray(values, dtype=float), rational(values))
