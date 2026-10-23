"""Finite reference-face visibility for multi-cut interface traces.

Segments cross a face only in their open interior. Coplanar segments and
endpoint touches do not separate fields. Triangle boundaries are included.
"""

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from ._exact_surface import cross
from .reference_coordinates import rational


@dataclass(frozen=True)
class ReferenceVisibility:
    triangles: np.ndarray
    cut_indices: np.ndarray
    lower: np.ndarray = field(init=False, repr=False)
    upper: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        triangles = np.asarray(self.triangles)
        cuts = np.asarray(self.cut_indices)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            raise ValueError("Reference faces must have shape (triangles, 3, 3)")
        if cuts.shape != (len(triangles),) or not np.issubdtype(cuts.dtype, np.integer) or np.any(cuts < 0):
            raise ValueError("Each reference triangle requires a nonnegative cut index")
        exact = np.asarray(
            [value if isinstance(value, Fraction) else Fraction(float(value)) for value in triangles.flat],
            dtype=object,
        ).reshape(triangles.shape)
        rounded = np.asarray(exact, dtype=float)
        if not np.isfinite(rounded).all():
            raise ValueError("Reference faces must be finite")
        for name, values in [
            ("triangles", exact),
            ("cut_indices", cuts.astype(np.int64, copy=True)),
            ("lower", np.nextafter(rounded.min(axis=1), -np.inf)),
            ("upper", np.nextafter(rounded.max(axis=1), np.inf)),
        ]:
            values.flags.writeable = False
            object.__setattr__(self, name, values)

    @classmethod
    def from_surface(cls, surface, reference, face_cut_indices):
        labels = np.asarray(face_cut_indices)
        if (
            labels.shape != (len(surface.polygons),)
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0)
        ):
            raise ValueError("Each surface polygon requires a cut index")
        reference = np.asarray(reference)
        if reference.shape != surface.reference_remainder.shape or not np.isfinite(reference).all():
            raise ValueError("Reference coordinates must match the surface particle set")
        centres = rational(reference) + surface.reference_remainder
        local = surface.coordinates()
        triangles, cuts = [], []
        for owner, polygon, cut in zip(surface.owners, surface.polygons, labels):
            for i in range(1, len(polygon) - 1):
                triangles.append(local[[polygon[0], polygon[i], polygon[i + 1]]] + centres[owner])
                cuts.append(cut)
        return cls(np.asarray(triangles, dtype=object).reshape(-1, 3, 3), np.asarray(cuts, dtype=np.int64))

    def particle_mask(self, points, particles, cut_indices):
        points, particles = np.asarray(points), np.asarray(particles)
        cuts = np.asarray(cut_indices)
        if any(a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all() for a in [points, particles]):
            raise ValueError("Visibility endpoints must be finite three-dimensional coordinates")
        if cuts.shape != (len(points),) or not np.issubdtype(cuts.dtype, np.integer) or np.any(cuts < 0):
            raise ValueError("Each surface point requires a nonnegative cut index")
        result = np.ones((len(points), len(particles)), dtype=bool)
        exact_points, exact_particles = rational(points), rational(particles)
        for q, start in enumerate(points):
            lower, upper = np.minimum(start, particles), np.maximum(start, particles)
            for face in np.flatnonzero(self.cut_indices != cuts[q]):
                candidates = np.flatnonzero(
                    result[q]
                    & np.all(upper >= self.lower[face], axis=1)
                    & np.all(lower <= self.upper[face], axis=1)
                )
                if not len(candidates):
                    continue
                a, b, c = self.triangles[face]
                normal = cross(b - a, c - a)
                initial = np.dot(normal, exact_points[q] - a)
                if initial == 0:
                    continue
                for p in candidates:
                    final = np.dot(normal, exact_particles[p] - a)
                    if initial * final >= 0:
                        continue
                    fraction = initial / (initial - final)
                    hit = exact_points[q] + fraction * (exact_particles[p] - exact_points[q])
                    if all(np.dot(normal, cross(v - u, hit - u)) >= 0 for u, v in [(a, b), (b, c), (c, a)]):
                        result[q, p] = False
        return result
