"""Outward-rounded float64 bounds for orientation predicates."""

import numpy as np

from .reference_coordinates import rational


def bounds(exact):
    head = np.asarray(exact, dtype=float)
    residual = exact - rational(head)
    return (
        np.where(residual < 0, np.nextafter(head, -np.inf), head),
        np.where(residual > 0, np.nextafter(head, np.inf), head),
    )


def _add(a, b):
    return np.nextafter(a[0] + b[0], -np.inf), np.nextafter(a[1] + b[1], np.inf)


def _subtract(a, b):
    return np.nextafter(a[0] - b[1], -np.inf), np.nextafter(a[1] - b[0], np.inf)


def _multiply(a, b):
    products = np.stack([x * y for x in a for y in b])
    return np.nextafter(products.min(axis=0), -np.inf), np.nextafter(products.max(axis=0), np.inf)


def _component(a, axis):
    return a[0][..., axis], a[1][..., axis]


def _orientation(a, b, c, d):
    u, v, w = _subtract(a, d), _subtract(b, d), _subtract(c, d)
    terms = []
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        minor = _subtract(
            _multiply(_component(v, j), _component(w, k)),
            _multiply(_component(v, k), _component(w, j)),
        )
        terms.append(_multiply(_component(u, i), minor))
    return _add(_add(terms[0], terms[1]), terms[2])


def segment_triangles(start, end, lower, upper):
    """Return proven crossings and misses; all remaining pairs need exact signs.

    Inputs lower/upper enclose each triangle vertex. Overflow and intervals
    containing zero remain undecided, including coplanar and boundary cases.
    """
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        a, b, c = [(lower[:, i], upper[:, i]) for i in range(3)]
        start, end = (start, start), (end, end)
        initial, final = _orientation(a, b, c, start), _orientation(a, b, c, end)
        positive, negative = initial[0] > 0, initial[1] < 0
        proper = (positive & (final[1] < 0)) | (negative & (final[0] > 0))
        outside = (positive & (final[0] > 0)) | (negative & (final[1] < 0))
        edges = [
            _orientation(start, end, a, b),
            _orientation(start, end, b, c),
            _orientation(start, end, c, a),
        ]
        edge_lower, edge_upper = np.stack([e[0] for e in edges]), np.stack([e[1] for e in edges])
        inside = np.all(edge_lower > 0, axis=0) | np.all(edge_upper < 0, axis=0)
        outside |= np.any(edge_lower > 0, axis=0) & np.any(edge_upper < 0, axis=0)
        finite = np.isfinite(np.stack([*initial, *final, *edge_lower, *edge_upper])).all(axis=0)
    return finite & proper & inside, finite & outside
