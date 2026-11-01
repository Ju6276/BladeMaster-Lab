"""Contact crossings along a quadratic material-position query, with guard derivatives."""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.spatial import cKDTree
import warp as wp

from ._cohesive_expansion import E, assign, divide, norm, positive
from ._contact_events import crossing_brackets
from .cohesive_pairing import pairs_at
from .compensated_state import CompensatedArray, as_compensated, add_gradient
from .reference_coordinates import rational
from .topology import compatible


@dataclass(frozen=True)
class PairEvent:
    pair: tuple[int, int]
    lower: Fraction
    upper: Fraction
    direction: int


def _rational_state(value):
    value = value.cpu()
    return rational(value.high) + rational(value.low)


def candidates(start, middle, end, reference, codes, diameter, spacing):
    """Bracket all entry/exit crossings of the quadratic query in (0, 1].

    The three positions correspond to times 0, 1/2 and 1. A Bezier control
    hull bounds the broad phase; exact quartic signs separate crossings from
    even tangencies. The material flow must subsequently locate each event.
    """
    start, middle, end = map(as_compensated, (start, middle, end))
    pairs_at(start, reference, codes, diameter, spacing)
    if any(
        value.high.dtype != start.high.dtype
        or value.high.shape != start.high.shape
        or value.high.device != start.high.device
        or not np.isfinite(value.high.numpy()).all()
        for value in [middle, end]
    ):
        raise ValueError("Contact query positions must share finite particle shapes, types and devices")
    x0, xm, x1 = map(_rational_state, (start, middle, end))
    if len(x0) < 2:
        return []
    reference, codes = np.asarray(reference), np.asarray(codes)
    centre = rational(start.high.numpy())
    control = 2 * xm - (x0 + x1) / 2
    span = max(np.max(abs(value - centre)) for value in [x0, control, x1])
    reach = Fraction(float(diameter)) + 2 * span
    radius = float(reach)
    if Fraction(radius) < reach:
        radius = np.nextafter(radius, np.inf)
    broad = cKDTree(start.high.numpy()).query_pairs(radius, p=np.inf, output_type="ndarray")
    a, b, c = 2 * (x0 + x1 - 2 * xm), 4 * xm - 3 * x0 - x1, x0
    result = []
    for p, q in broad:
        if compatible(codes[p], codes[q]) and positive(
            2 * E(spacing) - norm(E(reference[p]) - E(reference[q]))
        ):
            continue
        aa, bb, cc = a[p] - a[q], b[p] - b[q], c[p] - c[q]
        polynomial = [
            np.dot(cc, cc) - Fraction(float(diameter)) ** 2,
            2 * np.dot(bb, cc),
            np.dot(bb, bb) + 2 * np.dot(aa, cc),
            2 * np.dot(aa, bb),
            np.dot(aa, aa),
        ]
        result.extend(
            PairEvent((int(p), int(q)), lo, hi, direction)
            for lo, hi, direction in crossing_brackets(polynomial)
        )
    return sorted(result, key=lambda event: (event.lower, event.upper, event.pair))


def guard(position, pair, diameter):
    """Signed centre distance minus contact diameter in m, including remainders."""
    position = as_compensated(position)
    p, q = pair
    if (
        position.high.dtype != wp.vec3d
        or position.high.ndim != 1
        or not 0 <= p < q < len(position.high)
        or diameter.dtype != wp.float64
        or diameter.shape != (1,)
        or diameter.device != position.high.device
    ):
        raise ValueError("Contact guard requires a valid particle pair and matching scalar diameter")
    x, d = position.cpu(), E(diameter.numpy()[0])
    if not np.isfinite(x.high).all() or not np.isfinite(d.high) or not positive(d):
        raise ValueError("Contact guard geometry must be finite with positive diameter")
    delta = x[p] - x[q]
    distance = norm(delta)
    if not positive(distance):
        raise ValueError("Coincident particle centres do not define a contact normal")
    normal = divide(delta, distance)
    result = CompensatedArray.from_cpu(
        (distance - d).reshape(1), dtype=wp.float64, device=position.high.device
    )
    tape = wp._src.context.runtime.tape
    if tape is not None:

        def backward():
            seed = E(result.high.grad.numpy()[0])
            gradient = E(np.zeros_like(x.high))
            assign(gradient, p, seed * normal)
            assign(gradient, q, -seed * normal)
            add_gradient(position.high, gradient)
            add_gradient(diameter, -seed.reshape(1))

        tape.record_func(backward, [result.high, *[a for a in [position.high, diameter] if a.requires_grad]])
    return result
