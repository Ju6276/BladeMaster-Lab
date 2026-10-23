"""Scalar expansion arithmetic for ordered CPU matrix products."""

import math

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False, inline="always")
def _two_sum(a, b):
    value = a + b
    shifted = value - a
    return value, (a - (value - shifted)) + (b - shifted)


@njit(cache=True, fastmath=False, inline="always")
def _two_product(a, b):
    value = a * b
    ma, ea = math.frexp(a)
    mb, eb = math.frexp(b)
    ca, cb = 134217729.0 * ma, 134217729.0 * mb
    ah, bh = ca - (ca - ma), cb - (cb - mb)
    al, bl = ma - ah, mb - bh
    product = ma * mb
    error = ((ah * bh - product) + ah * bl + al * bh) + al * bl
    exponent = ea + eb
    return value, math.ldexp((product - math.ldexp(value, -exponent)) + error, exponent)


@njit(cache=True, fastmath=False)
def matrix_product(left_high, left_low, right_high, right_low):
    batches, rows, inner = left_high.shape
    columns = right_high.shape[2]
    high = np.zeros((batches, rows, columns))
    low = np.zeros_like(high)
    for batch in range(batches):
        for i in range(rows):
            for j in range(columns):
                h, l = 0.0, 0.0
                for k in range(inner):
                    ah, al = _two_sum(left_high[batch, i, k], left_low[batch, i, k])
                    bh, bl = _two_sum(right_high[batch, k, j], right_low[batch, k, j])
                    ph, pl = _two_product(ah, bh)
                    pl += ah * bl + al * bh
                    pl += al * bl
                    ph, pl = _two_sum(ph, pl)
                    nh, nl = _two_sum(h, ph)
                    nl += l + pl
                    h, l = _two_sum(nh, nl)
                high[batch, i, j], low[batch, i, j] = h, l
    return high, low
