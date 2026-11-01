"""Scalar expansion arithmetic with fixed CPU accumulation order."""

import numpy as np
from numba import float64, guvectorize, njit
from numba.extending import intrinsic


@njit(cache=True, fastmath=False, inline="always")
def _two_sum(a, b):
    value = a + b
    shifted = value - a
    return value, (a - (value - shifted)) + (b - shifted)


@intrinsic
def _fma(typing_context, a, b, c):
    if a != float64 or b != float64 or c != float64:
        return None

    def codegen(context, builder, signature, arguments):
        return builder.fma(*arguments)

    return float64(float64, float64, float64), codegen


@njit(cache=True, fastmath=False, inline="always")
def _two_product(a, b):
    value = a * b
    return value, _fma(a, b, -value)


@njit(cache=True, fastmath=False)
def ordered_sum(values_high, values_low):
    rows, columns = values_high.shape
    high = np.zeros(columns)
    low = np.zeros(columns)
    for column in range(columns):
        h, l = 0.0, 0.0
        for row in range(rows):
            ah, al = _two_sum(values_high[row, column], values_low[row, column])
            nh, nl = _two_sum(h, ah)
            nl += l + al
            h, l = _two_sum(nh, nl)
        high[column], low[column] = h, l
    return high, low


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


@njit(cache=True, fastmath=False, inline='always')
def _add_expansion(ah, al, bh, bl):
    h, l = _two_sum(ah, bh)
    l += al + bl
    return _two_sum(h, l)


@njit(cache=True, fastmath=False, inline='always')
def _multiply_expansion(ah, al, bh, bl):
    h, l = _two_product(ah, bh)
    l += ah * bl + al * bh
    l += al * bl
    return _two_sum(h, l)


@njit(cache=True, fastmath=False, inline='always')
def _divide_expansion(ah, al, bh, bl):
    h, l = ah / bh, 0.0
    for _ in range(2):
        ph, pl = _multiply_expansion(bh, bl, h, l)
        rh, rl = _add_expansion(ah, al, -ph, -pl)
        h, l = _add_expansion(h, l, rh / bh, 0.0)
    return h, l


@njit(cache=True, fastmath=False)
def lu_factor(high, low, rows, singletons):
    high, low, pivots = high.copy(), low.copy(), rows.copy()
    n = len(rows)
    for k in range(n):
        pivot = k
        if k >= singletons:
            largest = abs(high[k, k])
            for i in range(k + 1, n):
                if abs(high[i, k]) > largest:
                    pivot, largest = i, abs(high[i, k])
        if high[pivot, k] == 0.0:
            return high, low, pivots, k
        if pivot != k:
            for j in range(n):
                high[k, j], high[pivot, j] = high[pivot, j], high[k, j]
                low[k, j], low[pivot, j] = low[pivot, j], low[k, j]
            pivots[k], pivots[pivot] = pivots[pivot], pivots[k]
        for i in range(k + 1, n):
            if high[i, k] == 0.0 and low[i, k] == 0.0:
                continue
            ch, cl = _divide_expansion(high[i, k], low[i, k], high[k, k], low[k, k])
            if k >= singletons:
                for j in range(k + 1, n):
                    ph, pl = _multiply_expansion(ch, cl, high[k, j], low[k, j])
                    high[i, j], low[i, j] = _add_expansion(high[i, j], low[i, j], -ph, -pl)
            high[i, k], low[i, k] = ch, cl
    return high, low, pivots, -1


@njit(cache=True, fastmath=False)
def lu_substitute(lh, ll, high, low):
    high, low = high.copy(), low.copy()
    n, columns = high.shape
    for k in range(n):
        for j in range(columns):
            h, l = 0.0, 0.0
            for i in range(k):
                if (lh[k, i] == 0.0 and ll[k, i] == 0.0) or (high[i, j] == 0.0 and low[i, j] == 0.0):
                    continue
                ph, pl = _multiply_expansion(lh[k, i], ll[k, i], high[i, j], low[i, j])
                h, l = _add_expansion(h, l, ph, pl)
            high[k, j], low[k, j] = _add_expansion(high[k, j], low[k, j], -h, -l)
    for k in range(n - 1, -1, -1):
        for j in range(columns):
            h, l = 0.0, 0.0
            for i in range(k + 1, n):
                if (lh[k, i] == 0.0 and ll[k, i] == 0.0) or (high[i, j] == 0.0 and low[i, j] == 0.0):
                    continue
                ph, pl = _multiply_expansion(lh[k, i], ll[k, i], high[i, j], low[i, j])
                h, l = _add_expansion(h, l, ph, pl)
            rh, rl = _add_expansion(high[k, j], low[k, j], -h, -l)
            high[k, j], low[k, j] = _divide_expansion(rh, rl, lh[k, k], ll[k, k])
    return high, low


@njit(cache=True, fastmath=False, inline='always')
def _accumulate_four(p0, p1, p2, p3, value):
    p0, value = _two_sum(p0, value)
    p1, value = _two_sum(p1, value)
    p2, value = _two_sum(p2, value)
    p3, value = _two_sum(p3, value)
    return p0, p1, p2, p3


@njit(cache=True, fastmath=False)
def matrix_residual(ah, al, xh, xl, bh, bl):
    rows, inner = ah.shape
    columns = xh.shape[1]
    high = np.zeros((rows, columns))
    low = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            p0, p1, p2, p3 = _accumulate_four(0., 0., 0., 0., bh[i, j])
            p0, p1, p2, p3 = _accumulate_four(p0, p1, p2, p3, bl[i, j])
            for k in range(inner):
                for left in (ah[i, k], al[i, k]):
                    if left == 0.:
                        continue
                    for right in (xh[k, j], xl[k, j]):
                        if right == 0.:
                            continue
                        h, l = _two_product(left, right)
                        p0, p1, p2, p3 = _accumulate_four(p0, p1, p2, p3, -h)
                        p0, p1, p2, p3 = _accumulate_four(p0, p1, p2, p3, -l)
            h, l = _two_sum(p0, p1)
            for part in (p2, p3):
                nh, nl = _two_sum(h, part)
                nl += l
                h, l = _two_sum(nh, nl)
            high[i, j], low[i, j] = h, l
    return high, low


# Numba stores gufunc cores and wrappers separately; mixed cache generations can
# leave a wrapper calling an unresolved native symbol in a fresh process.
@guvectorize([(float64, float64, float64[:], float64[:])], "(),()->(),()", nopython=True, cache=False)
def array_sum(a, b, high, low):
    high[0], low[0] = _two_sum(a, b)


@guvectorize([(float64, float64, float64[:], float64[:])], "(),()->(),()", nopython=True, cache=False)
def array_product(a, b, high, low):
    high[0], low[0] = _two_product(a, b)


@guvectorize(
    [(float64, float64, float64, float64, float64[:], float64[:])],
    "(),(),(),()->(),()",
    nopython=True,
    cache=False,
)
def array_divide(ah, al, bh, bl, high, low):
    high[0], low[0] = _divide_expansion(ah, al, bh, bl)
