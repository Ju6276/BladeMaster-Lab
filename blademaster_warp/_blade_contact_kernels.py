"""Two-component forward derivatives for the 24 finite-blade query inputs."""

import numpy as np
from numba import njit

from ._compensated_kernels import (
    _add_expansion as plus,
    _multiply_expansion as times,
    _divide_expansion as quotient,
)


@njit(cache=True, fastmath=False)
def constant(value):
    result = np.zeros((2, 25))
    result[0, 0] = value
    return result


@njit(cache=True, fastmath=False)
def add(a, b):
    result = np.zeros_like(a)
    for i in range(25):
        result[0, i], result[1, i] = plus(a[0, i], a[1, i], b[0, i], b[1, i])
    return result


@njit(cache=True, fastmath=False)
def negative(a):
    return -a


@njit(cache=True, fastmath=False)
def subtract(a, b):
    return add(a, negative(b))


@njit(cache=True, fastmath=False)
def multiply(a, b):
    result = np.zeros_like(a)
    result[0, 0], result[1, 0] = times(a[0, 0], a[1, 0], b[0, 0], b[1, 0])
    for i in range(1, 25):
        ah, al = times(a[0, i], a[1, i], b[0, 0], b[1, 0])
        bh, bl = times(a[0, 0], a[1, 0], b[0, i], b[1, i])
        result[0, i], result[1, i] = plus(ah, al, bh, bl)
    return result


@njit(cache=True, fastmath=False)
def divide(a, b):
    result = np.zeros_like(a)
    h, l = quotient(a[0, 0], a[1, 0], b[0, 0], b[1, 0])
    result[0, 0], result[1, 0] = h, l
    for i in range(1, 25):
        ph, pl = times(h, l, b[0, i], b[1, i])
        rh, rl = plus(a[0, i], a[1, i], -ph, -pl)
        result[0, i], result[1, i] = quotient(rh, rl, b[0, 0], b[1, 0])
    return result


@njit(cache=True, fastmath=False)
def less(a, b):
    h, l = plus(b[0, 0], b[1, 0], -a[0, 0], -a[1, 0])
    return h > 0.0 or (h == 0.0 and l > 0.0)


@njit(cache=True, fastmath=False)
def absolute(a):
    return negative(a) if less(a, constant(0.0)) else a


@njit(cache=True, fastmath=False)
def square_root(a):
    first = np.sqrt(a[0, 0])
    if first == 0.0:
        raise ValueError('A differentiated square root requires a positive argument')
    ph, pl = times(first, 0.0, first, 0.0)
    rh, rl = plus(a[0, 0], a[1, 0], -ph, -pl)
    dh, dl = times(2.0, 0.0, first, 0.0)
    ch, cl = quotient(rh, rl, dh, dl)
    h, l = plus(first, 0.0, ch, cl)
    result = np.zeros_like(a)
    result[0, 0], result[1, 0] = h, l
    dh, dl = times(2.0, 0.0, h, l)
    for i in range(1, 25):
        result[0, i], result[1, i] = quotient(a[0, i], a[1, i], dh, dl)
    return result


@njit(cache=True, fastmath=False)
def clamp(a, lower, upper):
    if less(a, lower):
        return lower
    if less(upper, a):
        return upper
    return a


@njit(cache=True, fastmath=False)
def dot(a, b):
    result = constant(0.0)
    for i in range(len(a)):
        result = add(result, multiply(a[i], b[i]))
    return result


@njit(cache=True, fastmath=False)
def cross(a, b):
    return [subtract(multiply(a[1], b[2]), multiply(a[2], b[1])),
            subtract(multiply(a[2], b[0]), multiply(a[0], b[2])),
            subtract(multiply(a[0], b[1]), multiply(a[1], b[0]))]


@njit(cache=True, fastmath=False)
def rotate(q, vector):
    first = cross(q[:3], vector)
    second = cross(q[:3], first)
    return [add(vector[i], multiply(constant(2.0), add(multiply(q[3], first[i]), second[i])))
            for i in range(3)]


@njit(cache=True, fastmath=False)
def closest_blade(local, half, bevel):
    zero = constant(0.0)
    closest = [clamp(local[i], negative(half[i]), half[i]) for i in range(3)]
    delta = [subtract(local[i], closest[i]) for i in range(3)]
    distance_sq = dot(delta, delta)
    if bevel[0, 0] == 0.0 and bevel[1, 0] == 0.0:
        if less(zero, distance_sq):
            distance = square_root(distance_sq)
            normal = [divide(d, distance) for d in delta]
        else:
            margin = [subtract(half[i], absolute(local[i])) for i in range(3)]
            axis = 0
            for i in range(1, 3):
                if less(margin[i], margin[axis]):
                    axis = i
            normal = [zero, zero, zero]
            normal[axis] = constant(-1.0 if less(local[axis], zero) else 1.0)
            closest[axis] = multiply(normal[axis], half[axis])
            distance = negative(margin[axis])
        return closest, normal, distance
    sign = constant(-1.0 if less(local[0], zero) else 1.0)
    q = [absolute(local[0]), local[2]]
    edge = [half[0], bevel]
    edge_length_sq = dot(edge, edge)
    t = clamp(divide(dot([q[0], add(q[1], half[2])], edge), edge_length_sq), zero, constant(1.0))
    near = [multiply(t, half[0]), add(negative(half[2]), multiply(t, bevel))]
    delta = [subtract(q[i], near[i]) for i in range(2)]
    best_sq = dot(delta, delta)
    length = square_root(edge_length_sq)
    outward = [divide(bevel, length), divide(negative(half[0]), length)]
    candidates = [
        ([half[0], clamp(q[1], add(negative(half[2]), bevel), half[2])], [constant(1.0), zero]),
        ([clamp(q[0], zero, half[0]), half[2]], [zero, constant(1.0)]),
    ]
    for candidate, normal in candidates:
        delta = [subtract(q[i], candidate[i]) for i in range(2)]
        squared = dot(delta, delta)
        if less(squared, best_sq):
            best_sq, near, outward = squared, candidate, normal
    inside = (not less(half[0], q[0]) and not less(half[2], q[1])
              and not less(multiply(half[0], add(q[1], half[2])), multiply(bevel, q[0])))
    near = [multiply(sign, near[0]), near[1]]
    outward = [multiply(sign, outward[0]), outward[1]]
    closest = [near[0], clamp(local[1], negative(half[1]), half[1]), near[1]]
    if inside and not less(half[1], absolute(local[1])):
        normal = [outward[0], zero, outward[1]]
        if less(zero, best_sq):
            distance = negative(square_root(best_sq))
        else:
            distance = dot([subtract(local[i], closest[i]) for i in range(3)], normal)
        if less(subtract(half[1], absolute(local[1])), negative(distance)):
            side = constant(-1.0 if less(local[1], zero) else 1.0)
            closest = [local[0], multiply(side, half[1]), local[2]]
            normal = [zero, side, zero]
            distance = subtract(absolute(local[1]), half[1])
    else:
        if inside:
            closest[0], closest[2] = local[0], local[2]
        delta = [subtract(local[i], closest[i]) for i in range(3)]
        squared = dot(delta, delta)
        if less(zero, squared):
            distance = square_root(squared)
            normal = [divide(d, distance) for d in delta]
        else:
            normal = [outward[0], zero, outward[1]]
            distance = dot(delta, normal)
    return closest, normal, distance


@njit(cache=True, fastmath=False)
def query_kernel(high, low):
    variables = []
    for i in range(24):
        value = constant(high[i])
        value[1, 0] = low[i]
        value[0, i + 1] = 1.0
        variables.append(value)
    x, v, o, q = variables[:3], variables[3:6], variables[6:9], variables[9:13]
    vt, w, half = variables[13:16], variables[16:19], variables[19:22]
    bevel, offset = variables[22], variables[23]
    local = rotate([negative(q[0]), negative(q[1]), negative(q[2]), q[3]],
                   [subtract(x[i], o[i]) for i in range(3)])
    near, normal, distance = closest_blade(local, half, bevel)
    lever, normal = rotate(q, near), rotate(q, normal)
    closest = [add(o[i], lever[i]) for i in range(3)]
    spin = cross(w, lever)
    relative = [subtract(subtract(v[i], vt[i]), spin[i]) for i in range(3)]
    gap = subtract(distance, offset)
    shifted = [multiply(gap, n) for n in normal]
    output = shifted + normal + relative + closest + [gap]
    out_high, out_low = np.zeros((13, 25)), np.zeros((13, 25))
    for i in range(13):
        out_high[i], out_low[i] = output[i][0], output[i][1]
    return out_high, out_low
