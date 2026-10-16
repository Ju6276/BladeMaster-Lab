"""CUDA moving-segment CCD for the linked-volume surface."""

import numpy as np
import warp as wp

# Discrete surface intersection queries do not participate in autodiff.
wp.set_module_options({"enable_backward": False})

Candidates = wp.types.vector(length=64, dtype=wp.float64)


@wp.func
def polynomial(c: wp.vec4d, t: wp.float64):
    return ((c[3] * t + c[2]) * t + c[1]) * t + c[0]


@wp.func
def quadratic(a: wp.float64, b: wp.float64, c: wp.float64):
    values = wp.vec2d(wp.float64(0.0))
    count = int(0)
    if a == wp.float64(0.0):
        if b != wp.float64(0.0):
            values[0] = -c / b
            count = 1
    else:
        disc = b * b - wp.float64(4.0) * a * c
        if disc >= wp.float64(0.0):
            sign = wp.float64(1.0)
            if b < wp.float64(0.0):
                sign = wp.float64(-1.0)
            q = -wp.float64(0.5) * (b + sign * wp.sqrt(disc))
            if q == wp.float64(0.0):
                values[0] = -b / (wp.float64(2.0) * a)
                count = 1
            else:
                values[0] = q / a
                values[1] = c / q
                count = 2
    return values, count


@wp.func
def roots(c: wp.vec4d, relative_tolerance: wp.float64):
    threshold = wp.max(wp.max(wp.abs(c[0]), wp.abs(c[1])), wp.max(wp.abs(c[2]), wp.abs(c[3])))
    threshold = wp.max(threshold * relative_tolerance, wp.float64(1.0e-25))
    degree = int(0)
    for power_index in range(1, 4):
        if wp.abs(c[power_index]) > threshold:
            degree = power_index
    r = wp.vec3d(0.0)
    count = int(0)
    if degree == 1:
        r[0] = -c[0] / c[1]
        count = 1
    elif degree == 2:
        disc = c[1] * c[1] - wp.float64(4.0) * c[2] * c[0]
        if disc >= wp.float64(0.0):
            sign = wp.float64(1.0)
            if c[1] < wp.float64(0.0):
                sign = wp.float64(-1.0)
            q = -wp.float64(0.5) * (c[1] + sign * wp.sqrt(disc))
            if q == wp.float64(0.0):
                r[0] = -c[1] / (wp.float64(2.0) * c[2])
                count = 1
            else:
                r[0] = q / c[2]
                r[1] = c[0] / q
                count = 2
    elif degree == 3:
        # Only roots in the observation interval can be intersections. Split
        # at derivative roots, then solve each monotonic interval directly.
        lo = wp.float64(-1.0e-9)
        hi = wp.float64(1.0) + wp.float64(1.0e-9)
        boundaries = wp.vec4d(lo, hi, hi, hi)
        length = int(2)
        stationary, n = quadratic(wp.float64(3.0) * c[3], wp.float64(2.0) * c[2], c[1])
        for j in range(n):
            if stationary[j] > lo and stationary[j] < hi:
                boundaries[length] = stationary[j]
                length += 1
        for j in range(1, length):
            value = boundaries[j]
            k = j
            while k > 0 and boundaries[k - 1] > value:
                boundaries[k] = boundaries[k - 1]
                k -= 1
            boundaries[k] = value
        scale = wp.abs(c[0]) + wp.abs(c[1]) + wp.abs(c[2]) + wp.abs(c[3])
        # Roundoff tolerance for repeated roots, followed by the same geometric
        # residual check as the CPU implementation (not a contact thickness).
        epsilon = wp.float64(2.0e-15) * scale
        for j in range(length):
            if wp.abs(polynomial(c, boundaries[j])) <= epsilon and count < 3:
                if count == 0 or wp.abs(boundaries[j] - r[count - 1]) > wp.float64(1.0e-14):
                    r[count] = boundaries[j]
                    count += 1
        for j in range(length - 1):
            left = boundaries[j]
            right = boundaries[j + 1]
            fl = polynomial(c, left)
            fr = polynomial(c, right)
            if fl * fr < wp.float64(0.0) and wp.abs(fl) > epsilon and wp.abs(fr) > epsilon:
                for iteration in range(64):
                    middle = (left + right) / wp.float64(2.0)
                    fm = polynomial(c, middle)
                    if fl * fm > wp.float64(0.0):
                        left = middle
                        fl = fm
                    else:
                        right = middle
                if count < 3:
                    r[count] = (left + right) / wp.float64(2.0)
                    count += 1
    return r, count


@wp.func
def add_candidate(values: Candidates, count: int, value: wp.float64):
    if value >= wp.float64(-1.0e-9) and value <= wp.float64(1.0) + wp.float64(1.0e-9):
        values[count] = wp.clamp(value, wp.float64(0.0), wp.float64(1.0))
        count += 1
    return values, count


@wp.func
def segment_coordinates(u: wp.vec3d, v: wp.vec3d, w: wp.vec3d):
    # Least squares for [u,-v] @ [alpha,beta] == w, including rank-one lines.
    a, b, c = wp.dot(u, u), -wp.dot(u, v), wp.dot(v, v)
    d, e = wp.dot(u, w), -wp.dot(v, w)
    determinant = a * c - b * b
    alpha, beta = wp.float64(0.0), wp.float64(0.0)
    trace = a + c
    if determinant > wp.float64(1.0e-28) * trace * trace:
        alpha, beta = (c * d - b * e) / determinant, (a * e - b * d) / determinant
    elif trace > wp.float64(0.0):
        # For rank one, G^+ = G / trace(G)^2.
        alpha, beta = (a * d + b * e) / (trace * trace), (b * d + c * e) / (trace * trace)
    return alpha, beta


@wp.kernel
def moving_hits(
    a0: wp.array(dtype=wp.vec3d),
    b0: wp.array(dtype=wp.vec3d),
    a1: wp.array(dtype=wp.vec3d),
    b1: wp.array(dtype=wp.vec3d),
    blade: wp.array(dtype=wp.vec3d),
    hit: wp.array(dtype=int),
    alphas: wp.array(dtype=wp.float64),
    times: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    hit[i] = 0
    alphas[i] = wp.float64(0.0)
    times[i] = wp.float64(0.0)
    c0, d0, c1, d1 = blade[0], blade[1], blade[2], blade[3]
    for axis in range(3):
        lo = wp.min(wp.min(c0[axis], d0[axis]), wp.min(c1[axis], d1[axis]))
        hi = wp.max(wp.max(c0[axis], d0[axis]), wp.max(c1[axis], d1[axis]))
        low = wp.min(wp.min(a0[i][axis], b0[i][axis]), wp.min(a1[i][axis], b1[i][axis]))
        high = wp.max(wp.max(a0[i][axis], b0[i][axis]), wp.max(a1[i][axis], b1[i][axis]))
        if high < lo - wp.float64(1.0e-10) or low > hi + wp.float64(1.0e-10):
            return
    v0 = d0 - c0
    dv = (d1 - c1) - v0
    u0 = b0[i] - a0[i]
    du = (b1[i] - a1[i]) - u0
    w0 = c0 - a0[i]
    dw = c1 - a1[i] - w0
    cross0 = wp.cross(u0, v0)
    cross1 = wp.cross(du, v0) + wp.cross(u0, dv)
    cross2 = wp.cross(du, dv)
    coeff = wp.vec4d(
        wp.dot(cross0, w0),
        wp.dot(cross0, dw) + wp.dot(cross1, w0),
        wp.dot(cross1, dw) + wp.dot(cross2, w0),
        wp.dot(cross2, dw),
    )
    maximum = wp.max(wp.max(wp.abs(coeff[0]), wp.abs(coeff[1])), wp.max(wp.abs(coeff[2]), wp.abs(coeff[3])))
    candidates = Candidates(wp.float64(0.0))
    count = int(0)
    if maximum > wp.float64(1.0e-25):
        values, n = roots(coeff, wp.float64(1.0e-12))
        for j in range(n):
            candidates, count = add_candidate(candidates, count, values[j])
    else:
        candidates[0] = wp.float64(0.0)
        candidates[1] = wp.float64(1.0)
        count = int(2)
        for pair in range(4):
            p0, dp, q0, dq = w0, dw, u0, du
            if pair == 1:
                p0, dp = w0 + v0, dw + dv
            elif pair == 2:
                p0, dp, q0, dq = -w0, -dw, v0, dv
            elif pair == 3:
                p0, dp, q0, dq = u0 - w0, du - dw, v0, dv
            cc0, cc1, cc2 = wp.cross(p0, q0), wp.cross(dp, q0) + wp.cross(p0, dq), wp.cross(dp, dq)
            for axis in range(3):
                values, n = roots(wp.vec4d(cc0[axis], cc1[axis], cc2[axis], wp.float64(0.0)), wp.float64(0.0))
                for j in range(n):
                    candidates, count = add_candidate(candidates, count, values[j])
        # At most 2 + 4*3*2 = 26 candidates; adding interval middles fits 64.
        for j in range(1, count):
            value = candidates[j]
            k = j
            while k > 0 and candidates[k - 1] > value:
                candidates[k] = candidates[k - 1]
                k -= 1
            candidates[k] = value
        original_count = count
        for j in range(original_count - 1):
            candidates[count] = (candidates[j] + candidates[j + 1]) / wp.float64(2.0)
            count += 1
    best = wp.float64(2.0)
    for j in range(count):
        t = candidates[j]
        u, v, w = u0 + t * du, v0 + t * dv, w0 + t * dw
        alpha, beta = segment_coordinates(u, v, w)
        if (
            wp.length(u * alpha - v * beta - w) <= wp.float64(1.0e-9)
            and alpha >= wp.float64(-1.0e-8)
            and alpha <= wp.float64(1.0) + wp.float64(1.0e-8)
            and beta >= wp.float64(-1.0e-8)
            and beta <= wp.float64(1.0) + wp.float64(1.0e-8)
            and t < best
        ):
            hit[i] = 1
            alphas[i] = alpha
            times[i] = t
            best = t


class CudaSegmentCCD:
    def __init__(self, device="cuda:0"):
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("CUDA segment CCD requires a CUDA device")
        self.capacity = -1

    def __call__(self, a0, b0, a1, b1, edge0, edge1):
        arrays = [np.ascontiguousarray(v, dtype=np.float64) for v in (a0, b0, a1, b1)]
        n = len(arrays[0])
        if any(v.shape != (n, 3) or not np.isfinite(v).all() for v in arrays):
            raise ValueError("Finite moving segment endpoints are required")
        blade = np.asarray([*edge0, *edge1], dtype=np.float64)
        if blade.shape != (4, 3) or not np.isfinite(blade).all():
            raise ValueError("Expected two finite blade endpoints per observation")
        if not n:
            return np.zeros(0, bool), np.zeros(0), np.zeros(0)
        if self.capacity != n:
            self.inputs = [wp.empty(n, dtype=wp.vec3d, device=self.device) for _ in range(4)]
            self.blade = wp.empty(4, dtype=wp.vec3d, device=self.device)
            self.hit = wp.empty(n, dtype=int, device=self.device)
            self.alpha = wp.empty(n, dtype=wp.float64, device=self.device)
            self.times = wp.empty(n, dtype=wp.float64, device=self.device)
            self.capacity = n
        for target, value in zip(self.inputs, arrays):
            target.assign(value)
        self.blade.assign(blade)
        wp.launch(
            moving_hits, n, [*self.inputs, self.blade, self.hit, self.alpha, self.times], device=self.device
        )
        return self.hit.numpy().astype(bool), self.alpha.numpy(), self.times.numpy()
