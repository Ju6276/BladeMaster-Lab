"""Isochoric plastic increments minimizing fixed-corotated energy plus dissipation."""

import warp as wp


@wp.func
def deviator(value: wp.vec3d):
    return value - wp.vec3d((value[0] + value[1] + value[2]) / wp.float64(3.0))


@wp.func
def stretches_at(trial: wp.vec3d, increment: wp.vec3d):
    return wp.cw_mul(trial, wp.vec3d(wp.exp(increment[0]), wp.exp(increment[1]), wp.exp(increment[2])))


@wp.func
def expm1(value: wp.float64):
    if wp.abs(value) < wp.float64(1.0e-3):
        return value * (wp.float64(1.0) + value * (wp.float64(0.5) + value * (
            wp.float64(1.0 / 6.0) + value * (wp.float64(1.0 / 24.0) + value * wp.float64(1.0 / 120.0)))))
    return wp.exp(value) - wp.float64(1.0)


@wp.func
def exponential_remainder(value: wp.float64):
    if wp.abs(value) < wp.float64(1.0e-3):
        return value * value * (wp.float64(0.5) + value * (wp.float64(1.0 / 6.0)
                + value * (wp.float64(1.0 / 24.0) + value * wp.float64(1.0 / 120.0))))
    return wp.exp(value) - wp.float64(1.0) - value


@wp.func
def incremental_energy_change(stretches: wp.vec3d, increment: wp.vec3d, step: wp.vec3d, radius: wp.float64):
    # The increment has zero trace; separate linear and higher-order terms so
    # hydrostatic stress and near-stationary plastic work do not lose digits.
    ds = wp.cw_mul(stretches, wp.vec3d(expm1(step[0]), expm1(step[1]), expm1(step[2])))
    remainder = wp.cw_mul(stretches, wp.vec3d(exponential_remainder(step[0]),
                                            exponential_remainder(step[1]), exponential_remainder(step[2])))
    elastic = (wp.dot(stretches - wp.vec3d(wp.float64(1.0)), remainder)
               + wp.float64(0.5) * wp.dot(ds, ds))
    gradient = deviator(wp.cw_mul(stretches, stretches - wp.vec3d(wp.float64(1.0))))
    plastic = radius * wp.length(step)
    norm = wp.length(increment)
    if norm > wp.float64(0.0):
        unit = increment / norm
        gradient += radius * unit
        parallel = wp.dot(unit, step)
        perpendicular = step - parallel * unit
        next_norm = wp.length(increment + step)
        denominator = next_norm + norm + parallel
        plastic = radius * (next_norm - norm - parallel)
        if denominator > wp.float64(0.0):
            plastic = radius * wp.dot(perpendicular, perpendicular) / denominator
    return wp.dot(gradient, step) + elastic + plastic


@wp.func
def return_gradient(stretches: wp.vec3d, increment: wp.vec3d, radius: wp.float64):
    stress = wp.cw_mul(stretches, stretches - wp.vec3d(wp.float64(1.0)))
    return deviator(stress) + radius * increment / wp.length(increment)


@wp.func
def return_inverse(stretches: wp.vec3d, increment: wp.vec3d, radius: wp.float64, regularize: int):
    # In the radial/transverse basis, r/|d| appears only in the transverse
    # entry. Forming P-u*u^T first loses radial curvature as |d| approaches 0.
    norm = wp.length(increment)
    u = increment / norm
    v = wp.cross(wp.vec3d(wp.sqrt(wp.float64(1.0 / 3.0))), u)
    curvature = wp.cw_mul(stretches, wp.float64(2.0) * stretches - wp.vec3d(wp.float64(1.0)))
    a = wp.dot(u, wp.cw_mul(curvature, u))
    b = wp.dot(v, wp.cw_mul(curvature, v)) + radius / norm
    c = wp.dot(u, wp.cw_mul(curvature, v))
    if regularize != 0:
        discriminant = wp.sqrt((a - b) * (a - b) + wp.float64(4.0) * c * c)
        largest = wp.float64(0.5) * (a + b + discriminant)
        smallest = wp.float64(0.5) * (a + b - discriminant)
        if largest > wp.float64(0.0):
            smallest = (a * b - c * c) / largest
        floor = wp.float64(1.0e-6) * wp.max(wp.float64(1.0), wp.max(wp.abs(curvature)))
        shift = wp.max(wp.float64(0.0), floor - smallest)
        a += shift
        b += shift
    ratio = c / b
    radial_inverse = wp.float64(1.0) / (a - c * ratio)
    direction = u - ratio * v
    return radial_inverse * wp.outer(direction, direction) + wp.outer(v, v) / b


@wp.func
def return_state(trial: wp.vec3d, radius: wp.float64):
    """radius = sqrt(2/3) sigma_y / (2 mu); caller checks convergence before using the state or VJP."""
    valid = int(1)
    for i in range(3):
        if trial[i] <= wp.float64(0.0) or not wp.isfinite(trial[i]):
            valid = int(0)
    if radius < wp.float64(0.0) or not wp.isfinite(radius):
        valid = int(0)
    increment = wp.vec3d(wp.float64(0.0))
    if valid == 0:
        return trial, increment, int(0)
    trial_gradient = deviator(wp.cw_mul(trial, trial - wp.vec3d(wp.float64(1.0))))
    norm = wp.length(trial_gradient)
    if norm <= radius:
        return trial, increment, int(1)
    curvature = wp.cw_mul(trial, wp.float64(2.0) * trial - wp.vec3d(wp.float64(1.0)))
    scale = wp.max(wp.float64(1.0), wp.max(wp.abs(curvature)))
    direction = -trial_gradient / norm
    length = (norm - radius) / scale
    accepted = int(0)
    for _ in range(48):
        candidate = deviator(length * direction)
        change = incremental_energy_change(trial, increment, candidate, radius)
        if wp.isfinite(change) and change <= wp.float64(1.0e-4) * length * (radius - norm):
            increment = candidate
            accepted = int(1)
            break
        length *= wp.float64(0.5)
    if accepted == 0:
        return trial, increment, int(0)
    tolerance = wp.float64(2.0e-13) * wp.max(wp.float64(1.0), norm)
    result = trial
    converged = int(0)
    for _ in range(80):
        result = stretches_at(trial, increment)
        gradient = return_gradient(result, increment, radius)
        if wp.length(gradient) <= tolerance:
            converged = int(1)
            break
        direction = deviator(-return_inverse(result, increment, radius, 1) * gradient)
        slope = wp.dot(gradient, direction)
        if slope >= wp.float64(0.0) or not wp.isfinite(slope):
            break
        length = wp.float64(1.0)
        accepted = int(0)
        for _ in range(48):
            candidate = deviator(increment + length * direction)
            change = incremental_energy_change(result, increment, candidate - increment, radius)
            if wp.isfinite(change) and change <= wp.float64(1.0e-4) * length * slope:
                increment = candidate
                accepted = int(1)
                break
            length *= wp.float64(0.5)
        if accepted == 0:
            break
    return result, increment, converged


@wp.func
def corotated_return(trial: wp.vec3d, radius: wp.float64):
    projected, increment, success = return_state(trial, radius)
    return projected, success


@wp.func
def return_derivatives(trial: wp.vec3d, projected: wp.vec3d, delta: wp.vec3d, radius: wp.float64):
    identity = wp.identity(n=3, dtype=wp.float64)
    ones = wp.vec3d(wp.float64(1.0))
    stress = deviator(wp.cw_mul(trial, trial - ones))
    if wp.length(stress) <= radius:
        return identity, wp.vec3d(wp.float64(0.0))
    spherical = wp.outer(ones, ones) / wp.float64(3.0)
    projector = identity - spherical
    inverse = return_inverse(projected, delta, radius, 0)
    curvature = wp.cw_mul(projected, wp.float64(2.0) * projected - ones)
    log_derivative = identity - inverse * projector * wp.diag(curvature)
    derivative = wp.diag(projected) * log_derivative * wp.diag(wp.cw_div(ones, trial))
    radius_derivative = -wp.cw_mul(projected, inverse * (delta / wp.length(delta)))
    return derivative, radius_derivative


@wp.func_grad(corotated_return)
def corotated_return_grad(trial: wp.vec3d, radius: wp.float64, adj_result: wp.vec3d, adj_success: int):
    projected, delta, success = return_state(trial, radius)
    derivative, radius_derivative = return_derivatives(trial, projected, delta, radius)
    wp.adjoint[trial] += wp.transpose(derivative) * adj_result
    wp.adjoint[radius] += wp.dot(radius_derivative, adj_result)


@wp.func
def log_divided(a: wp.float64, b: wp.float64):
    ratio = (a - b) / (a + b)
    square = ratio * ratio
    if square < wp.float64(1.0e-4):
        series = wp.float64(1.0) + square * (wp.float64(1.0 / 3.0) + square * (
            wp.float64(1.0 / 5.0) + square * (wp.float64(1.0 / 7.0) + square * wp.float64(1.0 / 9.0))))
        return wp.float64(2.0) * series / (a + b)
    return wp.log(a / b) / (a - b)


@wp.func
def corotated_projection(elastic: wp.mat33d, radius: wp.float64):
    U, s, V = wp.svd3(elastic)
    projected, success = corotated_return(s, radius)
    return U * wp.diag(projected) * wp.transpose(V), success


@wp.func
def projection_vjp(elastic: wp.mat33d, radius: wp.float64, adj_projected: wp.mat33d):
    U, s, V = wp.svd3(elastic)
    projected, delta, success = return_state(s, radius)
    derivative, radius_derivative = return_derivatives(s, projected, delta, radius)
    local = wp.transpose(U) * adj_projected * V
    diagonal = wp.vec3d(local[0, 0], local[1, 1], local[2, 2])
    result = wp.diag(wp.transpose(derivative) * diagonal)
    norm = wp.length(delta)
    active = wp.length(deviator(wp.cw_mul(s, s - wp.vec3d(wp.float64(1.0))))) > radius
    for i in range(3):
        for j in range(3):
            if i != j:
                symmetric = wp.float64(1.0)
                if active:
                    # Subtract the two stationarity equations to evaluate
                    # (t_i-t_j)/(s_i-s_j) also at repeated singular values.
                    symmetric = radius * log_divided(s[i], s[j]) / (
                        radius * log_divided(projected[i], projected[j])
                        + norm * (projected[i] + projected[j] - wp.float64(1.0)))
                skew = (projected[i] + projected[j]) / (s[i] + s[j])
                result[i, j] = wp.float64(0.5) * (symmetric * (local[i, j] + local[j, i])
                                                 + skew * (local[i, j] - local[j, i]))
    return U * result * wp.transpose(V), wp.dot(radius_derivative, diagonal)


@wp.func_grad(corotated_projection)
def corotated_projection_grad(elastic: wp.mat33d, radius: wp.float64,
                               adj_projected: wp.mat33d, adj_success: int):
    derivative, radius_derivative = projection_vjp(elastic, radius, adj_projected)
    wp.adjoint[elastic] += derivative
    wp.adjoint[radius] += radius_derivative
