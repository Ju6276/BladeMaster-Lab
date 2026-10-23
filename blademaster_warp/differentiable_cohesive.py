"""Moving cohesive-interface equilibrium and its implicit Warp adjoint.

The pressure interface uses Pa and N/m; the integrated interface uses N and
N m. Both return energy densities per reference area and particle displacements.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from .cohesive import (
    cohesive_proximal_response,
    cohesive_measure_proximal_response,
    cohesive_contact_response,
    surface_normal,
    surface_normal_pullback,
)
from .precision import launch_double
from ._cohesive_solve import geometry, shared_intact_rows, solve_primal, refine_geometry


@wp.func
def gap_at(q: wp.array(dtype=wp.vec3), b: wp.array2d(dtype=float), u: wp.array(dtype=wp.vec3), i: int):
    correction = wp.vec3(0.0)
    for p in range(u.shape[0]):
        correction += b[i, p] * u[p]
    return q[i] + correction


@wp.func
def residual_force(
    b: wp.array2d(dtype=float),
    area: wp.array(dtype=float),
    traction: wp.array(dtype=wp.vec3),
    p: int,
    integrated_force: int,
):
    f = wp.vec3(0.0)
    for i in range(traction.shape[0]):
        weight = float(1.0)
        if integrated_force == 0:
            weight = area[i]
        f += b[i, p] * weight * traction[i]
    return f


@wp.func
def frame_rate(gradient: wp.array2d(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3), i: int):
    rate = wp.mat33(0.0)
    for p in range(velocity.shape[0]):
        rate += wp.outer(velocity[p], gradient[i, p])
    return rate


@wp.func
def frame_force(
    gradient: wp.array2d(dtype=wp.vec3),
    stress: wp.array(dtype=wp.mat33),
    area: wp.array(dtype=float),
    p: int,
    integrated_force: int,
):
    force = wp.vec3(0.0)
    for i in range(stress.shape[0]):
        weight = float(1.0)
        if integrated_force == 0:
            weight = area[i]
        force += weight * stress[i] * gradient[i, p]
    return force


@wp.kernel
def monolithic_momentum(
    b: wp.array2d(dtype=float),
    gradient: wp.array2d(dtype=wp.vec3),
    area: wp.array(dtype=float),
    traction: wp.array(dtype=wp.vec3),
    stress: wp.array(dtype=wp.mat33),
    mass: wp.array(dtype=float),
    dt: wp.array(dtype=float),
    u: wp.array(dtype=wp.vec3),
    integrated_force: int,
    residual: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    force = residual_force(b, area, traction, p, integrated_force)
    force += frame_force(gradient, stress, area, p, integrated_force)
    residual[p] = u[p] + dt[0] * dt[0] / mass[0] * force


@wp.kernel
def kinematics(
    q: wp.array(dtype=wp.vec3),
    free_frame: wp.array(dtype=wp.mat33),
    reference: wp.array(dtype=wp.vec3),
    history: wp.array(dtype=float),
    material: wp.array(dtype=wp.vec3),
    b: wp.array2d(dtype=float),
    gradient: wp.array2d(dtype=wp.vec3),
    area: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    dt: wp.array(dtype=float),
    u: wp.array(dtype=wp.vec3),
    solved_traction: wp.array(dtype=wp.vec3),
    compliance: float,
    integrated_force: int,
    gap: wp.array(dtype=wp.vec3),
    trial: wp.array(dtype=wp.vec3),
    frame: wp.array(dtype=wp.mat33),
    normal: wp.array(dtype=wp.vec3),
    parameters: wp.array(dtype=wp.vec3),
    traction: wp.array(dtype=wp.vec3),
    pressure: wp.array(dtype=float),
):
    i = wp.tid()
    g = gap_at(q, b, u, i)
    F = free_frame[i] + frame_rate(gradient, u, i)
    n = surface_normal(F, reference[i])
    z = g + compliance * solved_traction[i]
    gap[i] = g
    trial[i] = z
    frame[i] = F
    normal[i] = n
    parameters[i] = wp.vec3(material[i][0], material[i][1], compliance)
    traction[i] = solved_traction[i]
    pressure[i] = wp.min(wp.dot(z, n), 0.0) / compliance
    if integrated_force != 0 and area[i] == 0.0:
        pressure[i] = 0.0


@wp.kernel
def opening_residual(
    gap: wp.array(dtype=wp.vec3),
    proximal_gap: wp.array(dtype=wp.vec3),
    proximal_traction: wp.array(dtype=wp.vec3),
    traction: wp.array(dtype=wp.vec3),
    history: wp.array(dtype=float),
    material: wp.array(dtype=wp.vec3),
    area: wp.array(dtype=float),
    shared: wp.array(dtype=int),
    compliance: float,
    integrated_force: int,
    result: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    result[i] = compliance * (proximal_traction[i] - traction[i])
    gap_size = wp.max(wp.abs(gap[i][0]), wp.max(wp.abs(gap[i][1]), wp.abs(gap[i][2])))
    force_size = wp.max(wp.abs(traction[i][0]), wp.max(wp.abs(traction[i][1]), wp.abs(traction[i][2])))
    if history[i] < 2.0 * material[i][1] / material[i][0]:
        if wp.length(proximal_gap[i]) <= history[i] and gap_size <= compliance * force_size:
            result[i] = gap[i] - proximal_gap[i]
    closed = proximal_gap[i][0] == 0.0 and proximal_gap[i][1] == 0.0 and proximal_gap[i][2] == 0.0
    if history[i] == 0.0 and closed:
        if integrated_force == 0 or area[i] > 0.0:
            # The closed multiplier block is exactly zero. Keep the proximal
            # gap in the expression to retain the one-sided history derivative.
            result[i] = gap[i] - proximal_gap[i]
    if shared[i] != 0:
        result[i] = compliance * traction[i]


@wp.kernel
def normal_gap(gap: wp.array(dtype=wp.vec3), normal: wp.array(dtype=wp.vec3), result: wp.array(dtype=float)):
    i = wp.tid()
    result[i] = wp.dot(gap[i], normal[i])


@wp.kernel
def reaction_coordinates(
    q: wp.array(dtype=wp.vec3),
    free_frame: wp.array(dtype=wp.mat33),
    reference: wp.array(dtype=wp.vec3),
    b: wp.array2d(dtype=float),
    gradient: wp.array2d(dtype=wp.vec3),
    area: wp.array(dtype=float),
    u: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.vec3),
    points: wp.array(dtype=int),
    components: wp.array(dtype=int),
    integrated_force: int,
    matrix: wp.array2d(dtype=float),
    original: wp.array(dtype=float),
):
    column = wp.tid()
    i = points[column]
    component = components[column]
    g = gap_at(q, b, u, i)
    F = free_frame[i] + frame_rate(gradient, u, i)
    n = surface_normal(F, reference[i])
    moment = surface_normal_pullback(F, reference[i], g)
    root = wp.sqrt(area[i])
    value = force[i]
    if integrated_force == 0:
        value *= area[i]
    if component >= 0:
        original[column] = value[component] / root
    else:
        original[column] = wp.dot(value, n) / root
    for p in range(u.shape[0]):
        derivative = b[i, p] * n + moment * gradient[i, p]
        for axis in range(3):
            coefficient = float(0.0)
            if component == axis:
                coefficient = b[i, p]
            elif component < 0:
                coefficient = derivative[axis]
            matrix[3 * p + axis, column] = root * coefficient


@wp.kernel
def reconstruct_reaction(
    q: wp.array(dtype=wp.vec3),
    free_frame: wp.array(dtype=wp.mat33),
    reference: wp.array(dtype=wp.vec3),
    b: wp.array2d(dtype=float),
    gradient: wp.array2d(dtype=wp.vec3),
    area: wp.array(dtype=float),
    u: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.vec3),
    offsets: wp.array(dtype=int),
    closed: wp.array(dtype=int),
    reaction: wp.array(dtype=float),
    integrated_force: int,
    output: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    value = force[i]
    offset = offsets[i]
    if offset >= 0:
        root = wp.sqrt(area[i])
        if integrated_force == 0:
            value *= area[i]
        if closed[i] != 0:
            value = root * wp.vec3(reaction[offset], reaction[offset + 1], reaction[offset + 2])
        else:
            F = free_frame[i] + frame_rate(gradient, u, i)
            n = surface_normal(F, reference[i])
            value += (root * reaction[offset] - wp.dot(value, n)) * n
        if integrated_force == 0:
            value /= area[i]
    output[i] = value


@wp.kernel
def reaction_bounds(
    free_frame: wp.array(dtype=wp.mat33),
    reference: wp.array(dtype=wp.vec3),
    gradient: wp.array2d(dtype=wp.vec3),
    area: wp.array(dtype=float),
    material: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=wp.vec3),
    points: wp.array(dtype=int),
    offsets: wp.array(dtype=int),
    closed: wp.array(dtype=int),
    reaction: wp.array(dtype=float),
    bound: wp.array(dtype=float),
    derivative: wp.array2d(dtype=float),
):
    row = wp.tid()
    i = points[row]
    offset = offsets[i]
    if closed[i] != 0:
        F = free_frame[i] + frame_rate(gradient, u, i)
        n = surface_normal(F, reference[i])
        value = wp.vec3(reaction[offset], reaction[offset + 1], reaction[offset + 2])
        positive = value - wp.min(wp.dot(value, n), float(0.0)) * n
        length = wp.length(positive)
        bound[row] = wp.sqrt(area[i]) * material[i][0] - length
        for axis in range(3):
            derivative[row, offset + axis] = -positive[axis] / length
    else:
        bound[row] = -reaction[offset]
        derivative[row, offset] = -float(1.0)


def _evaluate(inputs, u, traction, compliance, *, integrated_force=False):
    count = len(inputs[0])
    device = inputs[0].device

    def zeros(n, vector=False):
        return wp.zeros(n, dtype=wp.vec3d if vector else wp.float64, device=device, requires_grad=True)

    kinds = [wp.vec3d, wp.vec3d, wp.mat33d, wp.vec3d, wp.vec3d, wp.vec3d, wp.float64]
    raw = [wp.zeros(count, dtype=kind, device=device, requires_grad=True) for kind in kinds]
    launch_double(
        kinematics,
        count,
        inputs=[*inputs, u, traction, compliance, int(integrated_force)],
        outputs=raw,
        device=device,
    )
    prox = [zeros(count, i < 2) for i in range(5)]
    proximal_kernel = cohesive_measure_proximal_response if integrated_force else cohesive_proximal_response
    proximal_inputs = [raw[1], raw[3], inputs[3], raw[4]]
    if integrated_force:
        proximal_inputs.append(inputs[7])
    launch_double(
        proximal_kernel,
        count,
        inputs=proximal_inputs,
        outputs=prox,
        device=device,
    )
    pressure_traction = zeros(count, True)
    stress = wp.zeros(count, dtype=wp.mat33d, device=device, requires_grad=True)
    constraint = zeros(count)
    launch_double(
        cohesive_contact_response,
        count,
        inputs=[prox[0], raw[2], inputs[2], raw[6]],
        outputs=[pressure_traction, stress],
        device=device,
    )
    launch_double(normal_gap, count, inputs=[prox[0], raw[3]], outputs=[constraint], device=device)
    (Ru, Rg) = (zeros(len(u), True), zeros(count, True))
    launch_double(
        monolithic_momentum,
        len(u),
        inputs=[
            inputs[5],
            inputs[6],
            inputs[7],
            raw[5],
            stress,
            inputs[8],
            inputs[9],
            u,
            int(integrated_force),
        ],
        outputs=[Ru],
        device=device,
    )
    shared_rows = shared_intact_rows(inputs[0].numpy(), inputs[3].numpy(), inputs[5].numpy())
    if integrated_force:
        shared_rows &= inputs[7].numpy() > 0.0
    shared = wp.array(
        shared_rows.astype(np.int32),
        dtype=int,
        device=device,
    )
    launch_double(
        opening_residual,
        count,
        inputs=[
            raw[0],
            prox[0],
            prox[1],
            traction,
            inputs[3],
            inputs[4],
            inputs[7],
            shared,
            compliance,
            int(integrated_force),
        ],
        outputs=[Rg],
        device=device,
    )
    result = [prox[0], raw[5], prox[2], prox[3], prox[4], raw[2], stress, constraint]
    return (result, Ru, Rg)


def _closed_pullback(inputs, values, u_numpy, traction_numpy, compliance, rhs_u, rhs_t):
    """Differentiate the fixed-rank closed constraint and its selected reaction."""
    q, _, _, _, _, b, _, area, mass, dt = values
    coefficient = mass[0] / dt[0] ** 2
    inverse = np.linalg.pinv(b, rcond=max(b.shape) * np.finfo(float).eps)
    seed_u = rhs_u.reshape(u_numpy.shape)
    seed_t = rhs_t.reshape(traction_numpy.shape)

    # First pull back the area-weighted minimum-pressure reaction. Then the
    # displacement constraint has no area dependence, even for tiny patches.
    root_area = np.sqrt(area)
    weighted = b.T * root_area[None, :]
    weighted_inverse = np.linalg.pinv(weighted, rcond=max(weighted.shape) * np.finfo(float).eps)
    pressure = root_area[:, None] * traction_numpy / coefficient
    seed_pressure = coefficient * seed_t / root_area[:, None]
    force_dual = weighted_inverse.T @ seed_pressure
    force_null = -u_numpy - weighted @ pressure
    pressure_null = seed_pressure - weighted_inverse @ (weighted @ seed_pressure)
    adj_weighted = -force_dual @ pressure.T
    adj_weighted += force_null @ (weighted_inverse @ (weighted_inverse.T @ seed_pressure)).T
    adj_weighted += (weighted_inverse.T @ pressure) @ pressure_null.T
    seed_u = seed_u - force_dual

    adj_q = -inverse.T @ seed_u
    residual = q + b @ u_numpy
    null_seed = seed_u - inverse @ (b @ seed_u)
    gradients = [np.zeros_like(value) for value in values]
    gradients[0] = adj_q
    gradients[5] = adj_q @ u_numpy.T
    gradients[5] -= residual @ (inverse @ (inverse.T @ seed_u)).T
    gradients[5] += (inverse.T @ u_numpy) @ null_seed.T
    gradients[5] += (adj_weighted * root_area[None, :]).T
    gradients[7] = np.sum(adj_weighted * b.T, axis=0) / (2 * root_area)
    gradients[7] -= np.sum(seed_t * traction_numpy, axis=1) / (2 * area)
    adj_coefficient = np.sum(seed_t * traction_numpy) / coefficient
    gradients[8][0] = adj_coefficient / dt[0] ** 2
    gradients[9][0] = -2 * adj_coefficient * coefficient / dt[0]

    # Positive prior damage can open a currently closed interface. Retain that
    # one-sided history derivative from the same proximal residual as the solve.
    if inputs[3].requires_grad:
        device = inputs[0].device
        clones = [
            wp.array(value, dtype=source.dtype, device=device, requires_grad=True)
            for value, source in zip(values, inputs)
        ]
        fixed_u = wp.array(u_numpy, dtype=wp.vec3d, device=device, requires_grad=True)
        fixed_t = wp.array(traction_numpy, dtype=wp.vec3d, device=device, requires_grad=True)
        momentum_dual = rhs_u.reshape(u_numpy.shape) + b.T @ adj_q
        with wp.Tape() as local:
            _, Ru, Rg = _evaluate(clones, fixed_u, fixed_t, compliance)
        local.backward(
            grads={
                Ru: wp.array(-momentum_dual, dtype=wp.vec3d, device=device),
                Rg: wp.array(adj_q, dtype=wp.vec3d, device=device),
            }
        )
        gradients[3] = clones[3].grad.numpy().copy()
    return gradients


def _implicit_pullback(
    inputs, values, u_numpy, traction_numpy, compliance, rhs_u, rhs_t, *, closed=False, integrated_force=False
):
    if closed and integrated_force and np.all(values[7] > 0.0):
        from ._cohesive_solve import setup

        area = values[7]
        pressure = traction_numpy / area[:, None]
        seed_force = (compliance * rhs_t).reshape(pressure.shape)
        density_compliance = setup(values)[2]
        gradients = _closed_pullback(
            inputs, values, u_numpy, pressure, density_compliance, rhs_u, (area[:, None] * seed_force).ravel()
        )
        gradients[7] += np.sum(seed_force * pressure, axis=1)
        return gradients
    if closed and not integrated_force:
        return _closed_pullback(
            inputs, values, u_numpy, traction_numpy, compliance, rhs_u, compliance * rhs_t
        )
    device = inputs[0].device

    def array(value, vector=False, grad=False):
        return wp.array(
            np.asarray(value), dtype=wp.vec3d if vector else wp.float64, device=device, requires_grad=grad
        )

    (_, _, weights) = geometry(values[0], values[1], values[2], values[5], values[6])
    matrix = weights.transpose(1, 0, 2).reshape(-1, values[5].shape[1])
    (_, singular, Vt) = np.linalg.svd(matrix, full_matrices=False)
    rank = np.count_nonzero(singular > singular[0] * max(matrix.shape) * np.finfo(float).eps)
    basis = np.kron(Vt[:rank].T, np.eye(3))
    size = basis.shape[1]
    rhs = np.r_[basis.T @ rhs_u, rhs_t]
    clones = [
        wp.array(v, dtype=a.dtype, device=a.device, requires_grad=True) for (v, a) in zip(values, inputs)
    ]
    fixed_u = array(u_numpy, True, True)
    fixed_t = array(traction_numpy, True, True)
    with wp.Tape() as local:
        (_, Ru, Rg) = _evaluate(clones, fixed_u, fixed_t, compliance, integrated_force=integrated_force)
    rows = []
    for column in basis.T:
        local.zero()
        local.backward(grads={Ru: array(column.reshape(-1, 3), True)})
        rows.append(np.r_[basis.T @ fixed_u.grad.numpy().ravel(), fixed_t.grad.numpy().ravel() / compliance])
    shape = Rg.numpy().shape
    for index in range(np.prod(shape)):
        local.zero()
        unit = np.zeros(shape)
        unit.ravel()[index] = 1.0
        local.backward(grads={Rg: array(unit, True)})
        rows.append(np.r_[basis.T @ fixed_u.grad.numpy().ravel(), fixed_t.grad.numpy().ravel() / compliance])
    J = np.array(rows)
    dual = np.linalg.lstsq(J.T, rhs, rcond=None)[0]
    # The orthogonal momentum block is identity; retain its map derivatives.
    orthogonal = rhs_u - basis @ (basis.T @ rhs_u)
    scale = max(np.max(np.abs(rhs_u)), np.max(np.abs(rhs_t)), 1.0)
    for _ in range(12):
        au = basis @ dual[:size] + orthogonal
        ag = dual[size:].reshape(shape)
        local.zero()
        local.backward(grads={Ru: array(au.reshape(-1, 3), True), Rg: array(ag, True)})
        error_u = rhs_u - fixed_u.grad.numpy().ravel()
        error_t = rhs_t - fixed_t.grad.numpy().ravel() / compliance
        full_error = np.max(np.abs(np.r_[error_u, error_t]))
        if not np.isfinite(full_error) or full_error <= 1e-10 * scale:
            break
        # Reevaluate the residual through the Warp VJP, including the momentum
        # complement, so matrix assembly roundoff is part of the correction.
        reduced_error = np.r_[basis.T @ error_u, error_t]
        dual += np.linalg.lstsq(J.T, reduced_error, rcond=None)[0]
        orthogonal += error_u - basis @ (basis.T @ error_u)
    if not np.isfinite(full_error) or full_error > 1e-10 * scale:
        raise RuntimeError(f"Full implicit adjoint residual: {full_error}")
    local.zero()
    local.backward(grads={Ru: array(-au.reshape(-1, 3), True), Rg: array(-ag, True)})
    return [a.grad.numpy().copy() for a in clones]


def _canonical_output(inputs, u, force, compliance, info, *, integrated_force=False):
    """Differentiate the selected minimum-pressure reaction before root pullback."""
    from ._cohesive_solve import canonical_reaction

    values = [a.numpy().copy() for a in inputs]
    if info.get("closed_linear_solution", False) and (not integrated_force or np.all(values[7] > 0.0)):
        return force
    selection_info = dict(info, closed_linear_solution=False)
    u_value, force_value = u.numpy().copy(), force.numpy().copy()
    _, _, context = canonical_reaction(
        values,
        u_value,
        force_value,
        compliance,
        selection_info,
        integrated_force=integrated_force,
        return_context=True,
    )
    if context is None:
        return force
    device = u.device
    output = wp.array(force_value, dtype=force.dtype, device=device, requires_grad=True)
    outer = wp._src.context.runtime.tape
    if outer is None:
        return output
    matrix, original, reaction = (context[key] for key in ["matrix", "original", "reaction"])
    count = len(reaction)
    points, components = [], []
    offsets = np.full(len(values[0]), -1, dtype=np.int32)
    for i, span in context["slices"]:
        offsets[i] = span.start
        points.extend([i] * (span.stop - span.start))
        components.extend(range(3) if context["closed"][i] else [-1])
    active, bound_hessians = [], []
    if context["constrained"]:
        for i, span in context["slices"]:
            value = reaction[span]
            hessian = np.zeros((count, count))
            if context["closed"][i]:
                n = context["normals"][i]
                compressed = value @ n < 0.0
                positive = value - min(value @ n, 0.0) * n
                radius = np.linalg.norm(positive)
                bound = context["root_area"][i] * values[4][i, 0] - radius
                if radius:
                    hessian[span, span] = (
                        -(np.eye(3) - compressed * np.outer(n, n) - np.outer(positive, positive) / radius**2)
                        / radius
                    )
            else:
                bound = -value[0]
            # SLSQP's objective tolerance identifies binding constraints;
            # the KKT residual below checks the resulting active set.
            if abs(bound) <= np.sqrt(2e-14) * max(np.linalg.norm(reaction), np.finfo(float).tiny):
                active.append(i)
                bound_hessians.append(hessian)

    def backward():
        clones = [
            wp.array(v, dtype=a.dtype, device=device, requires_grad=True) for v, a in zip(values, inputs)
        ]
        fixed_u = wp.array(u_value, dtype=u.dtype, device=device, requires_grad=True)
        fixed_force = wp.array(force_value, dtype=force.dtype, device=device, requires_grad=True)
        selected = wp.array(reaction, dtype=wp.float64, device=device, requires_grad=True)
        D = wp.zeros(matrix.shape, dtype=wp.float64, device=device, requires_grad=True)
        x = wp.zeros(count, dtype=wp.float64, device=device, requires_grad=True)
        result = wp.zeros(len(force), dtype=force.dtype, device=device, requires_grad=True)
        integer = lambda a: wp.array(np.asarray(a, dtype=np.int32), dtype=int, device=device)
        offset_array, closed_array = integer(offsets), integer(context["closed"])
        common = [clones[0], clones[1], clones[2], clones[5], clones[6], clones[7], fixed_u, fixed_force]
        bounds = wp.zeros(len(active), dtype=wp.float64, device=device, requires_grad=True)
        bound_gradient = wp.zeros((len(active), count), dtype=wp.float64, device=device, requires_grad=True)
        with wp.Tape() as local:
            launch_double(
                reaction_coordinates,
                count,
                inputs=[*common, integer(points), integer(components), int(integrated_force)],
                outputs=[D, x],
                device=device,
            )
            launch_double(
                reconstruct_reaction,
                len(force),
                inputs=[*common, offset_array, closed_array, selected, int(integrated_force)],
                outputs=[result],
                device=device,
            )
            if active:
                launch_double(
                    reaction_bounds,
                    len(active),
                    inputs=[
                        clones[1],
                        clones[2],
                        clones[6],
                        clones[7],
                        clones[4],
                        fixed_u,
                        integer(active),
                        offset_array,
                        closed_array,
                        selected,
                    ],
                    outputs=[bounds, bound_gradient],
                    device=device,
                )
        local.backward(grads={result: wp.array(output.grad.numpy(), dtype=force.dtype, device=device)})
        differentiands = [*clones, fixed_u, fixed_force]
        explicit = [a.grad.numpy().copy() for a in differentiands]
        seed = selected.grad.numpy().copy()
        if not active:
            inverse = np.linalg.pinv(matrix, rcond=max(matrix.shape) * np.finfo(float).eps)
            dual = inverse.T @ seed
            null_seed = seed - inverse @ (matrix @ seed)
            residual = matrix @ original - matrix @ reaction
            adj_matrix = np.outer(dual, original - reaction)
            adj_matrix += np.outer(residual, inverse @ (inverse.T @ seed))
            adj_matrix += np.outer(inverse.T @ reaction, null_seed)
            adj_original = matrix.T @ dual
            seeds = {}
        else:
            Jb = bound_gradient.numpy()
            multipliers = np.linalg.lstsq(np.c_[matrix.T, Jb.T], reaction, rcond=None)[0]
            multiplier, mu = multipliers[: matrix.shape[0]], multipliers[matrix.shape[0] :]
            stationarity = reaction - matrix.T @ multiplier - Jb.T @ mu
            scale = max(np.linalg.norm(reaction), 1.0)
            if np.max(abs(stationarity)) > 1e-10 * scale or np.min(mu) < -1e-10 * scale:
                raise RuntimeError("Canonical reaction KKT multipliers are inconsistent")
            H = np.eye(count) - np.einsum("i,ijk->jk", mu, np.array(bound_hessians))
            equations = matrix.shape[0] + len(active)
            constraint = np.r_[matrix, Jb]
            K = np.block([[H, -constraint.T], [constraint, np.zeros((equations, equations))]])
            adj = np.linalg.lstsq(K.T, np.r_[seed, np.zeros(equations)], rcond=None)[0]
            if np.max(abs(K.T @ adj - np.r_[seed, np.zeros(equations)])) > 1e-10 * max(
                np.max(abs(seed)), 1.0
            ):
                raise RuntimeError("Canonical reaction KKT adjoint residual")
            az, ae, ab = adj[:count], adj[count : count + matrix.shape[0]], adj[count + matrix.shape[0] :]
            adj_matrix = np.outer(multiplier, az) - np.outer(ae, reaction - original)
            adj_original = matrix.T @ ae
            seeds = {
                bounds: wp.array(-ab, dtype=wp.float64, device=device),
                bound_gradient: wp.array(np.outer(mu, az), dtype=wp.float64, device=device),
            }
        local.zero()
        seeds[D] = wp.array(adj_matrix, dtype=wp.float64, device=device)
        seeds[x] = wp.array(adj_original, dtype=wp.float64, device=device)
        local.backward(grads=seeds)
        for source, direct, indirect in zip([*inputs, u, force], explicit, differentiands):
            if source.requires_grad:
                source.grad.assign(source.grad.numpy() + direct + indirect.grad.numpy())

    outer.record_func(backward, [output, *[a for a in [*inputs, u, force] if a.requires_grad]])
    return output


def _solve_arrays(inputs, compliance_scale=1.0, *, integrated_force=False):
    device = inputs[0].device
    values = [a.numpy().copy() for a in inputs]
    if integrated_force:
        u_numpy, traction_numpy, compliance, info, _, _ = refine_geometry(
            values,
            np.zeros_like(values[0]),
            np.zeros_like(values[1]),
            compliance_scale,
            integrated_force=True,
        )
    else:
        u_numpy, traction_numpy, compliance, info = solve_primal(
            values, compliance_scale, return_traction=True
        )
    u = wp.array(u_numpy, dtype=wp.vec3d, device=device, requires_grad=True)
    traction = wp.array(traction_numpy, dtype=wp.vec3d, device=device, requires_grad=True)
    tape = wp._src.context.runtime.tape
    if tape is not None:

        def backward():
            gradients = _implicit_pullback(
                inputs,
                values,
                u_numpy,
                traction_numpy,
                compliance,
                u.grad.numpy().ravel(),
                traction.grad.numpy().ravel() / compliance,
                closed=info.get("closed_linear_solution", False),
                integrated_force=integrated_force,
            )
            for source, gradient in zip(inputs, gradients):
                if source.requires_grad:
                    source.grad.assign(source.grad.numpy() + gradient)

        tape.record_func(backward, [u, traction, *[a for a in inputs if a.requires_grad]])
    selected_traction = _canonical_output(
        inputs, u, traction, compliance, info, integrated_force=integrated_force
    )
    result, _, _ = _evaluate(inputs, u, selected_traction, compliance, integrated_force=integrated_force)
    result_type = CohesiveForceStep if integrated_force else CohesiveStep
    return result_type(*result, u, info)


@dataclass
class CohesiveStep:
    gap: wp.array
    traction: wp.array
    history: wp.array
    storage: wp.array
    fracture: wp.array
    frame: wp.array
    surface_stress: wp.array
    normal_gap: wp.array
    displacement: wp.array
    info: dict


@dataclass
class CohesiveForceStep:
    """Integrated force [N] and surface moment [N m]; energy densities [J/m²]."""

    gap: wp.array
    force: wp.array
    history: wp.array
    storage: wp.array
    fracture: wp.array
    frame: wp.array
    surface_moment: wp.array
    normal_gap: wp.array
    displacement: wp.array
    info: dict


def _validated_inputs(
    free_gap,
    free_frame,
    reference_normal,
    history,
    material,
    separation_map,
    frame_map,
    area,
    particle_mass,
    duration,
    *,
    compliance_scale=1.0,
    integrated_force=False,
):
    """Solve one inertial interface update in float64.

    material stores (strength [Pa], fracture energy [J/m²], unused).
    separation_map is dimensionless; frame_map is in 1/m; area is in m².
    particle_mass and duration each contain one scalar, in kg and s.
    The common particle mass matches MaterialTrajectory's reference sampling.
    """
    inputs = [
        free_gap,
        free_frame,
        reference_normal,
        history,
        material,
        separation_map,
        frame_map,
        area,
        particle_mass,
        duration,
    ]
    if not np.isfinite(compliance_scale) or compliance_scale <= 0:
        raise ValueError("Compliance scale must be finite and positive")
    expected = [
        wp.vec3d,
        wp.mat33d,
        wp.vec3d,
        wp.float64,
        wp.vec3d,
        wp.float64,
        wp.vec3d,
        wp.float64,
        wp.float64,
        wp.float64,
    ]
    if any(a.dtype != dtype or a.device != free_gap.device for a, dtype in zip(inputs, expected)):
        raise ValueError("Interface arrays must use float64 on a common device")
    count = len(free_gap)
    if count == 0:
        raise ValueError("An interface solve requires at least one quadrature point")
    if separation_map.ndim != 2 or separation_map.shape[1] == 0:
        raise ValueError("The separation map requires quadrature and particle axes")
    particles = separation_map.shape[1]
    shapes = [
        (count,),
        (count,),
        (count,),
        (count,),
        (count,),
        (count, particles),
        (count, particles),
        (count,),
        (1,),
        (1,),
    ]
    if any(a.shape != shape for a, shape in zip(inputs, shapes)):
        raise ValueError("Interface array shapes do not match the quadrature and particle counts")
    values = [a.numpy() for a in inputs]
    if any(not np.isfinite(a).all() for a in values):
        raise ValueError("Interface inputs must be finite")
    if (
        np.any(values[3] < 0)
        or np.any(values[4][:, :2] <= 0)
        or np.any(values[7] < 0 if integrated_force else values[7] <= 0)
        or values[8][0] <= 0
        or values[9][0] <= 0
    ):
        raise ValueError(
            "History and integrated areas must be nonnegative; other physical scales must be positive"
        )
    if np.any(np.linalg.norm(values[2], axis=1) == 0):
        raise ValueError("Reference normals must be nonzero")
    return inputs, values


def solve(
    free_gap,
    free_frame,
    reference_normal,
    history,
    material,
    separation_map,
    frame_map,
    area,
    particle_mass,
    duration,
    *,
    compliance_scale=1.0,
):
    """Return equilibrium displacement and response per reference interface area."""
    inputs, _ = _validated_inputs(
        free_gap,
        free_frame,
        reference_normal,
        history,
        material,
        separation_map,
        frame_map,
        area,
        particle_mass,
        duration,
        compliance_scale=compliance_scale,
    )
    return _solve_arrays(inputs, compliance_scale)


def solve_force(
    free_gap,
    free_frame,
    reference_normal,
    history,
    material,
    separation_map,
    frame_map,
    area,
    particle_mass,
    duration,
    *,
    compliance_scale=1.0,
):
    """Return equilibrium using integrated forces, including zero-area supports."""
    inputs, _ = _validated_inputs(
        free_gap,
        free_frame,
        reference_normal,
        history,
        material,
        separation_map,
        frame_map,
        area,
        particle_mass,
        duration,
        compliance_scale=compliance_scale,
        integrated_force=True,
    )
    return _solve_arrays(inputs, compliance_scale, integrated_force=True)
