"""Reduced nonlinear equations for moving cohesive interfaces."""

import numpy as np
from scipy.optimize import minimize

from ._cohesive_initial import fixed_normal_initial
from .reference_inertia import apply_inertia, solve_inertia


def skew(x):
    (a, b, c) = x
    return np.array([[0.0, -c, b], [c, 0.0, -a], [-b, a, 0.0]])


def geometry(q, frame, normal, b, gradient):
    (count, particles) = b.shape
    n = normal / np.linalg.norm(normal, axis=1)[:, None]
    axis = np.eye(3)[np.argmin(np.abs(n), axis=1)]
    t1 = np.cross(n, axis)
    t1 /= np.linalg.norm(t1, axis=1)[:, None]
    t2 = np.cross(n, t1)
    base = np.stack(
        [q, np.einsum("qij,qj->qi", frame, t1), np.einsum("qij,qj->qi", frame, t2)], axis=1
    ).reshape(count, 9)
    weights = np.stack(
        [b, np.einsum("qpa,qa->qp", gradient, t1), np.einsum("qpa,qa->qp", gradient, t2)], axis=1
    )
    L = np.einsum("qip,ab->qiapb", weights, np.eye(3)).reshape(count, 9, particles * 3)
    return (base, L, weights)


def _local_constraints(base, L, u):
    local = (base + np.einsum("qia,a->qi", L, u)).reshape(-1, 3, 3)
    values = []
    gradients = []
    hessians = []
    for g, a, b in local:
        area = np.cross(a, b)
        length = np.linalg.norm(area)
        n = area / length
        c = g @ n
        z = (g - c * n) / length
        dn = (np.eye(3) - np.outer(n, n)) / length
        h_aa = -(np.outer(g, n) + np.outer(n, g) + c * np.eye(3) - 3 * c * np.outer(n, n)) / length**2
        (Ja, Jb) = (-skew(b), skew(a))
        grad = np.r_[n, np.cross(b, z), np.cross(z, a)]
        H = np.block(
            [
                [np.zeros((3, 3)), dn @ Ja, dn @ Jb],
                [Ja.T @ dn, Ja.T @ h_aa @ Ja, Ja.T @ h_aa @ Jb - skew(z)],
                [Jb.T @ dn, Jb.T @ h_aa @ Ja + skew(z), Jb.T @ h_aa @ Jb],
            ]
        )
        values.append(c)
        gradients.append(grad)
        hessians.append(H)
    C = np.einsum("qi,qia->qa", gradients, L)
    H = np.einsum("qia,qij,qjb->qab", L, hessians, L)
    return (np.array(values), C, H)


def constraints(base, L, u):
    local = base + np.einsum("qia,a->qi", L, u)
    identity = np.broadcast_to(np.eye(9), (len(base), 9, 9))
    (c, gradient, hessian) = _local_constraints(local, identity, np.zeros(9))
    C = (gradient[:, None] @ L)[:, 0]
    H = np.swapaxes(L, 1, 2) @ (hessian @ L)
    return (c, C, H)


def constraint_gradient(base, L, u):
    local = (base + np.einsum("qia,a->qi", L, u)).reshape(-1, 3, 3)
    (gap, a, b) = local.transpose(1, 0, 2)
    normal = np.cross(a, b)
    length = np.linalg.norm(normal, axis=1)
    normal /= length[:, None]
    value = np.sum(gap * normal, axis=1)
    z = (gap - value[:, None] * normal) / length[:, None]
    local_gradient = np.concatenate([normal, np.cross(b, z), np.cross(z, a)], axis=1)
    gradient = np.einsum("qi,qia->qa", local_gradient, L)
    return (value, gradient, None)


def proximal(z, normal, history, material, compliance):
    count = len(z)
    compliance = np.broadcast_to(compliance, (count,))
    (strength, gc) = material[:, :2].T
    failure = 2 * gc / strength
    component = np.sum(z * normal, axis=1)
    closed = (component < 0) & (compliance > 0)
    compression = np.where(closed, component, 0.0)
    positive = z - compression[:, None] * normal
    radius = np.linalg.norm(positive, axis=1)
    opened = radius.copy()
    dtype = np.result_type(z, normal, history, material, compliance)
    derivative = np.ones(count, dtype=dtype)
    traction_ratio = np.zeros(count, dtype=dtype)
    traction_derivative = np.zeros(count, dtype=dtype)
    for i in range(count):
        if compliance[i] == 0.0 or history[i] >= failure[i]:
            continue
        opened[i] = 0.0
        derivative[i] = 0.0
        traction_ratio[i] = 1.0
        traction_derivative[i] = 1.0
        if history[i] > 0:
            stiffness = strength[i] * (1 - history[i] / failure[i]) / history[i]
            derivative[i] = 1 / (1 + compliance[i] * stiffness)
            opened[i] = radius[i] * derivative[i]
            traction_ratio[i] = compliance[i] * stiffness / (1 + compliance[i] * stiffness)
            traction_derivative[i] = traction_ratio[i]
        if opened[i] > history[i] or (history[i] == 0 and radius[i] > compliance[i] * strength[i]):
            derivative[i] = 1 / (1 - compliance[i] * strength[i] / failure[i])
            opened[i] = (radius[i] - compliance[i] * strength[i]) * derivative[i]
            traction_ratio[i] = compliance[i] * strength[i] * (1 / radius[i] - 1 / failure[i]) * derivative[i]
            traction_derivative[i] = -compliance[i] * strength[i] / failure[i] * derivative[i]
            if opened[i] >= failure[i]:
                opened[i] = radius[i]
                derivative[i] = 1.0
                traction_ratio[i] = 0.0
                traction_derivative[i] = 0.0
    gap = np.zeros_like(z)
    dz = np.zeros((count, 3, 3))
    dn = np.zeros((count, 3, 3))
    traction_dz = np.zeros((count, 3, 3))
    for i in range(count):
        radial_traction = traction_ratio[i] * np.eye(3)
        if radius[i] == 0:
            radial = derivative[i] * np.eye(3)
        else:
            direction = positive[i] / radius[i]
            ratio = opened[i] / radius[i]
            gap[i] = ratio * positive[i]
            radial = ratio * np.eye(3) + (derivative[i] - ratio) * np.outer(direction, direction)
            radial_traction += (traction_derivative[i] - traction_ratio[i]) * np.outer(direction, direction)
        projection = np.eye(3) - closed[i] * np.outer(normal[i], normal[i])
        dz[i] = radial @ projection
        dn[i] = -float(closed[i]) * radial @ (component[i] * np.eye(3) + np.outer(normal[i], z[i]))
        traction_dz[i] = radial_traction + float(closed[i]) * radial @ np.outer(normal[i], normal[i])
    dual_traction = compression[:, None] * normal + traction_ratio[:, None] * positive
    return (gap, dz, dn, compression, closed, dual_traction, traction_dz, -dn)


def shared_intact_rows(q, history, mapping):
    return (history == 0.0) & np.all(q == 0.0, axis=1) & np.all(mapping == 0.0, axis=1)


def setup(values, compliance_factor=1.0, *, extended=False, integrated_force=False, inertia=None):
    (q, frame, normal, history, material, b, gradient, area, mass, dt) = values
    coefficient = (
        np.longdouble(mass[0]) / np.longdouble(dt[0]) ** 2 if extended else float(mass[0] / dt[0] ** 2)
    )
    (base, full_L, weights) = geometry(q, frame, normal, b, gradient)
    matrix = weights.transpose(1, 0, 2).reshape(-1, b.shape[1])
    (_, singular, vt) = np.linalg.svd(matrix.astype(float), full_matrices=False)
    rank = np.count_nonzero(singular > singular.max() * max(matrix.shape) * np.finfo(float).eps)
    modes = vt[:rank].T
    if inertia is not None:
        modes = np.linalg.qr(solve_inertia(inertia, modes), mode="reduced")[0]
    basis = np.kron(modes, np.eye(3))
    reduced_inertia = np.eye(basis.shape[1])
    if inertia is not None:
        precise_modes = modes.astype(np.longdouble if extended else float)
        reduced_inertia = np.kron(precise_modes.T @ apply_inertia(inertia, precise_modes), np.eye(3))
    L = full_L @ basis
    softening = np.max(material[:, 0] ** 2 / (2 * material[:, 1]))
    mobility = b @ solve_inertia(inertia, b.T) / coefficient
    if integrated_force:
        softening = np.max(area * material[:, 0] ** 2 / (2 * material[:, 1]))
    else:
        mobility *= np.sqrt(area[:, None] * area[None, :])
    maximum_mobility = np.linalg.eigvalsh(mobility.astype(float)).max()
    # A shared field has B=0. Its proximal scale is set by the constitutive
    # convexity bound; there is no displacement stiffness to invert.
    inverse_mobility = 0.0 if maximum_mobility == 0.0 else 1 / maximum_mobility
    if integrated_force and inverse_mobility == 0.0 and softening == 0.0:
        inverse_mobility = coefficient
    compliance = compliance_factor / (inverse_mobility + 2 * softening)
    if compliance * softening >= 1:
        raise ValueError("Proximal subproblem is not strictly convex")
    length = np.min(2 * material[:, 1] / material[:, 0])
    size = basis.shape[1]
    shared = shared_intact_rows(q, history, b)
    if integrated_force:
        shared &= area > 0.0
    local_compliance = compliance * area if integrated_force else compliance

    def at(y):
        if extended:
            y = np.asarray(y, dtype=np.longdouble)
        u = length * y[:size]
        dual = length * y[size:].reshape(-1, 3)
        local = (base + np.einsum("qia,a->qi", L, u)).reshape(-1, 3, 3)
        (gap, tangent_a, tangent_b) = local.transpose(1, 0, 2)
        normals = np.cross(tangent_a, tangent_b)
        measure = np.linalg.norm(normals, axis=1)
        normals /= measure[:, None]
        normal_gradient = np.empty((len(q), 3, size), dtype=np.longdouble if extended else float)
        for i in range(len(q)):
            normal_gradient[i] = (
                (np.eye(3) - np.outer(normals[i], normals[i]))
                / measure[i]
                @ (-skew(tangent_b[i]) @ L[i, 3:6] + skew(tangent_a[i]) @ L[i, 6:9])
            )
        z = gap + dual
        (prox, D_z, D_n, compression, closed, dual_traction, T_z, T_n) = proximal(
            z, normals, history, material, local_compliance
        )
        proximal_base = base.copy()
        proximal_base[:, :3] += prox - gap
        (c, C, Hc) = constraints(proximal_base, L, u)
        face = np.einsum("qai,qa->qi", normal_gradient, prox)
        weights = (np.ones_like(area) if integrated_force else area) / (coefficient * compliance)
        Ru = reduced_inertia @ u + np.einsum("qai,qa,q->i", L[:, :3], dual, weights)
        Ru += np.einsum("qi,q,q->i", face, compression, weights)
        # Unloading can use small gap operands. Loading retains the traction
        # form because its small radial derivative is lost in I - Dprox.
        Rg = dual_traction - dual
        use_gap = np.max(np.abs(gap), axis=1) <= np.max(np.abs(dual), axis=1)
        use_gap &= (history < 2 * material[:, 1] / material[:, 0]) & (np.linalg.norm(prox, axis=1) <= history)
        Rg[use_gap] = gap[use_gap] - prox[use_gap]
        Juu = reduced_inertia.astype(float).copy()
        Jus = np.zeros((size, len(q), 3))
        Jgu = np.empty((len(q), 3, size))
        Jgs = np.zeros((len(q), 3, len(q), 3))
        for i in range(len(q)):
            (B, N) = (L[i, :3], normal_gradient[i])
            proximal_u = D_z[i] @ B + D_n[i] @ N
            face_gradient = Hc[i] - B.T @ N - N.T @ B + N.T @ proximal_u
            compression_gradient = closed[i] * (normals[i] @ B + z[i] @ N)
            Juu += weights[i] * (compression[i] * face_gradient + np.outer(face[i], compression_gradient))
            Jus[:, i] = weights[i] * (
                B.T + compression[i] * N.T @ D_z[i] + closed[i] * np.outer(face[i], normals[i])
            )
            Jgu[i] = T_z[i] @ B + T_n[i] @ N
            Jgs[i, :, i, :] = -D_z[i]
        # These opening constraints are identically zero. Their minimum-norm
        # reaction is zero; use that equation to fix the undetermined multiplier.
        Rg[shared] = dual[shared]
        Jgu[shared] = 0.0
        Jgs[shared] = 0.0
        for i in np.flatnonzero(shared):
            Jgs[i, :, i, :] = np.eye(3)
        dual_size = len(q) * 3
        J = np.block(
            [
                [Juu, Jus.reshape(size, dual_size)],
                [Jgu.reshape(dual_size, size), Jgs.reshape(dual_size, dual_size)],
            ]
        )
        R = np.r_[Ru, Rg.ravel()] / length
        return (R, J, dict(gap=gap, proximal_gap=prox, compression=compression, normal=normals))

    return (at, basis, float(compliance), length)


def canonical_reaction(
    values,
    u,
    traction,
    compliance,
    info,
    *,
    integrated_force=False,
    return_context=False,
    remainders=None,
    branches=None,
    map_remainders=None,
):
    if info.get("closed_linear_solution", False):
        return (traction, info, None) if return_context else (traction, info)
    q, frame, reference, history, material, b, gradient, area, mass, dt = values
    from ._cohesive_projection import reaction_geometry

    gap, mapped, normals, C, closed, contact = reaction_geometry(
        values, u, traction, compliance, integrated_force, remainders, map_remainders
    )
    if branches is not None:
        closed, contact = branches.closed, branches.contact
    if not np.any(closed | contact):
        return (traction, info, None) if return_context else (traction, info)
    root_area = np.sqrt(area)
    integrated = traction.copy() if integrated_force else area[:, None] * traction
    blocks, original, slices = [], [], []
    offset = 0
    for i in np.flatnonzero(closed | contact):
        if closed[i]:
            block = np.kron(b[i, :, None], np.eye(3))
            component = integrated[i]
        else:
            block = C[i, :, None]
            component = np.array([np.dot(integrated[i], normals[i])])
        blocks.append(root_area[i] * block)
        original.extend(component / root_area[i])
        slices.append((i, slice(offset, offset + block.shape[1])))
        offset += block.shape[1]
    matrix = np.concatenate(blocks, axis=1)
    original = np.asarray(original)
    force = matrix @ original
    U, singular, Vt = np.linalg.svd(matrix, full_matrices=True)
    rank = np.count_nonzero(singular > max(matrix.shape) * np.finfo(float).eps * singular[0])
    if rank == len(original):
        return (traction, info, None) if return_context else (traction, info)
    reaction = Vt[:rank].T @ ((U[:, :rank].T @ force) / singular[:rank])
    null = Vt[rank:].T

    def bounds(z, derivative=False):
        selected = reaction + null @ z
        result, rows = [], []
        for i, span in slices:
            value = selected[span]
            row = np.zeros(len(reaction))
            if closed[i]:
                positive = value - min(np.dot(value, normals[i]), 0.0) * normals[i]
                radius = np.linalg.norm(positive)
                result.append(root_area[i] * material[i, 0] - radius)
                if radius:
                    row[span] = -positive / radius
            else:
                result.append(-value[0])
                row[span] = -1.0
            rows.append(row @ null)
        return np.array(rows) if derivative else np.array(result)

    constrained = np.any(bounds(np.zeros(null.shape[1])) < 0.0)
    if constrained:
        reaction_scale = max(np.linalg.norm(original), np.finfo(float).tiny)
        result = minimize(
            lambda z: (0.5 * np.dot(z, z), z),
            null.T @ (original - reaction) / reaction_scale,
            jac=True,
            method="SLSQP",
            constraints=[
                dict(
                    type="ineq",
                    fun=lambda y: bounds(reaction_scale * y) / reaction_scale,
                    jac=lambda y: bounds(reaction_scale * y, derivative=True),
                )
            ],
            options=dict(ftol=1e-14, maxiter=200),
        )
        result.x *= reaction_scale
        scale = max(np.max(np.abs(original)), np.finfo(float).tiny)
        if not result.success or np.min(bounds(result.x)) < -256 * np.finfo(float).eps * scale:
            raise RuntimeError(f"Canonical reaction solve failed: {result.message}")
        reaction += null @ result.x
    for i, span in slices:
        value = root_area[i] * reaction[span]
        if closed[i]:
            integrated[i] = value
        else:
            integrated[i] += (value[0] - np.dot(integrated[i], normals[i])) * normals[i]
    error = np.max(np.abs(matrix @ reaction - force))
    scale = max(np.max(np.abs(force)), np.finfo(float).tiny)
    if error >= 256 * np.finfo(float).eps * scale:
        raise RuntimeError(f"Equivalent reaction changed momentum: {error}")
    info["reaction_nullity"] = int(len(original) - rank)
    output = integrated if integrated_force else integrated / area[:, None]
    if not return_context:
        return output, info
    context = dict(
        matrix=matrix,
        original=original,
        reaction=reaction,
        slices=slices,
        closed=closed,
        normals=normals,
        root_area=root_area,
        constrained=constrained,
    )
    return output, info, context


def solve_primal(
    values, compliance_factor=1.0, return_traction=False, *, integrated_force=False, inertia=None
):
    displacement, traction, compliance, info, _, _ = refine_geometry(
        values,
        np.zeros_like(values[0]),
        np.zeros_like(values[1]),
        compliance_factor,
        integrated_force=integrated_force,
        inertia=inertia,
    )
    result = traction if return_traction else compliance * traction
    return displacement.high, result, compliance, info


def refine_geometry(
    values,
    gap_remainder,
    frame_remainder,
    compliance_factor=1.0,
    *,
    integrated_force=False,
    closed_motion=None,
    inertia=None,
    return_context=False,
    map_remainders=None,
):
    from ._cohesive_equilibrium import solve
    from ._compensated import Expansion

    precise = list(values)
    precise[0] = values[0].astype(np.longdouble) + gap_remainder
    precise[1] = values[1].astype(np.longdouble) + frame_remainder
    if closed_motion is not None:
        if not integrated_force:
            raise ValueError("Factored closed motion requires integrated interface forces")
        motion = np.asarray(closed_motion, dtype=np.longdouble)
        mapping = values[5]
        if motion.shape != (mapping.shape[1], 3) or not np.isfinite(motion).all():
            raise ValueError("Closed motion must match the mass-normalized grid displacement")
        support = values[7] > 0.0
        gap = Expansion(values[0], gap_remainder)
        product = Expansion(
            mapping, 0.0 if map_remainders is None else map_remainders[0]
        ) @ Expansion.from_value(motion)
        bound = 64 * np.finfo(float).eps * (abs(mapping) @ abs(motion))
        if np.any(abs((gap - product).high[support]) > bound[support]):
            raise ValueError("Closed motion must generate the supported free gap")
    compliance = setup(values, compliance_factor, integrated_force=integrated_force, inertia=inertia)[2]
    _, force, _, info, displacement = fixed_normal_initial(
        precise,
        compliance,
        integrated_force=integrated_force,
        max_iterations=1024,
        inertia=inertia,
        return_primal=True,
    )
    info = dict(info, initializer="fixed_normal_admm")
    seed = (displacement, force, compliance, info, precise[0], precise[1])
    response = solve(
        values,
        gap_remainder,
        frame_remainder,
        seed,
        integrated_force=integrated_force,
        inertia=inertia,
        map_remainders=map_remainders,
    )
    info = dict(
        response.info,
        residual=response.info["paired_residual"],
        extended_residual=response.info["paired_residual"],
    )
    result = (
        response.displacement,
        response.traction.high,
        response.compliance,
        info,
        response.gap,
        response.frame,
    )
    return (*result, response.branches) if return_context else result
