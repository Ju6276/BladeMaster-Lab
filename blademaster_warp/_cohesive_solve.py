"""Reduced nonlinear equations for moving cohesive interfaces."""

import numpy as np
from scipy.optimize import least_squares, minimize


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
    (strength, gc) = material[:, :2].T
    failure = 2 * gc / strength
    component = np.sum(z * normal, axis=1)
    closed = component < 0
    compression = np.minimum(component, 0.0)
    positive = z - compression[:, None] * normal
    radius = np.linalg.norm(positive, axis=1)
    opened = radius.copy()
    derivative = np.ones(count)
    traction_ratio = np.zeros(count)
    traction_derivative = np.zeros(count)
    for i in range(count):
        if history[i] >= failure[i]:
            continue
        opened[i] = 0.0
        derivative[i] = 0.0
        traction_ratio[i] = 1.0
        traction_derivative[i] = 1.0
        if history[i] > 0:
            stiffness = strength[i] * (1 - history[i] / failure[i]) / history[i]
            derivative[i] = 1 / (1 + compliance * stiffness)
            opened[i] = radius[i] * derivative[i]
            traction_ratio[i] = compliance * stiffness / (1 + compliance * stiffness)
            traction_derivative[i] = traction_ratio[i]
        if opened[i] > history[i] or (history[i] == 0 and radius[i] > compliance * strength[i]):
            derivative[i] = 1 / (1 - compliance * strength[i] / failure[i])
            opened[i] = (radius[i] - compliance * strength[i]) * derivative[i]
            traction_ratio[i] = compliance * strength[i] * (1 / radius[i] - 1 / failure[i]) * derivative[i]
            traction_derivative[i] = -compliance * strength[i] / failure[i] * derivative[i]
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


def setup(values, compliance_factor=1.0, *, extended=False):
    (q, frame, normal, history, material, b, gradient, area, mass, dt) = values
    coefficient = (
        np.longdouble(mass[0]) / np.longdouble(dt[0]) ** 2 if extended else float(mass[0] / dt[0] ** 2)
    )
    (base, full_L, weights) = geometry(q, frame, normal, b, gradient)
    matrix = weights.transpose(1, 0, 2).reshape(-1, b.shape[1])
    (_, singular, vt) = np.linalg.svd(matrix.astype(float), full_matrices=False)
    rank = np.count_nonzero(singular > singular.max() * max(matrix.shape) * np.finfo(float).eps)
    basis = np.kron(vt[:rank].T, np.eye(3))
    L = full_L @ basis
    softening = np.max(material[:, 0] ** 2 / (2 * material[:, 1]))
    mobility = np.sqrt(area[:, None] * area[None, :]) * (b @ b.T) / coefficient
    maximum_mobility = np.linalg.eigvalsh(mobility.astype(float)).max()
    # A shared field has B=0. Its proximal scale is set by the constitutive
    # convexity bound; there is no displacement stiffness to invert.
    inverse_mobility = 0.0 if maximum_mobility == 0.0 else 1 / maximum_mobility
    compliance = compliance_factor / (inverse_mobility + 2 * softening)
    if compliance * softening >= 1:
        raise ValueError("Proximal subproblem is not strictly convex")
    length = np.min(2 * material[:, 1] / material[:, 0])
    size = basis.shape[1]
    shared = shared_intact_rows(q, history, b)

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
            z, normals, history, material, compliance
        )
        proximal_base = base.copy()
        proximal_base[:, :3] += prox - gap
        (c, C, Hc) = constraints(proximal_base, L, u)
        face = np.einsum("qai,qa->qi", normal_gradient, prox)
        weights = area / (coefficient * compliance)
        Ru = u + np.einsum("qai,qa,q->i", L[:, :3], dual, weights)
        Ru += np.einsum("qi,q,q->i", face, compression, weights)
        # g - prox(g + lambda*T) = lambda*(T_prox - T). Evaluate the
        # traction form to retain small corrections to a finite opening.
        Rg = dual_traction - dual
        Juu = np.eye(size)
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
            Jgs[i, :, i, :] = T_z[i] - np.eye(3)
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


def _solve_primal(values, compliance_factor=1.0, return_traction=False):
    (at, basis, compliance, length) = setup(values, compliance_factor)
    count = len(values[0])
    size = basis.shape[1]
    y = np.zeros(size + 3 * count)
    (q, frame, reference, history, material, b, gradient, area, mass, dt) = values
    coefficient = float(mass[0] / dt[0] ** 2)
    if np.all(history == 0.0):
        reaction = np.linalg.lstsq(b @ b.T, q, rcond=None)[0]
        candidate_u = -b.T @ reaction
        root_area = np.sqrt(area)
        weighted = b.T * root_area[None, :]
        pressure = np.linalg.lstsq(weighted, -candidate_u, rcond=None)[0]
        candidate_t = coefficient * pressure / root_area[:, None]
        candidate_frame = frame + np.einsum("pa,qpb->qab", candidate_u, gradient)
        candidate_n = (
            reference[:, :1] * np.cross(candidate_frame[:, :, 1], candidate_frame[:, :, 2])
            + reference[:, 1:2] * np.cross(candidate_frame[:, :, 2], candidate_frame[:, :, 0])
            + reference[:, 2:3] * np.cross(candidate_frame[:, :, 0], candidate_frame[:, :, 1])
        )
        candidate_n /= np.linalg.norm(candidate_n, axis=1)[:, None]
        component = np.sum(candidate_t * candidate_n, axis=1)
        positive = candidate_t - np.minimum(component, 0.0)[:, None] * candidate_n
        gap_error = np.max(np.abs(q + b @ candidate_u))
        roundoff = 64 * np.finfo(float).eps * np.max(np.abs(q))
        if gap_error <= roundoff and np.all(np.linalg.norm(positive, axis=1) <= material[:, 0]):
            y[:size] = basis.T @ candidate_u.ravel() / length
            y[size:] = compliance * candidate_t.ravel() / length
            error = float(np.max(np.abs(at(y)[0])))
            assert error < 1e-13
            result = candidate_t if return_traction else compliance * candidate_t
            return (
                candidate_u,
                result,
                compliance,
                dict(
                    iterations=0,
                    initial_evaluations=0,
                    initializer="closed_linear",
                    closed_linear_solution=True,
                    residual=error,
                    closed=count,
                    compressed=int(np.count_nonzero(component < 0)),
                ),
            )
    normals = (
        reference[:, :1] * np.cross(frame[:, :, 1], frame[:, :, 2])
        + reference[:, 1:2] * np.cross(frame[:, :, 2], frame[:, :, 0])
        + reference[:, 2:3] * np.cross(frame[:, :, 0], frame[:, :, 1])
    )
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    root_area = np.sqrt(area)[:, None]
    mobility = root_area * (b @ b.T) * root_area.T / coefficient
    inverse = np.linalg.inv(np.eye(count) + mobility / compliance)
    (gap, multiplier) = (np.zeros_like(q), np.zeros_like(q))
    for _ in range(256):
        raw = root_area * gap - multiplier + inverse @ (root_area * (q - gap) + multiplier)
        trial = (raw + multiplier) / root_area
        response = proximal(trial, normals, history, material, compliance)
        gap = response[0]
        multiplier += raw - root_area * gap
    traction = response[5] / compliance
    initial_u = -b.T @ (area[:, None] * traction) / coefficient
    initializer = "fixed_normal_admm"
    y[:size] = basis.T @ initial_u.ravel() / length
    y[size:] = compliance * traction.ravel() / length
    initial = least_squares(
        lambda y: at(y)[0], y, jac=lambda y: at(y)[1], ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=400
    )
    y = initial.x.copy()
    for iteration in range(80):
        (residual, jacobian, auxiliary) = at(y)
        error = float(np.max(np.abs(residual)))
        scale = max(np.max(np.abs(y)), np.max(np.abs(q)) / length, np.finfo(float).tiny)
        if error <= 64 * np.finfo(float).eps * scale:
            break
        step = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        norm = np.linalg.norm(residual)
        for backtrack in range(30):
            candidate = y + 2.0 ** (-backtrack) * step
            updated = at(candidate)[0]
            if np.linalg.norm(updated) <= (1 - 0.0001 * 2.0 ** (-backtrack)) * norm:
                y = candidate
                break
        else:
            raise RuntimeError(f"Semismooth line search failed at {iteration}: {error}")
    else:
        raise RuntimeError(f"Semismooth iteration limit: {error}")
    u = (length * basis @ y[:size]).reshape(-1, 3)
    dual = length * y[size:].reshape(count, 3)
    result = dual / compliance if return_traction else dual
    return (
        u,
        result,
        compliance,
        dict(
            iterations=iteration,
            initial_evaluations=initial.nfev,
            initializer=initializer,
            residual=error,
            closed=int(np.count_nonzero(np.linalg.norm(auxiliary["proximal_gap"], axis=1) == 0)),
            compressed=int(np.count_nonzero(auxiliary["compression"] < 0)),
        ),
    )


def canonical_reaction(values, u, traction, compliance, info):
    (q, frame, reference, history, material, b, gradient, area, mass, dt) = values
    if np.all(history > 0.0):
        (base, L, _) = geometry(q, frame, reference, b, gradient)
        (gap_normal, C, _) = constraint_gradient(base, L, u.ravel())
        gap = q + b @ u
        scale = np.linalg.norm(gap, axis=1) + np.linalg.norm(q, axis=1)
        active = np.flatnonzero(np.abs(gap_normal) <= 64 * np.finfo(float).eps * scale)
        if len(active):
            coefficient = mass[0] / dt[0] ** 2
            corrected_frame = frame + np.einsum("pa,qpb->qab", u, gradient)
            normals = (
                reference[:, :1] * np.cross(corrected_frame[:, :, 1], corrected_frame[:, :, 2])
                + reference[:, 1:2] * np.cross(corrected_frame[:, :, 2], corrected_frame[:, :, 0])
                + reference[:, 2:3] * np.cross(corrected_frame[:, :, 0], corrected_frame[:, :, 1])
            )
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            old_pressure = (
                np.minimum(np.sum((gap + compliance * traction) * normals, axis=1), 0.0) / compliance
            )
            weighted_C = C[active].T * np.sqrt(area[active])[None, :]
            old_reaction = np.sqrt(area[active]) * old_pressure[active] / coefficient
            force = weighted_C @ old_reaction
            (U, singular, Vt) = np.linalg.svd(weighted_C, full_matrices=False)
            rank = np.count_nonzero(singular > max(weighted_C.shape) * np.finfo(float).eps * singular[0])
            if rank < len(active):
                reaction = Vt[:rank].T @ (U[:, :rank].T @ force / singular[:rank])
                null = Vt[rank:].T
                if np.any(reaction > 0.0):
                    optimized = minimize(
                        lambda z: (0.5 * np.dot(z, z), z),
                        np.zeros(null.shape[1]),
                        jac=True,
                        method="SLSQP",
                        constraints=[
                            dict(type="ineq", fun=lambda z: -reaction - null @ z, jac=lambda z: -null)
                        ],
                        options=dict(ftol=1e-14, maxiter=200),
                    )
                    reaction += null @ optimized.x
                if np.all(reaction <= 0.0):
                    pressure = old_pressure.copy()
                    pressure[active] = coefficient * reaction / np.sqrt(area[active])
                    traction = traction + (pressure - old_pressure)[:, None] * normals
                    error = np.max(np.abs(weighted_C @ reaction - force))
                    if error >= 256 * np.finfo(float).eps * max(np.max(np.abs(force)), np.finfo(float).tiny):
                        raise RuntimeError(f"Equivalent reaction changed momentum: {error}")
                    info["reaction_nullity"] = int(len(active) - rank)
    return traction, info


def solve_primal(values, compliance_factor=1.0, return_traction=False):
    u, traction, compliance, info = _solve_primal(values, compliance_factor, return_traction=True)
    traction, info = canonical_reaction(values, u, traction, compliance, info)
    result = traction if return_traction else compliance * traction
    return u, result, compliance, info


def refine_geometry(values, gap_remainder, frame_remainder, compliance_factor=1.0):
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        raise RuntimeError("Interface refinement requires extended-precision CPU arithmetic")
    precise = [np.asarray(value, dtype=np.longdouble) for value in values]
    precise[0] = precise[0] + gap_remainder
    precise[1] = precise[1] + frame_remainder
    seed_u, seed_t, seed_compliance, info = _solve_primal(values, compliance_factor, return_traction=True)
    if info["initializer"] == "closed_linear":
        # With zero opening and history, equilibrium is independent of cohesive parameters.
        q, frame, reference, _, material, b, gradient, area, mass, dt = precise
        coefficient = mass[0] / dt[0] ** 2
        mobility = b @ b.T
        reaction = np.linalg.lstsq(mobility.astype(float), q.astype(float), rcond=None)[0].astype(
            np.longdouble
        )
        for _ in range(8):
            residual = q - mobility @ reaction
            reaction += np.linalg.lstsq(mobility.astype(float), residual.astype(float), rcond=None)[0]
        u = -b.T @ reaction
        root_area = np.sqrt(area)
        weighted = b.T * root_area[None, :]
        pressure = seed_t.astype(np.longdouble) * root_area[:, None] / coefficient
        for _ in range(8):
            residual = -u - weighted @ pressure
            pressure += np.linalg.lstsq(weighted.astype(float), residual.astype(float), rcond=None)[0]
        traction = coefficient * pressure / root_area[:, None]
        frame = frame + np.einsum("pa,qpb->qab", u, gradient)
        normal = (
            reference[:, :1] * np.cross(frame[:, :, 1], frame[:, :, 2])
            + reference[:, 1:2] * np.cross(frame[:, :, 2], frame[:, :, 0])
            + reference[:, 2:3] * np.cross(frame[:, :, 0], frame[:, :, 1])
        )
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        normal_force = np.sum(traction * normal, axis=1)
        active = traction - np.minimum(normal_force, 0.0)[:, None] * normal
        error = float(np.max(np.abs(q + b @ u)))
        scale = max(np.max(np.abs(q)), np.finfo(float).tiny)
        if error <= 64 * np.finfo(float).eps * scale and np.all(
            np.linalg.norm(active, axis=1) <= material[:, 0]
        ):
            info["extended_residual"] = error / float(np.min(2 * material[:, 1] / material[:, 0]))
            return u.astype(float), traction.astype(float), seed_compliance, info, np.zeros_like(q), frame
    info["closed_linear_solution"] = False
    at, basis, compliance, length = setup(precise, compliance_factor, extended=True)
    basis = basis.astype(np.longdouble)
    size = basis.shape[1]
    y = np.r_[
        basis.T @ seed_u.astype(np.longdouble).ravel() / length,
        compliance * seed_t.astype(np.longdouble).ravel() / length,
    ]
    for _ in range(8):
        residual, jacobian, _ = at(y)
        correction = np.linalg.lstsq(jacobian.astype(float), -residual.astype(float), rcond=None)[0]
        y += correction
    residual, _, response = at(y)
    error = float(np.max(np.abs(residual)))
    scale = max(np.max(np.abs(y)), np.max(np.abs(precise[0])) / length, np.finfo(float).tiny)
    if not np.isfinite(error) or error > 64 * np.finfo(float).eps * scale:
        raise RuntimeError(f"Refined interface residual: {error}")
    precise_u = (length * basis @ y[:size]).reshape(seed_u.shape)
    u = precise_u.astype(float)
    traction = (length * y[size:] / compliance).astype(float).reshape(seed_t.shape)
    traction, info = canonical_reaction(values, u, traction, compliance, info)
    info["extended_residual"] = error
    frame = precise[1] + np.einsum("pa,qpb->qab", precise_u, precise[6])
    return u, traction, compliance, info, response["proximal_gap"], frame
