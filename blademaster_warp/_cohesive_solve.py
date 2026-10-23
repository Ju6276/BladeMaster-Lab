"""Reduced nonlinear equations for moving cohesive interfaces."""

import numpy as np
from scipy.optimize import least_squares, minimize


def constitutive(gap, history, material):
    (strength, gc) = material[:, :2].T
    failure = 2 * gc / strength
    radius = np.linalg.norm(gap, axis=1)
    high = np.maximum(history, radius)
    if np.any(high <= 0):
        raise ValueError("Active-set radial response requires positive opening history")
    stiffness = np.where(high < failure, strength * (1 - high / failure) / high, 0.0)
    traction = stiffness[:, None] * gap
    potential = 0.5 * stiffness * radius**2 + 0.5 * strength * np.minimum(high, failure)
    hessian = stiffness[:, None, None] * np.eye(3)
    loading = (radius > history) & (radius < failure)
    hessian[loading] -= (strength[loading] / radius[loading] ** 3)[:, None, None] * np.einsum(
        "qi,qj->qij", gap[loading], gap[loading]
    )
    return (potential, traction, hessian)


def potential_increment(gap, reference, history, material, displacement=None):
    (strength, gc) = material[:, :2].T
    failure = 2 * gc / strength
    (radius, initial) = (np.linalg.norm(gap, axis=1), np.linalg.norm(reference, axis=1))
    displacement = gap - reference if displacement is None else displacement
    delta = np.divide(
        np.sum(displacement * (2 * reference + displacement), axis=1),
        radius + initial,
        out=np.zeros(len(gap)),
        where=radius + initial > 0.0,
    )
    stiffness = np.where(history < failure, strength * (1 - history / failure) / history, 0.0)
    (u, u0) = (np.minimum(radius, history), np.minimum(initial, history))
    du = np.where((radius <= history) & (initial <= history), delta, u - u0)
    (v, v0) = (np.clip(radius, history, failure), np.clip(initial, history, failure))
    both_loading = (radius >= history) & (radius <= failure) & (initial >= history) & (initial <= failure)
    dv = np.where(both_loading, delta, v - v0)
    return 0.5 * stiffness * du * (u + u0) + strength * dv * (1 - (v + v0) / (2 * failure))


def skew(x):
    (a, b, c) = x
    return np.array([[0.0, -c, b], [c, 0.0, -a], [-b, a, 0.0]])


def constrained_linear_solve(H, A, rhs_u, rhs_c):
    if not len(A):
        return (np.linalg.solve(H, rhs_u), np.empty(0))
    (U, s, Vt) = np.linalg.svd(A, full_matrices=True)
    rank = np.sum(s > s[0] * max(A.shape) * np.finfo(float).eps)
    (Ur, Vr) = (U[:, :rank], Vt[:rank].T)
    null = Vt[rank:].T
    particular = Vr @ (Ur.T @ rhs_c / s[:rank])
    u = particular + null @ np.linalg.solve(null.T @ H @ null, null.T @ (rhs_u - H @ particular))
    multiplier = Ur @ (Vr.T @ (rhs_u - H @ u) / s[:rank])
    return (u, multiplier)


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


def smooth_primal(values, initial_tolerance=1e-08):
    (q, frame, normal, history, material, b, gradient, area, mass, dt) = values
    coefficient = float(mass[0] / dt[0] ** 2)
    (base, L, weights) = geometry(q, frame, normal, b, gradient)
    basis_matrix = weights.transpose(1, 0, 2).reshape(-1, b.shape[1])
    (_, singular, vt) = np.linalg.svd(basis_matrix, full_matrices=False)
    rank = np.sum(singular > singular.max() * max(basis_matrix.shape) * np.finfo(float).eps)
    V = vt[:rank].T
    vector_basis = np.kron(V, np.eye(3))
    reduced = b @ V
    reduced_L = L @ vector_basis
    (_, base_t, _) = constitutive(q, history, material)
    c0 = constraints(base, reduced_L, np.zeros(rank * 3))[0]
    displacement = b @ (b.T @ (area[:, None] * base_t)) / coefficient
    scale = max(np.max(np.abs(c0)), np.max(np.abs(displacement)))
    if scale == 0:
        scale = np.max(2 * material[:, 1] / material[:, 0])
    energy_scale = coefficient * scale**2

    def at(y):
        displacement = scale * reduced @ y.reshape(rank, 3)
        g = q + displacement
        (phi, t, h) = constitutive(g, history, material)
        value = (
            0.5 * (y @ y)
            + np.sum(area * potential_increment(g, q, history, material, displacement)) / energy_scale
        )
        grad = y + (reduced.T @ (area[:, None] * t)).ravel() / (coefficient * scale)
        H = (
            np.eye(len(y))
            + np.einsum("qi,qab,qj,q->iajb", reduced, h, reduced, area).reshape(len(y), len(y)) / coefficient
        )
        (c, C, Hc) = constraints(base, reduced_L, scale * y)
        return (value, grad, H, c / scale, C, scale * Hc)

    optimized = minimize(
        lambda y: at(y)[:2],
        np.zeros(rank * 3),
        jac=True,
        method="SLSQP",
        constraints=[dict(type="ineq", fun=lambda y: at(y)[3], jac=lambda y: at(y)[4])],
        options=dict(ftol=initial_tolerance, maxiter=400),
    )
    y = optimized.x.copy()
    active = np.flatnonzero(optimized.multipliers > 0.0)
    pressure = optimized.multipliers[active].copy()
    for iteration in range(80):
        (value, grad, H, c, C, Hc) = at(y)
        A = C[active]
        hessian = H - np.einsum("q,qij->ij", pressure, Hc[active])
        residual = np.r_[grad - A.T @ pressure, c[active]]
        (du, dp) = constrained_linear_solve(hessian, A, -residual[: len(y)], -residual[len(y) :])
        increment = np.r_[du, -dp]
        y += increment[: len(y)]
        pressure += increment[len(y) :]
        if len(pressure) and pressure.min() < 0:
            delete = int(np.argmin(pressure))
            active = np.delete(active, delete)
            pressure = np.delete(pressure, delete)
            continue
        c = at(y)[3]
        violation = int(np.argmin(c))
        if c[violation] < -1e-12 and violation not in active:
            active = np.r_[active, violation]
            pressure = np.r_[pressure, 0.0]
            continue
        if np.max(np.abs(residual)) < 1e-11:
            break
    else:
        raise RuntimeError("Moving constraint KKT refinement failed")
    u = (scale * vector_basis @ y).reshape(-1, 3)
    nu = np.zeros(len(q))
    nu[active] = -scale * pressure
    g = q + b @ u
    (phi, t, h) = constitutive(g, history, material)
    (c, C, Hc) = constraints(base, reduced_L, scale * y)
    r_u = u.ravel() + (b.T @ (area[:, None] * t)).ravel() / coefficient + vector_basis @ (C.T @ nu)
    failure = 2 * material[:, 1] / material[:, 0]
    residual = float(np.max(np.abs(r_u)) / failure.min())
    active_error = np.max(np.abs(c[active]) / failure[active], initial=0.0)
    if residual > 1e-11 or np.min(c / failure) < -1e-11 or active_error > 1e-11:
        raise RuntimeError(f"Moving constraint residual {residual}, {np.min(c / failure)}")
    A = C[active]
    return (
        u,
        nu,
        active,
        None,
        dict(
            iterations=optimized.nit + iteration + 1,
            active=len(active),
            rank=int(rank),
            optimizer_success=bool(optimized.success),
            residual=residual,
            active_residual=float(active_error),
            max_pressure_pa=float(np.max(np.abs(coefficient * nu / area))),
            active_singular_values=np.linalg.svd(A, compute_uv=False).tolist(),
        ),
    )


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
    for i in range(count):
        if history[i] >= failure[i]:
            continue
        opened[i] = 0.0
        derivative[i] = 0.0
        if history[i] > 0:
            stiffness = strength[i] * (1 - history[i] / failure[i]) / history[i]
            derivative[i] = 1 / (1 + compliance * stiffness)
            opened[i] = radius[i] * derivative[i]
        if opened[i] > history[i] or (history[i] == 0 and radius[i] > compliance * strength[i]):
            derivative[i] = 1 / (1 - compliance * strength[i] / failure[i])
            opened[i] = (radius[i] - compliance * strength[i]) * derivative[i]
            if opened[i] >= failure[i]:
                opened[i] = radius[i]
                derivative[i] = 1.0
    gap = np.zeros_like(z)
    dz = np.zeros((count, 3, 3))
    dn = np.zeros((count, 3, 3))
    for i in range(count):
        if radius[i] == 0:
            radial = derivative[i] * np.eye(3)
        else:
            direction = positive[i] / radius[i]
            ratio = opened[i] / radius[i]
            gap[i] = ratio * positive[i]
            radial = ratio * np.eye(3) + (derivative[i] - ratio) * np.outer(direction, direction)
        projection = np.eye(3) - closed[i] * np.outer(normal[i], normal[i])
        dz[i] = radial @ projection
        dn[i] = -float(closed[i]) * radial @ (component[i] * np.eye(3) + np.outer(normal[i], z[i]))
    return (gap, dz, dn, compression, closed)


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
    inverse_mobility = 1 / np.linalg.eigvalsh(mobility.astype(float)).max()
    compliance = compliance_factor / (inverse_mobility + 2 * softening)
    if compliance * softening >= 1:
        raise ValueError("Proximal subproblem is not strictly convex")
    length = np.min(2 * material[:, 1] / material[:, 0])
    size = basis.shape[1]

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
        (prox, D_z, D_n, compression, closed) = proximal(z, normals, history, material, compliance)
        proximal_base = base.copy()
        proximal_base[:, :3] += prox - gap
        (c, C, Hc) = constraints(proximal_base, L, u)
        face = np.einsum("qai,qa->qi", normal_gradient, prox)
        weights = area / (coefficient * compliance)
        Ru = u + np.einsum("qai,qa,q->i", L[:, :3], dual, weights)
        Ru += np.einsum("qi,q,q->i", face, compression, weights)
        Rg = gap - prox
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
            Jgu[i] = B - proximal_u
            Jgs[i, :, i, :] = -D_z[i]
        J = np.block([[Juu, Jus.reshape(size, -1)], [Jgu.reshape(-1, size), Jgs.reshape(len(q) * 3, -1)]])
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
        candidate_t = coefficient * reaction / area[:, None]
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
        gap = proximal(trial, normals, history, material, compliance)[0]
        multiplier += raw - root_area * gap
    traction = (trial - gap) / compliance
    initial_u = -b.T @ (area[:, None] * traction) / coefficient
    initializer = "fixed_normal_admm"
    if np.all(history > 0):
        (initial_u, nu, _, _, _) = smooth_primal(values)
        gap = q + b @ initial_u
        updated_frame = frame + np.einsum("pa,qpb->qab", initial_u, gradient)
        normals = (
            reference[:, :1] * np.cross(updated_frame[:, :, 1], updated_frame[:, :, 2])
            + reference[:, 1:2] * np.cross(updated_frame[:, :, 2], updated_frame[:, :, 0])
            + reference[:, 2:3] * np.cross(updated_frame[:, :, 0], updated_frame[:, :, 1])
        )
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        traction = constitutive(gap, history, material)[1] + (coefficient * nu / area)[:, None] * normals
        initializer = "smooth_energy"
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
        reaction = seed_t.astype(np.longdouble) * area[:, None] / coefficient
        for _ in range(8):
            residual = q - mobility @ reaction
            reaction += np.linalg.lstsq(mobility.astype(float), residual.astype(float), rcond=None)[0]
        u = -b.T @ reaction
        traction = coefficient * reaction / area[:, None]
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
