"""Minimum-pressure reactions with strength and unilateral-contact constraints."""

from dataclasses import dataclass

import numpy as np

from ._compensated import Expansion as E
from ._compensated_linalg import column_basis, solve, minimum_norm
from ._cohesive_expansion import assign, cross, divide, norm, outer, positive, skew


@dataclass(frozen=True)
class ReactionBranches:
    closed: np.ndarray
    contact: np.ndarray
    constrain_normal: bool = True
    compressed: np.ndarray | None = None
    normal_active: np.ndarray | None = None
    strength_active: np.ndarray | None = None
    reaction_trial: np.ndarray | None = None

    def __post_init__(self):
        for name in ("closed", "contact", "compressed", "normal_active", "strength_active"):
            original = getattr(self, name)
            values = (
                np.zeros_like(self.closed) if original is None else np.asarray(original, dtype=bool).copy()
            )
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.reaction_trial is not None:
            trial = np.asarray(self.reaction_trial, dtype=float).copy()
            trial.setflags(write=False)
            object.__setattr__(self, "reaction_trial", trial)


def project(value, entries, length):
    output = E(value.high.copy(), value.low.copy())
    jacobian = E(np.zeros((value.shape[0], value.shape[0])))
    for _, _, span, closed, normal, factor, limit, normal_mode in entries:
        part = value[span]
        if normal_mode:
            component = (part * normal).sum()
            projected = normal_mode == 2 or positive(-component)
            tangent = E(np.eye(3)) - (outer(normal, normal) if projected else 0.0)
            tensile = part - (component * normal if projected else 0.0)
            radius = norm(tensile)
            bound = divide(limit, length * factor)
            if positive(radius - bound):
                direction = divide(tensile, radius)
                assign(output, span, bound * direction)
                assign(
                    jacobian, (span, span), divide(bound, radius) * (tangent - outer(direction, direction))
                )
            else:
                assign(output, span, tensile)
                assign(jacobian, (span, span), tangent)
            continue
        if not closed:
            if positive(-part[0]):
                assign(jacobian, (span, span), np.ones((1, 1)))
            else:
                assign(output, span, np.zeros(1))
            continue
        component = (part * normal).sum()
        compression = component * normal if positive(-component) else E(np.zeros(3))
        tensile = part - compression
        radius = norm(tensile)
        bound = divide(limit, length * factor)
        if not positive(radius - bound):
            assign(jacobian, (span, span), np.eye(3))
            continue
        direction = divide(tensile, radius)
        tangent = E(np.eye(3)) - (outer(normal, normal) if positive(-component) else 0.0)
        derivative = E(np.eye(3)) - tangent + divide(bound, radius) * (tangent - outer(direction, direction))
        assign(output, span, compression + bound * direction)
        assign(jacobian, (span, span), derivative)
    return output, jacobian


def select_constrained(mapping, old, initial, entries, length):
    projected, _ = project(initial, entries, length)
    if np.array_equal(projected.high, initial.high) and np.array_equal(projected.low, initial.low):
        return initial, initial
    tangent = E(np.eye(mapping.shape[1]))
    for _, _, span, closed, normal, factor, limit, normal_mode in entries:
        if normal_mode == 2:
            assign(tangent, (span, span), E(np.eye(3)) - outer(normal, normal))
    effective = mapping @ tangent
    basis = column_basis(effective.T)
    if not basis.shape[1]:
        zero = E(np.zeros_like(old.high))
        return zero, zero
    coordinates = minimum_norm(effective.T, basis).T
    matrix = coordinates @ mapping
    target = matrix @ old
    dual = solve(matrix @ matrix.T, target)

    def evaluate(value):
        trial = matrix.T @ value
        response, tangent = project(trial, entries, length)
        residual = matrix @ response - target
        objective = 0.5 * (residual * residual).sum()
        return objective, residual, matrix @ tangent @ matrix.T, response, trial

    for iteration in range(81):
        objective, residual, jacobian, response, trial = evaluate(dual)
        scale = max(
            float(np.max(abs(target.high))),
            float(np.max(abs(matrix.high) @ abs(response.high))),
            np.finfo(float).tiny,
        )
        relative = float(np.max(abs(residual.high))) / scale
        if relative <= 64 * np.finfo(float).eps ** 2:
            return response, trial
        if iteration == 80:
            break
        try:
            direction = solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            direction = minimum_norm(jacobian, -residual)
        slope = (residual * (jacobian @ direction)).sum()
        if not positive(-slope):
            direction = -divide(jacobian.T @ residual, (jacobian * jacobian).sum())
            slope = (residual * (jacobian @ direction)).sum()
        step = 1.0
        for trial in range(81):
            candidate = dual + step * direction
            candidate_objective, *_ = evaluate(candidate)
            if not positive(candidate_objective - objective - 1e-4 * step * slope):
                dual = candidate
                break
            step *= 0.5
        else:
            raise RuntimeError("Reaction projection line search failed")
    raise RuntimeError(f"Reaction projection equality residual: {relative}")


def select(equations, y, area, integrated):
    size = equations.size
    u = equations.length * y[:size]
    columns, original, entries = [], [], []
    compressed_rows = np.zeros(len(area), dtype=bool)
    normal_active = np.zeros(len(area), dtype=bool)
    strength_active = np.zeros(len(area), dtype=bool)
    reaction_trial = None

    def signature(offset, mapping, extent):
        return tuple(
            value.tobytes()
            for value in [
                offset.high[:extent],
                offset.low[:extent],
                mapping.high[:extent],
                mapping.low[:extent],
            ]
        )

    fixed_gaps, fixed_tangents = set(), set()
    if not equations.constrain_normal:
        for j, (offset, mapping) in enumerate(zip(equations.base, equations.maps)):
            if area[j] == 0 or equations.history[j] != 0:
                continue
            local = offset + mapping @ u
            normal = cross(local[3:6], local[6:9])
            normal = divide(normal, norm(normal))
            trial = local[:3] + equations.length * y[size + 3 * j : size + 3 * j + 3]
            normal_part = (trial * normal).sum()
            compressed = positive(-normal_part)
            opening_trial = trial - (normal_part * normal if compressed else 0)
            limit = equations.compliance[j] * equations.material[j, 0]
            if positive(limit - norm(opening_trial)):
                target = fixed_tangents if compressed else fixed_gaps
                target.add(signature(offset, mapping, 9 if compressed else 3))
    for i, (offset, mapping) in enumerate(zip(equations.base, equations.maps)):
        if area[i] == 0.0:
            continue
        local = offset + mapping @ u
        g, a, b = local[:3], local[3:6], local[6:9]
        oriented = cross(a, b)
        measure = norm(oriented)
        normal = divide(oriented, measure)
        normal_map = divide(E(np.eye(3)) - normal[:, None] * normal[None, :], measure)
        normal_map = normal_map @ (-skew(b) @ mapping[3:6] + skew(a) @ mapping[6:9])
        span = slice(size + 3 * i, size + 3 * i + 3)
        dual = y[span]
        trial = g + equations.length * dual
        component = (trial * normal).sum()
        compressed = positive(-component)
        compressed_rows[i] = compressed
        tensile = trial - (component * normal if compressed else 0.0)
        radius = norm(tensile)
        limit = equations.compliance[i] * equations.material[i, 0]
        constrained = (
            signature(offset, mapping, 3) in fixed_gaps or signature(offset, mapping, 9) in fixed_tangents
        )
        closed = equations.history[i] == 0.0 and (not positive(radius - limit) or constrained)
        local_scale = abs(offset.high) + abs(mapping.high) @ abs(u.high)
        oriented_scale = np.array([
            local_scale[4] * local_scale[8] + local_scale[5] * local_scale[7],
            local_scale[5] * local_scale[6] + local_scale[3] * local_scale[8],
            local_scale[3] * local_scale[7] + local_scale[4] * local_scale[6],
        ])
        gap_scale = float(local_scale[:3] @ oriented_scale)
        gap_roundoff = 64 * np.finfo(float).eps ** 2 * gap_scale
        at_boundary = abs(float((g * oriented).sum().numpy())) <= gap_roundoff
        if not closed and (not equations.constrain_normal or not (compressed or at_boundary)):
            continue
        root_area = E(np.sqrt(area[i]))
        root_area = root_area + divide(E(area[i]) - root_area * root_area, 2 * root_area)
        factor = root_area if integrated else divide(1.0, root_area)
        start = len(original)
        if closed:
            block = factor * equations.factors[i] * mapping[:3].T
            if not equations.constrain_normal and compressed:
                block = factor * equations.factors[i] * (mapping[:3].T - (g * normal).sum() * normal_map.T)
            old = divide(dual, factor)
            for j in range(3):
                columns.append(block[:, j])
                original.append(old[j])
        else:
            block = factor * equations.factors[i] * (mapping[:3].T @ normal + normal_map.T @ g)
            columns.append(block)
            original.append(divide((dual * normal).sum(), factor))
        normal_mode = 0 if equations.constrain_normal else (2 if compressed else 1)
        entries.append((i, span, slice(start, len(original)), closed, normal, factor, limit, normal_mode))
    closed_rows = np.zeros(len(area), dtype=bool)
    contact_rows = np.zeros(len(area), dtype=bool)
    for i, _, _, closed, *_ in entries:
        closed_rows[i] = closed
        contact_rows[i] = not closed

    def branches():
        return ReactionBranches(
            closed_rows,
            contact_rows,
            equations.constrain_normal,
            compressed_rows,
            normal_active,
            strength_active,
            reaction_trial,
        )

    if not original:
        return y, 0, branches()
    mapping = E(np.column_stack([v.high for v in columns]), np.column_stack([v.low for v in columns]))
    old = E(np.array([v.high for v in original]), np.array([v.low for v in original]))
    basis = column_basis(mapping.T)
    rank = basis.shape[1]
    nullity = int(len(original) - rank)
    if not nullity:
        return y, 0, branches()
    selected = basis @ (basis.T @ old)
    selected, projected_from = select_constrained(mapping, old, selected, entries, equations.length)
    if not equations.constrain_normal:
        first = entries[0][0]
        compliance = divide(equations.compliance[first], area[first] if integrated else 1.0)
        reaction_trial = (divide(equations.length, compliance) * projected_from).numpy()
    result = E(y.high.copy(), y.low.copy())
    for i, span, selected_span, closed, normal, factor, limit, normal_mode in entries:
        chosen = factor * selected[selected_span]
        if closed:
            load = equations.length * chosen
            compression = (load * normal).sum()
            tensile = (
                load if normal_mode else load - (compression * normal if positive(-compression) else 0.0)
            )
            if normal_mode:
                before = projected_from[selected_span]
                component = (before * normal).sum()
                normal_active[i] = normal_mode == 2 or positive(-component)
                tangent = before - (component * normal if normal_active[i] else 0.0)
                strength_active[i] = positive(norm(tangent) - divide(limit, equations.length * factor))
            bound_error = float(divide(norm(tensile) - limit, limit).numpy())
            if bound_error > 64 * np.finfo(float).eps ** 2:
                raise RuntimeError(f"Canonical strength bound residual: {bound_error}")
        else:
            if positive(chosen[0]):
                raise RuntimeError("Canonical contact bound requires a constrained projection")
            chosen = y[span] + (chosen[0] - (y[span] * normal).sum()) * normal
        assign(result, span, chosen)
    error = np.max(abs(equations(result)[0].high))
    scale = max(np.max(abs(result.high)), np.finfo(float).tiny)
    if error > 64 * np.finfo(float).eps ** 2 * scale:
        raise RuntimeError(f"Selected reaction changed equilibrium: {error / scale}")
    return result, nullity, branches()


def reaction_geometry(values, u, traction, compliance, integrated, remainders, map_remainders=None):
    from . import _cohesive_solve as solver
    from ._cohesive_expansion import stack

    q, frame, reference, history, material, b, gradient, area, mass, dt = values
    if remainders is None:
        remainders = (None, None, None)
    qlow, flow, ulow = [0.0 if a is None else a for a in remainders]
    low_b, low_g = (0.0, np.zeros_like(gradient)) if map_remainders is None else map_remainders
    displacement = E(u, ulow)
    gap = E(q, qlow) + E(b, low_b) @ displacement
    mapped = E(frame, flow) + stack(
        [E(gradient[:, :, c], low_g[:, :, c]) @ displacement for c in range(3)], axis=-1
    )
    normals = []
    for i in range(len(q)):
        F = mapped[i]
        vector = (
            reference[i, 0] * cross(F[:, 1], F[:, 2])
            + reference[i, 1] * cross(F[:, 2], F[:, 0])
            + reference[i, 2] * cross(F[:, 0], F[:, 1])
        )
        normals.append(divide(vector, norm(vector)))
    normals = stack(normals)
    trial = gap + compliance * E(traction)
    component = (trial * normals).sum(axis=1)
    compressed = component.high < 0.0
    tensile = (
        trial
        - E(np.where(compressed, component.high, 0.0), np.where(compressed, component.low, 0.0))[:, None]
        * normals
    )
    limit = E(compliance) * material[:, 0] * (area if integrated else 1.0)
    excess = (tensile * tensile).sum(axis=1) - limit * limit
    closed = (area > 0.0) & (history == 0.0) & (excess.high <= 0.0)
    contact = (area > 0.0) & ~closed & compressed
    base, mapping, _ = solver.geometry(gap.high, mapped.high, reference, b, gradient)
    _, C, _ = solver.constraint_gradient(base, mapping, np.zeros(u.size))
    return gap.high, mapped.high, normals.high, C, closed, contact
