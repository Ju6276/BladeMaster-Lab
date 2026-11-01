"""Compensated moving-interface equilibrium and explicit reaction branches."""

from dataclasses import dataclass

import numpy as np

from ._compensated import Expansion as E
from ._compensated_linalg import column_basis, minimum_norm, solve as solve_linear
from ._cohesive_solve import shared_intact_rows
from ._cohesive_expansion import Equations, divide, norm, root, cross, stack, assign, positive
from ._cohesive_refinement import refine
from ._cohesive_projection import ReactionBranches, select


@dataclass(frozen=True)
class EquilibriumResult:
    displacement: E
    traction: E
    gap: E
    frame: E
    compliance: float
    info: dict
    branches: ReactionBranches
    contacts: tuple = ()


def kron3(a):
    return E(np.kron(a.high, np.eye(3)), np.kron(a.low, np.eye(3)))


def closed_equilibrium(
    values, gap, frame, displacement, traction, branches, inertia, integrated_force, map_remainders=None
):
    support = values[7] > 0.0
    if not np.any(support) or not np.all(branches.closed[support]):
        return displacement, traction, False
    full_mapping = E(values[5], 0.0 if map_remainders is None else map_remainders[0])
    mapping = full_mapping[support]
    target = gap[support]
    if inertia is None:
        result = -minimum_norm(mapping, target)
    else:
        basis = column_basis(mapping.T)
        if basis.shape[1]:
            coordinates = minimum_norm(mapping @ basis, target)
            modes = inertia.solve_compensated(basis)
            result = -modes @ solve_linear(basis.T @ modes, coordinates)
        else:
            result = E(np.zeros_like(displacement.high))
    residual = target + mapping @ result
    scale = max(
        float(np.max(abs(target.high))),
        float(np.max(abs(mapping.high) @ abs(result.high))),
        np.finfo(float).tiny,
    )
    if np.max(abs(residual.high)) > 64 * np.finfo(float).eps ** 2 * scale:
        raise RuntimeError("Closed displacement constraints are inconsistent")
    coefficient = divide(E(values[8][0]), E(values[9][0]) * values[9][0])
    momentum = result if inertia is None else inertia.apply(result)
    weights = root(E(values[7][support]))
    reaction_map = full_mapping[support].T * weights[None, :]
    traction = E(traction.high.copy(), traction.low.copy())
    for iteration in range(5):
        force = traction if integrated_force else E(values[7][:, None]) * traction
        reaction = divide(full_mapping.T @ force, coefficient)
        residual = momentum + reaction
        scale = max(
            float(np.max(abs(momentum.high))),
            float(np.max(
                (abs(full_mapping.high) + abs(full_mapping.low)).T
                @ (abs(force.high) + abs(force.low)) / abs(coefficient.numpy())
            )),
            np.finfo(float).tiny,
        )
        if np.max(abs(residual.high)) <= 64 * np.finfo(float).eps ** 2 * scale:
            break
        if iteration == 4:
            raise RuntimeError("Closed displacement changed momentum balance")
        correction = weights[:, None] * minimum_norm(reaction_map, -coefficient * residual)
        if not integrated_force:
            correction = divide(correction, E(values[7][support, None]))
        assign(traction, support, traction[support] + correction)
    gradient = E(values[6], 0.0 if map_remainders is None else map_remainders[1])
    minimum = weights[:, None] * minimum_norm(reaction_map, -coefficient * momentum)
    if not integrated_force:
        minimum = divide(minimum, E(values[7][support, None]))
    minimum_traction = E(np.zeros_like(traction.high))
    assign(minimum_traction, support, minimum)
    linear = True
    for i in np.flatnonzero(support):
        mapped = frame[i] + result.T @ gradient[i]
        n = values[2][i]
        normal = (
            n[0] * cross(mapped[:, 1], mapped[:, 2])
            + n[1] * cross(mapped[:, 2], mapped[:, 0])
            + n[2] * cross(mapped[:, 0], mapped[:, 1])
        )
        normal = divide(normal, norm(normal))
        component = (traction[i] * normal).sum()
        tensile = traction[i] - (component * normal if positive(-component) else 0.0)
        limit = E(values[4][i, 0]) * (values[7][i] if integrated_force else 1.0)
        if float(divide(norm(tensile) - limit, limit).numpy()) > 64 * np.finfo(float).eps ** 2:
            raise RuntimeError("Closed refinement left the strength bound")
        component = (minimum_traction[i] * normal).sum()
        tensile = minimum_traction[i] - (component * normal if positive(-component) else 0.0)
        if not positive(limit - norm(tensile)):
            linear = False
    if linear:
        traction = minimum_traction
    return result, traction, linear and bool(np.all(support))


def constraint_geometry(reference, paired_b, paired_g):
    count = paired_b.shape[0]
    if not count:
        return [E(np.empty((0, 3))), E(np.empty((0, 3)))], [paired_b, paired_b, paired_b]
    tangents = [[], []]
    for original in reference:
        n = divide(E(original), norm(E(original)))
        axis = np.eye(3)[np.argmin(np.abs(n.high))]
        a = cross(n, E(axis))
        a = divide(a, norm(a))
        tangents[0].append(a)
        tangents[1].append(cross(n, a))
    normals = [stack(t) for t in tangents]
    weights = [paired_b]
    for tangent in normals:
        weights.append(stack([paired_g[i] @ tangent[i] for i in range(count)]))
    return normals, weights



def solve(
    values,
    gap_remainder,
    frame_remainder,
    seed,
    *,
    integrated_force=False,
    inertia=None,
    map_remainders=None,
    contacts=None,
    constrain_normal=True,
):
    seed_u, seed_t, compliance, info, _, _ = seed
    q, frame, reference, history, material, B, G, area, mass, dt = values
    count, particles = B.shape
    low_b, low_g = (0.0, np.zeros_like(G)) if map_remainders is None else map_remainders
    paired_b, paired_g = E(B, low_b), E(G, low_g)
    normals, weights = constraint_geometry(reference, paired_b, paired_g)
    all_weights = weights if contacts is None else [*weights, contacts.mapping.cpu()]
    whole = E(np.concatenate([w.high for w in all_weights]), np.concatenate([w.low for w in all_weights]))
    modes = column_basis(whole.T)
    rank = modes.shape[1]
    if inertia is not None:
        modes = inertia.solve_compensated(modes)
    for j in range(rank):
        vector = modes[:, j]
        for _ in range(2):
            for k in range(j):
                vector = vector - (modes[:, k] * vector).sum() * modes[:, k]
        vector = divide(vector, norm(vector))
        assign(modes, (slice(None), j), vector)
    basis = kron3(modes)
    reduced = kron3(modes.T @ (modes if inertia is None else inertia.apply(modes)))
    mapped = [w @ modes for w in weights]
    local_maps = [kron3(stack([w[i] for w in mapped])) for i in range(count)]
    frames = E(frame, frame_remainder)
    qpair = E(q, gap_remainder)
    base = []
    for i in range(count):
        terms = [qpair[i], frames[i] @ normals[0][i], frames[i] @ normals[1][i]]
        base.append(E(np.concatenate([t.high for t in terms]), np.concatenate([t.low for t in terms])))
    if count:
        failures = divide(2 * E(material[:, 1]), E(material[:, 0]))
        length = failures[np.argmin(failures.high)]
    else:
        length = E(contacts.length_scale(dt[0]))
    coefficient = divide(E(mass[0]), E(dt[0]) * dt[0])
    local_compliance = [E(compliance) * (a if integrated_force else 1.0) for a in area]
    factors = [divide(E(1.0 if integrated_force else a), coefficient * compliance) for a in area]
    shared = shared_intact_rows(qpair.high, history, paired_b.high)
    if integrated_force:
        shared &= area > 0
    first = divide(basis.T @ E.from_value(seed_u.ravel()), length)
    second = divide(compliance * E(seed_t.ravel()), length)
    y = E(np.r_[first.high, second.high], np.r_[first.low, second.low])
    arguments = (base, local_maps, reduced, length, local_compliance, history, material, factors, shared)
    if contacts is None:
        equations = Equations(*arguments, constrain_normal=constrain_normal)
    else:
        from ._contact_equilibrium import ContactEquations

        mapped_contacts = contacts.mapping.cpu() @ modes
        rows = [dict(row, map=kron3(mapped_contacts[i : i + 1])) for i, row in enumerate(contacts.rows())]
        equations = ContactEquations(
            *arguments,
            contacts=rows,
            evaluate_contact=contacts.evaluate,
            mass=mass[0],
            duration=dt[0],
            constrain_normal=constrain_normal,
        )
        rhs = E(np.zeros(equations.size))
        tangent = E(np.zeros((equations.size, equations.size)))
        for mapping, load, derivative, *_ in equations.contact_response(y):
            rhs = rhs + divide(mapping.T @ load, coefficient * length)
            tangent = tangent + divide(mapping.T @ derivative, coefficient)
        if equations.size:
            assign(y, slice(0, equations.size), y[: equations.size] + solve_linear(reduced - tangent, rhs))
    weights = np.ones(count)
    if integrated_force and count:
        np.divide(np.max(area), area, out=weights, where=area > 0)
    if y.high.size:
        y, gaps, trace = refine(equations, y, np.repeat(weights, 3))
    else:
        gaps, trace = [], []
    y, nullity, branches = select(equations, y, area, integrated_force)
    residual, _, gaps = equations(y)
    error = float(np.max(np.abs(residual.high), initial=0.0))
    iteration = sum(row["iterations"] for row in trace)
    size = basis.shape[1]
    displacement = (length * (basis @ y[:size])).reshape(particles, 3)
    force = divide(length * y[size:], E(compliance)).reshape(count, 3)
    closed_linear_solution = False
    if contacts is None and constrain_normal:
        displacement, force, closed_linear_solution = closed_equilibrium(
            values, qpair, frames, displacement, force, branches, inertia, integrated_force, map_remainders
        )
    frame_out = stack([frames[i] + displacement.T @ paired_g[i] for i in range(count)]) if count else frames
    details = dict(
        info,
        paired_iterations=iteration,
        paired_residual=error,
        regularization=trace,
        canonical_nullity=nullity,
        closed_linear_solution=closed_linear_solution,
    )
    contact_response = () if contacts is None else tuple(equations.contact_response(y))
    gap_out = stack(gaps) if count else qpair
    return EquilibriumResult(
        displacement, force, gap_out, frame_out, compliance, details, branches, contact_response
    )
