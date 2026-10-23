"""Moving cohesive-interface equilibrium and its implicit Warp adjoint.

Interface energies and tractions are measured per reference area. The input
maps carry the TLMPM transfer; the solve returns a particle displacement.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from .cohesive import cohesive_proximal_response, cohesive_contact_response, surface_normal
from .precision import launch_double
from ._cohesive_solve import geometry, solve_primal


@wp.func
def gap_at(q: wp.array(dtype=wp.vec3), b: wp.array2d(dtype=float), u: wp.array(dtype=wp.vec3), i: int):
    correction = wp.vec3(0.0)
    for p in range(u.shape[0]):
        correction += b[i, p] * u[p]
    return q[i] + correction


@wp.func
def residual_force(
    b: wp.array2d(dtype=float), area: wp.array(dtype=float), traction: wp.array(dtype=wp.vec3), p: int
):
    f = wp.vec3(0.0)
    for i in range(traction.shape[0]):
        f += b[i, p] * area[i] * traction[i]
    return f


@wp.func
def frame_rate(gradient: wp.array2d(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3), i: int):
    rate = wp.mat33(0.0)
    for p in range(velocity.shape[0]):
        rate += wp.outer(velocity[p], gradient[i, p])
    return rate


@wp.func
def frame_force(
    gradient: wp.array2d(dtype=wp.vec3), stress: wp.array(dtype=wp.mat33), area: wp.array(dtype=float), p: int
):
    force = wp.vec3(0.0)
    for i in range(stress.shape[0]):
        force += area[i] * stress[i] * gradient[i, p]
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
    residual: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    force = residual_force(b, area, traction, p) + frame_force(gradient, stress, area, p)
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


@wp.kernel
def subtract(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), result: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    result[i] = a[i] - b[i]


@wp.kernel
def normal_gap(gap: wp.array(dtype=wp.vec3), normal: wp.array(dtype=wp.vec3), result: wp.array(dtype=float)):
    i = wp.tid()
    result[i] = wp.dot(gap[i], normal[i])


def _evaluate(inputs, u, traction, compliance):
    count = len(inputs[0])
    device = inputs[0].device

    def zeros(n, vector=False):
        return wp.zeros(n, dtype=wp.vec3d if vector else wp.float64, device=device, requires_grad=True)

    kinds = [wp.vec3d, wp.vec3d, wp.mat33d, wp.vec3d, wp.vec3d, wp.vec3d, wp.float64]
    raw = [wp.zeros(count, dtype=kind, device=device, requires_grad=True) for kind in kinds]
    launch_double(kinematics, count, inputs=[*inputs, u, traction, compliance], outputs=raw, device=device)
    prox = [zeros(count, i < 2) for i in range(5)]
    launch_double(
        cohesive_proximal_response,
        count,
        inputs=[raw[1], raw[3], inputs[3], raw[4]],
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
        inputs=[inputs[5], inputs[6], inputs[7], raw[5], stress, inputs[8], inputs[9], u],
        outputs=[Ru],
        device=device,
    )
    launch_double(subtract, count, inputs=[raw[0], prox[0]], outputs=[Rg], device=device)
    result = [prox[0], raw[5], prox[2], prox[3], prox[4], raw[2], stress, constraint]
    return (result, Ru, Rg)


def _solve_arrays(inputs, compliance_scale=1.0):
    device = inputs[0].device

    def array(value, vector=False, grad=False):
        return wp.array(
            np.asarray(value), dtype=wp.vec3d if vector else wp.float64, device=device, requires_grad=grad
        )

    values = [a.numpy().copy() for a in inputs]
    (u_numpy, traction_numpy, compliance, info) = solve_primal(values, compliance_scale, return_traction=True)
    (u, traction) = (array(u_numpy, True, True), array(traction_numpy, True, True))
    tape = wp._src.context.runtime.tape
    if tape is not None:
        (_, _, weights) = geometry(values[0], values[1], values[2], values[5], values[6])
        matrix = weights.transpose(1, 0, 2).reshape(-1, values[5].shape[1])
        (_, singular, Vt) = np.linalg.svd(matrix, full_matrices=False)
        rank = np.count_nonzero(singular > singular[0] * max(matrix.shape) * np.finfo(float).eps)
        basis = np.kron(Vt[:rank].T, np.eye(3))
        size = basis.shape[1]

        def backward():
            rhs_u = u.grad.numpy().ravel()
            rhs_t = traction.grad.numpy().ravel() / compliance
            rhs = np.r_[basis.T @ rhs_u, rhs_t]
            clones = [
                wp.array(v, dtype=a.dtype, device=a.device, requires_grad=True)
                for (v, a) in zip(values, inputs)
            ]
            fixed_u = array(u_numpy, True, True)
            fixed_t = array(traction_numpy, True, True)
            with wp.Tape() as local:
                (_, Ru, Rg) = _evaluate(clones, fixed_u, fixed_t, compliance)
            rows = []
            for column in basis.T:
                local.zero()
                local.backward(grads={Ru: array(column.reshape(-1, 3), True)})
                rows.append(
                    np.r_[basis.T @ fixed_u.grad.numpy().ravel(), fixed_t.grad.numpy().ravel() / compliance]
                )
            shape = Rg.numpy().shape
            for index in range(np.prod(shape)):
                local.zero()
                unit = np.zeros(shape)
                unit.ravel()[index] = 1.0
                local.backward(grads={Rg: array(unit, True)})
                rows.append(
                    np.r_[basis.T @ fixed_u.grad.numpy().ravel(), fixed_t.grad.numpy().ravel() / compliance]
                )
            J = np.array(rows)
            dual = np.linalg.lstsq(J.T, rhs, rcond=None)[0]
            # The orthogonal momentum block is identity; retain its map derivatives.
            au = basis @ dual[:size] + rhs_u - basis @ (basis.T @ rhs_u)
            ag = dual[size:].reshape(shape)
            local.zero()
            local.backward(grads={Ru: array(au.reshape(-1, 3), True), Rg: array(ag, True)})
            full_error = max(
                np.max(np.abs(fixed_u.grad.numpy().ravel() - rhs_u)),
                np.max(np.abs(fixed_t.grad.numpy().ravel() / compliance - rhs_t)),
            )
            scale = max(np.max(np.abs(rhs_u)), np.max(np.abs(rhs_t)), 1.0)
            if full_error > 1e-10 * scale:
                raise RuntimeError(f"Full implicit adjoint residual: {full_error}")
            local.zero()
            local.backward(grads={Ru: array(-au.reshape(-1, 3), True), Rg: array(-ag, True)})
            for source, clone in zip(inputs, clones):
                if source.requires_grad:
                    source.grad.assign(source.grad.numpy() + clone.grad.numpy())

        tape.record_func(backward, [u, traction, *[a for a in inputs if a.requires_grad]])
    (result, _, _) = _evaluate(inputs, u, traction, compliance)
    return CohesiveStep(*result, u, info)


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
        or np.any(values[7] <= 0)
        or values[8][0] <= 0
        or values[9][0] <= 0
    ):
        raise ValueError("History must be nonnegative; material, area, mass and duration must be positive")
    if np.any(np.linalg.norm(values[2], axis=1) == 0):
        raise ValueError("Reference normals must be nonzero")
    return _solve_arrays(inputs, compliance_scale)
