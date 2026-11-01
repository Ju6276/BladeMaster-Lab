"""Minimum-norm adhesive reactions in tensile half-balls or compressed tangent disks."""

import numpy as np
import warp as wp
from scipy.linalg import qr

from ._compensated import Expansion as E
from ._compensated_linalg import equilibrated_solve
from .cohesive import surface_normal, surface_normal_pullback
from .compensated_state import CompensatedArray, add_gradient
from .precision import launch_double


@wp.kernel
def coordinates(
    gap: wp.array(dtype=wp.vec3),
    frame: wp.array(dtype=wp.mat33),
    reference: wp.array(dtype=wp.vec3),
    b: wp.array2d(dtype=float),
    gradient: wp.array2d(dtype=wp.vec3),
    area: wp.array(dtype=float),
    force: wp.array(dtype=wp.vec3),
    points: wp.array(dtype=int),
    matrix: wp.array2d(dtype=float),
    original: wp.array(dtype=float),
):
    column = wp.tid()
    point, axis = points[column // 3], column % 3
    n = surface_normal(frame[point], reference[point])
    compression = wp.min(wp.dot(gap[point], n), 0.0)
    unit = wp.vec3(0.0)
    if axis == 0:
        unit = wp.vec3(1.0, 0.0, 0.0)
    elif axis == 1:
        unit = wp.vec3(0.0, 1.0, 0.0)
    else:
        unit = wp.vec3(0.0, 0.0, 1.0)
    moment = surface_normal_pullback(frame[point], reference[point], -compression * unit)
    root = wp.sqrt(area[point])
    original[column] = force[point][axis] / root
    for particle in range(b.shape[1]):
        coefficient = b[point, particle] * unit + moment * gradient[point, particle]
        for component in range(3):
            matrix[3 * particle + component, column] = root * coefficient[component]


@wp.kernel
def reconstruct(
    area: wp.array(dtype=float),
    force: wp.array(dtype=wp.vec3),
    offsets: wp.array(dtype=int),
    selected: wp.array(dtype=float),
    result: wp.array(dtype=wp.vec3),
):
    point = wp.tid()
    offset = offsets[point]
    value = force[point]
    if offset >= 0:
        value = wp.sqrt(area[point]) * wp.vec3(selected[offset], selected[offset + 1], selected[offset + 2])
    result[point] = value


@wp.kernel
def constraints(
    frame: wp.array(dtype=wp.mat33),
    reference: wp.array(dtype=wp.vec3),
    area: wp.array(dtype=float),
    material: wp.array(dtype=wp.vec3),
    points: wp.array(dtype=int),
    offsets: wp.array(dtype=int),
    kinds: wp.array(dtype=int),
    selected: wp.array(dtype=float),
    bound: wp.array(dtype=float),
    jacobian: wp.array2d(dtype=float),
):
    row = wp.tid()
    point = points[row]
    offset = offsets[point]
    value = wp.vec3(selected[offset], selected[offset + 1], selected[offset + 2])
    derivative = surface_normal(frame[point], reference[point])
    bound[row] = wp.dot(value, derivative)
    if kinds[row] == 1:
        length = wp.length(value)
        bound[row] = wp.sqrt(area[point]) * material[point][0] - length
        derivative = -value / length
    for axis in range(3):
        jacobian[row, offset + axis] = derivative[axis]


def canonical_output(inputs, u, force, branches, *, remainders=None, map_remainders=None):
    if branches.reaction_trial is None:
        return force
    from .cohesive_geometry import InterfaceGeometry

    points = np.flatnonzero(branches.closed)
    count = 3 * len(points)
    offsets = np.full(len(force), -1, dtype=np.int32)
    offsets[points] = 3 * np.arange(len(points))
    active_points, kinds = [], []
    for i in points:
        if branches.normal_active[i]:
            active_points.append(i)
            kinds.append(0)
        if branches.strength_active[i]:
            active_points.append(i)
            kinds.append(1)
    device = u.device
    values = [a.numpy().copy() for a in inputs]
    u_value, force_value = u.numpy().copy(), force.numpy().copy()
    selected_value = (force_value[points] / np.sqrt(values[7][points, None])).ravel()
    output = wp.array(force_value, dtype=force.dtype, device=device, requires_grad=True)
    outer = wp._src.context.runtime.tape
    if outer is None:
        return output

    def backward():
        integer = lambda value: wp.array(np.asarray(value, dtype=np.int32), dtype=int, device=device)
        clones = [
            wp.array(v, dtype=a.dtype, device=device, requires_grad=True) for v, a in zip(values, inputs)
        ]
        fixed_u = wp.array(u_value, dtype=u.dtype, device=device, requires_grad=True)
        fixed_force = wp.array(force_value, dtype=force.dtype, device=device, requires_grad=True)
        selected = wp.array(selected_value, dtype=wp.float64, device=device, requires_grad=True)
        D = wp.zeros((u.size * 3, count), dtype=wp.float64, device=device, requires_grad=True)
        original = wp.zeros(count, dtype=wp.float64, device=device, requires_grad=True)
        result = wp.zeros_like(force, requires_grad=True)
        bounds = wp.zeros(len(kinds), dtype=wp.float64, device=device, requires_grad=True)
        Jb = wp.zeros((len(kinds), count), dtype=wp.float64, device=device, requires_grad=True)
        gap_low, frame_low, u_low = (None, None, None) if remainders is None else remainders
        low_b, low_g = (None, None) if map_remainders is None else map_remainders
        with wp.Tape() as local:
            geometry = InterfaceGeometry(clones[0], clones[1], gap_low, frame_low)
            moved = geometry.advance(
                CompensatedArray(clones[5], low_b),
                CompensatedArray(clones[6], low_g),
                CompensatedArray(fixed_u, u_low),
                wp.array([1.0], dtype=wp.float64, device=device),
            )
            launch_double(
                coordinates,
                count,
                inputs=[
                    moved.gap,
                    moved.frame,
                    clones[2],
                    clones[5],
                    clones[6],
                    clones[7],
                    fixed_force,
                    integer(points),
                ],
                outputs=[D, original],
                device=device,
            )
            launch_double(
                reconstruct,
                len(force),
                inputs=[clones[7], fixed_force, integer(offsets), selected],
                outputs=[result],
                device=device,
            )
            if kinds:
                launch_double(
                    constraints,
                    len(kinds),
                    inputs=[
                        moved.frame,
                        clones[2],
                        clones[7],
                        clones[4],
                        integer(active_points),
                        integer(offsets),
                        integer(kinds),
                        selected,
                    ],
                    outputs=[bounds, Jb],
                    device=device,
                )
        matrix = D.numpy()
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank = (
            np.count_nonzero(singular > max(matrix.shape) * np.finfo(float).eps * singular[0])
            if len(singular)
            else 0
        )
        rows = qr(matrix.T, pivoting=True, mode="economic")[2][:rank]
        reduced = matrix[rows]
        trial = branches.reaction_trial
        multiplier = np.linalg.lstsq(reduced.T, trial, rcond=None)[0]
        mu, hessians = [], []
        frames = moved.frame.numpy()
        for point, kind in zip(active_points, kinds):
            start = offsets[point]
            span = slice(start, start + 3)
            F, N = frames[point], values[2][point]
            normal = (
                N[0] * np.cross(F[:, 1], F[:, 2])
                + N[1] * np.cross(F[:, 2], F[:, 0])
                + N[2] * np.cross(F[:, 0], F[:, 1])
            )
            normal /= np.linalg.norm(normal)
            hessian = np.zeros((count, count))
            component = np.dot(trial[span], normal)
            if kind == 0:
                mu.append(-component)
            else:
                tangent = trial[span] - (component * normal if branches.normal_active[point] else 0.0)
                bound = np.sqrt(values[7][point]) * values[4][point, 0]
                mu.append(np.linalg.norm(tangent) - bound)
                value = selected_value[span]
                radius = np.linalg.norm(value)
                hessian[span, span] = -(np.eye(3) - np.outer(value, value) / radius**2) / radius
            hessians.append(hessian)
        mu = np.asarray(mu)
        constraint_jacobian = Jb.numpy()
        stationarity = selected_value - reduced.T @ multiplier - constraint_jacobian.T @ mu
        scale = max(np.linalg.norm(selected_value), 1.0)
        if np.max(abs(stationarity), initial=0.0) > 1e-10 * scale:
            raise RuntimeError("Adhesive reaction KKT stationarity residual")
        local.backward(grads={result: wp.array(output.grad.numpy(), dtype=force.dtype, device=device)})
        sources = [*clones, fixed_u, fixed_force]
        explicit = [a.grad.numpy().copy() for a in sources]
        seed = selected.grad.numpy().copy()
        H = np.eye(count)
        if len(mu):
            H -= np.einsum("i,ijk->jk", mu, np.asarray(hessians))
        C = np.r_[reduced, constraint_jacobian]
        K = np.block([[H, -C.T], [C, np.zeros((len(C), len(C)))]])
        rhs = np.r_[seed, np.zeros(len(C))]
        adjoint = equilibrated_solve(K.T, E(rhs)).numpy()
        if np.max(abs(K.T @ adjoint - rhs), initial=0.0) > 1e-10 * max(np.max(abs(rhs), initial=0.0), 1.0):
            raise RuntimeError("Adhesive reaction KKT adjoint residual")
        az, ae, ab = adjoint[:count], adjoint[count : count + rank], adjoint[count + rank :]
        matrix_gradient = np.zeros_like(matrix)
        matrix_gradient[rows] = np.outer(multiplier, az) - np.outer(ae, selected_value - original.numpy())
        seeds = {
            D: wp.array(matrix_gradient, dtype=wp.float64, device=device),
            original: wp.array(reduced.T @ ae, dtype=wp.float64, device=device),
        }
        if kinds:
            seeds[bounds] = wp.array(-ab, dtype=wp.float64, device=device)
            seeds[Jb] = wp.array(np.outer(mu, az), dtype=wp.float64, device=device)
        local.zero()
        local.backward(grads=seeds)
        for source, direct, indirect in zip([*inputs, u, force], explicit, sources):
            add_gradient(source, E(direct) + E(indirect.grad.numpy()))

    outer.record_func(backward, [output, *[a for a in [*inputs, u, force] if a.requires_grad and a.size]])
    return output
