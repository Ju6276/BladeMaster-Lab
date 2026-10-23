"""Reference-grid traces with polynomial extension of unsupported side nodes.

Positions and grid spacing are in metres. Each surface point selects two cut
codes. Compatible fields share a mass-weighted nodal value; missing nodes use
the trilinear polynomial of a supported cell.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
import warp as wp

from .geometry import grid_cells


CORNERS = np.array(list(product([0, 1], repeat=3)), dtype=np.int32)


@wp.kernel
def _basis(
    points: wp.array(dtype=wp.vec3d),
    bases: wp.array(dtype=wp.vec3i),
    corners: wp.array(dtype=wp.vec3i),
    origin: wp.vec3d,
    spacing: wp.float64,
    weights: wp.array2d(dtype=wp.float64),
    gradients: wp.array2d(dtype=wp.vec3d),
):
    q, a = wp.tid()
    local = (points[q] - origin) / spacing - wp.vec3d(bases[q])
    corner = corners[a]
    factor = wp.vec3d(0.0)
    direction = wp.vec3d(0.0)
    for axis in range(3):
        direction[axis] = wp.float64(2 * corner[axis] - 1)
        factor[axis] = wp.float64(1 - corner[axis]) + direction[axis] * local[axis]
    weights[q, a] = factor[0] * factor[1] * factor[2]
    gradients[q, a] = (
        wp.vec3d(
            direction[0] * factor[1] * factor[2],
            direction[1] * factor[0] * factor[2],
            direction[2] * factor[0] * factor[1],
        )
        / spacing
    )


@wp.kernel
def _extend(
    indices: wp.array(dtype=wp.vec3i),
    coefficients: wp.array2d(dtype=wp.float64),
    basis: wp.array2d(dtype=wp.float64),
    basis_gradient: wp.array2d(dtype=wp.vec3d),
    weights: wp.array3d(dtype=wp.float64),
    gradients: wp.array3d(dtype=wp.vec3d),
):
    entry = wp.tid()
    q, side, slot = indices[entry][0], indices[entry][1], indices[entry][2]
    value = wp.float64(0.0)
    derivative = wp.vec3d(0.0)
    for a in range(8):
        value += coefficients[entry, a] * basis[q, a]
        derivative += coefficients[entry, a] * basis_gradient[q, a]
    weights[q, side, slot] = value
    gradients[q, side, slot] = derivative


@wp.kernel
def _combine(
    weights: wp.array3d(dtype=wp.float64),
    gradients: wp.array3d(dtype=wp.vec3d),
    normals: wp.array(dtype=wp.vec3d),
    separation: wp.array2d(dtype=wp.float64),
    frame: wp.array2d(dtype=wp.vec3d),
):
    q, slot = wp.tid()
    separation[q, slot] = weights[q, 1, slot] - weights[q, 0, slot]
    average = wp.float64(0.5) * (gradients[q, 1, slot] + gradients[q, 0, slot])
    normal = wp.normalize(normals[q])
    frame[q, slot] = average - wp.dot(average, normal) * normal


@dataclass(frozen=True)
class ReferenceGridTrace:
    separation_map: wp.array
    frame_map: wp.array
    weights: wp.array
    gradients: wp.array


def _compatible(codes, side):
    codes = codes.copy()
    result = np.ones(codes.shape, dtype=bool)
    while side:
        digit = side % 3
        if digit:
            result &= (codes % 3 == 0) | (codes % 3 == digit)
        codes //= 3
        side //= 3
    return result


def _coefficients(model, topology, bases, sides, origin, cells):
    spacing = model.config.grid_spacing
    keys = topology.keys.numpy()
    slots = topology.slots.numpy()
    codes = keys // model.node_count
    particle_codes = codes[slots[:, 0]]
    masses = np.bincount(
        slots.ravel(), weights=model.weights.numpy().ravel() * model.mass, minlength=topology.count
    )
    particle_cells = np.floor((model.reference - origin) / spacing).astype(int)
    roots, nodal = {}, {}
    for code in np.unique(sides):
        code = int(code)
        roots[code] = np.unique(particle_cells[_compatible(particle_codes, code)], axis=0)
        groups = {}
        for slot in np.flatnonzero(_compatible(codes, code) & (masses > 0)):
            groups.setdefault(int(keys[slot] % model.node_count), []).append(int(slot))
        nodal[code] = {}
        for node, group in groups.items():
            total = np.sum(masses[group])
            nodal[code][node] = [(slot, masses[slot] / total) for slot in group]
    if any(len(value) == 0 for value in roots.values()):
        raise ValueError("Each trace side requires compatible particle support")
    extension = {}
    indices, coefficients = [], []
    for q, base in enumerate(bases):
        nodes = np.ravel_multi_index((base + CORNERS).T, tuple(cells + 1))
        for side, code in enumerate(sides[q]):
            code = int(code)
            row = {}
            for a, (node, index) in enumerate(zip(nodes, base + CORNERS)):
                key = (code, int(node))
                if key not in extension:
                    if node in nodal[code]:
                        extension[key] = nodal[code][node]
                    else:
                        candidates = roots[code]
                        distance = np.maximum(np.maximum(candidates - index, index - candidates - 1), 0)
                        root = candidates[np.argmin(np.sum(distance**2, axis=1))]
                        local = index - root
                        values = np.where(CORNERS, local, 1 - local).prod(axis=1)
                        root_nodes = np.ravel_multi_index((root + CORNERS).T, tuple(cells + 1))
                        extension[key] = [
                            (slot, float(value) * coefficient)
                            for node_id, value in zip(root_nodes, values)
                            if value != 0
                            for slot, coefficient in nodal[code][node_id]
                        ]
                for slot, coefficient in extension[key]:
                    row.setdefault(slot, np.zeros(8))[a] += coefficient
            for slot, values in sorted(row.items()):
                indices.append((q, side, slot))
                coefficients.append(values)
    return indices, coefficients


def build_trace(model, topology, points, normals, sides):
    """Return double-sided values, spatial derivatives and cohesive grid maps.

    Sides is an integer array (point, 2) of cut codes. Zero ternary digits are
    shared sides. Compatible fields at a node contribute in proportion to their
    reference mass. Missing nodes use the polynomial of the closest compatible
    particle-supported cell; ties use lexicographic cell order. Cell and root
    selection are rebuilt for each call.
    """
    count = len(points)
    sides = np.asarray(sides)
    if (
        model.precision != "float64"
        or points.dtype != wp.vec3d
        or normals.dtype != wp.vec3d
        or points.device != model.device
        or normals.device != model.device
        or points.shape != (count,)
        or normals.shape != (count,)
    ):
        raise ValueError("Reference-grid traces require matching float64 points and normals")
    if sides.shape != (count, 2) or not np.issubdtype(sides.dtype, np.integer) or np.any(sides < 0):
        raise ValueError("Each point requires two nonnegative integer cut codes")
    values, normal_values = points.numpy(), normals.numpy()
    if (
        not np.isfinite(values).all()
        or not np.isfinite(normal_values).all()
        or np.any(np.linalg.norm(normal_values, axis=1) == 0)
    ):
        raise ValueError("Trace coordinates and nonzero normals must be finite")
    spacing = model.config.grid_spacing
    origin = -np.array([model.config.size[0] / 2, model.config.size[1] / 2, 0.0])
    cells = grid_cells(model.config.size, spacing)
    if np.any(values < origin) or np.any(values > origin + cells * spacing):
        raise ValueError("Trace points must lie in the reference grid")
    bases = np.minimum(np.floor((values - origin) / spacing).astype(int), cells - 1)
    indices, coefficients = _coefficients(model, topology, bases, sides, origin, cells)
    weights = model.zeros((count, 2, topology.count), float)
    gradients = model.zeros((count, 2, topology.count), wp.vec3)
    separation = model.zeros((count, topology.count), float)
    frame = model.zeros((count, topology.count), wp.vec3)
    if count:
        basis = model.zeros((count, 8), float)
        derivative = model.zeros((count, 8), wp.vec3)
        wp.launch(
            _basis,
            (count, 8),
            inputs=[
                points,
                model.array(bases, wp.vec3i, False),
                model.array(CORNERS, wp.vec3i, False),
                wp.vec3d(origin),
                wp.float64(spacing),
            ],
            outputs=[basis, derivative],
            device=model.device,
        )
        wp.launch(
            _extend,
            len(indices),
            inputs=[
                model.array(indices, wp.vec3i, False),
                model.array(coefficients, float, False),
                basis,
                derivative,
            ],
            outputs=[weights, gradients],
            device=model.device,
        )
        wp.launch(
            _combine,
            (count, topology.count),
            inputs=[weights, gradients, normals],
            outputs=[separation, frame],
            device=model.device,
        )
    return ReferenceGridTrace(separation, frame, weights, gradients)
