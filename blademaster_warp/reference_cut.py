"""Differentiable reference-area quadrature for a plane cutting particle domains.

The current domain map is x_p + F_p (X-X_p). Its reference box has half extents
in metres. A plane coincident with a shared face belongs to the positive side.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
import warp as wp


CORNERS = np.array(list(product([-1.0, 1.0], repeat=3)))
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.count_nonzero(CORNERS[i] != CORNERS[j]) == 1]


@wp.kernel
def _intersection_vertices(
    position: wp.array(dtype=wp.vec3d),
    deformation: wp.array(dtype=wp.mat33d),
    reference: wp.array(dtype=wp.vec3d),
    half_extent: wp.array(dtype=wp.vec3d),
    plane_point: wp.array(dtype=wp.vec3d),
    plane_normal: wp.array(dtype=wp.vec3d),
    owners: wp.array(dtype=int),
    edges: wp.array(dtype=wp.vec2i),
    corners: wp.array(dtype=wp.vec3d),
    vertices: wp.array(dtype=wp.vec3d),
):
    v = wp.tid()
    p = owners[v]
    a = wp.cw_mul(corners[edges[v][0]], half_extent[p])
    b = wp.cw_mul(corners[edges[v][1]], half_extent[p])
    offset = position[p] - plane_point[0]
    ga = wp.dot(plane_normal[0], offset + deformation[p] * a)
    gb = wp.dot(plane_normal[0], offset + deformation[p] * b)
    fraction = ga / (ga - gb)
    vertices[v] = reference[p] + a + fraction * (b - a)


@wp.kernel
def _triangle_quadrature(
    vertices: wp.array(dtype=wp.vec3d),
    triangles: wp.array(dtype=wp.vec3i),
    owners: wp.array(dtype=int),
    deformation: wp.array(dtype=wp.mat33d),
    plane_normal: wp.array(dtype=wp.vec3d),
    points: wp.array(dtype=wp.vec3d),
    area: wp.array(dtype=wp.float64),
    normals: wp.array(dtype=wp.vec3d),
):
    triangle, local = wp.tid()
    ids = triangles[triangle]
    a, b, c = vertices[ids[0]], vertices[ids[1]], vertices[ids[2]]
    barycentric = wp.vec3d(wp.float64(1.0) / wp.float64(6.0))
    barycentric[local] = wp.float64(2.0) / wp.float64(3.0)
    q = 3 * triangle + local
    points[q] = barycentric[0] * a + barycentric[1] * b + barycentric[2] * c
    area[q] = wp.length(wp.cross(b - a, c - a)) / wp.float64(6.0)
    normals[q] = wp.normalize(wp.transpose(deformation[owners[triangle]]) * plane_normal[0])


@dataclass(frozen=True)
class CutQuadrature:
    points: wp.array
    area: wp.array
    normals: wp.array
    owners: np.ndarray


def _triangulate(position, deformation, half_extent, point, normal, selected):
    vertex_owners, vertex_edges, triangles, triangle_owners = [], [], [], []
    for p in selected:
        corners = CORNERS.astype(np.longdouble) * half_extent[p]
        current = position[p] - point + corners @ deformation[p].T
        levelset = current @ normal
        # Forward-error bound for the affine plane evaluation in float64.
        epsilon = np.finfo(np.float64).eps
        gamma = 16 * epsilon / (1 - 16 * epsilon)
        scale = np.abs(position[p]) + np.abs(point) + np.abs(deformation[p]) @ half_extent[p]
        roundoff = gamma * (scale @ np.abs(normal))
        levelset[np.abs(levelset) <= roundoff] = 0.0
        if not np.any(levelset > 0.0) or np.all(levelset > 0.0):
            continue
        cuts, edges, corner_ids = [], [], set()
        for i, j in EDGES:
            if (levelset[i] > 0.0) == (levelset[j] > 0.0):
                continue
            corner = i if levelset[i] == 0.0 else j if levelset[j] == 0.0 else None
            if corner is not None:
                if corner in corner_ids:
                    continue
                corner_ids.add(corner)
            t = levelset[i] / (levelset[i] - levelset[j])
            cuts.append(corners[i] + t * (corners[j] - corners[i]))
            edges.append((i, j))
        if len(cuts) < 3:
            continue
        values = np.asarray(cuts, dtype=float)
        n = deformation[p].T @ normal
        tangent = np.cross(n, np.eye(3)[np.argmin(np.abs(n))])
        tangent /= np.linalg.norm(tangent)
        second = np.cross(n, tangent)
        second /= np.linalg.norm(second)
        centered = values - values.mean(axis=0)
        order = np.argsort(np.arctan2(centered @ second, centered @ tangent))
        order = np.roll(order, -int(np.flatnonzero(order == 0)[0]))
        start = len(vertex_edges)
        vertex_edges.extend(edges[i] for i in order)
        vertex_owners.extend([p] * len(edges))
        for i in range(1, len(order) - 1):
            triangles.append([start, start + i, start + i + 1])
            triangle_owners.append(p)
    return vertex_owners, vertex_edges, triangles, triangle_owners


def quadrature(position, deformation, reference, half_extent, plane_point, plane_normal, *, owners=None):
    """Intersect a current plane with selected affine particle domains.

    Integrals use reference area and a degree-two triangle rule. Edge selection
    and triangulation are rebuilt at each call; their indices are fixed in a VJP.
    """
    sources = [position, deformation, reference, half_extent, plane_point, plane_normal]
    dtypes = [wp.vec3d, wp.mat33d, wp.vec3d, wp.vec3d, wp.vec3d, wp.vec3d]
    count = len(position)
    if any(a.dtype != dtype or a.device != position.device for a, dtype in zip(sources, dtypes)):
        raise ValueError("Cut quadrature requires float64 arrays on a common device")
    if any(a.shape != shape for a, shape in zip(sources, [(count,)] * 4 + [(1,), (1,)])):
        raise ValueError("Particle domains and the single cutting plane have inconsistent shapes")
    values = [a.numpy() for a in sources]
    if any(not np.isfinite(a).all() for a in values):
        raise ValueError("Cut quadrature inputs must be finite")
    if np.any(values[3] <= 0.0) or np.any(np.linalg.det(values[1]) <= 0.0):
        raise ValueError("Particle domains require positive half extents and deformation determinants")
    if not np.linalg.norm(values[5][0]) > 0.0:
        raise ValueError("The cutting plane requires a nonzero normal")
    selected = np.arange(count) if owners is None else np.asarray(owners)
    if (
        selected.ndim != 1
        or not np.issubdtype(selected.dtype, np.integer)
        or np.any(selected < 0)
        or np.any(selected >= count)
        or len(np.unique(selected)) != len(selected)
    ):
        raise ValueError("Selected particle IDs must be unique integers within the particle array")
    vertex_owners, edges, triangles, triangle_owners = _triangulate(
        values[0].astype(np.longdouble),
        values[1].astype(np.longdouble),
        values[3].astype(np.longdouble),
        values[4][0].astype(np.longdouble),
        values[5][0].astype(np.longdouble),
        selected,
    )
    device = position.device
    n = 3 * len(triangles)
    points = wp.empty(n, dtype=wp.vec3d, device=device, requires_grad=True)
    area = wp.empty(n, dtype=wp.float64, device=device, requires_grad=True)
    normals = wp.empty(n, dtype=wp.vec3d, device=device, requires_grad=True)
    if triangles:
        vertices = wp.empty(len(edges), dtype=wp.vec3d, device=device, requires_grad=True)
        wp.launch(
            _intersection_vertices,
            len(edges),
            inputs=[
                *sources,
                wp.array(vertex_owners, dtype=int, device=device),
                wp.array(edges, dtype=wp.vec2i, device=device),
                wp.array(CORNERS, dtype=wp.vec3d, device=device),
            ],
            outputs=[vertices],
            device=device,
        )
        wp.launch(
            _triangle_quadrature,
            (len(triangles), 3),
            inputs=[
                vertices,
                wp.array(triangles, dtype=wp.vec3i, device=device),
                wp.array(triangle_owners, dtype=int, device=device),
                deformation,
                plane_normal,
            ],
            outputs=[points, area, normals],
            device=device,
        )
    return CutQuadrature(points, area, normals, np.repeat(triangle_owners, 3).astype(np.int64))
