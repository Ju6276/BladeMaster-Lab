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
    vertices[v] = a + fraction * (b - a)


@wp.kernel
def _clip_vertices(
    previous: wp.array(dtype=wp.vec3d),
    edges: wp.array(dtype=wp.vec2i),
    owners: wp.array(dtype=int),
    position: wp.array(dtype=wp.vec3d),
    deformation: wp.array(dtype=wp.mat33d),
    clip_points: wp.array(dtype=wp.vec3d),
    clip_normals: wp.array(dtype=wp.vec3d),
    plane: int,
    result: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    pair = edges[i]
    value = previous[pair[0]]
    if pair[0] != pair[1]:
        p = owners[i]
        b = previous[pair[1]]
        offset = position[p] - clip_points[plane]
        ga = wp.dot(clip_normals[plane], offset + deformation[p] * value)
        gb = wp.dot(clip_normals[plane], offset + deformation[p] * b)
        value += ga / (ga - gb) * (b - value)
    result[i] = value


@wp.kernel
def _triangle_quadrature(
    vertices: wp.array(dtype=wp.vec3d),
    triangles: wp.array(dtype=wp.vec3i),
    owners: wp.array(dtype=int),
    reference: wp.array(dtype=wp.vec3d),
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
    points[q] = reference[owners[triangle]] + (barycentric[0] * a + barycentric[1] * b + barycentric[2] * c)
    area[q] = wp.length(wp.cross(b - a, c - a)) / wp.float64(6.0)
    normals[q] = wp.normalize(wp.transpose(deformation[owners[triangle]]) * plane_normal[0])


@wp.kernel
def _blade_planes(
    position: wp.array(dtype=wp.vec3d),
    rotation: wp.array(dtype=wp.quatd),
    half_size: wp.array(dtype=wp.vec3d),
    normal: wp.array(dtype=wp.vec3d),
    clip_points: wp.array(dtype=wp.vec3d),
    clip_normals: wp.array(dtype=wp.vec3d),
):
    q = wp.normalize(rotation[0])
    normal[0] = wp.quat_rotate(q, wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0)))
    for i in range(4):
        axis = 1 + i // 2
        local = wp.vec3d(wp.float64(0.0))
        sign = wp.float64(1.0)
        if i % 2 == 0:
            sign = wp.float64(-1.0)
        local[axis] = sign
        direction = wp.quat_rotate(q, local)
        clip_normals[i] = direction
        clip_points[i] = position[0] + half_size[0][axis] * direction


@dataclass(frozen=True)
class CutQuadrature:
    points: wp.array
    area: wp.array
    normals: wp.array
    owners: np.ndarray


def _sections(position, deformation, half_extent, point, normal, selected):
    vertex_owners, vertex_edges, vertex_values, polygons, polygon_owners = [], [], [], [], []
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
        vertex_values.extend(cuts[i] for i in order)
        vertex_owners.extend([p] * len(edges))
        polygons.append(list(range(start, start + len(order))))
        polygon_owners.append(p)
    return vertex_owners, vertex_edges, vertex_values, polygons, polygon_owners


def _clip_polygon(values, polygon, levelset, *, boundary_intersections=False, include_boundary=True):
    descriptors, locations = [], []

    def append(pair, value):
        if not descriptors or pair != descriptors[-1]:
            descriptors.append(pair)
            locations.append(value)

    for j, b in enumerate(polygon):
        a = polygon[j - 1]
        ga, gb = levelset[j - 1], levelset[j]
        inside_a = ga <= 0 if include_boundary else ga < 0
        inside_b = gb <= 0 if include_boundary else gb < 0
        if inside_a != inside_b:
            if ga == 0 and not boundary_intersections:
                append((a, a), values[a])
            elif gb == 0 and not boundary_intersections:
                append((b, b), values[b])
            else:
                append((a, b), values[a] + ga / (ga - gb) * (values[b] - values[a]))
        if inside_b:
            append((b, b), values[b])
    if len(descriptors) > 1 and descriptors[0] == descriptors[-1]:
        descriptors.pop()
        locations.pop()
    return descriptors, locations


def _clip_polygons(values, polygons, owners, position, deformation, point, normal):
    next_values, next_polygons, next_owners, edges, vertex_owners = [], [], [], [], []
    for polygon, p in zip(polygons, owners):
        coordinates = np.asarray([values[i] for i in polygon], dtype=np.longdouble)
        levelset = (position[p] - point + coordinates @ deformation[p].T) @ normal
        scale = np.abs(position[p]) + np.abs(point) + np.abs(coordinates) @ np.abs(deformation[p].T)
        epsilon = np.finfo(np.float64).eps
        roundoff = 16 * epsilon / (1 - 16 * epsilon) * (scale @ np.abs(normal))
        levelset[np.abs(levelset) <= roundoff] = 0.0
        descriptors, locations = _clip_polygon(values, polygon, levelset)
        if len(descriptors) < 3:
            continue
        start = len(next_values)
        next_values.extend(locations)
        edges.extend(descriptors)
        vertex_owners.extend([p] * len(locations))
        next_polygons.append(list(range(start, start + len(locations))))
        next_owners.append(p)
    return next_values, next_polygons, next_owners, edges, vertex_owners


def quadrature(
    position,
    deformation,
    reference,
    half_extent,
    plane_point,
    plane_normal,
    *,
    owners=None,
    clip_points=None,
    clip_normals=None,
):
    """Intersect a current plane with selected affine particle domains.

    Integrals use reference area and a degree-two triangle rule. Edge selection
    and triangulation are rebuilt at each call; their indices are fixed in a VJP.
    Optional current-space half planes retain n·(x-c) <= 0.
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
    if (clip_points is None) != (clip_normals is None):
        raise ValueError("Clipping requires both plane points and normals")
    clip_values = []
    if clip_points is not None:
        if (
            any(a.dtype != wp.vec3d or a.device != position.device for a in [clip_points, clip_normals])
            or clip_points.ndim != 1
            or clip_points.shape != clip_normals.shape
        ):
            raise ValueError("Clipping planes require matching float64 arrays on the particle device")
        clip_values = [a.numpy().astype(np.longdouble) for a in [clip_points, clip_normals]]
        if any(not np.isfinite(a).all() for a in clip_values) or np.any(
            np.linalg.norm(clip_values[1], axis=1) == 0
        ):
            raise ValueError("Clipping plane points and nonzero normals must be finite")
    vertex_owners, edges, vertex_values, polygons, polygon_owners = _sections(
        values[0].astype(np.longdouble),
        values[1].astype(np.longdouble),
        values[3].astype(np.longdouble),
        values[4][0].astype(np.longdouble),
        values[5][0].astype(np.longdouble),
        selected,
    )
    stages = []
    if clip_values:
        for point, normal in zip(*clip_values):
            vertex_values, polygons, polygon_owners, connections, stage_owners = _clip_polygons(
                vertex_values, polygons, polygon_owners, values[0], values[1], point, normal
            )
            stages.append((connections, stage_owners))
    triangles, triangle_owners = [], []
    for polygon, p in zip(polygons, polygon_owners):
        for i in range(1, len(polygon) - 1):
            ids = [polygon[0], polygon[i], polygon[i + 1]]
            a, b, c = [vertex_values[j] for j in ids]
            if np.any(np.cross(b - a, c - a) != 0):
                triangles.append(ids)
                triangle_owners.append(p)
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
                position,
                deformation,
                half_extent,
                plane_point,
                plane_normal,
                wp.array(vertex_owners, dtype=int, device=device),
                wp.array(edges, dtype=wp.vec2i, device=device),
                wp.array(CORNERS, dtype=wp.vec3d, device=device),
            ],
            outputs=[vertices],
            device=device,
        )
        for index, (connections, stage_owners) in enumerate(stages):
            clipped = wp.empty(len(connections), dtype=wp.vec3d, device=device, requires_grad=True)
            wp.launch(
                _clip_vertices,
                len(connections),
                inputs=[
                    vertices,
                    wp.array(connections, dtype=wp.vec2i, device=device),
                    wp.array(stage_owners, dtype=int, device=device),
                    position,
                    deformation,
                    clip_points,
                    clip_normals,
                    index,
                ],
                outputs=[clipped],
                device=device,
            )
            vertices = clipped
        wp.launch(
            _triangle_quadrature,
            (len(triangles), 3),
            inputs=[
                vertices,
                wp.array(triangles, dtype=wp.vec3i, device=device),
                wp.array(triangle_owners, dtype=int, device=device),
                reference,
                deformation,
                plane_normal,
            ],
            outputs=[points, area, normals],
            device=device,
        )
    return CutQuadrature(points, area, normals, np.repeat(triangle_owners, 3).astype(np.int64))


def blade_quadrature(
    position,
    deformation,
    reference,
    half_extent,
    tool_position,
    tool_rotation,
    tool_half_size,
    *,
    owners=None,
):
    """Intersect particle domains with the blade's current rectangular mid-surface.

    Blade-local x is normal, y spans the edge and z spans the blade height.
    Pose and half sizes use metres and an xyzw quaternion, normalized here.
    The result measures the instantaneous footprint; swept-area history is separate.
    """
    pose = [tool_position, tool_rotation, tool_half_size]
    dtypes = [wp.vec3d, wp.quatd, wp.vec3d]
    if any(
        a.dtype != dtype or a.device != position.device or a.shape != (1,) for a, dtype in zip(pose, dtypes)
    ):
        raise ValueError("Blade pose and half size require single float64 values on the particle device")
    values = [a.numpy() for a in pose]
    if (
        any(not np.isfinite(a).all() for a in values)
        or not np.linalg.norm(values[1][0]) > 0
        or np.any(values[2] <= 0)
    ):
        raise ValueError("Blade pose must be finite with nonzero rotation and positive half sizes")
    normal = wp.empty(1, dtype=wp.vec3d, device=position.device, requires_grad=True)
    clip_points = wp.empty(4, dtype=wp.vec3d, device=position.device, requires_grad=True)
    clip_normals = wp.empty(4, dtype=wp.vec3d, device=position.device, requires_grad=True)
    wp.launch(
        _blade_planes, 1, inputs=pose, outputs=[normal, clip_points, clip_normals], device=position.device
    )
    return quadrature(
        position,
        deformation,
        reference,
        half_extent,
        tool_position,
        normal,
        owners=owners,
        clip_points=clip_points,
        clip_normals=clip_normals,
    )
