"""Material-coordinate surfaces swept by a cutting edge.

Each accepted interval contributes two triangles per affine particle domain.
Coordinates are particle-local reference offsets in metres. Coplanar polygon
difference removes covered area while existing polygons remain unchanged.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from .reference_cut import CutQuadrature, _clip_polygon


EPS = np.finfo(np.float64).eps


@wp.kernel
def _pullback_edge(
    x: wp.array(dtype=wp.vec3d),
    F: wp.array(dtype=wp.mat33d),
    position: wp.array(dtype=wp.vec3d),
    rotation: wp.array(dtype=wp.quatd),
    half: wp.array(dtype=wp.vec3d),
    vertices: wp.array(dtype=wp.vec3d),
):
    p, side = wp.tid()
    sign = wp.float64(2 * side - 1)
    local = wp.vec3d(wp.float64(0.0), sign * half[0][1], -half[0][2])
    relative = (position[0] - x[p]) + wp.quat_rotate(wp.normalize(rotation[0]), local)
    vertices[2 * p + side] = wp.inverse(F[p]) * relative


@wp.kernel
def _join(a: wp.array(dtype=wp.vec3d), b: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if i < a.shape[0]:
        out[i] = a[i]
    else:
        out[i] = b[i - a.shape[0]]


@wp.kernel
def _subdivide(
    vertices: wp.array(dtype=wp.vec3d),
    edges: wp.array(dtype=wp.vec2i),
    fraction: wp.array(dtype=wp.float64),
    out: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    out[i] = (wp.float64(1.0) - fraction[i]) * vertices[edges[i][0]] + fraction[i] * vertices[edges[i][1]]


@wp.kernel
def _clip_vertices(
    previous: wp.array(dtype=wp.vec3d),
    edges: wp.array(dtype=wp.vec2i),
    owners: wp.array(dtype=int),
    point: wp.array(dtype=wp.vec3d),
    normal: wp.array(dtype=wp.vec3d),
    per_particle: int,
    sign: wp.float64,
    out: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    a, b = previous[edges[i][0]], previous[edges[i][1]]
    j = 0
    if per_particle != 0:
        j = owners[i]
    value = a
    if edges[i][0] != edges[i][1]:
        ga = sign * wp.dot(normal[j], a - point[j])
        gb = sign * wp.dot(normal[j], b - point[j])
        value = a + ga / (ga - gb) * (b - a)
    out[i] = value


@wp.kernel
def _box_plane(
    half: wp.array(dtype=wp.vec3d),
    axis: int,
    sign: wp.float64,
    point: wp.array(dtype=wp.vec3d),
    normal: wp.array(dtype=wp.vec3d),
):
    p = wp.tid()
    a, n = wp.vec3d(wp.float64(0.0)), wp.vec3d(wp.float64(0.0))
    a[axis] = sign * half[p][axis]
    n[axis] = sign
    point[p] = a
    normal[p] = n


@wp.kernel
def _polygon_plane(
    vertices: wp.array(dtype=wp.vec3d),
    normals: wp.array(dtype=wp.vec3d),
    face: int,
    edge: wp.vec2i,
    point: wp.array(dtype=wp.vec3d),
    normal: wp.array(dtype=wp.vec3d),
):
    point[0] = vertices[edge[0]]
    normal[0] = wp.cross(vertices[edge[1]] - vertices[edge[0]], normals[face])


@wp.kernel
def _normals(
    vertices: wp.array(dtype=wp.vec3d), faces: wp.array(dtype=wp.vec3i), normals: wp.array(dtype=wp.vec3d)
):
    i = wp.tid()
    face = faces[i]
    a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    normals[i] = wp.normalize(wp.cross(b - a, c - a))


@wp.kernel
def _select(values: wp.array(dtype=wp.vec3d), indices: wp.array(dtype=int), out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    out[i] = values[indices[i]]


def _gather(values, indices):
    out = wp.empty(len(indices), dtype=wp.vec3d, device=values.device, requires_grad=True)
    if len(indices):
        wp.launch(
            _select,
            len(indices),
            inputs=[values, wp.array(indices, dtype=int, device=values.device)],
            outputs=[out],
            device=values.device,
        )
    return out


@wp.kernel
def _quadrature(
    vertices: wp.array(dtype=wp.vec3d),
    triangles: wp.array(dtype=wp.vec3i),
    owners: wp.array(dtype=int),
    face_ids: wp.array(dtype=int),
    orientation: wp.array(dtype=wp.vec3d),
    reference: wp.array(dtype=wp.vec3d),
    points: wp.array(dtype=wp.vec3d),
    area: wp.array(dtype=wp.float64),
    normals: wp.array(dtype=wp.vec3d),
):
    i, j = wp.tid()
    ids = triangles[i]
    a, b, c = vertices[ids[0]], vertices[ids[1]], vertices[ids[2]]
    w = wp.vec3d(wp.float64(1.0) / wp.float64(6.0))
    w[j] = wp.float64(2.0) / wp.float64(3.0)
    cross = wp.cross(b - a, c - a)
    points[3 * i + j] = reference[owners[i]] + (w[0] * a + w[1] * b + w[2] * c)
    n = orientation[face_ids[i]]
    area[3 * i + j] = wp.dot(cross, n) / wp.float64(6.0)
    normals[3 * i + j] = n


def _face(values, polygon, error):
    """Select a resolved fan triangle using propagated coordinate roundoff."""
    best, magnitude = None, np.longdouble(0)
    for j in range(1, len(polygon) - 1):
        ids = [polygon[0], polygon[j], polygon[j + 1]]
        a, b, c = values[ids]
        u, v = b - a, c - a
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        bound = 2 * error * (lu + lv) + 4 * error**2 + 8 * EPS * lu * lv
        area = np.linalg.norm(np.cross(u, v))
        if area > bound and area > magnitude:
            best, magnitude = ids, area
    return best


@dataclass(frozen=True)
class ReferenceSurface:
    vertices: wp.array
    polygons: tuple
    owners: np.ndarray
    roundoff: np.ndarray
    normals: wp.array | None = None

    def __post_init__(self):
        if self.normals is None:
            normals = wp.empty(
                len(self.polygons), dtype=wp.vec3d, device=self.vertices.device, requires_grad=True
            )
            if self.polygons:
                values = self.vertices.numpy().astype(np.longdouble)
                faces = [_face(values, face, self.roundoff[p]) for face, p in zip(self.polygons, self.owners)]
                if any(face is None for face in faces):
                    raise ValueError("A collapsed polygon requires its parent surface orientation")
                wp.launch(
                    _normals,
                    len(faces),
                    inputs=[self.vertices, wp.array(faces, dtype=wp.vec3i, device=self.vertices.device)],
                    outputs=[normals],
                    device=self.vertices.device,
                )
            object.__setattr__(self, "normals", normals)

    @classmethod
    def empty(cls, particle_count, *, device):
        return cls(
            wp.empty(0, dtype=wp.vec3d, device=device, requires_grad=True),
            (),
            np.empty(0, dtype=np.int64),
            np.zeros(particle_count),
        )

    def select(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        polygons = tuple(self.polygons[i] for i in indices)
        used = sorted({vertex for face in polygons for vertex in face})
        lookup = {vertex: i for i, vertex in enumerate(used)}
        return ReferenceSurface(
            _gather(self.vertices, used),
            tuple(tuple(lookup[vertex] for vertex in face) for face in polygons),
            self.owners[indices],
            self.roundoff,
            _gather(self.normals, indices),
        )

    def join(self, other):
        if not self.polygons:
            return other
        if not other.polygons:
            return self
        if self.vertices.device != other.vertices.device or self.roundoff.shape != other.roundoff.shape:
            raise ValueError("Joined reference surfaces must use the same particle set and device")
        offset = len(self.vertices)
        vertices = wp.empty(
            offset + len(other.vertices), dtype=wp.vec3d, device=self.vertices.device, requires_grad=True
        )
        wp.launch(
            _join,
            len(vertices),
            inputs=[self.vertices, other.vertices],
            outputs=[vertices],
            device=vertices.device,
        )
        polygons = self.polygons + tuple(tuple(i + offset for i in face) for face in other.polygons)
        normals = wp.empty(len(polygons), dtype=wp.vec3d, device=vertices.device, requires_grad=True)
        wp.launch(
            _join,
            len(normals),
            inputs=[self.normals, other.normals],
            outputs=[normals],
            device=vertices.device,
        )
        return ReferenceSurface(
            vertices,
            polygons,
            np.r_[self.owners, other.owners],
            np.maximum(self.roundoff, other.roundoff),
            normals,
        )

    def clip(self, point, normal, *, sign=1, include_boundary=True):
        if sign not in (-1, 1):
            raise ValueError("Half-plane orientation must be -1 or 1")
        values = self.vertices.numpy().astype(np.longdouble)
        points, normals = point.numpy().astype(np.longdouble), normal.numpy().astype(np.longdouble)
        per_particle = len(point) != 1
        if point.shape != normal.shape or len(point) not in (1, len(self.roundoff)):
            raise ValueError("Clip planes must be global or indexed by particle")
        edges, owners, vertex_owners, polygons, parents = [], [], [], [], []
        for face, (polygon, p) in enumerate(zip(self.polygons, self.owners)):
            j = p if per_particle else 0
            coordinates = values[list(polygon)]
            n, c = sign * normals[j], points[j]
            level = (coordinates - c) @ n
            bound = 16 * EPS * ((np.abs(coordinates) + np.abs(c)) @ np.abs(n)) + self.roundoff[
                p
            ] * np.linalg.norm(n)
            level[np.abs(level) <= bound] = 0
            descriptors, locations = _clip_polygon(
                values, polygon, level, boundary_intersections=True, include_boundary=include_boundary
            )
            if len(descriptors) < 3:
                continue
            start = len(edges)
            polygons.append(tuple(range(start, start + len(descriptors))))
            edges.extend(descriptors)
            owners.append(p)
            parents.append(face)
            vertex_owners.extend([p] * len(descriptors))
        result = wp.empty(len(edges), dtype=wp.vec3d, device=self.vertices.device, requires_grad=True)
        if edges:
            wp.launch(
                _clip_vertices,
                len(edges),
                inputs=[
                    self.vertices,
                    wp.array(edges, dtype=wp.vec2i, device=result.device),
                    wp.array(vertex_owners, dtype=int, device=result.device),
                    point,
                    normal,
                    int(per_particle),
                    wp.float64(sign),
                ],
                outputs=[result],
                device=result.device,
            )
        return ReferenceSurface(
            result,
            tuple(polygons),
            np.asarray(owners, dtype=np.int64),
            self.roundoff,
            _gather(self.normals, parents),
        )

    def subtract(self, previous):
        """Keep new reference area; noncoplanar intersections have zero area."""
        result = self
        old_values = previous.vertices.numpy().astype(np.longdouble)
        old_normals = previous.normals.numpy().astype(np.longdouble)
        for old_face, (old_polygon, p) in enumerate(zip(previous.polygons, previous.owners)):
            if not result.polygons:
                break
            edge_ids = [
                (start, old_polygon[(j + 1) % len(old_polygon)]) for j, start in enumerate(old_polygon)
            ]
            edge_ids = [
                edge
                for edge in edge_ids
                if np.linalg.norm(old_values[edge[1]] - old_values[edge[0]]) > 2 * previous.roundoff[p]
            ]
            if len(edge_ids) < 2:
                continue
            a = old_values[old_polygon[0]]
            n = old_normals[old_face]
            old_points = old_values[list(old_polygon)]
            values = result.vertices.numpy().astype(np.longdouble)
            selected, other = [], []
            for i, (polygon, owner) in enumerate(zip(result.polygons, result.owners)):
                points = values[list(polygon)]
                error = result.roundoff[owner] + previous.roundoff[p]
                bound = 32 * EPS * ((np.abs(points) + np.abs(a)) @ np.abs(n)) + error * np.linalg.norm(n)
                overlap = (
                    owner == p
                    and np.all(np.abs((points - a) @ n) <= bound)
                    and np.all(points.max(axis=0) >= old_points.min(axis=0) - error)
                    and np.all(points.min(axis=0) <= old_points.max(axis=0) + error)
                )
                (selected if overlap else other).append(i)
            if not selected:
                continue
            remaining, kept = result.select(selected), result.select(other)
            for edge in edge_ids:
                point = wp.empty(1, dtype=wp.vec3d, device=self.vertices.device, requires_grad=True)
                normal = wp.empty(1, dtype=wp.vec3d, device=self.vertices.device, requires_grad=True)
                wp.launch(
                    _polygon_plane,
                    1,
                    inputs=[previous.vertices, previous.normals, old_face, wp.vec2i(edge)],
                    outputs=[point, normal],
                    device=self.vertices.device,
                )
                kept = kept.join(remaining.clip(point, normal, sign=-1))
                remaining = remaining.clip(point, normal, include_boundary=False)
                if not remaining.polygons:
                    break
            result = kept
        return result

    def quadrature(self, reference):
        triangles, owners, face_ids = [], [], []
        for i, (face, p) in enumerate(zip(self.polygons, self.owners)):
            for j in range(1, len(face) - 1):
                triangle = [face[0], face[j], face[j + 1]]
                triangles.append(triangle)
                owners.append(p)
                face_ids.append(i)
        count, device = 3 * len(triangles), self.vertices.device
        points = wp.empty(count, dtype=wp.vec3d, device=device, requires_grad=True)
        area = wp.empty(count, dtype=wp.float64, device=device, requires_grad=True)
        normals = wp.empty(count, dtype=wp.vec3d, device=device, requires_grad=True)
        if triangles:
            wp.launch(
                _quadrature,
                (len(triangles), 3),
                inputs=[
                    self.vertices,
                    wp.array(triangles, dtype=wp.vec3i, device=device),
                    wp.array(owners, dtype=int, device=device),
                    wp.array(face_ids, dtype=int, device=device),
                    self.normals,
                    reference,
                ],
                outputs=[points, area, normals],
                device=device,
            )
        return CutQuadrature(points, area, normals, np.repeat(owners, 3).astype(np.int64))


def swept_edge(
    position0,
    deformation0,
    position1,
    deformation1,
    half_extent,
    tool_position0,
    tool_rotation0,
    tool_position1,
    tool_rotation1,
    tool_half_size,
    *,
    edge_segments=None,
):
    """Pull both accepted edge poses into each particle's material coordinates.

    Each endpoint uses its own particle state. Space and time are triangulated.
    Default edge counts use blade width and reference particle width, so they
    stay fixed as the material deforms. Explicit counts support convergence tests.
    """
    count, device = len(position0), position0.device
    sources = [
        position0,
        deformation0,
        position1,
        deformation1,
        half_extent,
        tool_position0,
        tool_rotation0,
        tool_position1,
        tool_rotation1,
        tool_half_size,
    ]
    dtypes = [
        wp.vec3d,
        wp.mat33d,
        wp.vec3d,
        wp.mat33d,
        wp.vec3d,
        wp.vec3d,
        wp.quatd,
        wp.vec3d,
        wp.quatd,
        wp.vec3d,
    ]
    shapes = [(count,)] * 5 + [(1,)] * 5
    if any(
        a.device != device or a.dtype != dtype or a.shape != shape
        for a, dtype, shape in zip(sources, dtypes, shapes)
    ):
        raise ValueError("Swept edges require matching float64 particle states and single blade poses")
    data = [a.numpy() for a in sources]
    if any(not np.isfinite(a).all() for a in data):
        raise ValueError("Swept-edge inputs must be finite")
    if (
        np.any(data[4] <= 0)
        or np.any(data[9] <= 0)
        or any(np.linalg.norm(data[i][0]) == 0 for i in [6, 8])
        or any(np.any(np.linalg.det(data[i]) <= 0) for i in [1, 3])
    ):
        raise ValueError(
            "Swept edges require positive domain sizes, deformation determinants and nonzero rotations"
        )
    endpoints, errors = [], []
    for x, F, position, rotation in [
        (position0, deformation0, tool_position0, tool_rotation0),
        (position1, deformation1, tool_position1, tool_rotation1),
    ]:
        vertices = wp.empty(2 * count, dtype=wp.vec3d, device=device, requires_grad=True)
        wp.launch(
            _pullback_edge,
            (count, 2),
            inputs=[x, F, position, rotation, tool_half_size],
            outputs=[vertices],
            device=device,
        )
        endpoints.append(vertices)
        inverse = np.linalg.inv(F.numpy())
        condition = np.linalg.norm(F.numpy(), ord=np.inf, axis=(1, 2)) * np.linalg.norm(
            inverse, ord=np.inf, axis=(1, 2)
        )
        gamma = 64 * EPS * condition
        if np.any(gamma >= 1):
            raise FloatingPointError("Particle pullback cannot resolve a surface in float64")
        scale = np.linalg.norm(np.abs(x.numpy()) + np.abs(position.numpy()) + np.max(data[9]), axis=1)
        errors.append(gamma / (1 - gamma) * np.linalg.norm(inverse, ord=np.inf, axis=(1, 2)) * scale)
    vertices = wp.empty(4 * count, dtype=wp.vec3d, device=device, requires_grad=True)
    wp.launch(_join, len(vertices), inputs=endpoints, outputs=[vertices], device=device)
    if edge_segments is None:
        segments = np.maximum(1, np.ceil(data[9][0, 1] / data[4].min(axis=1)).astype(int))
    else:
        if (
            isinstance(edge_segments, (bool, np.bool_))
            or not isinstance(edge_segments, (int, np.integer))
            or edge_segments < 1
        ):
            raise ValueError("Edge subdivision requires a positive integer segment count")
        segments = np.full(count, edge_segments)
    edges, fractions, starts = [], [], []
    for p, pieces in enumerate(segments):
        starts.append(len(edges))
        for time in range(2):
            edges.extend([(2 * p + 2 * count * time, 2 * p + 2 * count * time + 1)] * (pieces + 1))
            fractions.extend(np.arange(pieces + 1) / pieces)
    refined = wp.empty(len(edges), dtype=wp.vec3d, device=device, requires_grad=True)
    wp.launch(
        _subdivide,
        len(edges),
        inputs=[
            vertices,
            wp.array(edges, dtype=wp.vec2i, device=device),
            wp.array(fractions, dtype=wp.float64, device=device),
        ],
        outputs=[refined],
        device=device,
    )
    vertices = refined
    values = vertices.numpy().astype(np.longdouble)
    roundoff = np.maximum(*errors)
    polygons, owners = [], []
    for p in range(count):
        for j in range(segments[p]):
            a = starts[p] + j
            b, c, d = a + 1, a + segments[p] + 1, a + segments[p] + 2
            for face in [(a, b, c), (c, b, d)]:
                offsets = values[list(face)]
                # Positive box faces are owned by the neighbouring particle domain.
                shared = np.any(np.all(np.abs(offsets - data[4][p]) <= roundoff[p], axis=0))
                if not shared and _face(values, face, roundoff[p]) is not None:
                    polygons.append(face)
                    owners.append(p)
    result = ReferenceSurface(vertices, tuple(polygons), np.asarray(owners, dtype=np.int64), roundoff)
    for axis in range(3):
        for sign in [-1, 1]:
            point = wp.empty(count, dtype=wp.vec3d, device=device, requires_grad=True)
            normal = wp.empty(count, dtype=wp.vec3d, device=device, requires_grad=True)
            wp.launch(
                _box_plane,
                count,
                inputs=[half_extent, axis, wp.float64(sign)],
                outputs=[point, normal],
                device=device,
            )
            result = result.clip(point, normal, include_boundary=sign < 0)
    return result
