"""Material-coordinate surfaces swept by a cutting edge.

Each accepted interval contributes two triangles per affine particle domain.
Coordinates are particle-local reference offsets in metres. Coplanar polygon
difference removes covered area while existing polygons remain unchanged.
"""

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from .reference_cut import CutQuadrature, _clip_polygon
from . import _exact_surface as exact


EPS = np.finfo(np.float64).eps


@wp.kernel
def _join(a: wp.array(dtype=wp.vec3d), b: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if i < a.shape[0]:
        out[i] = a[i]
    else:
        out[i] = b[i - a.shape[0]]


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
    exact_vertices: np.ndarray | None = None
    reference_remainder: np.ndarray | None = None
    vertex_head: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        coordinates = (
            exact.rational(self.vertices.numpy()) if self.exact_vertices is None else self.exact_vertices
        )
        coordinates = np.array(coordinates, dtype=object, copy=True)
        if coordinates.shape != (len(self.vertices), 3):
            raise ValueError("Exact surface coordinates must match the vertex array")
        coordinates.flags.writeable = False
        object.__setattr__(self, "exact_vertices", coordinates)
        head = self.vertices.numpy().copy()
        head.flags.writeable = False
        object.__setattr__(self, "vertex_head", head)
        remainder = self.reference_remainder
        if remainder is None:
            remainder = exact.rational(np.zeros((len(self.roundoff), 3)))
        remainder = np.array(remainder, dtype=object, copy=True)
        if remainder.shape != (len(self.roundoff), 3):
            raise ValueError("Reference remainders must match the particle set")
        remainder.flags.writeable = False
        object.__setattr__(self, "reference_remainder", remainder)
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

    def coordinates(self):
        """Apply edits to the float64 head while retaining construction remainders."""
        current = self.vertices.numpy()
        head = self.vertex_head
        if np.array_equal(current, head):
            return self.exact_vertices
        return self.exact_vertices + exact.rational(current) - exact.rational(head)

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
            self.coordinates()[used],
            self.reference_remainder,
        )

    def join(self, other):
        if not self.polygons:
            return other
        if not other.polygons:
            return self
        if self.vertices.device != other.vertices.device or self.roundoff.shape != other.roundoff.shape:
            raise ValueError("Joined reference surfaces must use the same particle set and device")
        if not np.array_equal(self.reference_remainder, other.reference_remainder):
            raise ValueError("Joined surfaces must use the same reference coordinate system")
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
            np.concatenate([self.coordinates(), other.coordinates()]),
            self.reference_remainder,
        )

    def clip(
        self,
        point,
        normal,
        *,
        sign=1,
        include_boundary=True,
        exact_point=None,
        exact_normal=None,
    ):
        if sign not in (-1, 1):
            raise ValueError("Half-plane orientation must be -1 or 1")
        values = self.coordinates()
        points = exact.rational(point.numpy()) if exact_point is None else exact_point
        normals = exact.rational(normal.numpy()) if exact_normal is None else exact_normal
        per_particle = len(point) != 1
        if point.shape != normal.shape or len(point) not in (1, len(self.roundoff)):
            raise ValueError("Clip planes must be global or indexed by particle")
        edges, owners, vertex_owners, polygons, parents, locations_all = [], [], [], [], [], []
        for face, (polygon, p) in enumerate(zip(self.polygons, self.owners)):
            j = p if per_particle else 0
            coordinates = values[list(polygon)]
            n, c = sign * normals[j], points[j]
            level = (coordinates - c) @ n
            descriptors, locations = _clip_polygon(
                values, polygon, level, boundary_intersections=True, include_boundary=include_boundary
            )
            if len(descriptors) < 3:
                continue
            start = len(edges)
            polygons.append(tuple(range(start, start + len(descriptors))))
            edges.extend(descriptors)
            locations_all.extend(locations)
            owners.append(p)
            parents.append(face)
            vertex_owners.extend([p] * len(descriptors))
        result, coordinates = exact.intersections(
            self.vertices, values, edges, vertex_owners, point, normal, points, normals, locations_all
        )
        return ReferenceSurface(
            result,
            tuple(polygons),
            np.asarray(owners, dtype=np.int64),
            self.roundoff,
            _gather(self.normals, parents),
            coordinates,
            self.reference_remainder,
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
                point, normal, exact_point, exact_normal = exact.polygon_plane(
                    previous.vertices, previous.coordinates(), previous.normals, old_face, edge
                )
                planes = dict(exact_point=exact_point, exact_normal=exact_normal)
                kept = kept.join(remaining.clip(point, normal, sign=-1, **planes))
                remaining = remaining.clip(point, normal, include_boundary=False, **planes)
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
        points, area, normals = exact.quadrature(
            self.vertices,
            self.coordinates(),
            triangles,
            owners,
            face_ids,
            self.normals,
            reference,
            self.reference_remainder,
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
    reference_coordinates=None,
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
    endpoints, errors, exact_endpoints = [], [], []
    remainder = (
        exact.rational(np.zeros((count, 3)))
        if reference_coordinates is None
        else reference_coordinates.remainder
    )
    if remainder.shape != (count, 3):
        raise ValueError("Reference coordinates must match the particle states")
    for x, F, position, rotation in [
        (position0, deformation0, tool_position0, tool_rotation0),
        (position1, deformation1, tool_position1, tool_rotation1),
    ]:
        vertices, coordinates = exact.pullback(x, F, position, rotation, tool_half_size, remainder)
        exact_endpoints.append(coordinates)
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
    vertices, coordinates = exact.subdivide(vertices, edges, fractions, np.concatenate(exact_endpoints))
    values = vertices.numpy().astype(np.longdouble)
    roundoff = np.maximum(*errors)
    polygons, owners = [], []
    for p in range(count):
        for j in range(segments[p]):
            a = starts[p] + j
            b, c, d = a + 1, a + segments[p] + 1, a + segments[p] + 2
            for face in [(a, b, c), (c, b, d)]:
                if _face(values, face, roundoff[p]) is not None:
                    polygons.append(face)
                    owners.append(p)
    result = ReferenceSurface(
        vertices,
        tuple(polygons),
        np.asarray(owners, dtype=np.int64),
        roundoff,
        exact_vertices=coordinates,
        reference_remainder=remainder,
    )
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
