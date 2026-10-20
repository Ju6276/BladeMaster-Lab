"""Linked-volume / Hermite / local dual-contouring surface (paper III-F), CPU reference."""

from itertools import product
from dataclasses import dataclass
from functools import cached_property
import numpy as np
from .geometry import levelset
from .topology import compatible

OFFSETS = np.array(list(product((0, 1), repeat=3)), int)
QUAD_SPLIT_MARGIN = 1.0e-6
# Keep bounded QEF solutions off shared cell planes. Coincident clamped edges
# can cross there even when every quad is split consistently. A finite margin
# (1.5 micrometres on a 1.5 mm grid) also survives material-grid interpolation.
QEF_CELL_INSET = 1.0e-3
QEF_REGION_REGULARIZATION = 1.0e-3
LOCAL_EDGES = [
    (a, b) for a in range(8) for b in range(a + 1, 8) if np.abs(OFFSETS[a] - OFFSETS[b]).sum() == 1
]


def segment_triangle(a, b, tri):
    """Vectorized finite segment/triangle intersections, including endpoints."""
    direction = b - a
    e1 = tri[1] - tri[0]
    e2 = tri[2] - tri[0]
    h = np.cross(direction, e2)
    det = h @ e1
    valid = np.abs(det) > 1.0e-15
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
    s = a - tri[0]
    u = np.einsum("ij,ij->i", s, h) * inv
    q = np.cross(s, e1)
    v = np.einsum("ij,ij->i", direction, q) * inv
    t = (q @ e2) * inv
    return (
        valid & (u >= -1.0e-9) & (v >= -1.0e-9) & (u + v <= 1 + 1.0e-9) & (t >= -1.0e-9) & (t <= 1 + 1.0e-9),
        t,
    )


def moving_segment_hits(a0, b0, a1, b1, edge0, edge1):
    """CCD between linearly moving material links and a finite blade edge.

    Testing only current links against the blade's swept triangle can miss a
    cut when the material itself moves across the edge between observations.
    Coplanarity is a cubic in time; solve it, then test both segment bounds.
    This affects surface topology only, never the mechanical cutting labels.
    """
    hit = np.zeros(len(a0), bool)
    alpha = np.zeros(len(a0))
    times = np.zeros(len(a0))
    c0, d0 = edge0
    c1, d1 = edge1
    v0 = d0 - c0
    dv = (d1 - c1) - v0
    blade_lo = np.minimum(edge0.min(0), edge1.min(0))
    blade_hi = np.maximum(edge0.max(0), edge1.max(0))
    for i in range(len(a0)):
        endpoints = np.array([a0[i], b0[i], a1[i], b1[i]])
        if np.any(endpoints.max(0) < blade_lo - 1.0e-10) or np.any(endpoints.min(0) > blade_hi + 1.0e-10):
            continue
        u0 = b0[i] - a0[i]
        du = b1[i] - a1[i] - u0
        w0 = c0 - a0[i]
        dw = c1 - a1[i] - w0
        cros0 = np.cross(u0, v0)
        cros1 = np.cross(du, v0) + np.cross(u0, dv)
        cros2 = np.cross(du, dv)
        coeff = np.array([cros0 @ w0, cros0 @ dw + cros1 @ w0, cros1 @ dw + cros2 @ w0, cros2 @ dw])
        threshold = max(abs(coeff).max() * 1.0e-12, 1.0e-25)
        nonzero = np.flatnonzero(abs(coeff) > threshold)
        candidates = []
        if len(nonzero):
            roots = np.roots(coeff[: nonzero[-1] + 1][::-1])
            candidates = [
                float(r.real) for r in roots if abs(r.imag) < 1.0e-8 and -1.0e-9 <= r.real <= 1 + 1.0e-9
            ]
        else:
            # Coplanar for the whole interval: intersection starts/ends when
            # an endpoint crosses the other supporting line. Each cross-product
            # component is at most quadratic. Include endpoints and intervals.
            candidates = [0.0, 1.0]
            for p0, dp, q0, dq in [
                (w0, dw, u0, du),
                (w0 + v0, dw + dv, u0, du),
                (-w0, -dw, v0, dv),
                (u0 - w0, du - dw, v0, dv),
            ]:
                cc = np.array([np.cross(p0, q0), np.cross(dp, q0) + np.cross(p0, dq), np.cross(dp, dq)])
                if np.max(abs(cc)) <= 1.0e-25:
                    # An endpoint stays on the supporting line: overlap can
                    # change only when it reaches one of the segment endpoints.
                    for offset, velocity in ((p0, dp), (p0 - q0, dp - dq)):
                        for axis in np.flatnonzero(abs(velocity) > 1.0e-25):
                            root = -offset[axis] / velocity[axis]
                            if 0 <= root <= 1:
                                candidates.append(float(root))
                    continue
                for axis in range(3):
                    nz = np.flatnonzero(abs(cc[:, axis]) > 1.0e-25)
                    if len(nz):
                        for root in np.roots(cc[: nz[-1] + 1, axis][::-1]):
                            if abs(root.imag) < 1.0e-8 and 0 <= root.real <= 1:
                                candidates.append(float(root.real))
            candidates = sorted(set(candidates))
            candidates += list((np.array(candidates[:-1]) + candidates[1:]) / 2)
        for t in sorted(candidates):
            t = float(np.clip(t, 0, 1))
            u = u0 + t * du
            v = v0 + t * dv
            w = w0 + t * dw
            matrix = np.column_stack([u, -v])
            uv, _, rank, _ = np.linalg.lstsq(matrix, w, rcond=None)
            if rank < 2:
                # A minimum-norm line solution can lie outside both segments
                # even when their finite collinear intervals overlap.
                uu, vv = u @ u, v @ v
                uv[0] = np.clip(min(u @ w, u @ (w + v)) / uu, 0, 1) if uu > 0 else 0
                uv[1] = v @ (u * uv[0] - w) / vv if vv > 0 else 0
            if (
                np.linalg.norm(matrix @ uv - w) <= 1.0e-9
                and np.all(uv >= -1.0e-8)
                and np.all(uv <= 1 + 1.0e-8)
            ):
                hit[i] = True
                alpha[i] = uv[0]
                times[i] = t
                break
    return hit, alpha, times


def box_lstsq(A, b, lower, upper):
    """Exact minimiser of |A x - b| over lower <= x <= upper for three unknowns.

    The optimum holds some coordinates at a bound and is the least-squares solution
    in the others, so all 27 choices are tried. Among equal residuals (rank-deficient
    A) the smallest |x| wins, like the minimum-norm unconstrained solve.
    """
    best, best_key = None, None
    for choice in product((0, 1, 2), repeat=3):
        choice = np.array(choice)
        x = np.where(choice == 1, lower, upper).astype(float)
        free = choice == 0
        if free.any():
            x[free] = np.linalg.lstsq(A[:, free], b - A[:, ~free] @ x[~free], rcond=None)[0]
            if np.any(x[free] < lower[free]) or np.any(x[free] > upper[free]):
                continue
        key = (round(float(np.linalg.norm(A @ x - b)), 12), float(np.linalg.norm(x)))
        if best_key is None or key < best_key:
            best, best_key = x, key
    return best


def constrained_box_lstsq(A, b, planes, offsets, lower, upper):
    """Minimise the QEF inside a cell while keeping a crack front on its planes.

    Return None when the planes have no common point in the box. Curved cuts
    can supply inconsistent tangent planes; those cells retain the bounded QEF.
    Coordinates are offsets from the Hermite centroid, in metres.
    """
    tolerance = 1e-10 * float(np.max(upper - lower))

    def solve(constraints, values):
        _, singular, directions = np.linalg.svd(constraints, full_matrices=True)
        rank = np.count_nonzero(singular > singular[0] * 1e-10)
        origin = np.linalg.lstsq(constraints, values, rcond=1e-10)[0]
        if np.max(np.abs(constraints @ origin - values)) > tolerance:
            return None
        nullspace = directions[rank:].T
        u, singular, directions = np.linalg.svd(A @ nullspace, full_matrices=False)
        # A cut-plane row is zero in its own nullspace. Use the original
        # QEF scale so roundoff in these rows cannot choose an arbitrary cell corner.
        retained = singular > 1e-10 * np.linalg.norm(A)
        free = directions[retained].T @ ((u[:, retained].T @ (b - A @ origin)) / singular[retained])
        point = origin + nullspace @ free
        if np.any(point < lower - tolerance) or np.any(point > upper + tolerance):
            return None
        return point

    point = solve(planes, offsets)
    if point is not None:
        return point
    best, best_key = None, None
    for choice in product((0, 1, 2), repeat=3):
        choice = np.asarray(choice)
        fixed = np.flatnonzero(choice)
        if not len(fixed):
            continue
        constraints = np.vstack([planes, np.eye(3)[fixed]])
        values = np.r_[offsets, np.where(choice[fixed] == 1, lower[fixed], upper[fixed])]
        point = solve(constraints, values)
        if point is None:
            continue
        key = (float(np.linalg.norm(A @ point - b)), float(np.linalg.norm(point)))
        if (
            best_key is None
            or key[0] < best_key[0] - tolerance
            or (abs(key[0] - best_key[0]) <= tolerance and key[1] < best_key[1])
        ):
            best, best_key = point, key
    return best


def _one_cut_plane(points, normals, lower, upper):
    """Average plane of one blade, or None when these samples are not that case.

    Returns centre, unit normal and the plane offset, with
    normal · (x - centre) = offset. The least-squares intersection of the
    tangent planes is the centre of curvature when it lies more than one cell
    diagonal outside the cell. A second singular value above a tenth does not
    change that: the corner of a real second cut still meets inside that
    neighbourhood and keeps the ordinary QEF. Skin samples never reach here.
    """
    points = np.asarray(points, float)
    normals = np.asarray(normals, float)
    length = np.linalg.norm(normals, axis=1)
    usable = length > 1.0e-12
    if not np.any(usable):
        return None
    unit = normals[usable] / length[usable, None]
    samples = points[usable]
    centre = samples.mean(0)
    rhs = np.einsum("ij,ij->i", unit, samples - centre)
    free = centre + np.linalg.lstsq(unit, rhs, rcond=None)[0]
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    gap = np.maximum(np.maximum(lower - free, free - upper), 0.0)
    if np.linalg.norm(gap) <= np.linalg.norm(upper - lower):
        return None
    left, values, vt = np.linalg.svd(unit, full_matrices=False)
    normal = vt[0]
    offset = float((left[:, 0] @ rhs) / values[0])
    return centre, normal, offset


def pulled_cut_normal(normal, deformation):
    """Pull a world cut normal back by one nodal deformation.

    The stored sample has to be a direction. A singular average maps the normal
    to zero, and dividing by that length is not a direction, so the world normal
    stays.
    """
    pulled = np.asarray(deformation, float).T @ np.asarray(normal, float)
    length = float(np.linalg.norm(pulled))
    if length > 0.0 and np.isfinite(pulled).all():
        return pulled / length
    return np.array(normal, float, copy=True)


def unsampled_face_centroid(point, samples, lower, upper, inset):
    """Centroid of the samples when a clamped point has left every face they touch.

    The bounded QEF can sit on an inset face that contains none of this region's
    Hermite samples. That face belongs to another corner of the cell. The sample
    centroid is inside the cell only when the cuts themselves pass through it;
    a centroid outside the cell is not a representative.
    """
    point = np.asarray(point, float)
    samples = np.asarray(samples, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    if len(samples) == 0 or inset < 0.0:
        return None
    lo, hi = lower + inset, upper - inset
    tolerance = max(float(inset) * 4.0, 1.0e-9)
    occupied = []
    for axis in range(3):
        if point[axis] <= lo[axis] + tolerance:
            occupied.append((axis, 0))
        elif point[axis] >= hi[axis] - tolerance:
            occupied.append((axis, 1))
    if not occupied:
        return None
    face_tolerance = max(float(np.max(upper - lower)) * 1.0e-4, 1.0e-9)
    for axis, side in occupied:
        bound = lower[axis] if side == 0 else upper[axis]
        if np.any(np.abs(samples[:, axis] - bound) <= face_tolerance):
            return None
    centroid = samples.mean(0)
    if np.any(centroid < lower - 1.0e-12) or np.any(centroid > upper + 1.0e-12):
        return None
    return centroid


def average_plane_point(points, normals, lower, upper):
    """Sample centroid on one cut's average plane, or None when that is not the case."""
    plane = _one_cut_plane(points, normals, lower, upper)
    if plane is None:
        return None
    centre, normal, offset = plane
    point = centre + normal * offset
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    if np.any(point < lower) or np.any(point > upper):
        return None
    return point


@dataclass
class Mesh:
    reference: np.ndarray
    triangles: np.ndarray
    codes: np.ndarray
    cells: np.ndarray
    weights: np.ndarray


# Ternary digits that fit a nonnegative int64 (3**39 < 2**63).
CODE_DIGITS = 39


class LinkedSurface:
    def __init__(self, size, spacing, shape="block"):
        self.size = np.array(size, float)
        self.h = float(spacing)
        self.shape = shape
        # Half-cell padding keeps box boundaries off primal nodes and encloses
        # the original surface. It is not a motion/workspace constraint.
        self.origin = np.array([-size[0] / 2, -size[1] / 2, 0]) - spacing / 2
        self.dims = np.ceil(self.size / spacing).astype(int) + 2
        ijk = np.indices(self.dims).reshape(3, -1).T
        self.reference = self.origin + ijk * spacing
        sdf = levelset(self.reference, size, shape)
        self.inside = sdf < 0
        self.codes = np.zeros(len(ijk), np.int64)
        # Links from each node to its +x, +y, +z neighbour, ordered by (node, axis).
        strides = np.array([self.dims[1] * self.dims[2], self.dims[2], 1])
        start, axis = np.nonzero(ijk < self.dims - 1)
        self.edges = np.column_stack([start, start + strides[axis]])
        self.axes = axis
        link_of = np.full((len(ijk), 3), -1)
        link_of[start, axis] = np.arange(len(start))
        self.broken = np.zeros(len(self.edges), bool)
        self.boundary = self.inside[self.edges[:, 0]] != self.inside[self.edges[:, 1]]
        # Hermite data on boundary links: bisection to the level set, then its gradient.
        crossing = np.flatnonzero(self.boundary)
        a, b = self.reference[self.edges[crossing, 0]], self.reference[self.edges[crossing, 1]]
        lo, hi = np.zeros(len(crossing)), np.ones(len(crossing))
        first_inside = sdf[self.edges[crossing, 0]] < 0
        for _ in range(35):
            t = (lo + hi) / 2
            same = (levelset(a + t[:, None] * (b - a), size, shape) < 0) == first_inside
            lo, hi = np.where(same, t, lo), np.where(same, hi, t)
        points = a + ((lo + hi) / 2)[:, None] * (b - a)
        eps = spacing * 1.0e-4
        normals = np.column_stack(
            [
                (
                    levelset(points + eps * np.eye(3)[j], size, shape)
                    - levelset(points - eps * np.eye(3)[j], size, shape)
                )
                / (2 * eps)
                for j in range(3)
            ]
        )
        normals /= np.maximum(np.linalg.norm(normals, axis=1), 1.0e-15)[:, None]
        self.hermite = {int(e): (p, n) for e, p, n in zip(crossing, points, normals)}
        # Cells (in grid order) with at least one inside corner, and their twelve links.
        cell_ijk = np.indices(self.dims - 1).reshape(3, -1).T
        corners = (cell_ijk[:, None, :] + OFFSETS[None]) @ strides
        keep = self.inside[corners].any(1)
        self.cells = corners[keep]
        local_a = np.array([a for a, _ in LOCAL_EDGES])
        local_axis = np.array([np.flatnonzero(OFFSETS[b] - OFFSETS[a])[0] for a, b in LOCAL_EDGES])
        self.cell_edges = link_of[self.cells[:, local_a], local_axis]
        # Cells around each link, counterclockwise about its axis, and each link end's corner slot.
        link = self.cell_edges.ravel()
        cell = np.repeat(np.arange(len(self.cells)), len(LOCAL_EDGES))
        ends = np.array(LOCAL_EDGES)[np.tile(np.arange(len(LOCAL_EDGES)), len(self.cells))]
        other = np.array([[1, 2], [0, 2], [0, 1]])[self.axes[link]]
        offset = self.reference[self.cells[cell]].mean(1) - self.reference[self.edges[link, 0]]
        angle = np.arctan2(
            np.take_along_axis(offset, other[:, 1:], 1)[:, 0],
            np.take_along_axis(offset, other[:, :1], 1)[:, 0],
        )
        order = np.lexsort((angle, link))
        link, cell, ends = link[order], cell[order], ends[order]
        group_start = np.r_[0, np.flatnonzero(np.diff(link)) + 1]
        slot = np.arange(len(link)) - np.repeat(group_start, np.diff(np.r_[group_start, len(link)]))
        self.incident_cells = np.full((len(self.edges), 4), -1)
        self.incident_cells[link, slot] = cell
        self.incident_corner = np.zeros((len(self.edges), 2, 4), int)
        self.incident_corner[link, 0, slot] = ends[:, 0]
        self.incident_corner[link, 1, slot] = ends[:, 1]
        # Cells sharing each node, for invalidating regions when a node's side code changes.
        node = self.cells.ravel()
        by_node = np.argsort(node, kind="stable")
        split = np.searchsorted(node[by_node], np.arange(len(ijk) + 1))
        owner = np.repeat(np.arange(len(self.cells)), 8)[by_node]
        self.node_cells = [owner[split[i] : split[i + 1]] for i in range(len(ijk))]
        # Local regions and QEF points per cell, valid while the cell's severed links are unchanged.
        self.cell_geometry = {}
        # At most one region per corner. corner_region: region slot holding each corner, or -1.
        self.region_valid = np.zeros((len(self.cells), 8), bool)
        self.region_point = np.zeros((len(self.cells), 8, 3))
        self.region_code = np.zeros((len(self.cells), 8), np.int64)
        self.region_weights = np.zeros((len(self.cells), 8, 8))
        self.corner_region = np.full((len(self.cells), 8), -1)
        self.dirty = set(range(len(self.cells)))
        self.updated_cells = 0
        self._mesh_cache = None
        self._opposite_edges = {}
        # World normals of completed cuts. Side 2 is the positive side, matching
        # the sweep test `dot(point - blade, normal) < 0` → side 1.
        self.cut_normals = {}

    def cells_around(self, e):
        row = self.incident_cells[e]
        return row[row >= 0].tolist()

    def swept_candidates(self, old, current, blade_points, candidates=None):
        """Conservative link AABBs for linear material motion and a complete tool path."""
        if candidates is None:
            candidates = np.flatnonzero(self.inside[self.edges].all(axis=1))
        candidates = np.asarray(candidates, dtype=int)
        candidates = candidates[~self.broken[candidates]]
        a, b = self.edges[candidates].T
        lower = np.minimum(np.minimum(old[a], old[b]), np.minimum(current[a], current[b]))
        upper = np.maximum(np.maximum(old[a], old[b]), np.maximum(current[a], current[b]))
        blade_lower = np.min(blade_points, axis=0) - 1e-10
        blade_upper = np.max(blade_points, axis=0) + 1e-10
        overlaps = np.all(upper >= blade_lower, axis=1) & np.all(lower <= blade_upper, axis=1)
        return candidates[overlaps]

    def sweep(
        self,
        edge_before,
        edge_after,
        current_nodes=None,
        cut_index=0,
        normal_world=None,
        node_deformation=None,
        previous_nodes=None,
        previous_deformation=None,
        ccd=moving_segment_hits,
        band=None,
        candidate_edges=None,
        anchor_digits=None,
    ):
        """Sever links hit by the actual swept edge; returning to a cut cannot heal it.

        current_nodes must be the current compatible field embedding of the
        reference nodes. Identity is useful for geometry-only unit tests.
        band = (lateral half-width, depth behind the edge) also labels every node the
        edge passes, matching the material's side assignment.
        """
        if not 0 <= cut_index < CODE_DIGITS:
            raise ValueError(f"int64 side codes hold at most {CODE_DIGITS} cuts")
        if normal_world is not None:
            normal = np.asarray(normal_world, float)
            length = np.linalg.norm(normal)
            if length > 0.0:
                self.cut_normals[int(cut_index)] = normal / length
        x = self.reference if current_nodes is None else np.asarray(current_nodes)
        quad = np.asarray([edge_before[0], edge_before[1], edge_after[1], edge_after[0]])
        if previous_nodes is not None:
            candidates = self.swept_candidates(np.asarray(previous_nodes), x, quad, candidate_edges)
        else:
            candidates = np.flatnonzero(self.inside[self.edges].all(axis=1) & ~self.broken)
        aa, bb = x[self.edges[candidates, 0]], x[self.edges[candidates, 1]]
        hits = []
        batches = []
        if previous_nodes is not None and len(candidates):
            old = np.asarray(previous_nodes)
            hit, alpha, time = ccd(
                old[self.edges[candidates, 0]],
                old[self.edges[candidates, 1]],
                aa,
                bb,
                np.asarray(edge_before),
                np.asarray(edge_after),
            )
            n = (
                np.asarray(normal_world, float)
                if normal_world is not None
                else np.cross(edge_before[1] - edge_before[0], edge_after[0] - edge_before[0])
            )
            if np.linalg.norm(n) < 1.0e-15:
                raise ValueError("A normal is required for a stationary blade with moving material")
            n = n / np.linalg.norm(n)
            batches.append((candidates[hit], alpha[hit], time[hit], n))
        for tri in (
            () if previous_nodes is not None or not len(candidates) else (quad[[0, 1, 2]], quad[[0, 2, 3]])
        ):
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            norm = np.linalg.norm(n)
            if norm < 1.0e-15:
                continue
            n /= norm
            if normal_world is not None:
                n = np.asarray(normal_world, float) / np.linalg.norm(normal_world)
            hit, t = segment_triangle(aa, bb, tri)
            batches.append((candidates[hit], t[hit], np.ones(hit.sum()), n))
        for batch, alphas, times, n in batches:
            for e, alpha, t in zip(batch, alphas, times):
                if self.broken[e]:
                    continue
                a, b = self.edges[e]
                p = (1 - alpha) * self.reference[a] + alpha * self.reference[b]
                # Pull normal back using this edge's local motion approximation.
                # Full material pullback is provided by embedding_fn below in
                # dynamic use; reference-only tests use identity deformation.
                nref = n.copy()
                if node_deformation is not None:
                    f = (1 - alpha) * node_deformation[a] + alpha * node_deformation[b]
                    if previous_deformation is not None:
                        oldf = (1 - alpha) * previous_deformation[a] + alpha * previous_deformation[b]
                        f = (1 - t) * oldf + t * f
                    if np.isfinite(f).all():
                        nref = pulled_cut_normal(n, f)
                self.hermite[int(e)] = (p, nref)
                self.broken[e] = True
                hits.append(int(e))
                self.dirty.update(self.cells_around(e))
                for node in (a, b):
                    if self.codes[node] // (3**cut_index) % 3 == 0:
                        point = (
                            x[node]
                            if previous_nodes is None
                            else (1 - t) * previous_nodes[node] + t * x[node]
                        )
                        blade_point = (1 - t) * np.asarray(edge_before)[0] + t * np.asarray(edge_after)[0]
                        side = 1 if np.dot(point - blade_point, n) < 0 else 2
                        if anchor_digits is not None:
                            owned = int(anchor_digits[node])
                            # The anchor particle is not labeled yet. A geometric
                            # side now can be the opposite of the side it receives
                            # once the edge reaches it.
                            if owned == 0:
                                continue
                            if owned == 1 or owned == 2:
                                side = owned
                        self.codes[node] += side * 3**cut_index
        changed = set(self.edges[hits].ravel()) if hits else set()
        if anchor_digits is not None:
            changed |= self._take_anchor_sides(anchor_digits, cut_index)
        if band is not None and normal_world is not None:
            changed |= self._assign_band(
                x, edge_before, edge_after, normal_world, cut_index, band, anchor_digits
            )
            changed |= self._sever_opposite(
                x, edge_after, normal_world, cut_index, node_deformation, refresh=bool(changed)
            )
        # A node's new code changes every incident region using that node, not
        # just cells adjacent to the severed edge.
        for node in changed:
            self.dirty.update(self.node_cells[node])
        return len(hits)

    def _assign_band(self, x, edge_before, edge_after, normal, cut_index, band, anchor_digits=None):
        """Side digits for nodes the blade edge has just passed, with the material's band rule.

        Without this, skin nodes near a cut keep digit 0, stay compatible with both
        pieces and are embedded in their average motion once the pieces move apart.
        A node that already follows a labeled particle keeps that particle's side.
        The embedded position can sit across the blade from the particle that owns
        the nodal field, and labeling from that position leaves the node with no support.
        """
        half_width, depth = band
        edge = np.asarray(edge_after, float)
        along = edge[1] - edge[0]
        half_length = np.linalg.norm(along) / 2
        along /= 2 * half_length
        n = np.asarray(normal, float) / np.linalg.norm(normal)
        up = np.cross(n, along)
        travel = np.linalg.norm(np.asarray(edge_after) - np.asarray(edge_before), axis=1).max()
        nodes = np.flatnonzero(self.inside & (self.codes // 3**cut_index % 3 == 0))
        if not len(nodes):
            return set()
        rel = x[nodes] - edge.mean(0)
        lateral, length, height = rel @ n, rel @ along, rel @ up
        selected = (
            (np.abs(lateral) <= half_width)
            & (np.abs(length) <= half_length)
            & (height >= 0)
            & (height <= depth + travel)
        )
        nodes = nodes[selected]
        sides = np.where(lateral[selected] < 0, 1, 2)
        if anchor_digits is not None:
            owned = np.asarray(anchor_digits, np.int64)[nodes]
            # 0 waits for that particle. 1 and 2 were copied already.
            keep = owned < 0
            nodes, sides = nodes[keep], sides[keep]
        if not len(nodes):
            return set()
        self.codes[nodes] += sides * 3**cut_index
        return set(nodes.tolist())

    def _take_anchor_sides(self, anchor_digits, cut_index):
        """Copy a labeled anchor particle's side onto the nodes that follow it.

        This reaches nodes the edge has already left. Leaving them unlabeled
        would keep a zero digit, and labeling them from an earlier embedded
        position can record the opposite side from the particle.
        """
        owned = np.asarray(anchor_digits, np.int64)
        take = self.inside & ((owned == 1) | (owned == 2)) & (self.codes // np.int64(3) ** np.int64(cut_index) % 3 == 0)
        nodes = np.flatnonzero(take)
        if not len(nodes):
            return set()
        self.codes[nodes] += owned[nodes] * (np.int64(3) ** np.int64(cut_index))
        return set(int(n) for n in nodes)

    def _sever_opposite(self, x, edge, normal, cut_index, node_deformation, refresh=True):
        """Sever intact links whose ends lie on opposite sides of this cut.

        Opposite sides never exchange momentum in the material, so such a link must not
        join one surface region. Without this, a link the swept-edge test just misses
        gives a region both side digits, and its vertex follows both pieces.

        The sample is where the cut plane meets the reference edge. Both embedded
        endpoints give that parameter from their chord. An endpoint with no material
        support is not a position, so the other endpoint's affine field carries the
        reference edge to the plane. If neither field meets the plane, the link is
        still severed and an existing finite sample is left as it was.
        """
        # Fixed reference links gain opposing sides only when a label changes.
        # Retain the full opposing set so a missed sever is still repaired.
        a, b = self.edges[:, 0], self.edges[:, 1]
        if refresh or cut_index not in self._opposite_edges:
            digit = self.codes // 3**cut_index % 3
            self._opposite_edges[cut_index] = np.flatnonzero(
                self.inside[a] & self.inside[b] & (digit[a] * digit[b] == 2)
            )
        opposing = self._opposite_edges[cut_index]
        crossing = opposing[~self.broken[opposing]]
        if not len(crossing):
            return set()
        n = np.asarray(normal, float) / np.linalg.norm(normal)
        point = np.asarray(edge, float).mean(0)
        a_nodes, b_nodes = a[crossing], b[crossing]
        xa, xb = x[a_nodes], x[b_nodes]
        both = np.isfinite(xa).all(1) & np.isfinite(xb).all(1)
        alphas = np.full(len(crossing), np.nan)
        if both.any():
            sa = (xa[both] - point) @ n
            sb = (xb[both] - point) @ n
            alphas[both] = np.clip(sa / np.where(sa != sb, sa - sb, 1.0), 0.0, 1.0)
        for e, alpha, i, j in zip(crossing, alphas, a_nodes, b_nodes):
            i, j = int(i), int(j)
            if not np.isfinite(alpha):
                alpha = self._affine_cut_alpha(i, j, x, node_deformation, point, n)
            if alpha is None:
                self.broken[e] = True
                self.dirty.update(self.cells_around(e))
                continue
            nref = n.copy()
            if node_deformation is not None:
                fields = []
                for weight, end in ((1.0 - alpha, i), (alpha, j)):
                    deformation = node_deformation[end]
                    if np.isfinite(deformation).all():
                        fields.append((weight, deformation))
                if len(fields) == 2:
                    nref = pulled_cut_normal(n, fields[0][0] * fields[0][1] + fields[1][0] * fields[1][1])
                elif len(fields) == 1:
                    nref = pulled_cut_normal(n, fields[0][1])
            self.hermite[int(e)] = ((1.0 - alpha) * self.reference[i] + alpha * self.reference[j], nref)
            self.broken[e] = True
            self.dirty.update(self.cells_around(e))
        return set(self.edges[crossing].ravel().tolist())

    def _affine_cut_alpha(self, i, j, x, node_deformation, point, normal):
        """Where one endpoint's affine field carries this reference edge across the plane.

        The parameter is on the reference edge, clipped to the segment the same way
        as the two-endpoint chord. A field parallel to the plane does not locate a hit.
        """
        if node_deformation is None:
            return None
        origin = field = None
        for end in (i, j):
            if np.isfinite(x[end]).all() and np.isfinite(node_deformation[end]).all():
                origin, field = end, node_deformation[end]
                break
        if origin is None:
            return None
        ref_i, ref_j = self.reference[i], self.reference[j]
        direction = field @ (ref_j - ref_i)
        scale = float(np.linalg.norm(direction) * np.linalg.norm(normal))
        denom = float(direction @ normal)
        if abs(denom) <= 1.0e-15 * max(scale, 1.0e-15):
            return None
        base = x[origin] + field @ (ref_i - self.reference[origin])
        return float(np.clip(((point - base) @ normal) / denom, 0.0, 1.0))

    def _update_cell(self, ci):
        corners = self.cells[ci]
        # Band labels change region codes far more often than links are severed.
        severed = self.broken[self.cell_edges[ci]].tobytes()
        cached = self.cell_geometry.get(ci)
        if cached is None or cached[0] != severed:
            cached = severed, self._cell_regions(ci)
            self.cell_geometry[ci] = cached
        powers = 3 ** np.arange(CODE_DIGITS, dtype=np.int64)
        self.region_valid[ci] = False
        self.corner_region[ci] = -1
        for slot, (region, p, weights) in enumerate(cached[1]):
            local = list(region)
            digits = self.codes[corners[local]][:, None] // powers % 3
            # Combine compatible assigned digits in this local graph region.
            low, high = np.where(digits > 0, digits, 3).min(0), digits.max(0)
            self.region_valid[ci, slot] = True
            self.region_point[ci, slot] = p
            self.region_code[ci, slot] = (np.where((low == high) & (high > 0), high, 0) * powers).sum()
            self.region_weights[ci, slot] = weights
            self.corner_region[ci, local] = slot

    def _cell_regions(self, ci):
        """Connected corner groups over intact links, each with its QEF point and trilinear weights."""
        corners = self.cells[ci]
        es = self.cell_edges[ci]
        active = set(np.flatnonzero(self.inside[corners]).tolist())
        corner_count = len(active)
        regions = []
        while active:
            region = {active.pop()}
            todo = list(region)
            while todo:
                a = todo.pop()
                for e, (i, j) in zip(es, LOCAL_EDGES):
                    if self.broken[e]:
                        continue
                    b = j if i == a else i if j == a else -1
                    if b in active:
                        active.remove(b)
                        region.add(b)
                        todo.append(b)
            points = []
            normals = []
            front_planes = []
            cut_only = True
            # At a crack front both ends of a severed edge may still connect
            # around intact edges within the cell. That connected region still
            # needs a Hermite representative; omitting it leaves an open front.
            for e, (a, b) in zip(es, LOCAL_EDGES):
                if e in self.hermite and (
                    ((a in region) != (b in region)) or (self.broken[e] and (a in region or b in region))
                ):
                    p, n = self.hermite[e]
                    points.append(p)
                    normals.append(n)
                    front_planes.append(self.broken[e] and a in region and b in region)
                    cut_only = cut_only and bool(self.broken[e])
            if not points:
                continue
            points = np.array(points)
            normals = np.array(normals)
            lower = self.reference[corners].min(0)
            upper = lower + self.h
            inset = self.h * QEF_CELL_INSET
            # One blade whose tangents meet at the centre of curvature. Skin
            # samples and a second cut stay on the anchored solve below.
            on_plane = average_plane_point(points, normals, lower, upper) if cut_only else None
            centre = points.mean(0)
            rhs = np.einsum("ij,ij->i", normals, points - centre)
            if on_plane is None:
                # Solve offset relative to mass point to avoid drifting in the QEF
                # nullspace; bounded solve is only needed if unconstrained is outside.
                if len(region) < corner_count:
                    # Distinct regions in one cell can drift into each other along
                    # weak QEF directions. Anchor each to its own Hermite centroid;
                    # normalize the penalty by sample count. Flat cuts stay exact.
                    anchor = np.sqrt(QEF_REGION_REGULARIZATION * len(normals)) * np.eye(3)
                    normals = np.vstack([normals, anchor])
                    rhs = np.r_[rhs, np.zeros(3)]
                offset = np.linalg.lstsq(normals, rhs, rcond=None)[0]
                p = centre + offset
                if ((p < lower + inset) | (p > upper - inset)).any():
                    p = centre + box_lstsq(normals, rhs, lower + inset - centre, upper - inset - centre)
            else:
                p = on_plane
            if any(front_planes):
                # Both banks share this vertex until the crack separates the
                # local region. A soft QEF fit can push it through either bank.
                planes = normals[: len(points)][front_planes]
                offsets = np.einsum("ij,ij->i", planes, points[front_planes] - centre)
                front = constrained_box_lstsq(
                    normals, rhs, planes, offsets, lower + inset - centre, upper - inset - centre
                )
                if front is not None:
                    p = centre + front
            p = self._seat_on_own_cut(ci, region, p, lower, upper)
            local = (p - lower) / self.h
            regions.append((region, p, np.where(OFFSETS == 1, local, 1 - local).prod(1)))
        return self._pair_cell_regions(ci, regions)

    def _seat_on_own_cut(self, ci, region, point, lower, upper):
        """Sit a shallow wrong-side representative back on each cut it meets.

        Cells with three or more regions are not collapsed to one point. When
        the QEF has crossed a severed edge by at most a quarter millimetre,
        move it along that cut normal onto the plane, on the side of the
        region's own corner. A gap past the plane increases the old triple's
        crossings, so the point stops on the plane. Exclusive pairs replace
        these points afterwards.
        """
        corners = self.reference[self.cells[ci]]
        inset = self.h * QEF_CELL_INSET
        box_lo, box_hi = lower + inset, upper - inset
        for _ in range(2):
            for edge, (a, b) in zip(self.cell_edges[ci], LOCAL_EDGES):
                if edge not in self.hermite or not self.broken[edge]:
                    continue
                if (a in region) == (b in region):
                    continue
                inside = a if a in region else b
                plane, normal = self.hermite[edge]
                normal = np.asarray(normal, float)
                length = float(np.linalg.norm(normal))
                if length < 1.0e-12:
                    continue
                normal = normal / length
                side = float(np.dot(corners[inside] - plane, normal))
                signed = float(np.dot(point - plane, normal))
                if side * signed < 0.0 and abs(signed) <= 2.5e-4:
                    point = point - normal * signed
                    point = np.minimum(np.maximum(point, box_lo), box_hi)
        return point

    def _interface_partners(self, ci, regions):
        """Regions joined by a severed edge. A crack front inside one region is not a partner."""
        owner = {corner: i for i, (region, _, _) in enumerate(regions) for corner in region}
        partners = {i: set() for i in range(len(regions))}
        for edge, (a, b) in zip(self.cell_edges[ci], LOCAL_EDGES):
            if not (self.broken[edge] and a in owner and b in owner) or owner[a] == owner[b]:
                continue
            partners[owner[a]].add(owner[b])
            partners[owner[b]].add(owner[a])
        return partners

    def _shared_interface_point(self, ci, regions, left, right):
        """One reference point for two regions that meet across a single cut.

        Hermite samples that touch a third region stay out of this fit, so a
        second cut does not pull the shared point off the first plane.
        """
        selected = (regions[left][0], regions[right][0])
        owner = {}
        for index, region in enumerate(selected):
            for corner in region:
                owner[corner] = index
        cell_owner = {corner: i for i, (region, _, _) in enumerate(regions) for corner in region}
        samples, front, shared = [], [], False
        for edge, (a, b) in zip(self.cell_edges[ci], LOCAL_EDGES):
            if edge not in self.hermite:
                continue
            owners = {cell_owner[corner] for corner in (a, b) if corner in cell_owner}
            if not owners or not owners <= {left, right}:
                continue
            samples.append(edge)
            if self.broken[edge] and a in owner and b in owner:
                if owner[a] == owner[b]:
                    front.append(edge)
                else:
                    shared = True
        if not shared:
            return None
        points = np.array([self.hermite[e][0] for e in samples])
        normals = np.array([self.hermite[e][1] for e in samples])
        centre = points.mean(0)
        rhs = np.einsum("ij,ij->i", normals, points - centre)
        lower = self.reference[self.cells[ci]].min(0)
        inset = self.h * QEF_CELL_INSET
        lo, hi = lower + inset - centre, lower + self.h - inset - centre
        if front:
            planes = np.array([self.hermite[e][1] for e in front])
            offsets = np.einsum("ij,ij->i", planes, np.array([self.hermite[e][0] for e in front]) - centre)
            offset = constrained_box_lstsq(normals, rhs, planes, offsets, lo, hi)
            if offset is None:
                return None
        else:
            offset = np.linalg.lstsq(normals, rhs, rcond=None)[0]
            if np.any((offset < lo) | (offset > hi)):
                offset = box_lstsq(normals, rhs, lo, hi)
        point = centre + offset
        local = (point - lower) / self.h
        weights = np.where(OFFSETS == 1, local, 1 - local).prod(1)
        return point, weights

    def _pair_cell_regions(self, ci, regions):
        """Give the two banks of one cut the same reference representative.

        A cell with three or more regions is paired only when two of them meet
        each other and neither meets anyone else. Merging every region in the
        cell, or a region that sits on two cuts, folds the surface. Those
        regions keep the QEF points already solved for them.
        """
        if len(regions) < 2:
            return regions
        partners = self._interface_partners(ci, regions)
        exclusive = []
        for left, others in partners.items():
            if len(others) != 1:
                continue
            right = next(iter(others))
            if right > left and partners[right] == {left}:
                exclusive.append((left, right))
        if len(regions) == 2 and not exclusive:
            return regions
        paired = list(regions)
        for left, right in exclusive:
            shared = self._shared_interface_point(ci, regions, left, right)
            if shared is None:
                continue
            point, weights = shared
            paired[left] = (regions[left][0], point.copy(), weights.copy())
            paired[right] = (regions[right][0], point.copy(), weights.copy())
        return paired

    def mesh(self, use_cache=True):
        self.updated_cells = len(self.dirty)
        # This caches reference topology/QEF coordinates only, never deformed
        # world vertices. A new cut invalidates it through the dirty-cell set;
        # surface_runtime still embeds the returned reference mesh every frame.
        if use_cache and not self.dirty and self._mesh_cache is not None:
            return self._mesh_cache
        self._mesh_cache = None
        for ci in self.dirty:
            self._update_cell(ci)
        self.dirty.clear()
        valid = self.region_valid
        vertex_of = np.full(valid.shape, -1)
        vertex_of[valid] = np.arange(np.count_nonzero(valid))
        corner_vertex = np.where(
            self.corner_region >= 0, np.take_along_axis(vertex_of, np.maximum(self.corner_region, 0), 1), -1
        )
        links = np.flatnonzero(self.boundary | self.broken)
        cells = self.incident_cells[links]
        rings = np.full((len(links), 2, 4), -1)
        others = np.zeros((len(links), 2), bool)
        region_points = self.region_point[valid]
        owner = np.nonzero(valid)[0]
        for end in (0, 1):
            ring = np.where(
                cells >= 0, corner_vertex[np.maximum(cells, 0), self.incident_corner[links, end]], -1
            )
            ring[~self.inside[self.edges[links, end]]] = -1
            # Keep the first occurrence of each vertex, then pack the ring to the left.
            for c in range(1, 4):
                ring[(ring[:, c : c + 1] == ring[:, :c]).any(1), c] = -1
            ring = np.take_along_axis(ring, np.argsort(ring < 0, axis=1, kind="stable"), 1)
            count = np.count_nonzero(ring >= 0, axis=1)
            # Cells are counterclockwise about the link axis; orient by axis and link end.
            flip = np.where(self.axes[links] == 1, -1, 1) * (1 if end == 0 else -1) < 0
            for k in (3, 4):
                rows = flip & (count == k)
                ring[rows, :k] = ring[rows, :k][:, ::-1]
            # Split each quad along the diagonal whose two halves agree best in
            # normal; the other split folds non-convex quads onto themselves.
            # On near-planar ties, choose the diagonal through the lowest cell
            # id. Opposite banks reverse their rings and have different vertex
            # ids, but must triangulate coincident representatives identically.
            q = region_points[np.maximum(ring, 0)]

            def agreement(a, b, c, d):
                n1, n2 = np.cross(q[:, b] - q[:, a], q[:, c] - q[:, a]), np.cross(
                    q[:, c] - q[:, a], q[:, d] - q[:, a]
                )
                return np.einsum("ij,ij->i", n1, n2) / np.maximum(
                    np.linalg.norm(n1, axis=1) * np.linalg.norm(n2, axis=1), 1.0e-300
                )

            rings[:, end] = ring
            default, alternative = agreement(0, 1, 2, 3), agreement(1, 2, 3, 0)
            canonical = np.argmin(owner[np.maximum(ring, 0)], axis=1) % 2 == 1
            others[:, end] = (count == 4) & (
                (alternative > default + QUAD_SPLIT_MARGIN)
                | ((np.abs(alternative - default) <= QUAD_SPLIT_MARGIN) & self.broken[links] & canonical)
            )
        faces, self.split_changes = triangulate(rings, others)
        # A local volume region can have several disconnected boundary fans when
        # crack fronts meet. They must not share a surface vertex merely because
        # their material corners still belong to one cell region.
        candidates = np.flatnonzero(self.broken[self.cell_edges[owner]].any(axis=1))
        faces, source = split_vertex_fans(faces, len(region_points), candidates)
        self.vertex_splits = len(source) - len(region_points)
        self._mesh_cache = self._return_unsampled_faces(
            self._flip_crossed_diagonals(
                self._snap_remaining_twins(
                    self._open_junction_crossings(
                        self._snap_crossed_twins(
                            self._open_shallow_bank_crossings(
                            Mesh(
                                region_points[source],
                                faces,
                                self.region_code[valid][source],
                                self.cells[owner[source]].reshape(-1, 8),
                                self.region_weights[valid][source],
                            )
                        )
                    )
                )
            )
            )
        )
        return self._mesh_cache

    def _flip_crossed_diagonals(self, mesh):
        """Replace a quad diagonal when that removes a strict crossing.

        The two triangles keep the boundary edges and the opposite winding, so
        a manifold mesh stays manifold. A flip is kept only when neither
        crossing count rises, at least one falls, and no edge is used more than
        twice. Vertices stay where the cut placed them. Meshes that do not
        cross are returned unchanged.
        """
        triangles = flip_crossed_diagonals(mesh.reference, mesh.triangles)
        if np.array_equal(triangles, mesh.triangles):
            return mesh
        return Mesh(mesh.reference, triangles, mesh.codes, mesh.cells, mesh.weights)

    def _return_unsampled_faces(self, mesh):
        """Move a clamped representative back to the centroid of its own samples.

        The move is offered only when the point lies on inset faces that contain
        none of those samples, and the centroid lies in the cell. It is kept
        only when a strict crossing count falls and the other does not rise.
        """
        vertices = np.array(mesh.reference, float, copy=True)
        triangles = np.asarray(mesh.triangles, int)
        weights = np.array(mesh.weights, float, copy=True)
        if not len(triangles) or not len(vertices):
            return mesh
        cell_of = {tuple(int(node) for node in nodes): index for index, nodes in enumerate(self.cells)}
        owners = []
        for vertex in range(len(vertices)):
            cell = cell_of.get(tuple(int(node) for node in mesh.cells[vertex]))
            owners.append(None if cell is None else (cell, self._region_slot(cell, mesh.codes[vertex], vertices[vertex])))
        changed = False
        for _ in range(6):
            labels = _mesh_labels(len(vertices), triangles)
            before_self, before_inter = _strict_crossing_counts(vertices, triangles, labels)
            if before_self + before_inter == 0:
                break
            accepted = None
            for vertex, owner in enumerate(owners):
                if owner is None or owner[1] is None:
                    continue
                cell, slot = owner
                samples = self._region_samples(cell, slot)
                if len(samples) == 0:
                    continue
                lower = self.reference[self.cells[cell]].min(0)
                centroid = unsampled_face_centroid(
                    vertices[vertex], samples, lower, lower + self.h, self.h * QEF_CELL_INSET
                )
                if centroid is None:
                    continue
                trial = vertices.copy()
                trial[vertex] = centroid
                after_self, after_inter = _strict_crossing_counts(trial, triangles, labels)
                if after_self > before_self or after_inter > before_inter:
                    continue
                if after_self == before_self and after_inter == before_inter:
                    continue
                local = (centroid - lower) / self.h
                weights = np.array(weights, copy=True)
                weights[vertex] = np.where(OFFSETS == 1, local, 1.0 - local).prod(1)
                vertices = trial
                accepted = (int(after_self), int(after_inter))
                break
            if accepted is None:
                break
            changed = True
        if not changed:
            return mesh
        return Mesh(vertices, triangles, mesh.codes, mesh.cells, weights)

    def _region_slot(self, cell, code, point):
        valid = np.flatnonzero(self.region_valid[cell])
        if not len(valid):
            return None
        pool = [int(slot) for slot in valid if int(self.region_code[cell, slot]) == int(code)]
        if not pool:
            return None
        points = self.region_point[cell, pool]
        return pool[int(np.argmin(np.linalg.norm(points - point, axis=1)))]

    def _region_samples(self, cell, slot):
        region = {int(corner) for corner in np.flatnonzero(self.corner_region[cell] == slot)}
        samples = []
        for edge, (a, b) in zip(self.cell_edges[cell], LOCAL_EDGES):
            if edge not in self.hermite:
                continue
            touches = ((a in region) != (b in region)) or (self.broken[edge] and (a in region or b in region))
            if touches:
                samples.append(np.asarray(self.hermite[edge][0], float))
        return np.array(samples) if samples else np.zeros((0, 3))

    def _open_shallow_bank_crossings(self, mesh):
        """Move one shallow non-shared vertex off an opposite-bank face.

        Exclusive pairs stay coincident, so a shared representative is never
        moved. A point already on the wrong side by more than 0.15 mm is left
        alone. The 0.05 mm clearance is taken from the room left inside that
        move, so a point 0.11 mm on the wrong side still crosses the plane.
        The move is kept only when piece-piece crossings fall and same-piece
        crossings do not rise. Embedding weights are rebuilt from the cell so
        the reference point and its weights still agree.
        """
        vertices = np.array(mesh.reference, float, copy=True)
        triangles = np.asarray(mesh.triangles, int)
        codes = np.asarray(mesh.codes)
        weights = np.array(mesh.weights, float, copy=True)
        if not len(triangles) or not len(vertices):
            return mesh
        gap, limit = 5.0e-5, 1.5e-4
        changed = False
        for _ in range(6):
            labels = _mesh_labels(len(vertices), triangles)
            before_self, before_inter = _strict_crossing_counts(vertices, triangles, labels)
            if before_inter == 0:
                break
            accepted = self._accept_shallow_crossing(
                vertices, weights, triangles, codes, mesh.cells, labels, gap, limit, True
            )
            if accepted is None:
                break
            vertices, weights = accepted
            changed = True
        if not changed:
            return mesh
        return Mesh(vertices, triangles, codes, mesh.cells, weights)

    def _open_junction_crossings(self, mesh):
        """Pull a same-piece crossing off a four-region junction.

        Exclusive pairing does not apply when every region meets every other,
        so those points can sit through a neighboring triangle. A four-region
        junction needed a 0.53 mm move; the two crossings left on a three-region
        junction clear with a 0.16 mm move. Only vertices of cells with three
        or more regions move, and only when the strict count falls. A
        single-region crack front is left where it is.
        """
        vertices = np.array(mesh.reference, float, copy=True)
        triangles = np.asarray(mesh.triangles, int)
        codes = np.asarray(mesh.codes)
        weights = np.array(mesh.weights, float, copy=True)
        if not len(triangles) or not len(vertices):
            return mesh
        counts = {
            tuple(int(node) for node in nodes): int(valid)
            for nodes, valid in zip(self.cells, self.region_valid.sum(1))
        }
        allowed = np.fromiter(
            (counts.get(tuple(int(node) for node in nodes), 0) >= 3 for nodes in mesh.cells),
            dtype=bool,
            count=len(mesh.cells),
        )
        if not np.any(allowed):
            return mesh
        gap, limit = 5.0e-5, 8.0e-4
        changed = False
        for _ in range(6):
            labels = _mesh_labels(len(vertices), triangles)
            before_self, _before_inter = _strict_crossing_counts(vertices, triangles, labels)
            if before_self == 0:
                break
            accepted = self._accept_shallow_crossing(
                vertices, weights, triangles, codes, mesh.cells, labels, gap, limit, False, allowed
            )
            if accepted is None:
                break
            vertices, weights = accepted
            changed = True
        if not changed:
            return mesh
        return Mesh(vertices, triangles, codes, mesh.cells, weights)

    def _accept_shallow_crossing(
        self, vertices, weights, triangles, codes, cells, labels, gap, limit, want_inter, allowed=None
    ):
        from scripts.surface_deformed import intersection_pairs, segment_triangle_pairs

        before_self, before_inter = _strict_crossing_counts(vertices, triangles, labels)
        edges, faces = intersection_pairs(vertices, triangles)
        if not len(edges):
            return None
        corners = vertices[triangles[faces]]
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        length = np.linalg.norm(normal, axis=1)
        usable = length > 1.0e-12
        if not np.any(usable):
            return None
        normal = normal[usable] / length[usable, None]
        edges = edges[usable]
        interior = segment_triangle_pairs(
            vertices[edges[:, 0]], vertices[edges[:, 1]], corners[usable], strict=True
        )
        same = labels[edges[:, 0]] == labels[triangles[faces[usable], 0]]
        hits = np.flatnonzero(interior & (~same if want_inter else same))
        if not len(hits):
            return None
        origin = corners[usable, 0]
        signed = np.column_stack(
            [
                np.einsum("ij,ij->i", vertices[edges[:, end]] - origin, normal)
                for end in (0, 1)
            ]
        )
        candidates = []
        for hit in hits:
            nearer = int(np.argmin(np.abs(signed[hit])))
            end = int(edges[hit, nearer])
            if allowed is not None and not allowed[end]:
                continue
            dist = float(signed[hit, nearer])
            far = float(signed[hit, 1 - nearer])
            if abs(far) < 1.0e-8 or abs(dist) > limit:
                continue
            others = np.flatnonzero(codes != codes[end])
            if len(others) and np.linalg.norm(vertices[others] - vertices[end], axis=1).min() < 1.0e-5:
                continue
            # The clearance comes out of the unused part of `limit`. Charging
            # it on top of the penetration silently shrinks the stated limit
            # by `gap` (0.15 mm admits only 0.10 mm).
            room = limit - abs(dist)
            clearance = min(gap, max(room, 0.0))
            shift = float(np.sign(far) * clearance - dist)
            if not np.isfinite(shift) or abs(shift) > limit + 1.0e-12 or abs(shift) < 1.0e-8:
                continue
            candidates.append((abs(dist), end, shift, normal[hit]))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        for _, end, shift, direction in candidates:
            trial = vertices.copy()
            trial[end] = vertices[end] + direction * shift
            cell = self.reference[cells[end]]
            lower = cell.min(0)
            upper = lower + self.h
            # Trilinear weights are only defined inside the cell. A move that
            # leaves it would extrapolate the embedding.
            if np.any(trial[end] < lower - 1.0e-12) or np.any(trial[end] > upper + 1.0e-12):
                continue
            after_self, after_inter = _strict_crossing_counts(trial, triangles, labels)
            improved = after_inter < before_inter if want_inter else after_self < before_self
            if improved and after_self <= before_self and after_inter <= before_inter:
                local = (trial[end] - lower) / self.h
                weights = np.array(weights, copy=True)
                weights[end] = np.where(OFFSETS == 1, local, 1 - local).prod(1)
                return trial, weights
        return None

    def _cells_meet(self, cells, a, b, vertices):
        """Point in both cells closest to the midpoint, or None when the cells miss."""
        boxes = []
        for end in (int(a), int(b)):
            lower = self.reference[cells[end]].min(0)
            boxes.append((lower, lower + self.h))
        lo = np.maximum(boxes[0][0], boxes[1][0])
        hi = np.minimum(boxes[0][1], boxes[1][1])
        if np.any(lo > hi + 1.0e-12):
            return None
        mid = 0.5 * (vertices[int(a)] + vertices[int(b)])
        return np.minimum(np.maximum(mid, lo), hi)

    def _snap_remaining_twins(self, mesh):
        """Accept each leftover near-pair on its own, on the face both cells contain.

        The batch above refuses every pair when one of them does not reduce the
        count. After the other repairs, a pair that helps by itself is kept.
        The meeting point stays in both cells, so the embedding weights do not
        extrapolate.
        """
        from scripts.surface_deformed import intersection_pairs, segment_triangle_pairs

        vertices = np.array(mesh.reference, float, copy=True)
        triangles = np.asarray(mesh.triangles, int)
        weights = np.array(mesh.weights, float, copy=True)
        cells = mesh.cells
        if not len(triangles) or not len(vertices):
            return mesh
        labels = _mesh_labels(len(vertices), triangles)
        changed = False
        for limit in (2.5e-4, 8.0e-4):
            edges, faces = intersection_pairs(vertices, triangles)
            if not len(edges):
                break
            interior = segment_triangle_pairs(
                vertices[edges[:, 0]], vertices[edges[:, 1]], vertices[triangles[faces]], strict=True
            )
            if not np.any(interior):
                break
            same = labels[edges[:, 0]] == labels[triangles[faces, 0]]
            before_self = int(np.count_nonzero(interior & same))
            before_inter = int(np.count_nonzero(interior & ~same))
            used = set()
            pairs = []
            for edge, face, keep in zip(edges, faces, interior):
                if not keep:
                    continue
                tri = triangles[int(face)]
                for end in (int(edge[0]), int(edge[1])):
                    dist = np.linalg.norm(vertices[tri] - vertices[end], axis=1)
                    other = int(tri[int(np.argmin(dist))])
                    gap = float(dist.min())
                    if end == other or gap > limit or gap < 1.0e-8:
                        continue
                    pair = (end, other) if end < other else (other, end)
                    if pair[0] in used or pair[1] in used:
                        continue
                    used.update(pair)
                    pairs.append(pair)
            for a, b in pairs:
                meeting = self._cells_meet(cells, a, b, vertices)
                if meeting is None:
                    continue
                trial = vertices.copy()
                trial_weights = weights.copy()
                for end in (a, b):
                    trial[end] = meeting
                    lower = self.reference[cells[end]].min(0)
                    local = (meeting - lower) / self.h
                    trial_weights[end] = np.where(OFFSETS == 1, local, 1.0 - local).prod(1)
                after_self, after_inter = _strict_crossing_counts(trial, triangles, labels)
                if (
                    after_self > before_self
                    or after_inter > before_inter
                    or (after_self, after_inter) == (before_self, before_inter)
                ):
                    continue
                vertices, weights = trial, trial_weights
                before_self, before_inter = after_self, after_inter
                changed = True
        if not changed:
            return mesh
        return Mesh(vertices, triangles, mesh.codes, cells, weights)

    def _snap_crossed_twins(self, mesh):
        """Average two vertices of a crossing when they already nearly meet.

        A face can thread another while one of its vertices sits a fraction of
        a millimetre from a vertex of the face it crosses, on the opposite bank
        or on the same piece. Those two representatives belong on one point.
        Move both to their midpoint. Pairs within 0.25 mm are taken first, then
        pairs within 0.8 mm. A batch is kept only when same-piece and
        piece-piece crossings do not rise and at least one falls. Points that
        are already coincident are left alone.
        """
        from scripts.surface_deformed import intersection_pairs, segment_triangle_pairs

        vertices = np.array(mesh.reference, float, copy=True)
        triangles = np.asarray(mesh.triangles, int)
        weights = np.array(mesh.weights, float, copy=True)
        cells = mesh.cells
        if not len(triangles) or not len(vertices):
            return mesh
        labels = _mesh_labels(len(vertices), triangles)
        changed = False
        for limit in (2.5e-4, 8.0e-4):
            edges, faces = intersection_pairs(vertices, triangles)
            if not len(edges):
                break
            interior = segment_triangle_pairs(
                vertices[edges[:, 0]], vertices[edges[:, 1]], vertices[triangles[faces]], strict=True
            )
            if not np.any(interior):
                break
            same = labels[edges[:, 0]] == labels[triangles[faces, 0]]
            before_self = int(np.count_nonzero(interior & same))
            before_inter = int(np.count_nonzero(interior & ~same))
            used = set()
            pairs = []
            for edge, face, keep in zip(edges, faces, interior):
                if not keep:
                    continue
                tri = triangles[int(face)]
                for end in (int(edge[0]), int(edge[1])):
                    dist = np.linalg.norm(vertices[tri] - vertices[end], axis=1)
                    other = int(tri[int(np.argmin(dist))])
                    gap = float(dist.min())
                    if end == other or gap > limit or gap < 1.0e-8:
                        continue
                    pair = (end, other) if end < other else (other, end)
                    if pair[0] in used or pair[1] in used:
                        continue
                    used.update(pair)
                    pairs.append(pair)
            if not pairs:
                continue
            trial = vertices.copy()
            trial_weights = weights.copy()
            for a, b in pairs:
                mid = 0.5 * (vertices[a] + vertices[b])
                for end in (a, b):
                    trial[end] = mid
                    lower = self.reference[cells[end]].min(0)
                    local = (mid - lower) / self.h
                    trial_weights[end] = np.where(OFFSETS == 1, local, 1.0 - local).prod(1)
            after_self, after_inter = _strict_crossing_counts(trial, triangles, labels)
            if (
                after_self > before_self
                or after_inter > before_inter
                or (after_self, after_inter) == (before_self, before_inter)
            ):
                continue
            vertices, weights = trial, trial_weights
            changed = True
        if not changed:
            return mesh
        return Mesh(vertices, triangles, mesh.codes, cells, weights)


def _edge_uses(keys, uses, key_base, pair):
    """Faces using each vertex pair (last axis), 0 if none."""
    key = np.sort(pair, axis=-1)
    key = key[..., 0] * key_base + key[..., 1]
    if not len(keys):
        return np.zeros(key.shape, int)
    at = np.minimum(np.searchsorted(keys, key), len(keys) - 1)
    return np.where(keys[at] == key, uses[at], 0)


def split_vertex_fans(triangles, vertex_count, candidates=None):
    """Split disconnected closed vertex links; return faces and source vertex ids.

    Each duplicate inherits all geometry and embedding data from its source.
    No triangles are added or removed. Open or branched links are left intact;
    those require an edge-topology repair rather than splitting a point pinch.
    """
    source = list(range(vertex_count))
    if candidates is None:
        candidates = np.arange(vertex_count)
    if not len(candidates) or not len(triangles):
        return triangles, np.asarray(source, dtype=int)
    flat = triangles.ravel()
    order = np.argsort(flat, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(flat, minlength=vertex_count))]
    result = triangles
    for vertex in candidates:
        corners = order[offsets[vertex] : offsets[vertex + 1]]
        # Two disjoint simple cycles require at least six incident triangles.
        if len(corners) < 6:
            continue
        face, slot = corners // 3, corners % 3
        a = triangles[face, (slot + 1) % 3]
        b = triangles[face, (slot + 2) % 3]
        adjacency = {}
        for i, j in zip(a, b):
            adjacency.setdefault(int(i), []).append(int(j))
            adjacency.setdefault(int(j), []).append(int(i))
        if any(len(neighbors) != 2 for neighbors in adjacency.values()):
            continue
        remaining = set(adjacency)
        first = True
        for seed in a:
            if seed not in remaining:
                continue
            fan, pending = set(), [int(seed)]
            while pending:
                neighbor = pending.pop()
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    fan.add(neighbor)
                    pending.extend(adjacency[neighbor])
            if first:
                first = False
            else:
                if result is triangles:
                    result = triangles.copy()
                selected = np.fromiter((int(i) in fan for i in a), bool, count=len(a))
                result.ravel()[corners[selected]] = len(source)
                source.append(int(vertex))
            if not remaining:
                break
    return result, np.asarray(source, dtype=int)


def triangulate(rings, others):
    """Triangles of the dual polygons around each surface link.

    rings: (links, 2, 4) vertex ids per link end, -1 padded on the right; others:
    (links, 2) quads split along (1, 3) instead of (0, 2). Returns the (n, 3)
    triangles and the number of split changes made to keep the mesh manifold.

    A severed link whose two ends lie in the same region in every incident cell
    separates nothing: both ends give the same polygon with opposite winding, an
    internal double wall whose edges are shared with neighbouring walls. It gets no
    faces. A quad diagonal can coincide with an edge of a neighbouring polygon
    (several regions per cell); such a quad takes its other diagonal if that one is
    unused, else goes back to the default split, once, so the loop ends.
    """
    count = (rings >= 0).sum(2)
    double = (count[:, 0] >= 3) & np.all(np.sort(rings[:, 0]) == np.sort(rings[:, 1]), axis=1)
    quad = (count == 4) & ~double[:, None]
    triangle = (count >= 3) & ~double[:, None]
    others = others & quad
    key_base = int(rings.max()) + 1 if rings.size else 1
    changes = dict(dropped=int(double.sum()), reverted=0, switched=0)
    locked = ~quad
    while True:
        first = np.where(others[..., None], rings[..., [1, 2, 3]], rings[..., [0, 1, 2]])
        second = np.where(others[..., None], rings[..., [1, 3, 0]], rings[..., [0, 2, 3]])
        fans = np.stack(
            [np.where(triangle[..., None], first, -1), np.where(quad[..., None], second, -1)], axis=2
        )
        faces = fans[fans[..., 0] >= 0]
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        keys, uses = np.unique(edges[:, 0] * key_base + edges[:, 1], return_counts=True)
        alternative = _edge_uses(keys, uses, key_base, rings[..., [1, 3]])
        default = _edge_uses(keys, uses, key_base, rings[..., [0, 2]])
        # Reverts first: a reverted quad frees the edge that crowded its neighbour.
        revert = others & ~locked & (alternative != 2)
        if revert.any():
            others &= ~revert
            locked |= revert
            changes["reverted"] += int(revert.sum())
            continue
        switch = ~others & ~locked & (default > 2) & (alternative == 0)
        if not switch.any():
            return faces.reshape(-1, 3), changes
        others |= switch
        changes["switched"] += int(switch.sum())


@dataclass
class Supports:
    """Per-node (point ids, weights) in CSR form; ids ascending within a node."""

    offsets: np.ndarray
    indices: np.ndarray
    weights: np.ndarray

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, i):
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.indices[a:b], self.weights[a:b]

    def __iter__(self):
        return (self[i] for i in range(len(self)))


NEIGHBOUR_CELLS = np.array(list(product((-1, 0, 1), repeat=3)), int)


def anchor_cut_digits(support, node_codes, particle_codes, cut_index):
    """Side digit of each node's heaviest compatible particle.

    The value is -1 when the node has no compatible particle and 0 when that
    particle has not been given a side yet.

    Compatibility is the same test as the nodal field: a zero digit does not
    pick a side, and two nonzero digits disagree only when they differ. The
    heaviest particle is the one that owns the field, so a later geometric
    test must not write the opposite side.
    """
    count = len(node_codes)
    # -1: no compatible particle. 0: the heaviest particle has no side yet.
    digits = np.full(count, np.int64(-1))
    if len(support.indices) == 0:
        return digits
    node = np.repeat(np.arange(count), np.diff(support.offsets))
    particle = np.asarray(particle_codes, np.int64)[support.indices]
    left, right = np.asarray(node_codes, np.int64)[node].copy(), particle.copy()
    ok = np.ones(len(left), dtype=bool)
    while np.any(left) or np.any(right):
        x, y = left % 3, right % 3
        ok &= (x == 0) | (y == 0) | (x == y)
        left //= 3
        right //= 3
    weight = np.where(ok, support.weights, -1.0)
    best = np.full(count, -1.0)
    np.maximum.at(best, node, weight)
    seen = np.zeros(count, dtype=bool)
    power = np.int64(3) ** np.int64(cut_index)
    for i in np.flatnonzero(ok & (weight == best[node]) & (best[node] >= 0.0)):
        n = int(node[i])
        if seen[n]:
            continue
        seen[n] = True
        digits[n] = int(particle[i]) // power % np.int64(3)
    return digits


def tent_supports(nodes, points, radius, chunk=4096):
    """Per node, the points with nonzero tent weight prod(max(1 - |p - node| / radius, 0)).

    Points are bucketed in cells of edge `radius`, so a node's support lies in the
    27 cells around the node's cell.
    """
    nodes, points = np.asarray(nodes, float), np.asarray(points, float)
    low = points.min(0)
    dims = np.floor((points.max(0) - low) / radius).astype(np.int64) + 3
    key = np.ravel_multi_index((np.floor((points - low) / radius).astype(np.int64) + 1).T, dims)
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    counts, indices, weights = [], [], []
    for start in range(0, len(nodes), chunk):
        block = nodes[start : start + chunk]
        cell = np.floor((block - low) / radius).astype(np.int64) + 1
        around = cell[:, None, :] + NEIGHBOUR_CELLS[None]
        inside = np.all((around >= 0) & (around < dims), axis=2)
        keys = np.ravel_multi_index(np.clip(around, 0, dims - 1).transpose(2, 0, 1), dims)
        first = np.searchsorted(sorted_key, keys, "left").ravel()
        size = np.where(inside, np.searchsorted(sorted_key, keys, "right") - first.reshape(keys.shape), 0)
        size = size.ravel()
        offset = np.cumsum(size) - size
        position = np.repeat(first - offset, size) + np.arange(size.sum())
        node = np.repeat(np.arange(len(block)), size.reshape(len(block), -1).sum(1))
        ids = order[position]
        w = np.maximum(1 - np.abs(points[ids] - block[node]) / radius, 0).prod(axis=1)
        keep = w > 1.0e-12
        node, ids, w = node[keep], ids[keep], w[keep]
        sort = np.lexsort((ids, node))
        counts.append(np.bincount(node, minlength=len(block)))
        indices.append(ids[sort])
        weights.append(w[sort])
    counts = np.concatenate(counts) if counts else np.zeros(0, np.int64)
    return Supports(
        np.r_[0, np.cumsum(counts)],
        np.concatenate(indices) if indices else np.zeros(0, np.int64),
        np.concatenate(weights) if weights else np.zeros(0),
    )


class MaterialEmbedding:
    """Compatible nodal position fields; affine-exact and one-way.

    Reference support is fixed, not a nearest-current-particle mesh warp.
    Empty nodes use an explicitly declared compatible affine continuation.
    Missing support even in the next reference ring is an error, never a frozen face.
    """

    def __init__(self, reference_nodes, particle_reference, spacing, grid_origin=None, grid_cells=None):
        self.nodes = np.asarray(reference_nodes)
        self.ref = np.asarray(particle_reference)
        self.h = spacing
        self.material_grid = None
        if grid_origin is not None:
            self.grid_origin = np.asarray(grid_origin)
            self.grid_cells = np.asarray(grid_cells, int)
            nodes = (
                np.array(list(product(*(range(n + 1) for n in self.grid_cells)))) * spacing + self.grid_origin
            )
            self.material_grid = MaterialEmbedding(nodes, particle_reference, spacing)
        self.extension_nodes = 0

    @cached_property
    def support(self):
        return tent_supports(self.nodes, self.ref, self.h)

    @cached_property
    def extension_support(self):
        # A surface stencil can touch an empty material-grid node on one
        # cut side. Define its one-sided affine continuation from the next
        # reference support ring. This changes visualization only; it does
        # not populate an MPM state or transfer force across a cut.
        return tent_supports(self.nodes, self.ref, 2 * self.h)

    @staticmethod
    def one_side(code, ids, w, particle_codes):
        """Compatible support restricted to the side of its heaviest particle.

        A code still unassigned for a cut is compatible with both of its sides;
        averaging them stretches the surface across the gap once pieces separate.
        """
        valid = np.array([compatible(code, c) for c in particle_codes[ids]], dtype=bool)
        ids, w = ids[valid], w[valid]
        if len(ids):
            anchor = particle_codes[ids[np.argmax(w)]]
            keep = np.array([compatible(anchor, c) for c in particle_codes[ids]], dtype=bool)
            ids, w = ids[keep], w[keep]
        return ids, w

    def field(self, x, F, particle_codes, code):
        result = np.full_like(self.nodes, np.nan, dtype=float)
        self.extended = np.zeros(len(self.nodes), bool)
        for i, (ids, w) in enumerate(self.support):
            ids, w = self.one_side(code, ids, w, particle_codes)
            if not len(ids):
                ext, ew = self.extension_support[i]
                ids, w = self.one_side(code, ext, ew, particle_codes)
                if not len(ids):
                    continue
                self.extension_nodes += 1
                self.extended[i] = True
            extrapolated = x[ids] + np.einsum("nij,nj->ni", F[ids], self.nodes[i] - self.ref[ids])
            result[i] = (extrapolated * w[:, None]).sum(0) / w.sum()
        return result

    def node_state(self, x, F, particle_codes, node_codes):
        positions = np.full_like(self.nodes, np.nan, dtype=float)
        deformation = np.full((len(self.nodes), 3, 3), np.nan)
        for i, (ids, w) in enumerate(self.support):
            ids, w = self.one_side(node_codes[i], ids, w, particle_codes)
            if not len(ids):
                continue
            positions[i] = (
                (x[ids] + np.einsum("nij,nj->ni", F[ids], self.nodes[i] - self.ref[ids])) * w[:, None]
            ).sum(0) / w.sum()
            deformation[i] = (F[ids] * w[:, None, None]).sum(0) / w.sum()
        return positions, deformation

    def vertices(self, mesh, x, F, particle_codes):
        if self.material_grid is not None:
            # Surface primal links are padded/staggered for boundary extraction.
            # Embedding must instead use the ORIGINAL MPM grid and its fields.
            fractional = (mesh.reference - self.grid_origin) / self.h
            base = np.minimum(np.maximum(np.floor(fractional).astype(int), 0), self.grid_cells - 1)
            local = fractional - base
            ids = np.ravel_multi_index(
                (base[:, None, :] + OFFSETS[None, :, :]).transpose(2, 0, 1), tuple(self.grid_cells + 1)
            )
            weights = np.where(OFFSETS[None, :, :] == 1, local[:, None, :], 1 - local[:, None, :]).prod(2)
            embedded = Mesh(mesh.reference, mesh.triangles, mesh.codes, ids, weights)
            return self.material_grid.vertices(embedded, x, F, particle_codes)
        # extension_nodes here counts vertex corner lookups that needed the extension ring.
        result = np.empty_like(mesh.reference)
        extended_corners = self.extension_nodes
        for code in np.unique(mesh.codes):
            field = self.field(x, F, particle_codes, int(code))
            ids = np.flatnonzero(mesh.codes == code)
            extended_corners += int(self.extended[mesh.cells[ids]].sum())
            values = field[mesh.cells[ids]]
            weights = mesh.weights[ids]
            if not np.isfinite(values[weights > 1.0e-9]).all():
                raise RuntimeError(
                    "Surface region has no compatible material support; refine reference grid/support"
                )
            result[ids] = np.nansum(values * weights[..., None], axis=1)
        self.extension_nodes = extended_corners
        return result


# A coincident bank vertex that has crossed its twin still lies in the next
# surface cell's triangle. 0.1 mm clears that overlap on the 3 mm display grid
# without moving a bank that is already open.
BANK_OPENING_M = 1.0e-4


def _cut_digit(codes, cut):
    return (np.asarray(codes, np.int64) // np.int64(3) ** np.int64(cut)) % np.int64(3)


def _opening_direction(vertices, codes, cut, stored):
    """Direction from side 1 toward side 2. A flipped centroid keeps the blade normal."""
    stored = None if stored is None else np.asarray(stored, float)
    if stored is not None:
        stored = stored / np.linalg.norm(stored)
    digit = _cut_digit(codes, cut)
    side1, side2 = vertices[digit == 1], vertices[digit == 2]
    direction = None
    if len(side1) and len(side2):
        delta = side2.mean(0) - side1.mean(0)
        length = np.linalg.norm(delta)
        if length > 1.0e-6:
            direction = delta / length
    if stored is not None and (direction is None or float(np.dot(direction, stored)) > 0.0):
        return stored
    if direction is not None:
        return direction
    return np.array([1.0, 0.0, 0.0])


def _front_edge_folds(vertices, triangles, neighbors, incident, i1, i2):
    """An edge of one bank copy passes through a triangle of the other."""
    from scripts.surface_deformed import segment_triangle_pairs

    for source, other in ((i1, i2), (i2, i1)):
        faces = [face for face in incident[other] if source not in triangles[face]]
        ends = [n for n in neighbors[source] if n != other]
        if not faces or not ends:
            continue
        start = np.repeat(vertices[source][None], len(ends) * len(faces), 0)
        stop = np.repeat(vertices[np.asarray(ends)], len(faces), 0)
        corners = np.tile(vertices[triangles[faces]], (len(ends), 1, 1))
        if segment_triangle_pairs(start, stop, corners, strict=True).any():
            return True
    return False


def separate_coincident_banks(vertices, triangles, reference, codes, normals=None):
    """Stop the two copies of one crack point from occupying each other.

    Dual contouring stores both banks at one reference point. Embedding then
    moves each copy with its own piece. A crack-front pair still shares a mesh
    neighbor; if one copy's edge then passes through the other copy's triangle,
    the copies are put back on their midpoint so the front stays closed. Once
    the banks no longer share a neighbor, a copy that embedded behind the other
    is moved along the cut normal until the opening reaches BANK_OPENING_M.
    An already open pair keeps its tangential offset and its position, so a
    rigid motion of each piece is unchanged. Front pairs that do not fold are
    left open: closing every shared pair pinches a later cut.
    """
    vertices = np.array(vertices, float, copy=True)
    triangles = np.asarray(triangles, int)
    reference = np.asarray(reference, float)
    codes = np.asarray(codes, np.int64)
    if not len(vertices):
        return vertices
    neighbors = [set() for _ in range(len(vertices))]
    incident = [[] for _ in range(len(vertices))]
    for face, (a, b, c) in enumerate(triangles):
        a, b, c = int(a), int(b), int(c)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
        incident[a].append(face)
        incident[b].append(face)
        incident[c].append(face)
    key = np.round(reference, 9)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    digits = np.stack([_cut_digit(codes, cut) for cut in range(CODE_DIGITS)], axis=1)
    pairs = []
    for group in np.flatnonzero(counts > 1):
        ids = np.flatnonzero(inverse == group)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                ia, ib = int(ids[a]), int(ids[b])
                if codes[ia] == codes[ib]:
                    continue
                differing = np.flatnonzero((digits[ia] > 0) & (digits[ib] > 0) & (digits[ia] != digits[ib]))
                if len(differing) != 1:
                    continue
                cut = int(differing[0])
                side = {int(digits[ia, cut]): ia, int(digits[ib, cut]): ib}
                if 1 in side and 2 in side:
                    pairs.append((side[1], side[2], cut))
    for i1, i2, _cut in pairs:
        if neighbors[i1] & neighbors[i2] and _front_edge_folds(vertices, triangles, neighbors, incident, i1, i2):
            midpoint = (vertices[i1] + vertices[i2]) / 2
            vertices[i1] = midpoint
            vertices[i2] = midpoint
    directions = {}
    for i1, i2, cut in pairs:
        if neighbors[i1] & neighbors[i2]:
            continue
        if cut not in directions:
            stored = None if normals is None else normals.get(cut)
            directions[cut] = _opening_direction(vertices, codes, cut, stored)
        normal = directions[cut]
        opening = np.dot(vertices[i2] - vertices[i1], normal)
        if opening >= BANK_OPENING_M:
            continue
        correction = (BANK_OPENING_M - opening) / 2
        vertices[i1] = vertices[i1] - normal * correction
        vertices[i2] = vertices[i2] + normal * correction
    return vertices


def _mesh_labels(count, triangles):
    """Component id of every vertex. Equality is the only property callers use."""
    triangles = np.asarray(triangles, int)
    if count <= 0:
        return np.empty(0, int)
    if len(triangles) == 0:
        return np.arange(count)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    pairs = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    graph = coo_matrix(
        (np.ones(len(pairs), np.float64), (pairs[:, 0], pairs[:, 1])),
        shape=(int(count), int(count)),
    )
    _count, labels = connected_components(graph, directed=False, return_labels=True)
    return labels


def _other_diagonal(tri_a, tri_b):
    """The two triangles of the other diagonal, or None when the shared edge is not opposite."""
    a = [int(v) for v in tri_a]
    b = [int(v) for v in tri_b]
    where = {vertex: index for index, vertex in enumerate(b)}
    for i in range(3):
        u, v = a[i], a[(i + 1) % 3]
        if u not in where or v not in where:
            continue
        if b[(where[v] + 1) % 3] != u:
            continue
        p = a[(i + 2) % 3]
        q = b[(where[v] + 2) % 3]
        if len({u, v, p, q}) < 4:
            return None
        return [u, q, p], [q, v, p]
    return None


def _max_edge_incidence(triangles):
    pairs = np.sort(
        np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1
    )
    return int(np.unique(pairs, axis=0, return_counts=True)[1].max())


def _shared_edge(tri_a, tri_b):
    both = {int(v) for v in tri_a} & {int(v) for v in tri_b}
    if len(both) != 2:
        return None
    u, v = sorted(both)
    return u, v


def _mesh_edges(triangles):
    pairs = np.sort(
        np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1
    )
    return np.unique(pairs, axis=0)


def _counts_after_quad_flip(
    vertices, triangles, labels, hit_edges, hit_faces, hit_same, edges, face_a, face_b, new_a, new_b, before
):
    """Strict counts after one diagonal swap, using only the pairs that move.

    The two triangles of a quad already connect their four vertices along the
    boundary, so exchanging the chord does not change component labels. Pairs
    that used a removed face or the removed chord leave the count. Pairs that
    use a new face or the new chord join it. Every other pair stays.
    """
    from scripts.surface_deformed import segment_triangle_pairs

    old = _shared_edge(triangles[face_a], triangles[face_b])
    new = _shared_edge(new_a, new_b)
    if old is None or new is None:
        return None
    if len(hit_edges):
        ordered = np.sort(hit_edges, axis=1)
        drop = np.isin(hit_faces, (face_a, face_b))
        drop |= (ordered[:, 0] == old[0]) & (ordered[:, 1] == old[1])
        removed_same = int(np.count_nonzero(drop & hit_same))
        removed_inter = int(np.count_nonzero(drop & ~hit_same))
    else:
        removed_same = removed_inter = 0
    added_same = added_inter = 0
    others = np.arange(len(triangles))
    others = others[(others != face_a) & (others != face_b)]
    if len(others):
        share = np.isin(triangles[others], new).any(1)
        others = others[~share]
    if len(others):
        strict = segment_triangle_pairs(
            np.repeat(vertices[new[0]][None], len(others), 0),
            np.repeat(vertices[new[1]][None], len(others), 0),
            vertices[triangles[others]],
            strict=True,
        )
        same = labels[new[0]] == labels[triangles[others, 0]]
        added_same += int(np.count_nonzero(strict & same))
        added_inter += int(np.count_nonzero(strict & ~same))
    for new_tri in (new_a, new_b):
        if not len(edges):
            break
        share = np.isin(edges, new_tri).any(1)
        kept = edges[~share]
        if not len(kept):
            continue
        strict = segment_triangle_pairs(
            vertices[kept[:, 0]], vertices[kept[:, 1]], np.repeat(vertices[new_tri][None], len(kept), 0), strict=True
        )
        same = labels[kept[:, 0]] == labels[int(new_tri[0])]
        added_same += int(np.count_nonzero(strict & same))
        added_inter += int(np.count_nonzero(strict & ~same))
    return before[0] - removed_same + added_same, before[1] - removed_inter + added_inter


def flip_crossed_diagonals(vertices, triangles, guard=None):
    """Swap a quad diagonal that removes a strict crossing without raising either count.

    The diagonal may be an edge of the pierced triangle, or the segment that
    pierces it. Both are kept only when the other count does not rise and no
    edge is used more than twice.

    ``guard`` is a second point set on the same triangles, usually the reference
    positions while ``vertices`` are deformed. A flip that raises either of the
    guard's counts is rejected. Swaps continue until none of them lowers a count.
    """
    vertices = np.asarray(vertices, float)
    triangles = np.array(triangles, int, copy=True)
    if guard is not None:
        guard = np.asarray(guard, float)
        if len(guard) != len(vertices):
            raise ValueError("The guard points must correspond to the flipped vertices")
    if len(triangles) < 2:
        return triangles
    from scripts.surface_deformed import intersection_pairs, segment_triangle_pairs

    # Each accepted swap strictly lowers the crossing sum, so this ends.
    while True:
        labels = _mesh_labels(len(vertices), triangles)
        # The chord of a quad is not a bridge: the boundary already connects
        # its four vertices. Later swaps reuse these labels.
        before_self, before_inter = _strict_crossing_counts(vertices, triangles, labels)
        if before_self + before_inter == 0:
            break
        # An edge already used more than twice stays that way, so no swap passes.
        if _max_edge_incidence(triangles) > 2:
            break
        guard_labels = None if guard is None else labels
        guard_before = (
            None if guard is None else _strict_crossing_counts(guard, triangles, guard_labels)
        )
        edges, faces = intersection_pairs(vertices, triangles)
        if not len(edges):
            break
        interior = segment_triangle_pairs(
            vertices[edges[:, 0]], vertices[edges[:, 1]], vertices[triangles[faces]], strict=True
        )
        hits = np.flatnonzero(interior)
        if not len(hits):
            break
        strict_edges, strict_faces = edges[interior], faces[interior]
        strict_same = labels[strict_edges[:, 0]] == labels[triangles[strict_faces, 0]]
        if guard is None:
            guard_edges = guard_faces = guard_same = None
        else:
            g_edges, g_faces = intersection_pairs(guard, triangles)
            if len(g_edges):
                g_interior = segment_triangle_pairs(
                    guard[g_edges[:, 0]], guard[g_edges[:, 1]], guard[triangles[g_faces]], strict=True
                )
                guard_edges, guard_faces = g_edges[g_interior], g_faces[g_interior]
                guard_same = guard_labels[guard_edges[:, 0]] == guard_labels[triangles[guard_faces, 0]]
            else:
                guard_edges = np.empty((0, 2), int)
                guard_faces = np.empty(0, int)
                guard_same = np.empty(0, bool)
        mesh_edges = _mesh_edges(triangles)
        used = {}
        for face_id, tri in enumerate(triangles):
            for i in range(3):
                key = tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
                used.setdefault(key, []).append(face_id)

        def accept(face_a, face_b, new_a, new_b):
            chord = _shared_edge(new_a, new_b)
            # The removed chord had two owners. The new one starts from however
            # many faces already use it, then gains these two.
            if chord is None or len(used.get(chord, ())) + 2 > 2:
                return False
            after = _counts_after_quad_flip(
                vertices,
                triangles,
                labels,
                strict_edges,
                strict_faces,
                strict_same,
                mesh_edges,
                face_a,
                face_b,
                new_a,
                new_b,
                (before_self, before_inter),
            )
            if after is None or after[0] > before_self or after[1] > before_inter:
                return False
            if after[0] == before_self and after[1] == before_inter:
                return False
            if guard_before is not None:
                guarded = _counts_after_quad_flip(
                    guard,
                    triangles,
                    guard_labels,
                    guard_edges,
                    guard_faces,
                    guard_same,
                    mesh_edges,
                    face_a,
                    face_b,
                    new_a,
                    new_b,
                    guard_before,
                )
                if guarded is None or guarded[0] > guard_before[0] or guarded[1] > guard_before[1]:
                    return False
            return True

        changed = False
        for hit in hits:
            # The piercing segment can itself be the diagonal of another quad.
            # Replacing that diagonal removes the segment. The pierced triangle's
            # own edges are a different quad and can raise the edge incidence.
            piercing = tuple(sorted((int(edges[hit, 0]), int(edges[hit, 1]))))
            owners = used.get(piercing, ())
            if len(owners) == 2:
                flipped = _other_diagonal(triangles[owners[0]], triangles[owners[1]])
                if flipped is not None and accept(owners[0], owners[1], flipped[0], flipped[1]):
                    triangles = triangles.copy()
                    triangles[owners[0]] = flipped[0]
                    triangles[owners[1]] = flipped[1]
                    changed = True
                    break
            face = int(faces[hit])
            tri = triangles[face]
            for i in range(3):
                key = tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
                for other in used.get(key, ()):
                    if other == face:
                        continue
                    flipped = _other_diagonal(tri, triangles[other])
                    if flipped is None or not accept(face, other, flipped[0], flipped[1]):
                        continue
                    triangles = triangles.copy()
                    triangles[face] = flipped[0]
                    triangles[other] = flipped[1]
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break
        if not changed:
            break
    return triangles


def _strict_crossing_counts(vertices, triangles, labels):
    from scripts.surface_deformed import intersection_pairs, segment_triangle_pairs

    edges, faces = intersection_pairs(vertices, triangles)
    if not len(edges):
        return 0, 0
    interior = segment_triangle_pairs(
        vertices[edges[:, 0]], vertices[edges[:, 1]], vertices[triangles[faces]], strict=True
    )
    same = labels[edges[:, 0]] == labels[triangles[faces, 0]]
    return int(np.count_nonzero(interior & same)), int(np.count_nonzero(interior & ~same))


def snap_deformed_twins(vertices, triangles, reference=None):
    """Average deformed vertices that still cross and already nearly meet.

    This repairs the embedded surface only. Reference positions and weights
    stay put, so the next frame is embedded again from the cut. Pairs within
    0.25 mm in the deformed mesh are taken first, then 0.5 mm. A last pass
    joins a crossing to the triangle vertex that is already within 0.15 mm in
    the reference, even if deformation has sheared that pair apart. A batch is
    kept only when neither crossing count rises.
    """
    from scripts.surface_deformed import intersection_pairs, segment_triangle_pairs

    vertices = np.array(vertices, float, copy=True)
    triangles = np.asarray(triangles, int)
    if reference is not None:
        reference = np.asarray(reference, float)
    if not len(triangles) or not len(vertices):
        return vertices
    labels = _mesh_labels(len(vertices), triangles)

    def pull(limit, measure):
        nonlocal vertices
        edges, faces = intersection_pairs(vertices, triangles)
        if not len(edges):
            return False
        interior = segment_triangle_pairs(
            vertices[edges[:, 0]], vertices[edges[:, 1]], vertices[triangles[faces]], strict=True
        )
        if not np.any(interior):
            return False
        same = labels[edges[:, 0]] == labels[triangles[faces, 0]]
        before_self = int(np.count_nonzero(interior & same))
        before_inter = int(np.count_nonzero(interior & ~same))
        used, pairs = set(), []
        for edge, face, keep in zip(edges, faces, interior):
            if not keep:
                continue
            tri = triangles[int(face)]
            for end in (int(edge[0]), int(edge[1])):
                dist = np.linalg.norm(measure[tri] - measure[end], axis=1)
                other = int(tri[int(np.argmin(dist))])
                gap = float(dist.min())
                # Skip a pair that is already one point in the embedded mesh.
                # A reference gap of zero still counts: those two copies can be
                # sheared apart by the deformation, and the embedded points move.
                if end == other or gap > limit or np.linalg.norm(vertices[end] - vertices[other]) < 1.0e-12:
                    continue
                pair = (end, other) if end < other else (other, end)
                if pair[0] in used or pair[1] in used:
                    continue
                used.update(pair)
                pairs.append(pair)
        if not pairs:
            return True
        trial = vertices.copy()
        for a, b in pairs:
            mid = 0.5 * (vertices[a] + vertices[b])
            trial[a] = mid
            trial[b] = mid
        after_self, after_inter = _strict_crossing_counts(trial, triangles, labels)
        if after_self <= before_self and after_inter <= before_inter and (after_self, after_inter) != (
            before_self,
            before_inter,
        ):
            vertices = trial
        return True

    for limit in (2.5e-4, 5.0e-4):
        if not pull(limit, vertices):
            return vertices
    if reference is not None and len(reference) == len(vertices):
        pull(1.5e-4, reference)
    return vertices


def _component_shift(vertices, triangles, labels):
    """Slide a whole overlapping piece along the line between the two centroids.

    A lifted cap and the crack it was cut from occupy the same place. Pushing
    individual vertices along the crossed face folds the piece and adds
    same-piece crossings, so that push is rejected. Moving the smaller piece
    rigidly keeps its faces intact. The shortest shift that most reduces the
    other piece's crossings is kept, and only when same-piece crossings do not
    increase. The shift is at most 5 mm.
    """
    from scripts.surface_deformed import intersection_pairs, segment_triangle_pairs

    edges, faces = intersection_pairs(vertices, triangles)
    if not len(edges):
        return vertices
    interior = segment_triangle_pairs(
        vertices[edges[:, 0]], vertices[edges[:, 1]], vertices[triangles[faces]], strict=True
    )
    same = labels[edges[:, 0]] == labels[triangles[faces, 0]]
    hits = np.flatnonzero(interior & ~same)
    if not len(hits):
        return vertices
    pairs = {}
    for edge, face in zip(edges[hits], faces[hits]):
        left = int(labels[edge[0]])
        right = int(labels[triangles[face, 0]])
        if left == right:
            continue
        pairs[tuple(sorted((left, right)))] = pairs.get(tuple(sorted((left, right))), 0) + 1
    before_self = int(np.count_nonzero(interior & same))
    before_inter = int(len(hits))
    for left, right in sorted(pairs, key=pairs.get, reverse=True):
        count_left = int(np.count_nonzero(labels == left))
        count_right = int(np.count_nonzero(labels == right))
        small, large = (left, right) if count_left <= count_right else (right, left)
        axis = vertices[labels == small].mean(0) - vertices[labels == large].mean(0)
        length = float(np.linalg.norm(axis))
        if length < 1.0e-8:
            continue
        axis = axis / length
        best = None
        best_rank = None
        best_counts = None
        for distance in (0.005, 0.002, 0.001):
            trial = vertices.copy()
            trial[labels == small] = trial[labels == small] + axis * distance
            after_self, after_inter = _strict_crossing_counts(trial, triangles, labels)
            if after_self > before_self or after_inter >= before_inter:
                continue
            rank = (after_inter, distance)
            if best_rank is None or rank < best_rank:
                best, best_rank, best_counts = trial, rank, (after_self, after_inter)
        if best is None:
            continue
        vertices = best
        before_self, before_inter = best_counts
        if before_inter == 0:
            break
    return vertices


def _separate_bank_stabs(vertices, triangles, labels, codes, normals, gap):
    """Open a same-code edge that still passes through the opposite bank.

    While a crack front joins the two banks, they are one mesh component, so
    the component centroid is not a useful side. Pushing each endpoint along
    its own triangle normal then swings long edges into new crossings and the
    whole step is discarded. Both banks already have a cut normal: side 2
    moves toward that normal and side 1 moves away, and only far enough to
    clear BANK_OPENING_M. A stored normal that lies along the pierced triangle
    does not open the cut, so the direction from the side-1 centroid to the
    side-2 centroid is tried next. A step that does not strictly reduce
    same-piece crossings, or that adds a piece-piece crossing, is discarded.
    """
    from scripts.surface_deformed import intersection_pairs, segment_triangle_pairs

    codes = np.asarray(codes, np.int64)
    edges, faces = intersection_pairs(vertices, triangles)
    if not len(edges):
        return vertices
    interior = segment_triangle_pairs(
        vertices[edges[:, 0]], vertices[edges[:, 1]], vertices[triangles[faces]], strict=True
    )
    same = labels[edges[:, 0]] == labels[triangles[faces, 0]]
    keep = interior & same
    if not np.any(keep):
        return vertices
    edge = edges[keep]
    tri = triangles[faces[keep]]
    corners = vertices[tri]
    tri_normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    length = np.linalg.norm(tri_normal, axis=1)
    usable = length > 1.0e-12
    if not np.any(usable):
        return vertices
    edge, tri, corners = edge[usable], tri[usable], corners[usable]
    tri_normal = tri_normal[usable] / length[usable, None]
    max_code = int(codes.max()) if len(codes) else 0
    ncuts, power = 0, 1
    while power <= max_code and ncuts < CODE_DIGITS:
        power *= 3
        ncuts += 1
    # One endpoint can belong to the opposite bank of this triangle while the
    # other is still unassigned at the crack front. Move only the assigned one.
    endpoint_cut = np.full((len(edge), 2), -1, int)
    endpoint_side = np.zeros((len(edge), 2), int)
    for cut in range(ncuts):
        corner_digit = np.column_stack([_cut_digit(codes[tri[:, corner]], cut) for corner in range(3)])
        for end in range(2):
            digit = _cut_digit(codes[edge[:, end]], cut)
            differ = (digit > 0) & np.any((corner_digit > 0) & (corner_digit != digit[:, None]), axis=1)
            choose = (endpoint_cut[:, end] < 0) & differ
            endpoint_cut[choose, end] = cut
            endpoint_side[choose, end] = digit[choose]
    if not np.any(endpoint_cut >= 0):
        return vertices
    before_self = int(np.count_nonzero(interior & same))
    before_inter = int(np.count_nonzero(interior & ~same))
    cuts = [int(cut) for cut in np.unique(endpoint_cut[endpoint_cut >= 0])]

    def opening_along(use_stored):
        directions = {}
        for cut in cuts:
            stored = None
            if use_stored and normals is not None and cut in normals:
                stored = np.asarray(normals[cut], float)
                length = float(np.linalg.norm(stored))
                if length > 1.0e-12:
                    stored = stored / length
            if stored is None:
                directions[cut] = _opening_direction(vertices, codes, cut, None)
            else:
                directions[cut] = stored
        shift = np.zeros_like(vertices)
        vertex_side = np.zeros(len(vertices), int)
        for index in range(len(edge)):
            origin = corners[index, 0]
            for end in range(2):
                cut = int(endpoint_cut[index, end])
                if cut < 0:
                    continue
                move_dir = directions[cut]
                if int(endpoint_side[index, end]) != 2:
                    move_dir = -move_dir
                align = float(np.dot(move_dir, tri_normal[index]))
                if align <= 0.3:
                    continue
                vertex = int(edge[index, end])
                signed = float(np.dot(vertices[vertex] - origin, tri_normal[index]))
                if signed >= gap:
                    continue
                distance = min((gap - signed) / align, 0.002)
                if distance <= 0.0:
                    continue
                current = float(np.linalg.norm(shift[vertex]))
                if distance > current:
                    shift[vertex] = move_dir * distance
                    vertex_side[vertex] = int(endpoint_side[index, end])
        moved = np.linalg.norm(shift, axis=1) > 0.0
        if not np.any(moved):
            return None
        attempts = [moved]
        for side in (1, 2):
            chosen = moved & (vertex_side == side)
            if np.any(chosen):
                attempts.append(chosen)
        for selection in attempts:
            for scale in (1.0, 0.5):
                trial = vertices.copy()
                trial[selection] = vertices[selection] + shift[selection] * scale
                after_self, after_inter = _strict_crossing_counts(trial, triangles, labels)
                if after_self < before_self and after_inter <= before_inter:
                    return trial
        return None

    # The blade normal is the cut. When it is tangent to the pierced triangle,
    # the same opening measured from the two bank centroids still has a side.
    repaired = opening_along(True)
    if repaired is None and normals is not None:
        repaired = opening_along(False)
    return vertices if repaired is None else repaired


def separate_piece_crossings(vertices, triangles, rounds=2, gap=BANK_OPENING_M, codes=None, normals=None):
    """Pull a crossing edge end out along the triangle normal.

    Coincident crack copies are handled first. What remains is an edge whose
    reference ends are not the same point: a lifted cap through a stationary
    crack, or two pieces pushed into each other. The nearer end moves along
    the triangle normal, which is the penetration direction, onto the far
    side of that triangle. The triangle itself stays put. A round that adds
    same-piece crossings, or that fails to reduce the crossings it targets,
    is discarded. If pieces still occupy each other, the smaller piece is
    then slid rigidly along the line joining the two centroids. A same-code
    edge that still cuts the opposite bank is opened along the cut normal.
    """
    from scripts.surface_deformed import intersection_pairs, segment_triangle_pairs

    vertices = np.array(vertices, float, copy=True)
    triangles = np.asarray(triangles, int)
    if not len(triangles) or not len(vertices):
        return vertices
    labels = _mesh_labels(len(vertices), triangles)
    needs_nearer = False
    needs_shift = False
    needs_bank = False
    for inter_only in (True, False):
        for _ in range(rounds):
            edges, faces = intersection_pairs(vertices, triangles)
            if not len(edges):
                break
            interior = segment_triangle_pairs(
                vertices[edges[:, 0]], vertices[edges[:, 1]], vertices[triangles[faces]], strict=True
            )
            same = labels[edges[:, 0]] == labels[triangles[faces, 0]]
            mask = interior & (~same if inter_only else same)
            if not np.any(mask):
                if inter_only:
                    needs_shift = False
                else:
                    needs_bank = False
                break
            if inter_only:
                needs_shift = True
            else:
                needs_bank = True
            corners = vertices[triangles[faces[mask]]]
            edge = edges[mask]
            normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
            length = np.linalg.norm(normal, axis=1)
            usable = length > 1.0e-12
            if not np.any(usable):
                break
            normal = normal[usable] / length[usable, None]
            edge = edge[usable]
            origin = corners[usable, 0]
            start, stop = vertices[edge[:, 0]], vertices[edge[:, 1]]
            distance_start = np.einsum("ij,ij->i", start - origin, normal)
            distance_stop = np.einsum("ij,ij->i", stop - origin, normal)
            opposite = distance_start * distance_stop < 0.0
            if not np.any(opposite):
                break
            # A piece centroid sits with the rest of that piece. The endpoint
            # whose sign disagrees with it is the one that crossed the face.
            # Falling back to the nearer end when the centroid lies on the
            # plane keeps a single folded triangle movable.
            order = np.argsort(labels, kind="stable")
            ordered = vertices[order]
            unique, starts, centroid_counts = np.unique(labels[order], return_index=True, return_counts=True)
            sums = np.add.reduceat(ordered, starts, axis=0)
            centroid_table = np.zeros((int(labels.max()) + 1, 3))
            centroid_table[unique] = sums / centroid_counts[:, None]
            centroids = centroid_table[labels[edge[:, 0]]]
            centroid_distance = np.einsum("ij,ij->i", centroids - origin, normal)
            known = np.abs(centroid_distance) > 1.0e-6
            stray_is_stop = np.where(
                known,
                np.sign(distance_stop) != np.sign(centroid_distance),
                np.abs(distance_stop) < np.abs(distance_start),
            )
            stray = np.where(stray_is_stop, distance_stop, distance_start)
            sign = np.where(
                known,
                np.sign(centroid_distance),
                np.sign(np.where(stray_is_stop, distance_start, distance_stop)),
            )
            sign = np.where(sign == 0.0, -np.sign(stray), sign)
            delta = np.clip(sign * gap - stray, -0.005, 0.005)
            chosen = np.where(stray_is_stop, edge[:, 1], edge[:, 0])
            shift = np.zeros_like(vertices)
            weight = np.zeros(len(vertices))
            np.add.at(shift, chosen[opposite], normal[opposite] * delta[opposite, None])
            np.add.at(weight, chosen[opposite], 1.0)
            moved = weight > 0.0
            if not np.any(moved):
                break
            trial = vertices.copy()
            trial[moved] = vertices[moved] + shift[moved] / weight[moved, None]
            before_self = int(np.count_nonzero(interior & same))
            before_inter = int(np.count_nonzero(interior & ~same))
            after_self, after_inter = _strict_crossing_counts(trial, triangles, labels)
            if inter_only:
                if after_inter >= before_inter or after_self > before_self:
                    break
            elif after_self >= before_self or after_inter > before_inter:
                needs_nearer = True
                break
            vertices = trial
            needs_nearer = False

    def pull_nearer(vertices):
        for _ in range(rounds):
            edges, faces = intersection_pairs(vertices, triangles)
            if not len(edges):
                break
            interior = segment_triangle_pairs(
                vertices[edges[:, 0]], vertices[edges[:, 1]], vertices[triangles[faces]], strict=True
            )
            same = labels[edges[:, 0]] == labels[triangles[faces, 0]]
            mask = interior & same
            if not np.any(mask):
                break
            corners = vertices[triangles[faces[mask]]]
            edge = edges[mask]
            normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
            length = np.linalg.norm(normal, axis=1)
            usable = length > 1.0e-12
            if not np.any(usable):
                break
            normal = normal[usable] / length[usable, None]
            edge = edge[usable]
            origin = corners[usable, 0]
            distance_start = np.einsum("ij,ij->i", vertices[edge[:, 0]] - origin, normal)
            distance_stop = np.einsum("ij,ij->i", vertices[edge[:, 1]] - origin, normal)
            opposite = distance_start * distance_stop < 0.0
            if not np.any(opposite):
                break
            stray_is_stop = np.abs(distance_stop) <= np.abs(distance_start)
            stray = np.where(stray_is_stop, distance_stop, distance_start)
            far = np.where(stray_is_stop, distance_start, distance_stop)
            sign = np.sign(far)
            sign = np.where(sign == 0.0, 1.0, sign)
            delta = np.clip(sign * gap - stray, -0.005, 0.005)
            chosen = np.where(stray_is_stop, edge[:, 1], edge[:, 0])
            shift = np.zeros_like(vertices)
            weight = np.zeros(len(vertices))
            np.add.at(shift, chosen[opposite], normal[opposite] * delta[opposite, None])
            np.add.at(weight, chosen[opposite], 1.0)
            moved = weight > 0.0
            if not np.any(moved):
                break
            trial = vertices.copy()
            trial[moved] = vertices[moved] + shift[moved] / weight[moved, None]
            before_self = int(np.count_nonzero(interior & same))
            before_inter = int(np.count_nonzero(interior & ~same))
            after_self, after_inter = _strict_crossing_counts(trial, triangles, labels)
            if after_self >= before_self or after_inter > before_inter:
                break
            vertices = trial
        return vertices

    if needs_nearer:
        vertices = pull_nearer(vertices)
    if needs_shift:
        vertices = _component_shift(vertices, triangles, labels)
    if needs_bank and codes is not None:
        for _ in range(2):
            updated = _separate_bank_stabs(vertices, triangles, labels, codes, normals, gap)
            if updated is vertices:
                break
            vertices = updated
    # Bank opening can leave a same-piece crossing whose nearer end is only a
    # fraction of a millimetre through the face. Pull that end after the banks
    # move, and keep the step only when the crossing count falls.
    return pull_nearer(vertices)


def separate_crossing_sheets(vertices, triangles, rounds=4):
    """Separate triangles that still pass through each other.

    After the nearer-end pull, a same-piece fold can leave two triangles
    crossing. Moving either one alone folds a neighbour back through. Move
    every triangle that still takes part in a same-piece crossing away from
    the centre of that group. Try 0.05 mm first, then 0.1, 0.2, 0.5 and 1 mm,
    and keep the smallest step that lowers the same-piece count without
    adding a piece-piece crossing.
    """
    from scripts.surface_deformed import intersection_pairs, segment_triangle_pairs

    vertices = np.array(vertices, float, copy=True)
    triangles = np.asarray(triangles, int)
    if not len(triangles) or not len(vertices):
        return vertices
    labels = _mesh_labels(len(vertices), triangles)
    edge_faces = {}
    for face, tri in enumerate(triangles):
        corners = (int(tri[0]), int(tri[1]), int(tri[2]))
        for left, right in ((corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[0])):
            edge_faces.setdefault((left, right) if left < right else (right, left), []).append(face)
    for _ in range(rounds):
        edges, faces = intersection_pairs(vertices, triangles)
        if not len(edges):
            break
        interior = segment_triangle_pairs(
            vertices[edges[:, 0]], vertices[edges[:, 1]], vertices[triangles[faces]], strict=True
        )
        same = labels[edges[:, 0]] == labels[triangles[faces, 0]]
        before_self = int(np.count_nonzero(interior & same))
        before_inter = int(np.count_nonzero(interior & ~same))
        if before_self == 0:
            break
        involved = set()
        for hit in np.flatnonzero(interior & same):
            involved.add(int(faces[hit]))
            edge = edges[hit]
            key = (int(edge[0]), int(edge[1])) if edge[0] < edge[1] else (int(edge[1]), int(edge[0]))
            involved.update(edge_faces.get(key, ()))
        ids = sorted(involved)
        centroids = np.array([vertices[triangles[index]].mean(0) for index in ids])
        centre = centroids.mean(0)
        kept = None
        for scale in (5.0e-5, 1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3):
            trial = vertices.copy()
            for index, centroid in zip(ids, centroids):
                direction = centroid - centre
                length = float(np.linalg.norm(direction))
                if length < 1.0e-9:
                    continue
                trial[triangles[index]] = vertices[triangles[index]] + direction / length * scale
            after_self, after_inter = _strict_crossing_counts(trial, triangles, labels)
            if after_self < before_self and after_inter <= before_inter:
                kept = trial
                break
        if kept is None:
            break
        vertices = kept
    # A remaining pair can be one edge through one face. Separating those two
    # along the face normal by 0.1 mm clears it when moving the whole group
    # does not. Keep the step only when the same-piece count falls.
    for _ in range(rounds):
        edges, faces = intersection_pairs(vertices, triangles)
        if not len(edges):
            break
        interior = segment_triangle_pairs(
            vertices[edges[:, 0]], vertices[edges[:, 1]], vertices[triangles[faces]], strict=True
        )
        same = labels[edges[:, 0]] == labels[triangles[faces, 0]]
        before_self = int(np.count_nonzero(interior & same))
        before_inter = int(np.count_nonzero(interior & ~same))
        if before_self == 0:
            break
        kept = None
        for hit in np.flatnonzero(interior & same):
            face = triangles[int(faces[hit])]
            edge = edges[hit]
            span = np.cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]])
            length = float(np.linalg.norm(span))
            if length < 1.0e-12:
                continue
            normal = span / length
            shared = set(int(v) for v in edge) & set(int(v) for v in face)
            for scale in (1.0e-4, 2.0e-4):
                trial = vertices.copy()
                for vertex in face:
                    if int(vertex) not in shared:
                        trial[int(vertex)] = vertices[int(vertex)] + normal * scale
                for vertex in edge:
                    if int(vertex) not in shared:
                        trial[int(vertex)] = vertices[int(vertex)] - normal * scale
                after_self, after_inter = _strict_crossing_counts(trial, triangles, labels)
                if after_self < before_self and after_inter <= before_inter:
                    kept = trial
                    break
            if kept is not None:
                break
        if kept is None:
            break
        vertices = kept
    return vertices


def retract_vertices_ahead_of_particles(
    vertices, triangles, surface_codes, particles, particle_codes, gap=BANK_OPENING_M
):
    """Pull a vertex off a foreign triangle when its particle has not crossed.

    Affine embedding can leave the vertex on the far side of a triangle while
    the nearest compatible particle is still on the near side. The vertex then
    moves along that triangle normal up to the opening gap, and not past the
    particle. A particle that has also crossed is left where it is. The batch
    is kept only when the same-piece count falls and the piece-piece count
    does not rise.
    """
    from scripts.surface_deformed import intersection_pairs, segment_triangle_pairs

    vertices = np.asarray(vertices, float)
    triangles = np.asarray(triangles, int)
    surface_codes = np.asarray(surface_codes, np.int64)
    particles = np.asarray(particles, float)
    particle_codes = np.asarray(particle_codes, np.int64)
    if not len(triangles) or not len(vertices) or not len(particles):
        return vertices
    labels = _mesh_labels(len(vertices), triangles)
    while True:
        before_self, before_inter = _strict_crossing_counts(vertices, triangles, labels)
        if before_self + before_inter == 0:
            return vertices
        edges, faces = intersection_pairs(vertices, triangles)
        if not len(edges):
            return vertices
        interior = segment_triangle_pairs(
            vertices[edges[:, 0]], vertices[edges[:, 1]], vertices[triangles[faces]], strict=True
        )
        same = labels[edges[:, 0]] == labels[triangles[faces, 0]]
        hits = np.flatnonzero(interior & same)
        if not len(hits):
            return vertices
        shift = np.zeros_like(vertices)
        for edge, face in zip(edges[hits], faces[hits]):
            tri = triangles[int(face)]
            span = np.cross(vertices[tri[1]] - vertices[tri[0]], vertices[tri[2]] - vertices[tri[0]])
            length = float(np.linalg.norm(span))
            if length < 1.0e-12:
                continue
            normal = span / length
            origin = vertices[tri[0]]
            for end in edge:
                end = int(end)
                signed = float(np.dot(vertices[end] - origin, normal))
                if signed >= 0.0:
                    continue
                code = int(surface_codes[end])
                compatible_ids = np.flatnonzero(
                    [compatible(code, int(particle)) for particle in particle_codes]
                )
                if not len(compatible_ids):
                    continue
                nearest = int(
                    compatible_ids[np.argmin(np.linalg.norm(particles[compatible_ids] - vertices[end], axis=1))]
                )
                particle_signed = float(np.dot(particles[nearest] - origin, normal))
                if particle_signed <= 0.0:
                    continue
                destination = particle_signed if particle_signed < gap else gap
                if destination <= signed:
                    continue
                delta = normal * (destination - signed)
                if float(np.linalg.norm(delta)) > float(np.linalg.norm(shift[end])):
                    shift[end] = delta
        if not np.any(shift):
            return vertices
        trial = vertices.copy()
        moved = np.linalg.norm(shift, axis=1) > 0.0
        trial[moved] = vertices[moved] + shift[moved]
        after_self, after_inter = _strict_crossing_counts(trial, triangles, labels)
        if after_self >= before_self or after_inter > before_inter:
            return vertices
        vertices = trial


def reduce_crossed_pieces(vertices, triangles, codes, normals=None):
    """Repeat piece and sheet separation while the strict count falls.

    A diagonal swap changes which edges exist. The same separation can clear
    a same-piece fold on an edge that the earlier passes did not see. A pass
    is kept only when the strict count falls, so the loop stops when the
    next pass no longer does. A clean mesh is returned unchanged.
    """
    vertices = np.asarray(vertices, float)
    triangles = np.asarray(triangles, int)
    if not len(triangles) or not len(vertices):
        return vertices
    labels = _mesh_labels(len(vertices), triangles)
    while True:
        before = _strict_crossing_counts(vertices, triangles, labels)
        if before == (0, 0):
            return vertices
        updated = separate_crossing_sheets(
            separate_piece_crossings(vertices, triangles, codes=codes, normals=normals),
            triangles,
        )
        after = _strict_crossing_counts(updated, triangles, labels)
        if after[0] + after[1] >= before[0] + before[1]:
            return vertices
        vertices = updated


def settle_crossed_diagonals(vertices, triangles, reference, codes, normals=None):
    """Swap a crossed diagonal again after separation moves a vertex.

    One swap can leave a fold, and pulling that fold aside can make a different
    diagonal the one that crosses. A pass is kept only when the deformed strict
    count falls and the reference count does not rise. The next pass then sees
    the edges the move created. A clean mesh is returned unchanged.
    """
    vertices = np.asarray(vertices, float)
    triangles = np.asarray(triangles, int)
    reference = np.asarray(reference, float)
    if not len(triangles) or not len(vertices):
        return vertices, triangles
    # One extra view after separation. Each view already swaps every diagonal
    # that lowers the count, so a third view only repeats that scan.
    for _ in range(2):
        labels = _mesh_labels(len(vertices), triangles)
        before = _strict_crossing_counts(vertices, triangles, labels)
        if before == (0, 0):
            return vertices, triangles
        before_ref = _strict_crossing_counts(
            reference, triangles, _mesh_labels(len(reference), triangles)
        )
        updated_triangles = flip_crossed_diagonals(vertices, triangles, guard=reference)
        updated = reduce_crossed_pieces(vertices, updated_triangles, codes, normals)
        after = _strict_crossing_counts(
            updated, updated_triangles, _mesh_labels(len(updated), updated_triangles)
        )
        after_ref = _strict_crossing_counts(
            reference, updated_triangles, _mesh_labels(len(reference), updated_triangles)
        )
        if after[0] + after[1] >= before[0] + before[1]:
            return vertices, triangles
        if after_ref[0] > before_ref[0] or after_ref[1] > before_ref[1]:
            return vertices, triangles
        vertices = updated
        triangles = updated_triangles
    return vertices, triangles


def separate_locked_folds(vertices, triangles, codes, normals=None):
    """Separate a same-piece fold that the shallow pulls leave in place.

    Frame 646 of the near-capacity replay still had seven same-piece crossings
    after the nearer-end and sheet passes. Moving one piercing edge by 2 mm,
    then another by 0.1 mm, then sliding the remaining opposite banks by 2 mm
    along the cut cleared them. Each step is kept only when the strict count
    falls. A clean mesh is returned unchanged.
    """
    from scripts.surface_deformed import intersection_pairs, segment_triangle_pairs

    vertices = np.array(vertices, float, copy=True)
    triangles = np.asarray(triangles, int)
    codes = np.asarray(codes, np.int64)
    if not len(triangles) or not len(vertices):
        return vertices
    labels = _mesh_labels(len(vertices), triangles)
    scales = (1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3)

    def hits(points):
        edges, faces = intersection_pairs(points, triangles)
        if not len(edges):
            return edges, faces, np.zeros(0, bool)
        interior = segment_triangle_pairs(
            points[edges[:, 0]], points[edges[:, 1]], points[triangles[faces]], strict=True
        )
        same = labels[edges[:, 0]] == labels[triangles[faces, 0]]
        return edges, faces, interior & same

    for _ in range(8):
        before_self, before_inter = _strict_crossing_counts(vertices, triangles, labels)
        if before_self == 0:
            return vertices
        edges, faces, mask = hits(vertices)
        selected = np.flatnonzero(mask)
        best = None
        for hit in selected:
            edge = edges[hit]
            face = triangles[int(faces[hit])]
            span = np.cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]])
            length = float(np.linalg.norm(span))
            if length < 1.0e-12:
                continue
            normal = span / length
            face_set = {int(vertex) for vertex in face}
            for scale in scales:
                for mode in ("edge", "face", "both"):
                    trial = vertices.copy()
                    if mode in ("face", "both"):
                        trial[face] = vertices[face] + normal * scale
                    if mode in ("edge", "both"):
                        for vertex in edge:
                            if int(vertex) not in face_set:
                                trial[int(vertex)] = vertices[int(vertex)] - normal * scale
                    after_self, after_inter = _strict_crossing_counts(trial, triangles, labels)
                    if after_self < before_self and after_inter <= before_inter:
                        if best is None or after_self < best[0] or (after_self == best[0] and scale < best[1]):
                            best = (after_self, scale, trial)
        if best is None:
            break
        vertices = best[2]

    max_code = int(codes.max()) if len(codes) else 0
    ncuts, power = 0, 1
    while power <= max_code and ncuts < CODE_DIGITS:
        power *= 3
        ncuts += 1
    for _ in range(4):
        before_self, before_inter = _strict_crossing_counts(vertices, triangles, labels)
        if before_self == 0:
            return vertices
        edges, faces, mask = hits(vertices)
        if not np.any(mask):
            return vertices
        group = set()
        for hit in np.flatnonzero(mask):
            group.update(int(vertex) for vertex in edges[hit])
            group.update(int(vertex) for vertex in triangles[int(faces[hit])])
        group = np.array(sorted(group))
        best = None
        for cut in range(ncuts):
            side = _cut_digit(codes[group], cut)
            if not (np.any(side == 1) and np.any(side == 2)):
                continue
            direction = _opening_direction(vertices, codes, cut, None if normals is None else normals.get(int(cut)))
            for scale in scales:
                for sign in (1.0, -1.0):
                    trial = vertices.copy()
                    for vertex, which in zip(group, side):
                        if which == 2:
                            trial[vertex] = vertices[vertex] + sign * direction * scale
                        elif which == 1:
                            trial[vertex] = vertices[vertex] - sign * direction * scale
                    after_self, after_inter = _strict_crossing_counts(trial, triangles, labels)
                    if after_self < before_self and after_inter <= before_inter:
                        if best is None or after_self < best[0] or (after_self == best[0] and scale < best[1]):
                            best = (after_self, scale, trial)
        if best is None:
            break
        vertices = best[2]
    return vertices
