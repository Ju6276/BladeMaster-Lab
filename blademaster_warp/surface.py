"""Linked-volume / Hermite / local dual-contouring surface (paper III-F), CPU reference."""

from itertools import product
from dataclasses import dataclass
from functools import cached_property
import numpy as np
from .geometry import levelset
from .topology import compatible

OFFSETS = np.array(list(product((0, 1), repeat=3)), int)
QUAD_SPLIT_MARGIN = 1.0e-6
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
            uv = np.linalg.lstsq(matrix, w, rcond=None)[0]
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

    def cells_around(self, e):
        row = self.incident_cells[e]
        return row[row >= 0].tolist()

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
    ):
        """Sever links hit by the actual swept edge; returning to a cut cannot heal it.

        current_nodes must be the current compatible field embedding of the
        reference nodes. Identity is useful for geometry-only unit tests.
        band = (lateral half-width, depth behind the edge) also labels every node the
        edge passes, matching the material's side assignment.
        """
        if not 0 <= cut_index < CODE_DIGITS:
            raise ValueError(f"int64 side codes hold at most {CODE_DIGITS} cuts")
        x = self.reference if current_nodes is None else np.asarray(current_nodes)
        quad = np.asarray([edge_before[0], edge_before[1], edge_after[1], edge_after[0]])
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
                    nref = f.T @ n
                    nref /= np.linalg.norm(nref)
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
                        self.codes[node] += side * 3**cut_index
        changed = set(self.edges[hits].ravel()) if hits else set()
        if band is not None and normal_world is not None:
            changed |= self._assign_band(x, edge_before, edge_after, normal_world, cut_index, band)
            changed |= self._sever_opposite(x, edge_after, normal_world, cut_index, node_deformation)
        # A node's new code changes every incident region using that node, not
        # just cells adjacent to the severed edge.
        for node in changed:
            self.dirty.update(self.node_cells[node])
        return len(hits)

    def _assign_band(self, x, edge_before, edge_after, normal, cut_index, band):
        """Side digits for nodes the blade edge has just passed, with the material's band rule.

        Without this, skin nodes near a cut keep digit 0, stay compatible with both
        pieces and are embedded in their average motion once the pieces move apart.
        """
        half_width, depth = band
        edge = np.asarray(edge_after, float)
        along = edge[1] - edge[0]
        half_length = np.linalg.norm(along) / 2
        along /= 2 * half_length
        n = np.asarray(normal, float) / np.linalg.norm(normal)
        up = np.cross(n, along)
        travel = np.linalg.norm(np.asarray(edge_after) - np.asarray(edge_before), axis=1).max()
        rel = x - edge.mean(0)
        lateral, length, height = rel @ n, rel @ along, rel @ up
        unassigned = self.codes // 3**cut_index % 3 == 0
        nodes = np.flatnonzero(
            self.inside
            & unassigned
            & (np.abs(lateral) <= half_width)
            & (np.abs(length) <= half_length)
            & (height >= 0)
            & (height <= depth + travel)
        )
        self.codes[nodes] += np.where(lateral[nodes] < 0, 1, 2) * 3**cut_index
        return set(nodes.tolist())

    def _sever_opposite(self, x, edge, normal, cut_index, node_deformation):
        """Sever intact links whose ends lie on opposite sides of this cut.

        Opposite sides never exchange momentum in the material, so such a link must not
        join one surface region. Without this, a link the swept-edge test just misses
        gives a region both side digits, and its vertex follows both pieces.
        """
        digit = self.codes // 3**cut_index % 3
        a, b = self.edges[:, 0], self.edges[:, 1]
        crossing = np.flatnonzero(~self.broken & self.inside[a] & self.inside[b] & (digit[a] * digit[b] == 2))
        if not len(crossing):
            return set()
        n = np.asarray(normal, float) / np.linalg.norm(normal)
        point = np.asarray(edge, float).mean(0)
        sa, sb = (x[a[crossing]] - point) @ n, (x[b[crossing]] - point) @ n
        alphas = np.clip(sa / np.where(sa != sb, sa - sb, 1.0), 0.0, 1.0)
        for e, alpha in zip(crossing, alphas):
            i, j = self.edges[e]
            nref = n.copy()
            if node_deformation is not None:
                f = (1 - alpha) * node_deformation[i] + alpha * node_deformation[j]
                if np.isfinite(f).all():
                    nref = f.T @ n
                    nref /= np.linalg.norm(nref)
            self.hermite[int(e)] = ((1 - alpha) * self.reference[i] + alpha * self.reference[j], nref)
            self.broken[e] = True
            self.dirty.update(self.cells_around(e))
        return set(self.edges[crossing].ravel().tolist())

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
            if not points:
                continue
            points = np.array(points)
            normals = np.array(normals)
            centre = points.mean(0)
            rhs = np.einsum("ij,ij->i", normals, points - centre)
            # Solve offset relative to mass point to avoid drifting in the QEF
            # nullspace; bounded solve is only needed if unconstrained is outside.
            offset = np.linalg.lstsq(normals, rhs, rcond=None)[0]
            lower = self.reference[corners].min(0)
            upper = lower + self.h
            p = centre + offset
            if ((p < lower - 1.0e-10) | (p > upper + 1.0e-10)).any():
                p = centre + box_lstsq(normals, rhs, lower - centre - 1.0e-12, upper - centre + 1.0e-12)
            local = (p - lower) / self.h
            regions.append((region, p, np.where(OFFSETS == 1, local, 1 - local).prod(1)))
        return regions

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
            # The margin keeps near-planar quads (ties up to rounding) on the first split.
            q = region_points[np.maximum(ring, 0)]

            def agreement(a, b, c, d):
                n1, n2 = np.cross(q[:, b] - q[:, a], q[:, c] - q[:, a]), np.cross(q[:, c] - q[:, a], q[:, d] - q[:, a])
                return np.einsum("ij,ij->i", n1, n2) / np.maximum(
                    np.linalg.norm(n1, axis=1) * np.linalg.norm(n2, axis=1), 1.0e-300
                )

            rings[:, end] = ring
            others[:, end] = (count == 4) & (agreement(1, 2, 3, 0) > agreement(0, 1, 2, 3) + QUAD_SPLIT_MARGIN)
        quad = (rings >= 0).sum(2) == 4
        triangle = (rings >= 0).sum(2) >= 3
        key_base = int(rings.max()) + 1 if rings.size else 1
        while True:
            first = np.where(others[..., None], rings[..., [1, 2, 3]], rings[..., [0, 1, 2]])
            second = np.where(others[..., None], rings[..., [1, 3, 0]], rings[..., [0, 2, 3]])
            fans = np.stack(
                [np.where(triangle[..., None], first, -1), np.where(quad[..., None], second, -1)], axis=2
            )
            faces = fans[fans[..., 0] >= 0]
            # The alternative diagonal can coincide with an edge of a neighbouring
            # polygon (several regions per cell); keep the default split there.
            edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
            keys, uses = np.unique(edges[:, 0] * key_base + edges[:, 1], return_counts=True)
            diagonal = np.sort(rings[..., [1, 3]], axis=2)
            at = np.searchsorted(keys, diagonal[..., 0] * key_base + diagonal[..., 1])
            bad = others & (uses[np.minimum(at, len(uses) - 1)] != 2)
            if not bad.any():
                break
            others &= ~bad
        owner = np.nonzero(valid)[0]
        self._mesh_cache = Mesh(
            region_points,
            faces.reshape(-1, 3),
            self.region_code[valid],
            self.cells[owner].reshape(-1, 8),
            self.region_weights[valid],
        )
        return self._mesh_cache


def tent_supports(tree, nodes, points, radius):
    """Per node, the points with nonzero tent weight prod(max(1 - |p - node| / radius, 0))."""
    result = []
    for node, ids in zip(nodes, tree.query_ball_point(nodes, radius * np.sqrt(3) + 1.0e-10)):
        ids = np.array(ids, int)
        w = np.maximum(1 - np.abs(points[ids] - node) / radius, 0).prod(axis=1)
        result.append((ids[w > 1.0e-12], w[w > 1.0e-12]))
    return result


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
    def _tree(self):
        from scipy.spatial import cKDTree

        return cKDTree(self.ref)

    # With a material grid, vertices() embeds through the grid's supports;
    # surface-node supports are only built if field()/node_state() are called.
    @cached_property
    def support(self):
        return tent_supports(self._tree, self.nodes, self.ref, self.h)

    @cached_property
    def extension_support(self):
        # A surface stencil can touch an empty material-grid node on one
        # cut side. Define its one-sided affine continuation from the next
        # reference support ring. This changes visualization only; it does
        # not populate an MPM state or transfer force across a cut.
        return tent_supports(self._tree, self.nodes, self.ref, 2 * self.h)

    def field(self, x, F, particle_codes, code):
        result = np.full_like(self.nodes, np.nan, dtype=float)
        self.extended = np.zeros(len(self.nodes), bool)
        for i, (ids, w) in enumerate(self.support):
            valid = np.array([compatible(code, c) for c in particle_codes[ids]], dtype=bool)
            ids, w = ids[valid], w[valid]
            if not len(ids):
                ext, ew = self.extension_support[i]
                valid = np.array([compatible(code, c) for c in particle_codes[ext]], dtype=bool)
                ids, w = ext[valid], ew[valid]
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
            valid = np.array([compatible(node_codes[i], c) for c in particle_codes[ids]], dtype=bool)
            ids, w = ids[valid], w[valid]
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
