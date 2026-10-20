"""Embedded surface under known deformations: vertex error and self-intersections.

Particles are moved by a smooth map phi with its exact gradient as F; each
surface vertex is compared with phi of its reference position (the embedding is
exact for affine maps, so the error measures the nonlinear part at the grid
scale). For cut pieces moved apart rigidly, every vertex must follow the motion
of its own piece. Contact checks include coplanar overlap and endpoints between
mesh edges and triangles sharing no vertex. Reports distinguish contacts within
a component from contacts between pieces, and check manifoldness and volume.

    python -m scripts.surface_deformed
"""

import argparse
import json
from pathlib import Path
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation
from blademaster_warp.geometry import BANANA_DEMO_SIZE_M, grid_cells, points
from blademaster_warp.surface import LinkedSurface, MaterialEmbedding


def bend(radius):
    """Bend about the y axis: X -> ((R + z) sin(x/R), y, (R + z) cos(x/R) - R)."""

    def phi(X):
        a, r = X[:, 0] / radius, radius + X[:, 2]
        x = np.stack([r * np.sin(a), X[:, 1], r * np.cos(a) - radius], 1)
        F = np.zeros((len(X), 3, 3))
        F[:, 0, 0], F[:, 0, 2] = r / radius * np.cos(a), np.sin(a)
        F[:, 1, 1] = 1
        F[:, 2, 0], F[:, 2, 2] = -r / radius * np.sin(a), np.cos(a)
        return x, F

    return phi


def twist(rate, axis_point=(0.0, 0.0, 0.012)):
    """Rotate each x-section about the x axis through axis_point by rate * x."""
    c0 = np.asarray(axis_point)

    def phi(X):
        a = rate * X[:, 0]
        y, z = X[:, 1] - c0[1], X[:, 2] - c0[2]
        cos, sin = np.cos(a), np.sin(a)
        x = np.stack([X[:, 0], c0[1] + cos * y - sin * z, c0[2] + sin * y + cos * z], 1)
        F = np.zeros((len(X), 3, 3))
        F[:, 0, 0] = 1
        F[:, 1, 0], F[:, 1, 1], F[:, 1, 2] = rate * (-sin * y - cos * z), cos, -sin
        F[:, 2, 0], F[:, 2, 1], F[:, 2, 2] = rate * (cos * y - sin * z), sin, cos
        return x, F

    return phi


# A box whose longest side equals the median face, once expanded by the
# contact tolerance, crosses at most one extra grid plane on each side.
_ORDINARY_BOX_SPAN = 2


@njit(cache=True)
def _hash_box_pairs(edge_lo, edge_hi, tri_lo, tri_hi, edges, triangles, cell, origin):
    """Non-adjacent edge/triangle ids whose expanded boxes overlap.

    Each ordinary box is stored in the hash buckets of the cells it touches.
    An overlapping pair shares one of those cells, so the walk cannot miss it;
    a bucket collision only adds candidates. Boxes that cross more than the
    ordinary span are compared directly, with the same overlap test.
    """
    ntri = len(triangles)
    nedge = len(edges)
    capacity = ntri * 27
    if capacity < 1:
        capacity = 1
    table = 1
    while table < capacity * 2:
        table *= 2
    mask = table - 1
    head = np.full(table, -1, np.int64)
    nxt = np.empty(capacity, np.int64)
    which = np.empty(capacity, np.int64)
    long_tris = np.empty(ntri, np.int64)
    long_edges = np.empty(nedge, np.int64)
    stored = 0
    n_long_tri = 0
    n_long_edge = 0
    inv = 1.0 / cell
    ox, oy, oz = origin[0], origin[1], origin[2]

    def overlaps(e, t):
        return not (
            edge_hi[e, 0] < tri_lo[t, 0]
            or edge_lo[e, 0] > tri_hi[t, 0]
            or edge_hi[e, 1] < tri_lo[t, 1]
            or edge_lo[e, 1] > tri_hi[t, 1]
            or edge_hi[e, 2] < tri_lo[t, 2]
            or edge_lo[e, 2] > tri_hi[t, 2]
        )

    def shares_vertex(e, t):
        i = edges[e, 0]
        j = edges[e, 1]
        tri = triangles[t]
        return i == tri[0] or i == tri[1] or i == tri[2] or j == tri[0] or j == tri[1] or j == tri[2]

    for t in range(ntri):
        ix0 = int(np.floor((tri_lo[t, 0] - ox) * inv))
        iy0 = int(np.floor((tri_lo[t, 1] - oy) * inv))
        iz0 = int(np.floor((tri_lo[t, 2] - oz) * inv))
        ix1 = int(np.floor((tri_hi[t, 0] - ox) * inv))
        iy1 = int(np.floor((tri_hi[t, 1] - oy) * inv))
        iz1 = int(np.floor((tri_hi[t, 2] - oz) * inv))
        if ix1 - ix0 > _ORDINARY_BOX_SPAN or iy1 - iy0 > _ORDINARY_BOX_SPAN or iz1 - iz0 > _ORDINARY_BOX_SPAN:
            long_tris[n_long_tri] = t
            n_long_tri += 1
            continue
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                for iz in range(iz0, iz1 + 1):
                    bucket = (ix * 73856093 ^ iy * 19349663 ^ iz * 83492791) & mask
                    nxt[stored] = head[bucket]
                    which[stored] = t
                    head[bucket] = stored
                    stored += 1
    for e in range(nedge):
        ix0 = int(np.floor((edge_lo[e, 0] - ox) * inv))
        iy0 = int(np.floor((edge_lo[e, 1] - oy) * inv))
        iz0 = int(np.floor((edge_lo[e, 2] - oz) * inv))
        ix1 = int(np.floor((edge_hi[e, 0] - ox) * inv))
        iy1 = int(np.floor((edge_hi[e, 1] - oy) * inv))
        iz1 = int(np.floor((edge_hi[e, 2] - oz) * inv))
        if ix1 - ix0 > _ORDINARY_BOX_SPAN or iy1 - iy0 > _ORDINARY_BOX_SPAN or iz1 - iz0 > _ORDINARY_BOX_SPAN:
            long_edges[n_long_edge] = e
            n_long_edge += 1
    pe = np.empty(max(nedge, 1) * 4, np.int64)
    pt = np.empty(max(nedge, 1) * 4, np.int64)
    found = 0

    def append(e, t):
        nonlocal pe, pt
        if found == len(pe):
            grown_e = np.empty(len(pe) * 2, np.int64)
            grown_t = np.empty(len(pt) * 2, np.int64)
            grown_e[:found] = pe
            grown_t[:found] = pt
            pe, pt = grown_e, grown_t
        pe[found] = e
        pt[found] = t

    for e in range(nedge):
        ix0 = int(np.floor((edge_lo[e, 0] - ox) * inv))
        iy0 = int(np.floor((edge_lo[e, 1] - oy) * inv))
        iz0 = int(np.floor((edge_lo[e, 2] - oz) * inv))
        ix1 = int(np.floor((edge_hi[e, 0] - ox) * inv))
        iy1 = int(np.floor((edge_hi[e, 1] - oy) * inv))
        iz1 = int(np.floor((edge_hi[e, 2] - oz) * inv))
        if ix1 - ix0 > _ORDINARY_BOX_SPAN or iy1 - iy0 > _ORDINARY_BOX_SPAN or iz1 - iz0 > _ORDINARY_BOX_SPAN:
            continue
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                for iz in range(iz0, iz1 + 1):
                    link = head[(ix * 73856093 ^ iy * 19349663 ^ iz * 83492791) & mask]
                    while link != -1:
                        t = which[link]
                        link = nxt[link]
                        if shares_vertex(e, t) or not overlaps(e, t):
                            continue
                        append(e, t)
                        found += 1
    for k in range(n_long_tri):
        t = long_tris[k]
        for e in range(nedge):
            if shares_vertex(e, t) or not overlaps(e, t):
                continue
            append(e, t)
            found += 1
    for k in range(n_long_edge):
        e = long_edges[k]
        for t in range(ntri):
            if shares_vertex(e, t) or not overlaps(e, t):
                continue
            append(e, t)
            found += 1
    return pe[:found], pt[:found]


def _overlapping_box_pairs(edges, triangles, a, b, tri_lo, tri_hi, tolerance):
    """Every non-adjacent edge/triangle pair whose boxes meet within ``tolerance``."""
    edge_lo = np.ascontiguousarray(np.minimum(a, b) - tolerance)
    edge_hi = np.ascontiguousarray(np.maximum(a, b) + tolerance)
    tri_lo = np.ascontiguousarray(np.asarray(tri_lo, dtype=float) - tolerance)
    tri_hi = np.ascontiguousarray(np.asarray(tri_hi, dtype=float) + tolerance)
    extent = np.concatenate([(edge_hi - edge_lo).max(1), (tri_hi - tri_lo).max(1)])
    positive = extent[extent > 0]
    cell = float(np.median(positive)) if len(positive) else 1.0
    origin = np.ascontiguousarray(np.minimum(edge_lo.min(0), tri_lo.min(0)), dtype=np.float64)
    return _hash_box_pairs(
        edge_lo,
        edge_hi,
        tri_lo,
        tri_hi,
        np.ascontiguousarray(edges, dtype=np.int64),
        np.ascontiguousarray(triangles, dtype=np.int64),
        cell,
        origin,
    )


def intersection_pairs(vertices, triangles, chunk=2048):
    """Return contacting (edge vertex ids, triangle ids), excluding shared vertices.

    Coplanar overlap and endpoint contact count. Distinct cut pieces may touch
    legitimately before separation; geometry_report counts those separately.
    This edge/face check does not test overlap of faces sharing a mesh vertex.
    ``chunk`` is accepted so existing callers keep working; it does not change
    which pairs are returned.
    """
    vertices, triangles = np.asarray(vertices, float), np.asarray(triangles, int)
    if not len(triangles):
        return np.empty((0, 2), int), np.empty(0, int)
    if not np.isfinite(vertices).all():
        raise ValueError("Intersection checks require finite vertices")
    edges = np.unique(
        np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1),
        axis=0,
    )
    corners = vertices[triangles]
    reach = float(np.linalg.norm(corners - corners.mean(1)[:, None], axis=2).max())
    a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
    half = np.linalg.norm(b - a, axis=1) / 2
    lo, hi = corners.min(1), corners.max(1)
    tolerance = max(reach, float(half.max()) if len(half) else 0.0) * 1e-10
    pe, pt = _overlapping_box_pairs(edges, triangles, a, b, lo, hi, tolerance)
    if not len(pe):
        return np.empty((0, 2), int), np.empty(0, int)
    code = pe.astype(np.int64) * np.int64(len(triangles)) + pt.astype(np.int64)
    _unique, first = np.unique(code, return_index=True)
    first.sort()
    pe, pt = pe[first], pt[first]
    shared = (triangles[pt] == edges[pe, :1]).any(1) | (triangles[pt] == edges[pe, 1:]).any(1)
    overlap = np.all(np.maximum(a[pe], b[pe]) >= lo[pt] - tolerance, axis=1) & np.all(
        np.minimum(a[pe], b[pe]) <= hi[pt] + tolerance, axis=1
    )
    pe, pt = pe[~shared & overlap], pt[~shared & overlap]
    if not len(pe):
        return np.empty((0, 2), int), np.empty(0, int)
    hit = segment_triangle_pairs(a[pe], b[pe], corners[pt])
    return edges[pe[hit]], pt[hit]


def self_intersections(vertices, triangles):
    return len(intersection_pairs(vertices, triangles)[1])


def segment_triangle_pairs(a, b, tri, strict=False):
    """Segment i against triangle i, including coplanar and endpoint contacts.

    Tolerances are relative to pair size, so millimetres and metres agree.
    Degenerate triangles are reported separately by geometry_report.
    strict=True counts only transverse hits inside both primitives; coplanar,
    endpoint and triangle-edge contacts remain in the default inclusive count.
    """
    a, b, tri = np.asarray(a, float), np.asarray(b, float), np.asarray(tri, float)
    d, e1, e2 = b - a, tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    scale = np.maximum.reduce(
        [np.linalg.norm(d, axis=1), np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1)]
    )
    scale = np.maximum(scale, np.finfo(float).tiny)[:, None]
    d, e1, e2, s = d / scale, e1 / scale, e2 / scale, (a - tri[:, 0]) / scale
    n = np.cross(e1, e2)
    area = np.linalg.norm(n, axis=1)
    h = np.cross(d, e2)
    det = np.einsum("ij,ij->i", h, e1)
    roundoff = 64 * np.finfo(float).eps
    nondegenerate = area > roundoff
    ok = nondegenerate & (np.abs(det) > roundoff * area * np.linalg.norm(d, axis=1))
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=ok)
    u = np.einsum("ij,ij->i", s, h) * inv
    q = np.cross(s, e1)
    v = np.einsum("ij,ij->i", d, q) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    eps = 1.0e-10
    if strict:
        # A few thousand ulps of det is still a coplanar edge. Duplicate
        # vertices that sit on a triangle's corners produce that noise; the
        # resulting barycentric coordinates are not a pierce.
        separated = np.abs(det) > (4096 * np.finfo(float).eps) * area * np.linalg.norm(d, axis=1)
        on_corner = (np.linalg.norm(a[:, None, :] - tri, axis=2).min(1) <= 1.0e-9) & (
            np.linalg.norm(b[:, None, :] - tri, axis=2).min(1) <= 1.0e-9
        )
        return (
            nondegenerate
            & separated
            & (u > eps)
            & (v > eps)
            & (u + v < 1 - eps)
            & (t > eps)
            & (t < 1 - eps)
            & ~on_corner
        )
    hit = ok & (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps) & (t >= -eps) & (t <= 1 + eps)
    coplanar = nondegenerate & ~ok & (np.abs(np.einsum("ij,ij->i", s, n)) <= roundoff * area)
    ids = np.flatnonzero(coplanar)
    if len(ids):
        # Clip the segment against the three inward half-planes of the triangle.
        normal = n[ids] / area[ids, None]
        corners = [np.zeros_like(e1[ids]), e1[ids], e2[ids]]
        lower, upper = np.zeros(len(ids)), np.ones(len(ids))
        possible = np.ones(len(ids), bool)
        for j in range(3):
            edge = corners[(j + 1) % 3] - corners[j]
            f0 = np.einsum("ij,ij->i", np.cross(edge, s[ids] - corners[j]), normal)
            df = np.einsum("ij,ij->i", np.cross(edge, d[ids]), normal)
            tolerance = eps * np.linalg.norm(edge, axis=1)
            parallel = np.abs(df) <= roundoff
            possible &= ~parallel | (f0 >= -tolerance)
            bound = np.divide(-tolerance - f0, df, out=np.zeros_like(df), where=~parallel)
            lower = np.where(~parallel & (df > 0), np.maximum(lower, bound), lower)
            upper = np.where(~parallel & (df < 0), np.minimum(upper, bound), upper)
        hit[ids] = possible & (lower <= upper)
    return hit


def geometry_report(vertices, triangles, intersections=True):
    """Topology, component-wise signed volume, and geometric contacts in metres."""
    vertices, triangles = np.asarray(vertices, float), np.asarray(triangles, int)
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    _, inverse, incidence = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    orientation = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices))
    )
    _, labels = connected_components(graph, directed=False)
    used = np.unique(triangles)
    components = np.unique(labels[used])
    # Edge incidence alone misses two closed fans pinched together at one vertex.
    links = [[] for _ in vertices]
    for a, b, c in triangles:
        links[a].append((b, c))
        links[b].append((c, a))
        links[c].append((a, b))
    nonmanifold_vertices = 0
    for vertex in used:
        adjacency = {}
        for a, b in links[vertex]:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        seen, pending = set(), [next(iter(adjacency))]
        while pending:
            neighbor = pending.pop()
            if neighbor not in seen:
                seen.add(neighbor)
                pending.extend(adjacency[neighbor])
        if len(seen) != len(adjacency) or any(len(neighbors) != 2 for neighbors in adjacency.values()):
            nonmanifold_vertices += 1
    volumes = []
    for component in components:
        faces = triangles[labels[triangles[:, 0]] == component]
        # Translate each component before the tetrahedron sum to avoid cancellation.
        corners = vertices[faces] - vertices[np.unique(faces)].mean(0)
        volumes.append(
            float(np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])).sum() / 6)
        )
    corners = vertices[triangles]
    e1, e2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    area2 = np.linalg.norm(np.cross(e1, e2), axis=1)
    edge_scale2 = np.maximum(np.einsum("ij,ij->i", e1, e1), np.einsum("ij,ij->i", e2, e2))
    report = dict(
        vertices=len(vertices),
        triangles=len(triangles),
        finite_vertices=bool(np.isfinite(vertices).all()),
        components=len(components),
        unused_vertices=len(vertices) - len(used),
        boundary_edges=int(np.count_nonzero(incidence == 1)),
        nonmanifold_edges=int(np.count_nonzero(incidence > 2)),
        nonmanifold_vertices=nonmanifold_vertices,
        inconsistent_edge_orientations=int(np.count_nonzero(orientation)),
        degenerate_triangles=int(np.count_nonzero(area2 <= 64 * np.finfo(float).eps * edge_scale2)),
        signed_volume_m3=float(sum(volumes)),
        component_volumes_m3=volumes,
    )
    if intersections:
        hit_edges, hit_triangles = intersection_pairs(vertices, triangles)
        same = labels[hit_edges[:, 0]] == labels[triangles[hit_triangles, 0]]
        interior = segment_triangle_pairs(
            vertices[hit_edges[:, 0]],
            vertices[hit_edges[:, 1]],
            vertices[triangles[hit_triangles]],
            strict=True,
        )
        report.update(
            self_intersections=int(same.sum()),
            inter_component_contacts=int((~same).sum()),
            self_interior_crossings=int((same & interior).sum()),
            inter_component_interior_crossings=int((~same & interior).sum()),
        )
    return report


def cut(surface, index, x0, spacing):
    """Vertical through-cut at x = x0 with the runtime's side band (surface_runtime: 6 dx by 2 dx)."""
    edge = lambda z: [[x0, -0.05, z], [x0, 0.05, z]]  # noqa: E731
    surface.sweep(
        edge(0.06), edge(-0.02), cut_index=index, normal_world=[1, 0, 0], band=(6 * spacing, 2 * spacing)
    )


def rigid_pieces(codes, reference, seed=0):
    """A different rotation about its centroid and an outward shift for each piece."""
    rng = np.random.default_rng(seed)
    motions = {}
    for k, c in enumerate(np.unique(codes)):
        centre = reference[codes == c].mean(0)
        r = Rotation.from_rotvec(rng.normal(size=3) * 0.3)
        motions[c] = (centre, r, np.array([0.02 * (k - 1), 0.0, 0.01 * k]))
    return motions


def apply(motion, X):
    centre, r, shift = motion
    return r.apply(X - centre) + centre + shift


def run(spacing):
    size = BANANA_DEMO_SIZE_M
    ref, origin, cells = points(size, spacing, "banana")
    grid = grid_cells(size, spacing)
    rows = []
    cases = [("bend R = 150 mm", bend(0.15)), ("bend R = 60 mm", bend(0.06)), ("twist 10 rad/m", twist(10.0))]
    surface = LinkedSurface(size, spacing / 2, "banana")
    emb = MaterialEmbedding(surface.reference, ref, spacing, origin, grid)
    mesh = surface.mesh()
    for name, phi in cases:
        x, F = phi(ref)
        v = emb.vertices(mesh, x, F, np.zeros(len(ref), np.int64))
        exact, _ = phi(mesh.reference)
        rows.append((name, v, exact, mesh.triangles))
    surface = LinkedSurface(size, spacing / 2, "banana")
    for k, x0 in enumerate((-0.03, 0.0)):
        cut(surface, k, x0, spacing)
    mesh = surface.mesh()
    digits = [np.where(ref[:, 0] < x0, 1, 2) for x0 in (-0.03, 0.0)]
    codes = (digits[0] + 3 * digits[1]).astype(np.int64)
    motions = rigid_pieces(codes, ref)
    x = np.empty_like(ref)
    F = np.empty((len(ref), 3, 3))
    for c, m in motions.items():
        x[codes == c] = apply(m, ref[codes == c])
        F[codes == c] = m[1].as_matrix()
    emb = MaterialEmbedding(surface.reference, ref, spacing, origin, grid)
    v = emb.vertices(mesh, x, F, codes)
    # A vertex follows the piece of its side digits; a digit it does not carry
    # (far from that cut) is the side of its reference position.
    owner = np.zeros(len(v), np.int64)
    for d, x0 in enumerate((-0.03, 0.0)):
        digit = mesh.codes // 3**d % 3
        owner += 3**d * np.where(digit > 0, digit, np.where(mesh.reference[:, 0] < x0, 1, 2))
    exact = np.empty_like(v)
    for c in np.unique(owner):
        exact[owner == c] = apply(motions[c], mesh.reference[owner == c])
    rows.append(("3 pieces, rigid motions", v, exact, mesh.triangles))
    h = spacing / 2
    results = []
    for name, v, exact, tri in rows:
        e = np.linalg.norm(v - exact, axis=1) / h
        geometry, exact_geometry = geometry_report(v, tri), geometry_report(exact, tri)
        results.append(
            dict(
                case=name,
                material_spacing_m=spacing,
                surface_spacing_m=h,
                mean_error_cells=float(e.mean()),
                p99_error_cells=float(np.quantile(e, 0.99)),
                max_error_cells=float(e.max()),
                geometry=geometry,
                exact_geometry=exact_geometry,
                relative_volume_error=geometry["signed_volume_m3"] / exact_geometry["signed_volume_m3"] - 1,
            )
        )
        print(
            f"{spacing * 1e3:.0f} mm grid  {name:24s} vertices {len(v):6d}  error mean {e.mean():.3f}"
            f"  p99 {np.quantile(e, 0.99):.3f}  max {e.max():.3f} (surface cells)"
            f"  contacts {geometry['self_intersections'] + geometry['inter_component_contacts']}"
            f" (exact map: {exact_geometry['self_intersections'] + exact_geometry['inter_component_contacts']})"
            f"  volume error {results[-1]['relative_volume_error']:+.3%}",
            flush=True,
        )
    return results


def recorded(directory, frames=None):
    """Check saved world vertices and reference geometry without rerunning physics."""
    directory = Path(directory)
    if (directory / "surface").is_dir():
        directory = directory / "surface"
    paths = (
        sorted(directory.glob("*.npz")) if frames is None else [directory / f"{i:05d}.npz" for i in frames]
    )
    if not paths:
        raise ValueError(f"No surface snapshots in {directory}")
    results = []
    for path in paths:
        with np.load(path) as mesh:
            result = dict(frame=path.name, geometry=geometry_report(mesh["vertices"], mesh["triangles"]))
            if "reference" in mesh:
                result["reference_geometry"] = geometry_report(mesh["reference"], mesh["triangles"])
        results.append(result)
        print(json.dumps(result), flush=True)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spacing", type=float, nargs="+", default=[0.006, 0.003])
    p.add_argument("--recorded", type=Path, help="Run directory or surface snapshot directory")
    p.add_argument("--frames", type=int, nargs="+", help="Selected recorded frame indices; default all")
    p.add_argument("--output", type=Path, help="Write the full metrics as JSON")
    args = p.parse_args()
    results = (
        recorded(args.recorded, args.frames)
        if args.recorded
        else [row for s in args.spacing for row in run(s)]
    )
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")
