"""Embedded surface under known deformations: vertex error and self-intersections.

Particles are moved by a smooth map phi with its exact gradient as F; each
surface vertex is compared with phi of its reference position (the embedding is
exact for affine maps, so the error measures the nonlinear part at the grid
scale). For cut pieces moved apart rigidly, every vertex must follow the motion
of its own piece. Self-intersections: mesh edges crossing a triangle that shares
no vertex with them.

    python -m scripts.surface_deformed
"""

import argparse
import numpy as np
from scipy.spatial import cKDTree
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


def self_intersections(vertices, triangles):
    edges = np.unique(
        np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1), axis=0
    )
    corners = vertices[triangles]
    centroid = corners.mean(1)
    reach = np.linalg.norm(corners - centroid[:, None], axis=2).max()
    a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
    half = np.linalg.norm(b - a, axis=1).max() / 2
    near = cKDTree(centroid).query_ball_point((a + b) / 2, reach + half)
    pe = np.repeat(np.arange(len(edges)), [len(n) for n in near])
    pt = np.concatenate([np.asarray(n, int) for n in near])
    shared = (triangles[pt] == edges[pe, :1]).any(1) | (triangles[pt] == edges[pe, 1:]).any(1)
    pe, pt = pe[~shared], pt[~shared]
    return int(segment_triangle_pairs(a[pe], b[pe], corners[pt]).sum())


def segment_triangle_pairs(a, b, tri):
    """Segment i against triangle i (Moller-Trumbore), endpoints included."""
    d, e1, e2 = b - a, tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    h = np.cross(d, e2)
    det = np.einsum("ij,ij->i", h, e1)
    ok = np.abs(det) > 1.0e-18
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=ok)
    s = a - tri[:, 0]
    u = np.einsum("ij,ij->i", s, h) * inv
    q = np.cross(s, e1)
    v = np.einsum("ij,ij->i", d, q) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    eps = 1.0e-9
    return ok & (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps) & (t >= -eps) & (t <= 1 + eps)


def cut(surface, index, x0, spacing):
    """Vertical through-cut at x = x0 with the runtime's side band (surface_runtime: 6 dx by 2 dx)."""
    edge = lambda z: [[x0, -0.05, z], [x0, 0.05, z]]  # noqa: E731
    surface.sweep(edge(0.06), edge(-0.02), cut_index=index, normal_world=[1, 0, 0], band=(6 * spacing, 2 * spacing))


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
    for name, phi in cases:
        surface = LinkedSurface(size, spacing / 2, "banana")
        emb = MaterialEmbedding(surface.reference, ref, spacing, origin, grid)
        mesh = surface.mesh()
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
    for name, v, exact, tri in rows:
        e = np.linalg.norm(v - exact, axis=1) / h
        print(
            f"{spacing * 1e3:.0f} mm grid  {name:24s} vertices {len(v):6d}  error mean {e.mean():.3f}"
            f"  p99 {np.quantile(e, 0.99):.3f}  max {e.max():.3f} (surface cells)"
            f"  self-intersections {self_intersections(v, tri)} (reference mesh: {self_intersections(exact, tri)})"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spacing", type=float, nargs="+", default=[0.006, 0.003])
    for s in p.parse_args().spacing:
        run(s)
