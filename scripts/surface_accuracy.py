"""Reconstructed surface vs analytic pieces after planar through-cuts (reference configuration).

Each vertex belongs to the piece selected by its side code; the exact piece is the
banana level set intersected with one half-space per assigned digit. Reported:
|piece level set| at the vertices in surface cells, and closedness of the mesh.

    python -m scripts.surface_accuracy --spacing 0.0015
"""

import argparse
import numpy as np
from blademaster_warp.geometry import BANANA_DEMO_SIZE_M, levelset
from blademaster_warp.surface import LinkedSurface

# (point on the cut plane, unit normal). Two vertical cuts, an oblique cut and three thin slices.
CUTS = {
    "vertical": [((-0.03, 0, 0.012), (1, 0, 0)), ((0.0, 0, 0.012), (1, 0, 0))],
    "oblique": [((-0.03, 0, 0.012), (np.cos(0.5), 0, np.sin(0.5)))],
    "thin slices": [((x, 0, 0.012), (1, 0, 0)) for x in (0.02, 0.026, 0.032)],
}


def cut(surface, index, point, normal):
    """Sweep a long straight edge through the whole banana within the plane (point, normal)."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    down = np.cross(n, [0.0, 1.0, 0.0])
    down *= -np.sign(down[2])
    centre = np.asarray(point, float)
    edge = lambda c: [c - [0, 0.05, 0], c + [0, 0.05, 0]]  # noqa: E731
    surface.sweep(edge(centre - 0.05 * down), edge(centre + 0.05 * down), cut_index=index, normal_world=n)


def piece_levelset(x, codes, planes):
    value = levelset(x, BANANA_DEMO_SIZE_M, "banana")
    for k, (point, normal) in enumerate(planes):
        digit = codes // 3**k % 3
        signed = (x - point) @ (np.asarray(normal) / np.linalg.norm(normal))
        # Digit 1: the side with negative signed distance; digit 2: positive.
        value = np.where(digit == 1, np.maximum(value, signed), value)
        value = np.where(digit == 2, np.maximum(value, -signed), value)
    return value


def accuracy(spacing, cuts):
    s = LinkedSurface(BANANA_DEMO_SIZE_M, spacing, "banana")
    for k, (point, normal) in enumerate(cuts):
        cut(s, k, point, normal)
    mesh = s.mesh()
    error = np.abs(piece_levelset(mesh.reference, mesh.codes, cuts)) / spacing
    tri = mesh.triangles
    edges = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return dict(
        vertices=len(mesh.reference),
        triangles=len(tri),
        mean_error_cells=float(error.mean()),
        p99_error_cells=float(np.quantile(error, 0.99)),
        max_error_cells=float(error.max()),
        open_edges=int(np.count_nonzero(counts != 2)),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spacing", type=float, default=0.0015)
    a = p.parse_args()
    for name, cuts in [("uncut", [])] + list(CUTS.items()):
        print(f"{name:12s}", accuracy(a.spacing, cuts))
