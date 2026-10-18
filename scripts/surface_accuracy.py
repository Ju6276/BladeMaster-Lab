"""Reconstructed surface vs analytic pieces after planar through-cuts (reference configuration).

Each vertex belongs to the piece selected by its side code; the exact piece is the
banana level set intersected with one half-space per assigned digit. Reported:
|piece level set| at vertices and face centres, manifold/orientation checks,
contacts within and between pieces, and signed volume against the analytic solid.

    python -m scripts.surface_accuracy --spacing 0.0015
"""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import beta
from blademaster_warp.geometry import BANANA_DEMO_SIZE_M, levelset
from blademaster_warp.surface import LinkedSurface
from scripts.surface_deformed import geometry_report

# (point on the cut plane, unit normal). Two vertical cuts, an oblique cut, 6 mm and 3 mm slices.
CUTS = {
    "vertical": [((-0.03, 0, 0.012), (1, 0, 0)), ((0.0, 0, 0.012), (1, 0, 0))],
    "oblique": [((-0.03, 0, 0.012), (np.cos(0.5), 0, np.sin(0.5)))],
    "thin slices": [((x, 0, 0.012), (1, 0, 0)) for x in (0.02, 0.026, 0.032)],
    "3 mm slices": [((x, 0, 0.012), (1, 0, 0)) for x in (0.02, 0.023, 0.026, 0.029)],
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


def banana_volume():
    """Integral of pi * radius(x)**2 over the analytic banana's length, in m^3."""
    a, b, exponent = 0.003, 0.009, 0.55
    return float(
        np.pi
        * BANANA_DEMO_SIZE_M[0]
        / 2
        * (2 * a * a + 2 * a * b * beta(0.5, exponent + 1) + b * b * beta(0.5, 2 * exponent + 1))
    )


def accuracy(spacing, cuts, intersections=True):
    s = LinkedSurface(BANANA_DEMO_SIZE_M, spacing, "banana")
    for k, (point, normal) in enumerate(cuts):
        cut(s, k, point, normal)
    mesh = s.mesh()
    error = np.abs(piece_levelset(mesh.reference, mesh.codes, cuts)) / spacing
    tri = mesh.triangles
    # A face interior can deviate even if all its vertices fit the implicit surface.
    centres = mesh.reference[tri].mean(1)
    # Every face belongs to the same material piece; combine its assigned digits.
    codes = np.zeros(len(tri), np.int64)
    for k in range(len(cuts)):
        digits = mesh.codes[tri] // 3**k % 3
        if np.any((digits == 1).any(1) & (digits == 2).any(1)):
            raise ValueError("A surface triangle joins opposite cut sides")
        codes += digits.max(1) * 3**k
    centre_error = np.abs(piece_levelset(centres, codes, cuts)) / spacing
    geometry = geometry_report(mesh.reference, tri, intersections=intersections)
    return dict(
        **geometry,
        mean_error_cells=float(error.mean()),
        p99_error_cells=float(np.quantile(error, 0.99)),
        max_error_cells=float(error.max()),
        mean_face_error_cells=float(centre_error.mean()),
        max_face_error_cells=float(centre_error.max()),
        open_edges=geometry["boundary_edges"] + geometry["nonmanifold_edges"],
        analytic_volume_m3=banana_volume(),
        relative_volume_error=geometry["signed_volume_m3"] / banana_volume() - 1,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spacing", type=float, default=0.0015)
    p.add_argument("--output", type=Path, help="Write the full metrics as JSON")
    p.add_argument("--skip-intersections", action="store_true", help="Skip edge/face contact checks")
    a = p.parse_args()
    results = {}
    for name, cuts in [("uncut", [])] + list(CUTS.items()):
        results[name] = accuracy(a.spacing, cuts, intersections=not a.skip_intersections)
        print(f"{name:12s}", results[name], flush=True)
    if a.output:
        a.output.write_text(json.dumps(results, indent=2) + "\n")
