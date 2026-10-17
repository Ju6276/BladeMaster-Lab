"""Reconstructed surfaces of scenario runs at chosen times, one row per run.

python -m scripts.scene_surfaces runs/fig7-shallow runs/fig7-deep --times 3 5.5 10 13.4 --output fig7.png
"""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation


def knife_edges(position, orientation, half):
    hx, hy, hz = half
    corners = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    world = Rotation.from_quat(orientation).apply(corners) + position
    pairs = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
    return [world[[a, b]] for a, b in pairs]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--times", type=float, nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--elev", type=float, default=35)
    p.add_argument("--azim", type=float, default=-55)
    p.add_argument("--columns", type=int, default=None, help="Wrap the times of a run into rows")
    p.add_argument("--xlim", type=float, nargs=2, default=None, help="mm")
    a = p.parse_args()
    light = np.array([0.3, -0.4, 0.87])
    light /= np.linalg.norm(light)
    columns = a.columns or len(a.times)
    rows_per_run = -(-len(a.times) // columns)
    fig = plt.figure(figsize=(3.4 * columns, 3.3 * rows_per_run * len(a.runs)))
    for r, run in enumerate(a.runs):
        directory = Path(run)
        manifest = json.loads((directory / "manifest.json").read_text())
        half = manifest["config"]["blade_half_size"]
        size = np.array(manifest["config"]["size"]) * 1000
        frames = json.loads((directory / "surface/frames.json").read_text())
        times = np.array([f["time_s"] for f in frames])
        for c, t in enumerate(a.times):
            fi = int(np.argmin(abs(times - t)))
            with np.load(directory / "surface" / f"{fi:05d}.npz") as z:
                v, tri = z["vertices"] * 1000, z["triangles"]
                tool_p, tool_q = z["tool_position"] * 1000, z["tool_orientation"]
            faces = v[tri]
            normal = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
            shade = 0.35 + 0.65 * np.clip(np.abs(normal @ light), 0, 1)
            colors = np.column_stack([0.93 * shade, 0.86 * shade, 0.70 * shade, np.ones_like(shade)])
            ax = fig.add_subplot(
                rows_per_run * len(a.runs), columns, r * rows_per_run * columns + c + 1, projection="3d"
            )
            ax.add_collection3d(Poly3DCollection(faces, facecolors=colors, edgecolor="none"))
            for edge in knife_edges(tool_p, tool_q, np.array(half) * 1000):
                ax.plot(*edge.T, color="#33414f", lw=0.7)
            span = size.max() * 0.6
            xlim = a.xlim or (-span, span)
            ax.set_xlim(*xlim)
            ax.set_ylim(-span, span)
            ax.set_zlim(0, size[2] * 1.4)
            ax.set_box_aspect((xlim[1] - xlim[0], 2 * span, size[2] * 1.4))
            ax.view_init(elev=a.elev, azim=a.azim)
            ax.set_axis_off()
            ax.set_title(f"{directory.name}  {times[fi]:.2f} s", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.output, dpi=120)
    print(a.output)


if __name__ == "__main__":
    main()
