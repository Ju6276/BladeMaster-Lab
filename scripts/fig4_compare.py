"""Paper Fig. 4: BladeMaster vs single-field baseline during slice lifting.

python -m scripts.fig4_compare runs/fig4_bm runs/fig4_sf --output fig4.png
"""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from blademaster_warp.geometry import points, stencil
from blademaster_warp.topology import components


def load(directory):
    d = np.load(Path(directory) / "states.npz")
    m = json.loads((Path(directory) / "manifest.json").read_text())
    cfg = m["config"]
    ref, origin, cells = points(cfg["size"], cfg["grid_spacing"], cfg["shape"], cfg["sampling_seed"])
    nodes, w, _ = stencil(ref, origin, cfg["grid_spacing"], cells)
    labels, sizes = components(nodes, w, d["sides"][-1])
    return d, m, labels == int(np.argmin(sizes))


def knife_outline(d, i, half):
    hx, hy, hz = half
    corners = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    world = Rotation.from_quat(d["blade_orientation"][i, 0]).apply(corners) + d["blade_position"][i, 0]
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
    return [(world[a], world[b]) for a, b in edges]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("blademaster")
    p.add_argument("single_field")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    runs = [("BladeMaster", *load(a.blademaster)), ("Single-field baseline", *load(a.single_field))]
    keys = runs[0][2]["keyframes"]
    moments = [
        ("Two radial cuts", keys[-5][0]),
        ("Lifting", (keys[-3][0] + keys[-2][0]) / 2),
        ("Lifted", keys[-1][0]),
    ]
    fig = plt.figure(figsize=(13.5, 8.2))
    for r, (name, d, m, piece) in enumerate(runs):
        half = m["config"]["blade_half_size"]
        for c, (title, t) in enumerate(moments):
            i = int(np.argmin(abs(d["time"] - t)))
            x = d["position"][i] * 1000
            ax = fig.add_subplot(2, 3, 3 * r + c + 1, projection="3d")
            ax.scatter(*x[~piece].T, s=1.2, c="#9aa5b1", depthshade=False, linewidths=0)
            ax.scatter(*x[piece].T, s=1.2, c="#e07a1f", depthshade=False, linewidths=0)
            for u, v in knife_outline(d, i, half):
                ax.plot(*np.array([u, v]).T * 1000, color="#33414f", lw=0.8)
            ax.set_xlim(-80, 140)
            ax.set_ylim(-110, 110)
            ax.set_zlim(0, 110)
            ax.set_box_aspect((220, 220, 110))
            ax.view_init(elev=18, azim=-60)
            ax.set_axis_off()
            ax.set_title(f"{name}: {title} ({d['time'][i]:.1f} s)", fontsize=10)
    fig.suptitle("Slice lifting (paper Fig. 4): orange = slice by cut-history codes", fontsize=12)
    fig.tight_layout()
    fig.savefig(a.output, dpi=110)
    print(a.output)


if __name__ == "__main__":
    main()
