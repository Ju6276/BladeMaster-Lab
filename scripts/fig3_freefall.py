"""Paper Fig. 3: snapshots of the free-falling knife and its edge height and tilt.

python -m scripts.fig3_freefall runs/fig3 --output fig3.png
"""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.fig4_compare import knife_outline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--output", required=True)
    p.add_argument("--times", type=float, nargs="+", default=[0.0, 0.11, 0.14, 0.17, 0.25, 2.0])
    a = p.parse_args()
    d = np.load(Path(a.run) / "states.npz")
    cfg = json.loads((Path(a.run) / "manifest.json").read_text())["config"]
    half = np.array(cfg["blade_half_size"])
    t = d["time"]
    rot = Rotation.from_quat(d["blade_orientation"][:, 0])
    ends = np.stack([rot.apply([0.0, s * half[1], -half[2]]) for s in (-1, 1)], 1) + d["blade_position"]
    tilt = np.degrees(np.arcsin(np.clip(np.abs(ends[:, 1, 2] - ends[:, 0, 2]) / (2 * half[1]), 0, 1)))
    colors = np.array(["#9aa5b1", "#3d7fc1", "#e07a1f"])
    fig = plt.figure(figsize=(15, 8.5))
    for k, when in enumerate(a.times):
        i = int(np.argmin(abs(t - when)))
        x = d["position"][i] * 1000
        ax = fig.add_subplot(2, 4, k + 1, projection="3d")
        ax.scatter(*x.T, s=1.0, c=colors[d["sides"][i] % 3], depthshade=False, linewidths=0)
        for u, v in knife_outline(d, i, half):
            ax.plot(*np.array([u, v]).T * 1000, color="#33414f", lw=0.9)
        ax.set_xlim(-70, 70)
        ax.set_ylim(-70, 70)
        ax.set_zlim(0, 150)
        ax.set_box_aspect((140, 140, 150))
        ax.view_init(elev=12, azim=-35)
        ax.set_axis_off()
        ax.set_title(f"t = {t[i]:.2f} s", fontsize=10)
    ax = fig.add_subplot(2, 4, 7)
    ax.plot(t, ends[:, :, 2].min(1) * 1000, label="lower edge end")
    ax.plot(t, ends[:, :, 2].max(1) * 1000, label="upper edge end")
    ax.axhline(cfg["size"][2] * 1000, color="0.6", ls="--", lw=0.8, label="block top (initial)")
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("height [mm]")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(2, 4, 8)
    ax.plot(t, tilt)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("edge tilt [deg]")
    fig.suptitle("Free-falling 0.1 kg knife, 25° initial edge tilt (paper Fig. 3); blue / orange = cut sides", fontsize=12)
    fig.tight_layout()
    fig.savefig(a.output, dpi=110)
    print(a.output)


if __name__ == "__main__":
    main()
