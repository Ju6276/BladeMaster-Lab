"""Close-up of recorded surface and particles at chosen times (no robot), for checking surface quality."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from blademaster_warp.replay_style import ReplayScene
from blademaster_warp.render_franka import set_camera


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--times", type=float, nargs="+", default=[0.0, 3.6, 18.0])
    p.add_argument("--output", type=Path)
    p.add_argument("--camera", type=float, nargs=3, default=[60, -160, 110])
    p.add_argument("--scale", type=float, default=45)
    args = p.parse_args()
    d = args.directory
    manifest = json.loads((d / "manifest.json").read_text())
    frames = json.loads((d / "surface/frames.json").read_text())
    frame_times = np.array([f["time_s"] for f in frames])
    z = np.load(d / "trajectory.npz")
    times = z["control_tcp"][:, 0]
    geom = ET.parse(d / "scene.xml").find(".//geom[@name='board']")
    board = dict(
        size_mm=(np.fromstring(geom.get("size"), sep=" ") * 2000).tolist(),
        center_mm=((np.fromstring(geom.get("pos"), sep=" ") - z["origin"]) * 1000).tolist(),
    )
    scene = ReplayScene(manifest["config"], board, 640, 420, style="paper")
    set_camera(scene, position=args.camera, focal=[15, 0, 10], scale=args.scale)
    rows = []
    for t in args.times:
        fi = int(np.argmin(abs(frame_times - t)))
        idx = int(np.argmin(abs(times - frame_times[fi])))
        with np.load(d / "surface" / f"{fi:05d}.npz") as snapshot:
            scene.update(snapshot, z["particles"][idx], z["codes"][idx])
            rows.append(np.hstack([scene.frame(), scene.frame(particles=True)]))
    output = args.output or d / "surface_closeup.png"
    Image.fromarray(np.vstack(rows)).save(output)
    print(output)


if __name__ == "__main__":
    main()
