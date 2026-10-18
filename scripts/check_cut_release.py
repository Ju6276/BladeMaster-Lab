"""Reproduce a Franka teleop cut through the banana, hold, withdraw and settle.

python -m scripts.check_cut_release --spacing 0.003 --seed 1 --output runs/release-3mm
Use the same seed and protocol at another spacing to compare breakthrough motion.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from blademaster_warp.franka_run import FrankaCuttingRun, parser as run_parser
from blademaster_warp.teleop_input import Command


class CutReleaseRun(FrankaCuttingRun):
    def _teleop_input(self, t):
        # These are operator velocity commands; the impedance controller, material
        # reaction and board contact determine the actual blade trajectory.
        vz = -0.06 if 0.4 <= t < 2.4 else 0.06 if 3.2 <= t < 4.2 else 0.0
        return Command(round(t / self.args.interval), (0.0, 0.0, vz), (0.0, 0.0, 0.0)), []


def summarize(folder):
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text())
    labels = np.load(folder / "component_labels.npy")
    with np.load(folder / "trajectory.npz") as trace:
        times = trace["control_tcp"][:, 0]
        particles = trace["particles"].astype(float)
        edge_heights = trace["edge_heights"].min(axis=1)
        near_board = np.flatnonzero(edge_heights <= float(trace["radius"]))
        if not len(near_board):
            raise ValueError("The blade did not reach the bottom particle layer")
        bottom_time = float(trace["log"][near_board[0], 0])
        window = (times[:-1] >= bottom_time - 0.1) & (times[1:] <= bottom_time + 0.3)
        if not window.any():
            raise ValueError("No recorded intervals in the breakthrough window")
        pieces = []
        for label in np.unique(labels):
            members = labels == label
            centers = particles[:, members].mean(axis=1)
            # Position transport uses grid velocity. FLIP particle velocity can
            # contain unresolved residuals, so measure visible motion from x(t).
            velocity = np.diff(centers, axis=0) / np.diff(times)[:, None]
            before = np.array([np.interp(0.4, times, centers[:, j]) for j in range(3)])
            held = np.array([np.interp(1.8, times, centers[:, j]) for j in range(3)])
            pieces.append(
                dict(
                    particles=int(members.sum()),
                    peak_centroid_speed_m_s=float(np.linalg.norm(velocity[window], axis=1).max()),
                    peak_lateral_speed_m_s=float(np.abs(velocity[window, 0]).max()),
                    displacement_at_hold_m=(held - before).tolist(),
                )
            )
        result = dict(
            bottom_time_s=bottom_time,
            centroid_sampling_interval_s=float(np.median(np.diff(times))),
            peak_force_n=report["peak_material_force_n"],
            peak_sampled_kinetic_energy_j=float(trace["energy"][:, 1].max()),
            pieces=pieces,
            gpu_errors=report["gpu_errors"],
            mujoco_warnings=report["warnings"],
        )
    if (
        report["status"] != "completed"
        or len(pieces) != 2
        or any(result["gpu_errors"] + result["mujoco_warnings"])
    ):
        raise RuntimeError(f"Cut release failed: {result}")
    (folder / "cut_release.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--surface", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use a fresh result directory: {args.output}")
    options = [
        "--output",
        str(args.output),
        "--motion",
        "teleop",
        "--cut-events",
        "auto",
        "--material",
        "paper-banana",
        "--controller",
        "libfranka",
        "--stiffness",
        "1000",
        "--spacing",
        str(args.spacing),
        "--sampling-seed",
        str(args.seed),
        "--duration",
        "5.4",
        "--record",
    ]
    if args.surface:
        options += ["--surface", "--surface-embedding", "cuda", "--surface-ccd", "cuda"]
    run = CutReleaseRun(run_parser().parse_args(options))
    run.execute()
    protocol = dict(
        input="Franka velocity commands, automatic cutting",
        phases_s=dict(down=[0.4, 2.4], hold=[2.4, 3.2], up=[3.2, 4.2], end=5.4),
        commanded_speed_m_s=0.06,
        centroid_window_relative_to_bottom_s=[-0.1, 0.3],
        displacement_samples_s=[0.4, 1.8],
    )
    (args.output / "input_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(summarize(args.output), indent=2))


if __name__ == "__main__":
    main()
