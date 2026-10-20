"""Cutting force at a fixed 0.5 mm contact radius across grids and seeds.

The interactive default still uses the particle radius. This series is the older
fixed-distance configuration. A dirty work tree is recorded, not rejected.
"""

import json
from pathlib import Path

import numpy as np

from scripts.force_convergence import DESCENT_S, INTERVAL, cut_trace

OUT = Path("runs/radius_series_20261019")
SPACINGS = (0.006, 0.004, 0.003)
SEEDS = range(1, 6)
RADIUS = 0.0005


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rows.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["spacing"], row["seed"]))
    print("cases", len(SPACINGS) * len(list(SEEDS)), "done", len(done), flush=True)
    for spacing in SPACINGS:
        for seed in SEEDS:
            if (spacing, seed) in done:
                continue
            run = cut_trace(spacing, seed, RADIUS)
            fz = np.asarray(run["force_n"])[:, 2]
            t = (np.arange(len(fz)) + 1) * INTERVAL
            inside = (t > 0.4) & (t <= 1.4)
            row = dict(
                spacing=spacing,
                seed=seed,
                dt=run["dt"],
                blade_contact_radius=run["blade_contact_radius"],
                particles=run["particles"],
                pieces=run["pieces"],
                steady_force_n=float(fz[inside].mean()),
                peak_force_n=float(np.max(np.abs(fz))),
                descent_impulse_ns=float(fz[t <= DESCENT_S].sum() * INTERVAL),
                directional_work_j=run["directional_work_j"],
                damaged_particles=run["damaged_particles"],
                gpu_errors=run["gpu_errors"],
                wall_s=run["wall_s"],
                revision=run["revision"],
            )
            with path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                f"h {spacing*1e3:g} seed {seed} steady {row['steady_force_n']:.3f} "
                f"peak {row['peak_force_n']:.3f} pieces {row['pieces']} wall {row['wall_s']:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
