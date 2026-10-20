"""Twenty sampling seeds at 2.5, 2 and 1.5 mm, default particle-radius contact.

Same vertical cut as force_convergence. A dirty tree is recorded in each row
and does not satisfy that script's clean-commit cache.
"""

import json
from pathlib import Path

import numpy as np

from blademaster_warp.provenance import git_revision
from scripts.force_convergence import DESCENT_S, INTERVAL, cut_trace

OUT = Path("runs/force_seed_scan_20261019")
SPACINGS = (0.0025, 0.002, 0.0015)
SEEDS = range(1, 21)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rows.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["spacing"], row["seed"]))
    print("cases", len(SPACINGS) * len(list(SEEDS)), "done", len(done), "revision", git_revision(), flush=True)
    for spacing in SPACINGS:
        for seed in SEEDS:
            if (spacing, seed) in done:
                continue
            run = cut_trace(spacing, seed, None)
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
                f"peak {row['peak_force_n']:.3f} pieces {len(row['pieces'])} "
                f"damaged {row['damaged_particles']} wall {row['wall_s']:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
