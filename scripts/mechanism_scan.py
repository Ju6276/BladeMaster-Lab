"""Fig. 3 free-fall with each cutting mechanism removed, at high and low sharpness.

Defaults stay the paper path: no edge hold, a six-cell work band, board-layer
labels on. Each row is one change.
"""

import json
from pathlib import Path

from blademaster_warp.paper_freefall import run

OUT = Path("runs/mechanism_scan_20261019")
CASES = (
    ("base", {}),
    ("hold", {"cut_hold": True}),
    ("noband", {"cut_band_cells": 0.0}),
    ("noboard", {"label_board_layer": False}),
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = OUT / "summary.jsonl"
    done = set()
    if summary.exists():
        for line in summary.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["name"])
    for sharpness in (0.9, 0.2):
        for name, extra in CASES:
            key = f"s{sharpness:g}_{name}"
            if key in done:
                continue
            result = run(OUT / key, sharpness=sharpness, **extra)
            row = dict(
                name=key,
                sharpness=sharpness,
                **extra,
                pieces=result["component_sizes"],
                edge_mm=round(result["final_lowest_edge_m"] * 1e3, 3),
                damaged=result["released_particles"],
                work_j=result["directional_contact_work_j"],
                assigned=result["assigned_particles"],
                gpu_errors=result["gpu_errors"],
            )
            with summary.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
