"""Axial cutting resistance of recorded feedback runs (compare with paper Fig. 9).

.venv/bin/python -m scripts.cut_force runs/a runs/b ...

Each directory must come from franka_run --motion feedback --record.
"""

import argparse
import json
from pathlib import Path
import numpy as np


def summarize(directory):
    directory = Path(directory)
    z = np.load(directory / "trajectory.npz")
    report = json.loads((directory / "report.json").read_text())
    interval = float(z["interval"])
    log, trace = z["log"], z["cycle_trace"]
    descend = trace[:, 1] == 1
    # Force of the material on the knife, world z; resistance is the upward part.
    resistance = log[descend, 9]
    board = trace[descend, -1]
    cut_time = descend.sum() * interval
    return dict(
        run=directory.name,
        particles=report["particle_count"],
        descend_s=cut_time,
        mean_resistance_n=float(resistance.mean()),
        plateau_resistance_n=float(np.median(resistance[: int(0.8 * len(resistance))])),
        peak_resistance_n=float(resistance.max()),
        impulse_ns=float(resistance.sum() * interval),
        peak_board_force_n=float(board.max()),
        cutting_work_j=float(z["energy"][-1, -1]),
        components=report["components"],
        success=report["task_evaluation"]["success"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    rows = [summarize(d) for d in p.parse_args().runs]
    for row in rows:
        print(json.dumps(row))
    if len(rows) > 1:
        means = {k: float(np.mean([r[k] for r in rows])) for k in ("mean_resistance_n", "impulse_ns")}
        print("mean over runs:", json.dumps(means))


if __name__ == "__main__":
    main()
