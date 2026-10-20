"""Split the 6 mm five-offset vertical force into contact sources.

The reported force is the tool reaction. This records, at the peak vertical
sample of each lateral offset:

- broad-face normal
- edge or bevel normal
- friction
- grid-face constraint
- directional cutting work over the whole cut, in joules

    python -m scripts.force_components --output runs/force-components
"""

import argparse
import json
from pathlib import Path

import numpy as np

from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.solver import Solver
from blademaster_warp.topology import components
from scripts.force_convergence import (
    DESCENT_S,
    HOLD_S,
    INTERVAL,
    SPEED,
    START_EDGE,
    effective_config,
)


def cut(spacing, fraction):
    cfg = effective_config(spacing, 0)
    solver = Solver(cfg)
    port = ExternalToolPort(solver)
    half = cfg.blade_half_size[2]
    steps = round((DESCENT_S + HOLD_S) / INTERVAL)
    force = np.empty((steps, 3))
    parts = np.empty((steps, 4, 3))
    work = np.empty(steps)
    for i in range(steps):
        t = i * INTERVAL
        moving = t < DESCENT_S
        edge = START_EDGE - SPEED * t if moving else 0.0
        before_parts = solver.contact_parts.numpy().copy()
        before_work = float(solver.work.numpy()[0])
        state = ToolState(
            [fraction * spacing / 2, 0.0, edge + half],
            [0, 0, 0, 1],
            [0.0, 0.0, -SPEED if moving else 0.0],
            [0, 0, 0],
        )
        force[i] = port.advance(state, INTERVAL, cutting=True).force_n
        parts[i] = solver.contact_parts.numpy() - before_parts
        work[i] = float(solver.work.numpy()[0]) - before_work
    component_force = parts / INTERVAL
    peak = int(np.argmax(force[:, 2]))
    _, pieces = components(solver.node_numpy, solver.weight_numpy, solver.codes.numpy())
    names = ("face_normal_n", "edge_normal_n", "friction_n", "grid_constraint_n")
    at_peak = {name: component_force[peak, index].tolist() for index, name in enumerate(names)}
    summed = component_force.sum(axis=1)
    return dict(
        x_offset_mm=fraction * spacing / 2 * 1000,
        particles=solver.n,
        pieces=pieces.tolist(),
        peak_force_n=force[peak].tolist(),
        peak_time_s=(peak + 1) * INTERVAL,
        components_at_peak_n=at_peak,
        component_sum_at_peak_n=summed[peak].tolist(),
        sum_minus_total_n=(summed[peak] - force[peak]).tolist(),
        cutting_work_j=float(work.sum()),
        gpu_errors=solver.errors.numpy().tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=0.006)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0, 0.125, 0.25, 0.375, 0.5])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir()
    rows = []
    for fraction in args.fractions:
        row = cut(args.spacing, fraction)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (args.output / "components.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
