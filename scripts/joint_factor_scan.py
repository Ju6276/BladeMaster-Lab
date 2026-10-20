"""Joint cut scan: grid, timestep, seed, blade thickness, contact distance.

Contact stiffness stays at the 25 us reference when the step changes. This does
not require a clean git commit; the dirty revision is stored with each row.
"""

import json
import math
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.presets import banana_config, banana_dt
from blademaster_warp.provenance import git_revision
from blademaster_warp.solver import Solver
from blademaster_warp.topology import components

INTERVAL = 0.001
DESCENT_S, HOLD_S, SPEED = 1.8, 0.2, 0.02
START_EDGE = DESCENT_S * SPEED
OUT = Path("runs/joint_scan_20261019")


def compatible(nominal, robot_step=INTERVAL):
    steps = max(1, math.ceil(robot_step / nominal - 1e-12))
    return robot_step / steps


def cases():
    rows = []
    for spacing in (0.006, 0.003):
        nominal = banana_dt(banana_config(), spacing, INTERVAL)
        seeds = (0, 1)
        for seed in seeds:
            for dt in (nominal, compatible(nominal / 2)):
                for thickness in (0.0005, 0.001):
                    for radius in (0.0, 0.0005, 0.003):
                        rows.append(
                            dict(
                                spacing=spacing,
                                seed=seed,
                                dt=dt,
                                thickness=thickness,
                                radius=radius,
                            )
                        )
    return rows


def run_case(case):
    base = banana_config()
    half = list(base.blade_half_size)
    half[0] = case["thickness"]
    cfg = replace(
        base,
        grid_spacing=case["spacing"],
        dt=case["dt"],
        sampling_seed=case["seed"],
        blade_half_size=tuple(half),
        blade_contact_radius=case["radius"],
        external_tool=True,
        grid_storage="compact",
        cache_topology=True,
    )
    solver = Solver(cfg)
    port = ExternalToolPort(solver)
    half_z = cfg.blade_half_size[2]
    steps = round((DESCENT_S + HOLD_S) / INTERVAL)
    edge_z = []
    actual_z = []
    peak = 0.0
    impulse = 0.0
    begin = time.perf_counter()
    for i in range(steps):
        t = i * INTERVAL
        moving = t < DESCENT_S
        edge = START_EDGE - SPEED * t if moving else 0.0
        state = ToolState(
            [0.0, 0.0, edge + half_z],
            [0, 0, 0, 1],
            [0.0, 0.0, -SPEED if moving else 0.0],
            [0, 0, 0],
        )
        force = port.advance(state, INTERVAL, cutting=True).force_n
        fz = float(force[2])
        peak = max(peak, abs(fz))
        if moving:
            impulse += fz * INTERVAL
        if i % 50 == 0 or i == steps - 1:
            edge_z.append(round(edge, 6))
            actual_z.append(round(float(solver.tool_pos.numpy()[0, 2] - half_z), 6))
    codes = solver.codes.numpy()
    sides = np.stack([(codes // 3**k) % 3 for k in range(6)], axis=1)
    _, pieces = components(solver.node_numpy, solver.weight_numpy, codes)
    return dict(
        **case,
        contact_reference_dt=cfg.contact_reference_dt,
        particles=int(solver.n),
        pieces=[int(n) for n in pieces],
        final_edge_z_m=edge_z[-1],
        final_actual_edge_z_m=actual_z[-1],
        min_actual_edge_z_m=min(actual_z),
        edge_z_m=edge_z,
        actual_edge_z_m=actual_z,
        peak_abs_fz_n=peak,
        descent_impulse_ns=impulse,
        directional_work_j=float(solver.work.numpy()[0]),
        damaged_particles=int((solver.damage.numpy() >= 1).sum()),
        side_counts=[int((sides[:, 0] == label).sum()) for label in (0, 1, 2)],
        gpu_errors=int(solver.errors.numpy().sum()),
        device=str(solver.device),
        wall_s=round(time.perf_counter() - begin, 3),
        revision=git_revision(),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rows.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["spacing"], row["seed"], row["dt"], row["thickness"], row["radius"]))
    print("cases", len(cases()), "done", len(done), flush=True)
    for index, case in enumerate(cases()):
        key = (case["spacing"], case["seed"], case["dt"], case["thickness"], case["radius"])
        if key in done:
            continue
        row = run_case(case)
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            index,
            "h",
            case["spacing"],
            "dt_us",
            round(case["dt"] * 1e6, 2),
            "seed",
            case["seed"],
            "thick_mm",
            case["thickness"] * 1e3,
            "radius_mm",
            case["radius"] * 1e3,
            "work",
            round(row["directional_work_j"], 6),
            "damaged",
            row["damaged_particles"],
            "sides",
            row["side_counts"],
            "edge_mm",
            round(row["final_actual_edge_z_m"] * 1e3, 3),
            "pieces",
            row["pieces"],
            "wall",
            row["wall_s"],
            flush=True,
        )


if __name__ == "__main__":
    main()
