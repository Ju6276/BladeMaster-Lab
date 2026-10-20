"""Fine-grid volume, indentation force and cutting force at two timesteps.

Contact stiffness stays at the 25 us reference. Both steps are at or below the
grid CFL limit used by banana_dt. A dirty tree is recorded, not rejected.
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

INTERVAL = 0.001
OUT = Path("runs/timestep_reference_20261019")
SPACINGS = (0.0025, 0.002, 0.0015)


def compatible(nominal, robot_step=INTERVAL):
    steps = max(1, math.ceil(robot_step / nominal - 1e-12))
    return robot_step / steps


def state(solver):
    det = np.linalg.det(solver.F.numpy())
    reference = solver.n * solver.volume
    current = float(det.sum() * solver.volume)
    velocity = solver.v.numpy()
    return dict(
        volume_m3=current,
        reference_volume_m3=float(reference),
        volume_ratio=current / reference,
        min_det_F=float(det.min()),
        kinetic_J=0.5 * float(solver.mass) * float(np.square(velocity).sum()),
        gpu_errors=int(solver.errors.numpy().sum()),
    )


def configure(spacing, dt, seed):
    base = banana_config()
    return replace(
        base,
        grid_spacing=spacing,
        dt=dt,
        sampling_seed=seed,
        external_tool=True,
        grid_storage="compact",
        cache_topology=True,
    )


def indent(spacing, dt, seed):
    cfg = configure(spacing, dt, seed)
    solver = Solver(cfg)
    port = ExternalToolPort(solver)
    half = cfg.blade_half_size[2]
    top = solver.reference[np.abs(solver.reference[:, 0]) < spacing, 2].max() + solver.blade_offset
    edge, forces, edges = 0.03, [], []
    begin = time.perf_counter()
    while edge > top - 0.006:
        edge = 0.03 - 0.02 * len(edges) * INTERVAL
        state_tool = ToolState([0.0, 0.0, edge + half], [0, 0, 0, 1], [0.0, 0.0, -0.02], [0, 0, 0])
        forces.append(float(port.advance(state_tool, INTERVAL, cutting=False).force_n[2]))
        edges.append(edge)
    depth = top - np.array(edges)
    fz = np.asarray(forces)
    measured = state(solver)
    return dict(
        kind="indent",
        spacing=spacing,
        dt=dt,
        seed=seed,
        contact_reference_dt=cfg.contact_reference_dt,
        particles=int(solver.n),
        forces_n={str(d): float(np.interp(d, depth, fz)) for d in (0.002, 0.004, 0.006)},
        peak_force_n=float(np.max(np.abs(fz))),
        wall_s=round(time.perf_counter() - begin, 3),
        revision=git_revision(),
        **measured,
    )


def cut(spacing, dt, seed):
    cfg = configure(spacing, dt, seed)
    solver = Solver(cfg)
    port = ExternalToolPort(solver)
    half = cfg.blade_half_size[2]
    steps = round(2.0 / INTERVAL)
    fz = np.empty(steps)
    begin = time.perf_counter()
    for i in range(steps):
        t = i * INTERVAL
        moving = t < 1.8
        edge = 0.036 - 0.02 * t if moving else 0.0
        tool = ToolState(
            [0.0, 0.0, edge + half],
            [0, 0, 0, 1],
            [0.0, 0.0, -0.02 if moving else 0.0],
            [0, 0, 0],
        )
        fz[i] = port.advance(tool, INTERVAL, cutting=True).force_n[2]
    clock = (np.arange(steps) + 1) * INTERVAL
    inside = (clock > 0.4) & (clock <= 1.4)
    measured = state(solver)
    return dict(
        kind="cut",
        spacing=spacing,
        dt=dt,
        seed=seed,
        contact_reference_dt=cfg.contact_reference_dt,
        particles=int(solver.n),
        steady_force_n=float(fz[inside].mean()),
        peak_force_n=float(np.max(np.abs(fz))),
        directional_work_j=float(solver.work.numpy()[0]),
        damaged_particles=int((solver.damage.numpy() >= 1).sum()),
        wall_s=round(time.perf_counter() - begin, 3),
        revision=git_revision(),
        **measured,
    )


def cases():
    rows = []
    for spacing in SPACINGS:
        nominal = compatible(banana_dt(banana_config(), spacing, INTERVAL))
        for dt in (nominal, compatible(nominal / 2)):
            for kind in ("indent", "cut"):
                rows.append((kind, spacing, dt))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rows.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["kind"], row["spacing"], row["dt"]))
    print("cases", len(cases()), "done", len(done), flush=True)
    for kind, spacing, dt in cases():
        if (kind, spacing, dt) in done:
            continue
        row = indent(spacing, dt, 1) if kind == "indent" else cut(spacing, dt, 1)
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            kind,
            f"h {spacing*1e3:g}",
            f"dt {dt*1e6:.2f}us",
            "particles",
            row["particles"],
            "vol",
            round(row["volume_ratio"], 5),
            "ke",
            f"{row['kinetic_J']:.3e}",
            "force",
            row["steady_force_n"] if kind == "cut" else row["forces_n"],
            "wall",
            row["wall_s"],
            flush=True,
        )


if __name__ == "__main__":
    main()
