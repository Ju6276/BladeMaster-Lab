"""Blade force under grid refinement and particle resampling (paper Fig. 9 style).

Each run is the same prescribed vertical cut through the banana middle: the edge
moves from 36 mm to the board at 20 mm/s, then holds for 0.2 s. For every grid
spacing the cut is repeated with different particle sampling seeds; the traces
are averaged per spacing. Material step: the banana default for the spacing.

    python -m scripts.force_convergence --output runs/force_convergence \
        --spacings 0.006 0.005 0.004 0.003 --seeds 20
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.presets import banana_config, banana_dt
from blademaster_warp.solver import Solver
from blademaster_warp.topology import components

INTERVAL = 0.001
DESCENT_S, HOLD_S, SPEED = 1.8, 0.2, 0.02
START_EDGE = DESCENT_S * SPEED


def cut_trace(spacing, seed):
    base = banana_config()
    cfg = replace(
        base,
        grid_spacing=spacing,
        dt=banana_dt(base, spacing, INTERVAL),
        sampling_seed=seed,
        external_tool=True,
        grid_storage="compact",
        cache_topology=True,
    )
    s = Solver(cfg)
    port = ExternalToolPort(s)
    half = cfg.blade_half_size[2]
    steps = round((DESCENT_S + HOLD_S) / INTERVAL)
    force = np.empty((steps, 3))
    begin = time.perf_counter()
    for i in range(steps):
        t = i * INTERVAL
        moving = t < DESCENT_S
        edge = START_EDGE - SPEED * t if moving else 0.0
        state = ToolState(
            [0.0, 0.0, edge + half], [0, 0, 0, 1], [0.0, 0.0, -SPEED if moving else 0.0], [0, 0, 0]
        )
        force[i] = port.advance(state, INTERVAL, cutting=True).force_n
    wall = time.perf_counter() - begin
    _, pieces = components(s.node_numpy, s.weight_numpy, s.codes.numpy())
    return dict(
        spacing=spacing,
        seed=seed,
        dt=cfg.dt,
        particles=s.n,
        force_n=force.tolist(),
        pieces=pieces.tolist(),
        wall_s=wall,
        gpu_errors=s.errors.numpy().tolist(),
        min_det_F=float(np.linalg.det(s.F.numpy()).min()),
    )


def summarize(runs, window=(0.4, 1.4), bin_s=0.02):
    """Per spacing: mean/std over seeds of the binned vertical force and window averages."""
    out = {}
    for spacing in sorted({r["spacing"] for r in runs}, reverse=True):
        group = [r for r in runs if r["spacing"] == spacing]
        fz = np.array([np.asarray(r["force_n"])[:, 2] for r in group])
        per_bin = round(bin_s / INTERVAL)
        binned = fz[:, : fz.shape[1] // per_bin * per_bin].reshape(len(group), -1, per_bin).mean(2)
        t = (np.arange(fz.shape[1]) + 1) * INTERVAL
        inside = (t > window[0]) & (t <= window[1])
        steady = fz[:, inside].mean(1)
        impulse = fz[:, t <= DESCENT_S].sum(1) * INTERVAL
        out[f"{spacing * 1000:g} mm"] = dict(
            particles=group[0]["particles"],
            dt_us=group[0]["dt"] * 1.0e6,
            seeds=len(group),
            steady_force_n=dict(
                mean=float(steady.mean()), std=float(steady.std(ddof=1)) if len(group) > 1 else 0.0
            ),
            descent_impulse_ns=dict(
                mean=float(impulse.mean()), std=float(impulse.std(ddof=1)) if len(group) > 1 else 0.0
            ),
            peak_single_run_n=dict(mean=float(fz.max(1).mean()), max=float(fz.max())),
            peak_of_mean_trace_n=float(binned.mean(0).max()),
            severed_runs=int(sum(len(r["pieces"]) >= 2 for r in group)),
            wall_s_per_run=float(np.mean([r["wall_s"] for r in group])),
            gpu_errors=int(sum(sum(r["gpu_errors"]) for r in group)),
            mean_trace_n=binned.mean(0).tolist(),
            std_trace_n=binned.std(0, ddof=1).tolist() if len(group) > 1 else [0.0] * binned.shape[1],
        )
    return dict(window_s=list(window), bin_s=bin_s, by_spacing=out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True)
    p.add_argument("--spacings", type=float, nargs="+", default=[0.006, 0.005, 0.004, 0.003])
    p.add_argument("--seeds", type=int, default=20, help="Sampling seeds 1..N per spacing")
    p.add_argument("--seeds-for", type=float, nargs=2, action="append", default=[], metavar=("SPACING", "N"))
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    counts = {round(s, 6): int(n) for s, n in a.seeds_for}
    runs = []
    for spacing in a.spacings:
        for seed in range(1, counts.get(round(spacing, 6), a.seeds) + 1):
            path = out / f"h{spacing * 1000:g}mm_seed{seed}.json"
            if path.exists():
                run = json.loads(path.read_text())
            else:
                run = cut_trace(spacing, seed)
                path.write_text(json.dumps(run))
            runs.append(run)
            print(
                f"h={spacing*1000:g} mm seed={seed} n={run['particles']} wall={run['wall_s']:.1f}s",
                flush=True,
            )
    summary = summarize(runs)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for name, row in summary["by_spacing"].items():
        print(
            f"{name}: n={row['particles']} dt={row['dt_us']:.1f}us seeds={row['seeds']} "
            f"steady={row['steady_force_n']['mean']:.3f}±{row['steady_force_n']['std']:.3f} N "
            f"impulse={row['descent_impulse_ns']['mean']:.3f}±{row['descent_impulse_ns']['std']:.3f} Ns "
            f"peak(single)={row['peak_single_run_n']['mean']:.2f} N peak(mean)={row['peak_of_mean_trace_n']:.2f} N "
            f"severed={row['severed_runs']}/{row['seeds']} errors={row['gpu_errors']}"
        )


if __name__ == "__main__":
    main()
