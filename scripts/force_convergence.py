"""Blade force under grid refinement and particle resampling (paper Fig. 9 style).

Each run is the same prescribed vertical cut through the banana middle: the edge
moves from 36 mm to the board at 20 mm/s, then holds for 0.2 s. For every grid
spacing the cut is repeated with different particle sampling seeds; the traces
are averaged per spacing. Material step: the banana default for the spacing.

    python -m scripts.force_convergence --output runs/force_convergence \
        --spacings 0.006 0.005 0.004 0.003 --seeds 20
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
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


def effective_config(spacing, seed, blade_contact_radius=None):
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
    if blade_contact_radius is not None:
        cfg = replace(cfg, blade_contact_radius=blade_contact_radius)
    return cfg


def clean_revision():
    revision = git_revision()
    if not revision.get("commit") or revision.get("dirty") is not False:
        raise ValueError("Numeric study caches require a clean Git commit; commit changes before running")
    return revision


def cache_identity(spacing, seed, blade_contact_radius, provenance, x_offset_fraction=0.0):
    return json.loads(
        json.dumps(
            dict(
                config=asdict(effective_config(spacing, seed, blade_contact_radius)),
                provenance=provenance,
                protocol=dict(
                    interval=INTERVAL,
                    descent_s=DESCENT_S,
                    hold_s=HOLD_S,
                    speed=SPEED,
                    start_edge=START_EDGE,
                    x_offset_m=x_offset_fraction * spacing / 2,
                ),
            )
        )
    )


def validate_cache(run, expected, path):
    # Fresh Config snapshots contain tuples; JSON checkpoints contain lists.
    run = json.loads(json.dumps(run))
    for key, value in expected.items():
        if run.get("cache_identity", {}).get(key) != value:
            raise ValueError(f"Cache mismatch in {path}: {key}; use a separate output directory")
    config = expected["config"]
    for key, value in dict(
        config=config,
        spacing=config["grid_spacing"],
        seed=config["sampling_seed"],
        dt=config["dt"],
        blade_contact_radius=config["blade_contact_radius"],
        revision=expected["provenance"],
        x_offset_m=expected["protocol"]["x_offset_m"],
    ).items():
        if run.get(key) != value:
            raise ValueError(f"Cache mismatch in {path}: {key}; use a separate output directory")


def cut_trace(spacing, seed, blade_contact_radius=None, x_offset_fraction=0.0):
    cfg = effective_config(spacing, seed, blade_contact_radius)
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
            [x_offset_fraction * spacing / 2, 0.0, edge + half],
            [0, 0, 0, 1],
            [0.0, 0.0, -SPEED if moving else 0.0],
            [0, 0, 0],
        )
        force[i] = port.advance(state, INTERVAL, cutting=True).force_n
    wall = time.perf_counter() - begin
    _, pieces = components(s.node_numpy, s.weight_numpy, s.codes.numpy())
    return dict(
        spacing=spacing,
        seed=seed,
        x_offset_m=x_offset_fraction * spacing / 2,
        dt=cfg.dt,
        blade_contact_radius=cfg.blade_contact_radius,
        particles=s.n,
        force_n=force.tolist(),
        pieces=pieces.tolist(),
        wall_s=wall,
        gpu_errors=s.errors.numpy().tolist(),
        min_det_F=float(np.linalg.det(s.F.numpy()).min()),
        directional_work_j=float(s.work.numpy()[0]),
        damaged_particles=int((s.damage.numpy() >= 1).sum()),
        config=asdict(cfg),
        revision=git_revision(),
    )


def summarize(runs, window=(0.4, 1.4), bin_s=0.02):
    """Per spacing: mean/std over sampling seeds and offsets of force and impulse."""
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
            seeds=len({r["seed"] for r in group}),
            samples=len(group),
            x_offsets_m=sorted({r.get("x_offset_m", 0.0) for r in group}),
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


def plot(studies, path):
    """studies: [(label, summary)]. Left: mean +- std force traces; right: window force vs spacing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(studies) + 1, figsize=(5.2 * (len(studies) + 1), 3.8))
    for ax, (label, summary) in zip(axes, studies):
        rows = summary["by_spacing"]
        colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(rows)))
        for color, (name, row) in zip(colors, rows.items()):
            mean, std = np.array(row["mean_trace_n"]), np.array(row["std_trace_n"])
            t = (np.arange(len(mean)) + 0.5) * summary["bin_s"]
            ax.plot(t, mean, color=color, lw=1.4, label=f"{name} ({row.get('samples', row['seeds'])}×)")
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.18, lw=0)
        ax.axvspan(*summary["window_s"], color="0.9", zorder=-1)
        ax.set(
            title=label, xlabel="time (s)", ylabel="vertical blade force (N)", xlim=(0, DESCENT_S + HOLD_S)
        )
        ax.legend(fontsize=7, frameon=False)
    ax = axes[-1]
    for label, summary in studies:
        rows = summary["by_spacing"].items()
        h = [float(name.split()[0]) for name, _ in rows]
        mean = [row["steady_force_n"]["mean"] for _, row in rows]
        std = [row["steady_force_n"]["std"] for _, row in rows]
        ax.errorbar(h, mean, yerr=std, marker="o", ms=4, capsize=3, label=label)
    ax.set(xlabel="grid spacing (mm)", ylabel="mean force in shaded window (N)", ylim=(0, None))
    ax.invert_xaxis()
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--plot",
        nargs="+",
        metavar="LABEL=DIR",
        help="Only plot finished studies (summary.json in each DIR) to --output, a .png path",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--spacings", type=float, nargs="+", default=[0.006, 0.005, 0.004, 0.003])
    p.add_argument("--seeds", type=int, default=20, help="Sampling seeds 1..N per spacing")
    p.add_argument("--sampling-seeds", type=int, nargs="+", help="Explicit seeds; use 0 for regular sampling")
    p.add_argument(
        "--x-offset-fractions",
        type=float,
        nargs="+",
        default=[0.0],
        help="Knife offsets as fractions of the particle spacing h/2, in [0, 1)",
    )
    p.add_argument("--seeds-for", type=float, nargs=2, action="append", default=[], metavar=("SPACING", "N"))
    p.add_argument(
        "--blade-contact-radius",
        type=float,
        default=None,
        help="Metres; 0: particle radius h/4; default: preset",
    )
    a = p.parse_args()
    if a.plot:
        studies = [
            (s.split("=", 1)[0], json.loads(Path(s.split("=", 1)[1], "summary.json").read_text()))
            for s in a.plot
        ]
        plot(studies, a.output)
        return
    if any(not np.isfinite(f) or not 0 <= f < 1 for f in a.x_offset_fractions):
        p.error("--x-offset-fractions must be finite and in [0, 1)")
    if a.sampling_seeds is not None and (min(a.sampling_seeds) < 0 or a.seeds_for):
        p.error("--sampling-seeds must be non-negative and cannot be combined with --seeds-for")
    provenance = clean_revision()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    counts = {round(s, 6): int(n) for s, n in a.seeds_for}
    requests = []
    # Validate every existing cache before launching any simulations.
    for spacing in a.spacings:
        seeds = a.sampling_seeds or range(1, counts.get(round(spacing, 6), a.seeds) + 1)
        for seed in dict.fromkeys(seeds):
            for fraction in dict.fromkeys(a.x_offset_fractions):
                suffix = f"_phase{fraction!r}" if fraction else ""
                path = out / f"h{spacing * 1000:g}mm_seed{seed}{suffix}.json"
                identity = cache_identity(spacing, seed, a.blade_contact_radius, provenance, fraction)
                run = None
                if path.exists():
                    try:
                        run = json.loads(path.read_text())
                        validate_cache(run, identity, path)
                    except (ValueError, TypeError, AttributeError) as exc:
                        raise ValueError(f"Cannot reuse cache {path}: {exc}") from exc
                requests.append((spacing, seed, fraction, path, identity, run))
    runs = []
    for spacing, seed, fraction, path, identity, run in requests:
        if run is None:
            run = cut_trace(spacing, seed, a.blade_contact_radius, fraction)
            if git_revision() != provenance:
                raise ValueError("Code changed during computation; results were not cached")
            run["cache_identity"] = identity
            validate_cache(run, identity, path)
            path.write_text(json.dumps(run))
        runs.append(run)
        print(
            f"h={spacing*1000:g} mm seed={seed} phase={fraction:g} "
            f"n={run['particles']} wall={run['wall_s']:.1f}s",
            flush=True,
        )
    if git_revision() != provenance:
        raise ValueError("Code changed during the study; summary was not written")
    summary = dict(summarize(runs), revision=provenance, command=" ".join(sys.argv))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for name, row in summary["by_spacing"].items():
        print(
            f"{name}: n={row['particles']} dt={row['dt_us']:.1f}us seeds={row['seeds']} "
            f"steady={row['steady_force_n']['mean']:.3f}±{row['steady_force_n']['std']:.3f} N "
            f"impulse={row['descent_impulse_ns']['mean']:.3f}±{row['descent_impulse_ns']['std']:.3f} Ns "
            f"peak(single)={row['peak_single_run_n']['mean']:.2f} N peak(mean)={row['peak_of_mean_trace_n']:.2f} N "
            f"severed={row['severed_runs']}/{row['samples']} errors={row['gpu_errors']}"
        )


if __name__ == "__main__":
    main()
