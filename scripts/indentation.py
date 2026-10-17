"""Knife indentation without cutting: grid dependence of the banana's stiffness, and force relaxation.

The knife edge is pressed 6 mm into the banana top at 20 mm/s (--speed) with cutting disabled.
Reported: vertical force at 2, 4 and 6 mm indentation, averaged over sampling seeds.
With --hold the knife then stays at 6 mm for that many seconds; the force during the
hold is fitted with F(t) = F_inf + (F_0 - F_inf) exp(-t / tau).

    python -m scripts.indentation --spacings 0.006 0.004 0.003 0.002 --seeds 3
    python -m scripts.indentation --spacings 0.003 --seeds 1 --hold 10 --maxwell-tau 2.5 --maxwell-fraction 0.2
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.presets import banana_config, banana_dt
from blademaster_warp.solver import Solver

INTERVAL, SPEED, START, DEPTHS = 0.001, 0.02, 0.03, (0.002, 0.004, 0.006)


def indent(spacing, seed, hold=0.0, maxwell_tau=0.0, maxwell_fraction=0.0, speed=SPEED):
    base = banana_config()
    cfg = replace(
        base,
        grid_spacing=spacing,
        dt=banana_dt(base, spacing, INTERVAL),
        sampling_seed=seed,
        external_tool=True,
        grid_storage="compact",
        cache_topology=True,
        maxwell_tau=maxwell_tau,
        maxwell_fraction=maxwell_fraction,
    )
    s = Solver(cfg)
    port, half = ExternalToolPort(s), cfg.blade_half_size[2]
    # Contact starts when the edge is within the knife contact distance of a particle centre.
    top = s.reference[np.abs(s.reference[:, 0]) < spacing, 2].max() + s.blade_offset
    edges, forces = [], []
    edge = START
    while edge > top - max(DEPTHS):
        edge = START - speed * len(edges) * INTERVAL
        state = ToolState([0.0, 0.0, edge + half], [0, 0, 0, 1], [0.0, 0.0, -speed], [0, 0, 0])
        forces.append(port.advance(state, INTERVAL, cutting=False).force_n[2])
        edges.append(edge)
    depth = top - np.array(edges)
    held = []
    state = ToolState([0.0, 0.0, edge + half], [0, 0, 0, 1], [0.0, 0.0, 0.0], [0, 0, 0])
    for _ in range(round(hold / INTERVAL)):
        held.append(port.advance(state, INTERVAL, cutting=False).force_n[2])
    s.state()
    return [float(np.interp(d, depth, forces)) for d in DEPTHS], top, float(depth[-1]), np.array(held)


def fit_relaxation(force, skip=0.02):
    """Fit F_inf + (F_0 - F_inf) exp(-t/tau) after the first `skip` seconds of the hold."""
    t = (np.arange(len(force)) + 1) * INTERVAL
    keep = t > skip
    t, f = t[keep] - skip, force[keep]

    def model(t, f_inf, f_0, tau):
        return f_inf + (f_0 - f_inf) * np.exp(-t / tau)

    guess = (f[-1], f[0], t[-1] / 5)
    (f_inf, f_0, tau), _ = curve_fit(model, t, f, p0=guess, maxfev=20000)
    rms = float(np.sqrt(np.mean((model(t, f_inf, f_0, tau) - f) ** 2)))
    return dict(f_0=float(f_0), f_inf=float(f_inf), tau_s=float(tau), rms_n=rms, skip_s=skip)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spacings", type=float, nargs="+", default=[0.006, 0.004, 0.003, 0.002])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--speed", type=float, default=SPEED, help="Indentation speed, m/s")
    p.add_argument("--hold", type=float, default=0.0, help="Hold at the final depth, s")
    p.add_argument("--maxwell-tau", type=float, default=0.0, help="Config.maxwell_tau, s")
    p.add_argument("--maxwell-fraction", type=float, default=0.0, help="Config.maxwell_fraction")
    p.add_argument("--output", help="Optional JSON path with the hold force traces")
    a = p.parse_args()
    results = []
    for spacing in a.spacings:
        rows = [
            indent(spacing, seed, a.hold, a.maxwell_tau, a.maxwell_fraction, a.speed)
            for seed in range(1, a.seeds + 1)
        ]
        f = np.array([r[0] for r in rows])
        print(
            f"{spacing * 1000:g} mm: top {1000 * np.mean([r[1] for r in rows]):.1f} mm, force at "
            + ", ".join(f"{1000 * d:g} mm {m:.2f}±{s:.2f} N" for d, m, s in zip(DEPTHS, f.mean(0), f.std(0))),
            flush=True,
        )
        for seed, (_, _, depth, held) in enumerate(rows, start=1):
            if not len(held):
                continue
            fit = fit_relaxation(held)
            results.append(dict(spacing=spacing, seed=seed, depth_m=depth, hold_force_n=held.tolist(), **fit))
            print(
                f"  seed {seed}: hold at {1000 * depth:.2f} mm, force {held[0]:.3f} -> {held[-1]:.3f} N, "
                f"fit F0 {fit['f_0']:.3f} N, F_inf {fit['f_inf']:.3f} N "
                f"(ratio {fit['f_inf'] / fit['f_0']:.3f}), tau {fit['tau_s']:.3f} s, "
                f"rms {fit['rms_n']:.3f} N",
                flush=True,
            )
    if a.output:
        Path(a.output).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
