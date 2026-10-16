"""Effect of Newtonian viscosity on cutting force and post-cut ringing of the banana.

Same prescribed vertical cut as scripts/force_convergence.py, then the knife is
withdrawn at 0.1 m/s and the pieces are left alone for 1 s. Reported per viscosity:
mean force in the 0.4-1.4 s window, peak force, and the RMS particle speed after
withdrawal (how long the pieces keep wobbling).

    python -m scripts.viscosity_feel --spacing 0.003 --viscosities 0 2 5 20 50
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.presets import banana_config, banana_dt
from blademaster_warp.solver import Solver

INTERVAL = 0.001
DESCENT_S, HOLD_S, SPEED = 1.8, 0.2, 0.02
WITHDRAW_S, WITHDRAW_SPEED, REST_S = 0.4, 0.1, 1.0


def edge_command(t):
    """Edge height and vertical velocity of the knife at time t."""
    if t < DESCENT_S:
        return DESCENT_S * SPEED - SPEED * t, -SPEED
    if t < DESCENT_S + HOLD_S:
        return 0.0, 0.0
    if t < DESCENT_S + HOLD_S + WITHDRAW_S:
        return WITHDRAW_SPEED * (t - DESCENT_S - HOLD_S), WITHDRAW_SPEED
    return WITHDRAW_SPEED * WITHDRAW_S, 0.0


def run(spacing, viscosity, seed=1):
    base = banana_config()
    cfg = replace(
        base,
        grid_spacing=spacing,
        dt=banana_dt(base, spacing, INTERVAL),
        sampling_seed=seed,
        external_tool=True,
        grid_storage="compact",
        cache_topology=True,
        dynamic_viscosity=viscosity,
    )
    s = Solver(cfg)
    port, half = ExternalToolPort(s), cfg.blade_half_size[2]
    steps = round((DESCENT_S + HOLD_S + WITHDRAW_S + REST_S) / INTERVAL)
    force, speed = np.empty(steps), np.full(steps, np.nan)
    for i in range(steps):
        edge, v = edge_command(i * INTERVAL)
        state = ToolState([0.0, 0.0, edge + half], [0, 0, 0, 1], [0.0, 0.0, v], [0, 0, 0])
        force[i] = port.advance(state, INTERVAL, cutting=True).force_n[2]
        if i % 5 == 4:
            speed[i] = float(np.sqrt(np.mean(np.sum(s.v.numpy().astype(float) ** 2, axis=1))))
    t = (np.arange(steps) + 1) * INTERVAL
    window = (t > 0.4) & (t <= 1.4)
    after = DESCENT_S + HOLD_S + WITHDRAW_S
    rest = (t > after) & np.isfinite(speed)
    return dict(
        viscosity_pa_s=viscosity,
        spacing=spacing,
        steady_force_n=float(force[window].mean()),
        peak_force_n=float(force.max()),
        rms_speed_mm_s={
            f"{dt:g} s after withdrawal": float(
                1000 * speed[rest & (t > after + dt - 0.05) & (t <= after + dt)].mean()
            )
            for dt in (0.05, 0.2, 0.5, 1.0)
        },
        gpu_errors=s.errors.numpy().tolist(),
        speed_trace_mm_s=(1000 * speed[np.isfinite(speed)]).tolist(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spacing", type=float, default=0.003)
    p.add_argument("--viscosities", type=float, nargs="+", default=[0.0, 2.0, 5.0, 20.0, 50.0])
    p.add_argument("--output", help="Optional JSON path")
    a = p.parse_args()
    results = []
    for eta in a.viscosities:
        r = run(a.spacing, eta)
        results.append(r)
        print(
            f"eta={eta:g} Pa s: steady {r['steady_force_n']:.3f} N, peak {r['peak_force_n']:.2f} N, "
            f"RMS speed after withdrawal {', '.join(f'{k}: {v:.2f}' for k, v in r['rms_speed_mm_s'].items())} mm/s, "
            f"errors {r['gpu_errors']}",
            flush=True,
        )
    if a.output:
        Path(a.output).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
