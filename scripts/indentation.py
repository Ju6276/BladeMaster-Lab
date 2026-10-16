"""Knife indentation without cutting: grid dependence of the banana's stiffness (volumetric locking).

The knife edge is pressed 6 mm into the banana top at 20 mm/s with cutting disabled.
Reported: vertical force at 2, 4 and 6 mm indentation, averaged over sampling seeds.

    python -m scripts.indentation --spacings 0.006 0.004 0.003 0.002 --seeds 3
"""

import argparse
from dataclasses import replace
import numpy as np
from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.presets import banana_config, banana_dt
from blademaster_warp.solver import Solver

INTERVAL, SPEED, START, DEPTHS = 0.001, 0.02, 0.03, (0.002, 0.004, 0.006)


def indent(spacing, seed):
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
    port, half = ExternalToolPort(s), cfg.blade_half_size[2]
    # Contact starts when the edge is within the knife contact distance of a particle centre.
    top = s.reference[np.abs(s.reference[:, 0]) < spacing, 2].max() + s.blade_offset
    edges, forces = [], []
    edge = START
    while edge > top - max(DEPTHS):
        edge = START - SPEED * len(edges) * INTERVAL
        state = ToolState([0.0, 0.0, edge + half], [0, 0, 0, 1], [0.0, 0.0, -SPEED], [0, 0, 0])
        forces.append(port.advance(state, INTERVAL, cutting=False).force_n[2])
        edges.append(edge)
    depth = top - np.array(edges)
    return [float(np.interp(d, depth, forces)) for d in DEPTHS], top


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spacings", type=float, nargs="+", default=[0.006, 0.004, 0.003, 0.002])
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()
    for spacing in a.spacings:
        rows = [indent(spacing, seed) for seed in range(1, a.seeds + 1)]
        f = np.array([r[0] for r in rows])
        print(
            f"{spacing * 1000:g} mm: top {1000 * np.mean([r[1] for r in rows]):.1f} mm, force at "
            + ", ".join(f"{1000 * d:g} mm {m:.2f}±{s:.2f} N" for d, m, s in zip(DEPTHS, f.mean(0), f.std(0))),
            flush=True,
        )


if __name__ == "__main__":
    main()
