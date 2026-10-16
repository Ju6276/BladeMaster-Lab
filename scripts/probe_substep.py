"""Material substep cost with an external tool, per spacing / dt / pair search.

.venv/bin/python -m scripts.probe_substep --spacing .006 .003 --pair-search all hash
.venv/bin/python -m scripts.probe_substep --kernels   # per-kernel GPU time, knife inside the banana
"""

import argparse
from dataclasses import replace
import itertools
import time
import warp as wp
from blademaster_warp.presets import banana_config
from blademaster_warp.solver import Solver
from blademaster_warp.coupling import ExternalToolPort, ToolState


def probe(spacing, dt, pair_search, intervals=200, interval=0.001):
    cfg = replace(
        banana_config(),
        external_tool=True,
        grid_storage="compact",
        cache_topology=True,
        grid_spacing=spacing,
        dt=dt,
        pair_search=pair_search,
    )
    s = Solver(cfg)
    port = ExternalToolPort(s)
    state = ToolState(
        position=[0, 0, 0.08],
        orientation_xyzw=[0, 0, 0, 1],
        linear_velocity=[0, 0, -0.02],
        angular_velocity=[0, 0, 0],
    )
    for _ in range(3):
        port.advance(state, interval, cutting=False)
    wp.synchronize()
    start = time.perf_counter()
    for _ in range(intervals):
        port.advance(state, interval, cutting=False)
    wp.synchronize()
    elapsed = time.perf_counter() - start
    substeps = intervals * round(interval / dt)
    return dict(
        particles=s.n,
        all_pairs=s.all_pairs,
        substep_us=elapsed / substeps * 1e6,
        wall_per_sim=elapsed / (intervals * interval),
    )


def kernel_profile(spacing, dt, intervals=20, interval=0.001):
    """Average GPU time per substep of each kernel, with the knife edge inside the material."""
    cfg = replace(
        banana_config(),
        external_tool=True,
        grid_storage="compact",
        cache_topology=True,
        grid_spacing=spacing,
        dt=dt,
    )
    s = Solver(cfg)
    port = ExternalToolPort(s)
    state = ToolState(
        position=[0, 0, 0.012 + cfg.blade_half_size[2]],
        orientation_xyzw=[0, 0, 0, 1],
        linear_velocity=[0, 0, -0.02],
        angular_velocity=[0, 0, 0],
    )
    port.advance(state, interval, graph=False)
    wp.synchronize()
    wp.timing_begin(cuda_filter=wp.TIMING_KERNEL)
    for _ in range(intervals):
        port.advance(state, interval, graph=False)
    results = wp.timing_end()
    substeps = intervals * round(interval / dt)
    totals = {}
    for r in results:
        totals[r.name] = totals.get(r.name, 0.0) + r.elapsed
    print(f"spacing={spacing} dt={dt:g} particles={s.n}: GPU microseconds per substep")
    for name, ms in sorted(totals.items(), key=lambda item: -item[1]):
        print(f"  {ms * 1e3 / substeps:7.2f}  {name}")
    print(f"  {sum(totals.values()) * 1e3 / substeps:7.2f}  total")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spacing", type=float, nargs="+", default=[0.006])
    p.add_argument("--dt", type=float, nargs="+", default=[6.25e-5])
    p.add_argument("--pair-search", nargs="+", default=["auto"])
    p.add_argument("--kernels", action="store_true", help="Per-kernel GPU time instead of wall time")
    args = p.parse_args()
    wp.config.quiet = True
    if args.kernels:
        for spacing, dt in itertools.product(args.spacing, args.dt):
            kernel_profile(spacing, dt)
        return
    for spacing, dt, search in itertools.product(args.spacing, args.dt, args.pair_search):
        r = probe(spacing, dt, search)
        print(
            f"spacing={spacing} dt={dt:g} search={search} particles={r['particles']} "
            f"substep_us={r['substep_us']:.1f} material_wall/sim={r['wall_per_sim']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
