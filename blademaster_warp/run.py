"""Standalone guided-blade runs (DexLab cut-push trajectory)."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import warp as wp
from .solver import Solver
from .presets import banana_config, DEFAULT_BANANA_PRESET
from .provenance import git_revision


def banana_target(t, push=True):
    """DexLab b955d85 cut-push trajectory, adapted under Apache-2.0."""
    cut = min(int(t // 2), 2)
    x = (-0.03, 0.0, 0.03)[cut]
    local = t - 2 * cut
    z = np.interp(local, [0, 0.4, 1.1, 1.3, 1.8, 3], [0.036, 0.036, -0.004, -0.004, 0.036, 0.036])
    cutting = 0.4 <= local < 1.3
    if t >= 5.3:
        times = [5.3, 5.5, 6.7, 7.3, 8.0]
        x = np.interp(
            t, times, [0.03, 0.03, 0.042 if push else 0.03, 0.042 if push else 0.03, 0.042 if push else 0.03]
        )
        z = np.interp(t, times, [-0.004, 0.002, 0.002, 0.036, 0.036])
        cutting = False
    return (float(x), 0.0, float(z)), cut, cutting


def preset(name):
    return banana_config(paper_material=name == "paper-material")


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run(
    out,
    name=DEFAULT_BANANA_PRESET,
    duration=8.0,
    push=True,
    cutting=True,
    dt=None,
    graph=True,
    time_scale=1.0,
    spacing=None,
):
    cfg = preset(name)
    if dt is not None:
        cfg = replace(cfg, dt=dt)
    if spacing is not None:
        cfg = replace(cfg, grid_spacing=spacing)
    if not np.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("Invalid trajectory time scale")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    steps = round(duration / cfg.dt)
    if steps <= 0 or not np.isclose(steps * cfg.dt, duration):
        raise ValueError("Invalid duration")
    # Warm compilation on a separate state, never advance the recorded initial state.
    warm = Solver(cfg)
    warm.step()
    warm.state()
    del warm
    s = Solver(cfg)
    s.tool_pos.assign([[-0.03, 0.0, 0.036 + cfg.blade_half_size[2]]])
    initial = s.state()
    manifest = {
        "status": "running",
        "preset": name,
        "config": asdict(cfg),
        "device": str(s.device),
        "device_name": s.device.name,
        "warp_version": wp.__version__,
        "particles": s.n,
        "simulation_duration_s": duration,
        "precision": "float32",
        "push": push,
        "cutting": cutting,
        "revision": git_revision(),
        "parameter_provenance": (
            "DexLab synthetic baseline"
            if name == "dexlab-baseline"
            else "paper E,nu,rho,yield,blade mass,friction,Gc,sharpness; independent step size, geometry, contact scales, return map and drive"
        ),
        "sampling_hz": 100.0,
        "max_cut_index": s.max_cut_index,
        "grid_hash_capacity": s.capacity,
        "pair_history_capacity": s.pair_capacity,
        "cuda_graph": graph,
        "trajectory_time_scale": time_scale,
    }
    write(out / "manifest.json", manifest)
    snapshots = {k: [v] for k, v in initial.items()}
    trace = []
    stride = max(1, round(0.01 / cfg.dt))
    start = time.perf_counter()
    min_det = 1.0
    previous_codes = initial["sides"]
    try:
        for begin in range(0, steps, stride):
            end = min(begin + stride, steps)
            commands = [banana_target(i * cfg.dt / time_scale, push) for i in range(begin, end)]
            a = 0
            while a < len(commands):
                target, cut, active = commands[a]
                b = a + 1
                while b < len(commands) and commands[b][1:] == (cut, active):
                    b += 1
                s.advance(
                    [v[0] for v in commands[a:b]], cutting=cutting and active, cut_index=cut, graph=graph
                )
                a = b
            step = end - 1
            if (step + 1) % stride == 0 or step + 1 == steps:
                state = s.state()
                for digit in (1, 3, 9):
                    old = (previous_codes // digit) % 3
                    new = (state["sides"] // digit) % 3
                    if np.any((old != 0) & (old != new)):
                        raise RuntimeError("Persistent cut history changed")
                previous_codes = state["sides"]
                for k, v in state.items():
                    snapshots[k].append(v)
                d = float(np.linalg.det(state["deformation"]).min())
                min_det = min(min_det, d)
                trace.append(
                    {
                        "time_s": state["time"],
                        "minimum_det_f": d,
                        "labelled_particles": int(np.count_nonzero(state["sides"])),
                        "blade_reaction_n": s.reaction.numpy()[0].tolist(),
                        "contact_work_j": float(s.work.numpy()[0]),
                    }
                )
                trace[-1]["blade_reaction_n"] = (np.asarray(trace[-1]["blade_reaction_n"]) / cfg.dt).tolist()
            if (step + 1) % max(1, round(0.5 / cfg.dt)) == 0:
                print(
                    json.dumps(
                        {
                            "time_s": (step + 1) * cfg.dt,
                            "min_det_f": min_det,
                            "wall_s": time.perf_counter() - start,
                        }
                    ),
                    flush=True,
                )
        wp.synchronize_device(s.device)
        manifest.update(
            status="complete",
            completed_steps=s.step_count,
            wall_s_excluding_jit=time.perf_counter() - start,
            graph_build_s=s.graph_build_s,
        )
    except Exception as e:
        manifest.update(
            status="failed",
            error=repr(e),
            completed_steps=s.step_count,
            wall_s_excluding_jit=time.perf_counter() - start,
        )
        raise
    finally:
        np.savez_compressed(
            out / "states.npz", **{k: np.asarray(v) for k, v in snapshots.items()}, reference=s.reference
        )
        write(out / "trace.json", trace)
        manifest["minimum_det_f_sampled"] = min_det
        manifest["artifacts_sha256"] = {
            p: hashlib.sha256((out / p).read_bytes()).hexdigest() for p in ["states.npz", "trace.json"]
        }
        write(out / "manifest.json", manifest)
    codes, counts = np.unique(snapshots["sides"][-1], return_counts=True)
    result = {
        "status": "complete",
        "wall_s": manifest["wall_s_excluding_jit"],
        "realtime_factor": duration / manifest["wall_s_excluding_jit"],
        "final_code_counts": dict(zip(map(str, codes), map(int, counts))),
        "minimum_det_f_sampled": min_det,
        "cut_history_persistent": True,
        "maximum_sampled_reaction_n": float(max(np.linalg.norm(row["blade_reaction_n"]) for row in trace)),
        "contact_work_j": float(s.work.numpy()[0]),
    }
    result["graph_build_s"] = s.graph_build_s
    result["warm_execution_s"] = result["wall_s"] - s.graph_build_s
    result["warm_realtime_factor"] = duration / result["warm_execution_s"]
    write(out / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--preset",
        choices=["dexlab-baseline", "paper-material"],
        default=DEFAULT_BANANA_PRESET,
        help="Default: paper-material; dexlab-baseline is the explicitly selected soft-material regression",
    )
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--dt", type=float)
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--no-cut", action="store_true")
    p.add_argument(
        "--graph", dest="graph", action="store_true", default=True, help="CUDA Graph is the default"
    )
    p.add_argument("--no-graph", dest="graph", action="store_false", help="Eager reference scheduling")
    p.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Slow the target protocol without changing material or forces",
    )
    p.add_argument("--spacing", type=float)
    a = p.parse_args()
    run(a.output, a.preset, a.duration, not a.no_push, not a.no_cut, a.dt, a.graph, a.time_scale, a.spacing)
