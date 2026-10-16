"""Paper Fig. 3: 0.1 kg knife released at 25 degrees, gravity and material reaction only."""

import argparse
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import warp as wp
from .solver import Config, Solver
from .run import write
from .topology import components
from .provenance import git_revision


def run(output, sharpness=0.9, dt=0.002, duration=1.0, sampling_seed=0):
    cfg = Config(
        size=(0.14, 0.10, 0.08),
        grid_spacing=0.01,
        dt=dt,
        young_modulus=6000.0,
        poisson_ratio=0.4,
        density=1100.0,
        blade_mass=0.1,
        blade_half_size=(0.0005, 0.06, 0.02),
        free_blade=True,
        sharpness=sharpness,
        sampling_seed=sampling_seed,
    )
    steps = round(duration / dt)
    if steps < 1 or not np.isclose(steps * dt, duration):
        raise ValueError("Invalid duration")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    warm = Solver(cfg)
    warm.step()
    warm.state()
    del warm
    s = Solver(cfg)
    angle = np.deg2rad(25.0)
    q = np.array([0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)])
    height = 0.08 + 0.015 + cfg.blade_half_size[2] * np.cos(angle) + cfg.blade_half_size[0] * np.sin(angle)
    s.tool_rot.assign([q])
    s.tool_pos.assign([[0.0, 0.0, height]])
    meta = {
        "status": "running",
        "experiment": "fig3_freefall",
        "config": asdict(cfg),
        "device": s.device.name,
        "published_values": [
            "E=6kPa",
            "nu=0.4",
            "rho=1100kg/m3",
            "knife_mass=0.1kg",
            "tilt=25deg",
            "Table I: 8960 particles, dt=2ms",
        ],
        "independent_choices": [
            "block size 140x100x80mm",
            "grid spacing 10mm",
            "blade geometry/inertia",
            "initial clearance 15mm",
            "sharpness/Gc",
            "contact/transfer coefficients",
        ],
        "dt_is_table_value": dt == 0.002,
        "actuation": "none; gravity plus material/board reaction",
        "surface": "particles only; reconstruction pending",
        "revision": git_revision(),
    }
    saved = {k: [v] for k, v in s.state().items()}
    start = time.perf_counter()
    minimum = 1.0
    try:
        stride = max(1, round(0.01 / dt))
        for i in range(0, steps, stride):
            count = min(stride, steps - i)
            # No driver uses these targets in free_blade mode.
            s.advance(np.zeros((count, 3)), cutting=True)
            state = s.state()
            minimum = min(minimum, float(np.linalg.det(state["deformation"]).min()))
            for k, v in state.items():
                saved[k].append(v)
        wp.synchronize_device(s.device)
        meta.update(status="complete", wall_s_excluding_jit=time.perf_counter() - start)
    except Exception as exc:
        meta.update(status="failed", error=repr(exc), wall_s_excluding_jit=time.perf_counter() - start)
        raise
    finally:
        meta.update(completed_steps=s.step_count, minimum_det_f_sampled=minimum)
        np.savez_compressed(
            out / "states.npz", **{k: np.asarray(v) for k, v in saved.items()}, reference=s.reference
        )
        write(out / "manifest.json", meta)
    _, counts = components(s.node_numpy, s.weight_numpy, saved["sides"][-1])
    qfinal = saved["blade_orientation"][-1][0]
    turn = 2 * np.arccos(np.clip(abs(np.dot(qfinal, q)), 0.0, 1.0))
    result = {
        "status": "complete",
        "component_sizes": counts.tolist(),
        "wall_s": meta["wall_s_excluding_jit"],
        "orientation_change_deg": float(np.rad2deg(turn)),
        "horizontal_displacement_m": saved["blade_position"][-1][0, :2].tolist(),
        "final_blade_height_m": float(saved["blade_position"][-1][0, 2]),
        "minimum_det_f_sampled": minimum,
    }
    write(out / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--sharpness", type=float, default=0.9)
    p.add_argument("--dt", type=float, default=0.002)
    p.add_argument("--duration", type=float, default=1.0)
    p.add_argument("--sampling-seed", type=int, default=0)
    a = p.parse_args()
    run(a.output, a.sharpness, a.dt, a.duration, a.sampling_seed)
