"""Paper Fig. 6: three free knives launched into one block at 0.5, 0.8 and 1.1 m/s."""

import argparse
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
from .solver import Config, Solver
from .run import write
from .topology import components
from .provenance import git_revision
from .scenarios import FIG8_SIZE, FIG8_KNIFE_HALF_SIZE
from .geometry import DEMO_KNIFE_BEVEL_HEIGHT_M

SPEEDS = (0.5, 0.8, 1.1)
KNIFE_X = (-0.03, 0.0, 0.03)
# Not given in the paper (Fig. 3's knife is 0.1 kg).
KNIFE_MASS = 0.1


def config(dt, cutting_energy, cut_work="relative", cut_hold=False, knife_mass=KNIFE_MASS):
    return Config(
        size=FIG8_SIZE,
        grid_spacing=0.006,
        dt=dt,
        young_modulus=20000.0,
        poisson_ratio=0.4,
        density=1200.0,
        yield_stress=2000.0,
        cutting_energy=cutting_energy,
        cut_work=cut_work,
        cut_hold=cut_hold,
        # Particle radius: a free knife between two 3 mm-apart particle columns must
        # still touch them (see Solver's slipping warning).
        blade_contact_radius=0.0,
        blade_mass=knife_mass,
        blade_half_size=FIG8_KNIFE_HALF_SIZE,
        blade_bevel_height=DEMO_KNIFE_BEVEL_HEIGHT_M,
        free_blade=True,
        tool_count=len(SPEEDS),
        grid_storage="compact",
    )


def edge_height(position, orientation, half):
    """Lowest edge point height and tilt from vertical, per frame and knife (xyzw quaternions)."""
    position, orientation = np.asarray(position), np.asarray(orientation)
    rotation = Rotation.from_quat(orientation.reshape(-1, 4))
    ends = [rotation.apply([0.0, y, -half[2]])[:, 2] for y in (-half[1], half[1])]
    lowest = np.minimum(*ends).reshape(position.shape[:2]) + position[..., 2]
    up = rotation.apply([0.0, 0.0, 1.0])[:, 2].reshape(position.shape[:2])
    return lowest, np.degrees(np.arccos(np.clip(up, -1.0, 1.0)))


def run(
    output,
    dt=0.0005,
    duration=0.5,
    cutting_energy=100.0,
    cut_work="relative",
    cut_hold=False,
    knife_mass=KNIFE_MASS,
):
    cfg = config(dt, cutting_energy, cut_work, cut_hold, knife_mass)
    steps = round(duration / dt)
    if steps < 1 or not np.isclose(steps * dt, duration):
        raise ValueError("Invalid duration")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    hz = cfg.blade_half_size[2]
    start = [[x, 0.0, FIG8_SIZE[2] + 0.002 + hz] for x in KNIFE_X]
    velocity = [[0.0, 0.0, -v] for v in SPEEDS]

    def place(s):
        s.tool_pos.assign(start)
        s.tool_vel.assign(velocity)

    warm = Solver(cfg)
    place(warm)
    warm.advance(np.zeros((1, 3)), cutting=True)
    warm.state()
    del warm
    s = Solver(cfg)
    place(s)
    meta = {
        "status": "running",
        "experiment": "fig6_moving_knives",
        "config": asdict(cfg),
        "particles": s.n,
        "device": s.device.name,
        "published_values": [
            "E=20kPa",
            "nu=0.4",
            "rho=1200kg/m3",
            "yield stress 2kPa (von Mises)",
            "initial speeds 0.5, 0.8, 1.1 m/s, left to right",
            "Table I: 7922 particles, dt=1ms, 3 cuts",
        ],
        "independent_choices": [
            "block 100x50x42mm on a 6mm grid",
            "knife geometry and 0.1kg mass",
            "vertical launch 2mm above the block, 30mm apart",
            "Gc",
            "contact/transfer coefficients",
        ],
        "revision": git_revision(),
    }
    saved = {k: [v] for k, v in s.state().items()}
    physics = 0.0
    try:
        stride = max(1, round(0.005 / dt))
        for i in range(0, steps, stride):
            count = min(stride, steps - i)
            tick = time.perf_counter()
            s.advance(np.zeros((count, 3)), cutting=True)
            wp.synchronize_device(s.device)
            physics += time.perf_counter() - tick
            for k, v in s.state().items():
                saved[k].append(v)
        meta.update(status="complete")
    except Exception as exc:
        meta.update(status="failed", error=repr(exc))
        raise
    finally:
        simulated = s.step_count * dt
        meta.update(
            completed_steps=s.step_count,
            physics_wall_s=physics,
            physics_wall_per_sim=physics / simulated if simulated else None,
            graph_build_s=s.graph_build_s,
        )
        np.savez_compressed(
            out / "states.npz", **{k: np.asarray(v) for k, v in saved.items()}, reference=s.reference
        )
        write(out / "manifest.json", meta)
    t = np.asarray(saved["time"])
    edge, tilt = edge_height(saved["blade_position"], saved["blade_orientation"], cfg.blade_half_size)
    speed = np.asarray(saved["blade_velocity"])[:, :, 2]
    _, counts = components(s.node_numpy, s.weight_numpy, saved["sides"][-1])
    knives = []
    for k, v0 in enumerate(SPEEDS):
        moving = np.nonzero(np.abs(speed[:, k]) > 0.01)[0]
        knives.append(
            dict(
                initial_speed_m_s=v0,
                max_penetration_m=float(FIG8_SIZE[2] - edge[:, k].min()),
                final_edge_height_m=float(edge[-1, k]),
                final_tilt_deg=float(tilt[-1, k]),
                final_speed_m_s=float(speed[-1, k]),
                stopped_at_s=float(t[moving[-1] + 1]) if len(moving) and moving[-1] + 1 < len(t) else None,
            )
        )
    result = dict(
        status="complete",
        component_sizes=counts.tolist(),
        knives=knives,
        physics_wall_per_sim=meta["physics_wall_per_sim"],
    )
    write(out / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--dt", type=float, default=0.0005)
    p.add_argument("--duration", type=float, default=0.5)
    p.add_argument("--cutting-energy", type=float, default=100.0, help="Gc in J/m^2; not given in the paper")
    p.add_argument("--cut-work", choices=("relative", "advance"), default="relative")
    p.add_argument("--cut-hold", action="store_true")
    p.add_argument("--knife-mass", type=float, default=KNIFE_MASS, help="kg; not given in the paper")
    a = p.parse_args()
    run(a.output, a.dt, a.duration, a.cutting_energy, a.cut_work, a.cut_hold, a.knife_mass)
