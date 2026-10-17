"""Cut-history scenarios after paper Figs. 4, 7, 8 with a prescribed tool."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .coupling import ToolState, ExternalToolPort
from .geometry import DEMO_KNIFE_HALF_SIZE_M, DEMO_KNIFE_BEVEL_HEIGHT_M
from .provenance import git_revision

# Paper Fig. 4 cylinder: 160 mm x 56 mm on a 1 cm grid, about the 8960 particles of Table I.
FIG4_RADIUS = 0.08
FIG4_HEIGHT = 0.056
# Long enough that each radial stroke runs 10 mm past the axis and past the rim, so
# the two cuts overlap at the apex; 80 mm tall to pass through the full height.
FIG4_KNIFE_HALF_SIZE = (0.0005, 0.05, 0.04)
FIG4_CUT_ANGLES_DEG = (60.0, -60.0)
FIG4_CUT_CENTRE_RADIUS = 0.04


# Paper Fig. 7 block: 80 x 80 x 70 mm on a 1 cm grid is the 3584 particles of Table I.
FIG7_SIZE = (0.08, 0.08, 0.07)
# Settled top surface the depths are measured from: under gravity the top particle
# centres drop from 67.5 to 64.0 mm, plus the 2.5 mm particle radius.
FIG7_TOP = 0.0665
FIG7_ROLL_DEG = -30.0
FIG7_DEPTHS = {"fig7-shallow": 0.006, "fig7-deep": 0.020}


def _tip_offset(rotation):
    """Lowest blade corner relative to the tool origin, in world axes."""
    hx, hy, hz = FIG4_KNIFE_HALF_SIZE
    corners = np.array([[0.0, sy * hy, -hz] for sy in (-1, 1)])
    world = Rotation.from_euler("ZYX", rotation, degrees=True).apply(corners)
    return world[np.argmin(world[:, 2])]


def _score(t0, start, end, index, depth, speed=0.02, clearance=0.01):
    """Tilted knife whose tip plunges at start, follows the line to end and withdraws."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    direction = (end - start) / np.linalg.norm(end - start)
    # Knife length along the path, tip leading.
    rotation = [np.degrees(np.arctan2(-direction[0], direction[1])), 0.0, FIG7_ROLL_DEG]
    tip = _tip_offset(rotation)
    top = FIG7_TOP

    def at(xy, z):
        return (np.array([*xy, z]) - tip).tolist()

    t1 = t0 + 0.2
    t2 = t1 + (clearance + depth) / speed
    t3 = t2 + np.linalg.norm(end - start) / speed
    return [
        (t0, at(start, top + clearance), rotation, index),
        (t1, at(start, top + clearance), rotation, index),
        (t2, at(start, top - depth), rotation, index),
        (t3, at(end, top - depth), rotation, -1),
        (t3 + 1.0, at(end, top + clearance + 0.02), rotation, -1),
    ]


# Paper Fig. 8 block on a 6 mm grid: 7854 particles (Table I: 7922).
FIG8_SIZE = (0.10, 0.05, 0.042)
FIG8_KNIFE_HALF_SIZE = (0.0005, 0.04, 0.03)
FIG8_SLICE = 0.012
FIG8_CUTS = 4


def _slice(t0, x, index, speed=0.03, push=0.02):
    """Cut through at x, rise to 12 mm below the top and tip the slice over towards +x."""
    hz = FIG8_KNIFE_HALF_SIZE[2]
    height = FIG8_SIZE[2]
    top, bottom, tip_over = height + 0.01 + hz, 0.0003 + hz, height - 0.012 + hz
    t1 = t0 + 0.2
    t2 = t1 + (top - bottom) / speed
    t3 = t2 + 0.2
    t4 = t3 + 0.6
    t5 = t4 + push / 0.025
    t6 = t5 + 0.6
    return [
        (t0, [x, 0, top], [0, 0, 0], index),
        (t1, [x, 0, top], [0, 0, 0], index),
        (t2, [x, 0, bottom], [0, 0, 0], index),
        (t3, [x, 0, bottom], [0, 0, 0], -1),
        (t4, [x, 0, tip_over], [0, 0, 0], -1),
        (t5, [x + push, 0, tip_over], [0, 0, 0], -1),
        (t6, [x + push, 0, top + 0.01], [0, 0, 0], -1),
        (t6 + 0.8, [x + push, 0, top + 0.01], [0, 0, 0], -1),
    ]


def _radial_cut(t0, angle_deg, index, top, bottom, descend_s, hold_s=0.2, withdraw_s=1.0):
    """Vertical stroke whose knife length lies along the radius at angle_deg."""
    a = np.deg2rad(angle_deg)
    xy = [FIG4_CUT_CENTRE_RADIUS * np.cos(a), FIG4_CUT_CENTRE_RADIUS * np.sin(a)]
    rotation = [angle_deg - 90.0, 0, 0]
    t1 = t0 + 0.2
    t2 = t1 + descend_s
    t3 = t2 + hold_s
    return [
        (t0, [*xy, top], rotation, index),
        (t1, [*xy, top], rotation, index),
        (t2, [*xy, bottom], rotation, index),
        (t3, [*xy, bottom], rotation, -1),
        (t3 + withdraw_s, [*xy, top], rotation, -1),
    ]


def keyframes(name):
    # time, position, rotation-euler-z-y-x, cut event (-1 means passive contact)
    if name == "parallel":
        return [
            (0, [-0.012, 0, 0.06], [0, 0, 0], 0),
            (0.3, [-0.012, 0, 0.06], [0, 0, 0], 0),
            (2.3, [-0.012, 0, 0.0202], [0, 0, 0], 0),
            (2.5, [-0.012, 0, 0.0202], [0, 0, 0], -1),
            (3.5, [-0.012, 0, 0.07], [0, 0, 0], -1),
            (4.2, [0.012, 0, 0.07], [0, 0, 0], 1),
            (6.5, [0.012, 0, 0.0202], [0, 0, 0], 1),
            (6.7, [0.012, 0, 0.0202], [0, 0, 0], -1),
            (8, [0.012, 0, 0.07], [0, 0, 0], -1),
        ]
    if name == "cross-shallow":
        return [
            (0, [0, 0, 0.06], [0, 0, 0], 0),
            (2, [0, 0, 0.035], [0, 0, 0], 0),
            (2.2, [0, 0, 0.035], [0, 0, 0], -1),
            (3.2, [0, 0, 0.07], [0, 0, 0], -1),
            (4.2, [0, 0, 0.07], [90, 0, 0], 1),
            (6.2, [0, 0, 0.035], [90, 0, 0], 1),
            (6.4, [0, 0, 0.035], [90, 0, 0], -1),
            (7.5, [0, 0, 0.07], [90, 0, 0], -1),
        ]
    if name == "slice-lift":
        base = keyframes("parallel")
        return base + [
            (9, [0.065, 0, 0.07], [0, 0, 0], -1),
            (10, [0.065, 0, 0.05], [0, 90, 0], -1),
            (11, [0.065, 0, 0.003], [0, 90, 0], -1),
            (13, [0, 0, 0.003], [0, 90, 0], -1),
            (15, [0, 0, 0.035], [0, 90, 0], -1),
            (16, [0, 0, 0.035], [0, 90, 0], -1),
        ]
    if name == "fig4":
        hz = FIG4_KNIFE_HALF_SIZE[2]
        top = FIG4_HEIGHT + 0.01 + hz
        bottom = 0.0003 + hz
        descend = (top - bottom) / 0.04
        first = _radial_cut(0.0, FIG4_CUT_ANGLES_DEG[0], 0, top, bottom, descend)
        second = _radial_cut(first[-1][0] + 0.8, FIG4_CUT_ANGLES_DEG[1], 1, top, bottom, descend)
        # Flat knife, edge towards the axis; turned outside the rim, then slid in with its
        # top face at board level, under the slice only (|y| < 50 mm lies inside the
        # 120 degree slice for x > 29 mm), then lifted 40 mm. The prescribed knife does
        # not collide with the board.
        flat = [0, 90, 0]
        t = second[-1][0]
        outside, low = [0.15, 0, top], -FIG4_KNIFE_HALF_SIZE[0]
        return (
            first
            + second
            + [
                (t + 0.6, outside, second[-1][2], -1),
                (t + 1.2, outside, flat, -1),
                (t + 2.0, [0.15, 0, low], flat, -1),
                (t + 4.0, [0.07, 0, low], flat, -1),
                (t + 4.3, [0.07, 0, low], flat, -1),
                (t + 5.8, [0.07, 0, low + 0.04], flat, -1),
                (t + 6.8, [0.07, 0, low + 0.04], flat, -1),
            ]
        )
    if name in FIG7_DEPTHS:
        depth = FIG7_DEPTHS[name]
        first = _score(0.0, [-0.03, 0.0], [0.03, 0.0], 0, depth)
        second = _score(first[-1][0] + 1.0, [0.0, -0.03], [0.0, 0.03], 1, depth)
        return first + second + [(second[-1][0] + 0.5, second[-1][1], second[-1][2], -1)]
    if name == "fig8":
        keys = []
        for k in range(FIG8_CUTS):
            t0 = keys[-1][0] + 0.6 if keys else 0.0
            keys += _slice(t0, FIG8_SIZE[0] / 2 - (k + 1) * FIG8_SLICE, k)
        return keys + [(keys[-1][0] + 1.5, keys[-1][1], keys[-1][2], -1)]
    raise ValueError(name)


def scene_config(name, dt=None):
    """Material, geometry and knife of a scene; dt defaults to the paper's Table I step.

    Another dt keeps the contact stiffness of the default step (contact_reference_dt):
    with a fixed per-step factor s_n the stiffness s_n m / dt^2 would grow as 1/dt^2.
    """
    cfg = _scene_config(name)
    if dt is None or dt == cfg.dt:
        return cfg
    return replace(cfg, dt=dt, contact_reference_dt=cfg.dt)


def _scene_config(name, dt=None):
    from .solver import Config

    if name == "fig4":
        return Config(
            size=(2 * FIG4_RADIUS, 2 * FIG4_RADIUS, FIG4_HEIGHT),
            shape="cylinder",
            grid_spacing=0.01,
            dt=dt or 0.002,
            # Table I step. With s_n = 0.1 at 2 ms the bottom layer sinks ~4 mm into the
            # board (below the knife edge, the slice stays attached) and the slice weight
            # pushes particles through the flat knife; s_n = 1 removes the whole predicted
            # penetration each step.
            normal_scale=1.0,
            board_normal_scale=1.0,
            young_modulus=10000.0,
            poisson_ratio=0.4,
            density=1100.0,
            cutting_energy=10.0,
            # Particle radius (spacing / 4): with 5 mm particle spacing a 0.5 mm contact
            # distance lets the flat knife push particles past its mid-plane while lifting.
            blade_contact_radius=0.0,
            external_tool=True,
            grid_storage="compact",
            blade_half_size=FIG4_KNIFE_HALF_SIZE,
            blade_bevel_height=DEMO_KNIFE_BEVEL_HEIGHT_M,
        )
    if name in FIG7_DEPTHS:
        return Config(
            size=FIG7_SIZE,
            shape="block",
            grid_spacing=0.01,
            # Table I step and contact factors as Fig. 4.
            dt=dt or 0.002,
            normal_scale=1.0,
            board_normal_scale=1.0,
            young_modulus=6000.0,
            poisson_ratio=0.35,
            density=1100.0,
            cutting_energy=10.0,
            blade_contact_radius=0.0,
            external_tool=True,
            grid_storage="compact",
            blade_half_size=FIG4_KNIFE_HALF_SIZE,
            blade_bevel_height=DEMO_KNIFE_BEVEL_HEIGHT_M,
        )
    if name == "fig8":
        return Config(
            size=FIG8_SIZE,
            shape="block",
            grid_spacing=0.006,
            # Table I step, about the CFL limit of this material on a 6 mm grid; with
            # s_n = 0.1 only 2 of the 4 slices separate at the bottom (board penetration).
            dt=dt or 0.001,
            normal_scale=1.0,
            board_normal_scale=1.0,
            young_modulus=20000.0,
            poisson_ratio=0.4,
            density=1200.0,
            yield_stress=2000.0,
            cutting_energy=10.0,
            # Particle radius: with 0.5 mm the knife passes between particle columns.
            blade_contact_radius=0.0,
            external_tool=True,
            grid_storage="compact",
            blade_half_size=FIG8_KNIFE_HALF_SIZE,
            blade_bevel_height=DEMO_KNIFE_BEVEL_HEIGHT_M,
        )
    return Config(
        size=(0.06, 0.03, 0.03),
        grid_spacing=0.006,
        dt=dt or 0.00025,
        young_modulus=6000,
        poisson_ratio=0.35,
        external_tool=True,
        blade_half_size=DEMO_KNIFE_HALF_SIZE_M,
        blade_bevel_height=DEMO_KNIFE_BEVEL_HEIGHT_M,
    )


PROVENANCE = {
    "fig4": "Paper Fig. 4 E/nu/rho; local cylinder size, knife, path, Gc=10, 0.5 ms step and contact numerics",
    "fig7-shallow": "Paper Fig. 7 E/nu/rho; local block size, knife, roll, path, depth, Gc=10 and step",
    "fig7-deep": "Paper Fig. 7 E/nu/rho; local block size, knife, roll, path, depth, Gc=10 and step",
    "fig8": "Paper Fig. 6/8 E/nu/rho/yield; local block size, knife, slice width, tip-over push, Gc=10, step and contact",
}


def command_at(t, keys):
    times = np.array([k[0] for k in keys])
    j = min(np.searchsorted(times, t, side="right") - 1, len(keys) - 2)
    j = max(j, 0)
    a, b = keys[j : j + 2]
    duration = b[0] - a[0]
    u = np.clip((t - a[0]) / duration, 0, 1)
    p = (1 - u) * np.array(a[1]) + u * np.array(b[1])
    v = (np.array(b[1]) - a[1]) / duration
    rotations = Rotation.from_euler("ZYX", [a[2], b[2]], degrees=True)
    r = Slerp([0, 1], rotations)(u)
    w = rotations[0].apply((rotations[0].inv() * rotations[1]).as_rotvec()) / duration
    active = a[3] >= 0
    index = max(k[3] for k in keys[: j + 1])
    return ToolState(p, r.as_quat(), v, w), active, index


def run(args):
    from .solver import Solver
    from .topology import components
    from .diagnostics import Metrics
    from .surface_runtime import SurfaceRecorder

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = scene_config(args.scenario, args.dt)
    if getattr(args, "board_normal_scale", None) is not None:
        cfg = replace(cfg, board_normal_scale=args.board_normal_scale)
    if getattr(args, "normal_scale", None) is not None:
        cfg = replace(cfg, normal_scale=args.normal_scale)
    keys = keyframes(args.scenario)
    interval = max(0.002, cfg.dt)
    duration = keys[-1][0]
    warm = Solver(cfg)
    ExternalToolPort(warm).advance(command_at(0, keys)[0], interval)
    del warm
    s = Solver(cfg)
    port = ExternalToolPort(s)
    meter = Metrics(s)
    surface = (
        SurfaceRecorder(
            s,
            out / "surface",
            args.surface_spacing,
            getattr(args, "surface_embedding", "cpu"),
            getattr(args, "surface_ccd", "cpu"),
        )
        if args.surface
        else None
    )
    meta = {
        "scenario": args.scenario,
        "config": asdict(cfg),
        "particles": s.n,
        "keyframes": keys,
        "driver": "prescribed tool (infinite effective mass)",
        "parameter_provenance": PROVENANCE.get(
            args.scenario, "Fig.7 elastic E/nu/rho; local geometry, path, Gc=.5 and contact numerics"
        ),
        "single_field_after_withdrawal": bool(args.single_field),
        "revision": git_revision(),
        "no_presplit": True,
        "surface_interval_s": 0.04 if surface else None,
        "status": "running",
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    saved = {k: [] for k in s.state()}
    forces = []
    energy = []
    events = []
    minimum = 1.0
    status = "completed"
    error = None
    physics = 0.0
    begin = time.perf_counter()
    try:
        for i in range(round(duration / interval)):
            t = i * interval
            tool, active, index = command_at(t, keys)
            s.set_single_field(args.single_field and not active)
            tick = time.perf_counter()
            reaction = port.advance(tool, interval, cutting=active, cut_index=index)
            physics += time.perf_counter() - tick
            forces.append([t, *reaction.force_n, *reaction.torque_nm])
            if i % 20 == 0 or i == round(duration / interval) - 1:
                state = s.state()
                for k, v in state.items():
                    saved[k].append(v)
                energy.append([t, *meter.read()])
                minimum = min(minimum, energy[-1][-1])
                if surface:
                    surface.update(
                        t, state["position"], state["deformation"], state["sides"], tool, active, index
                    )
            if (i + 1) % 250 == 0:
                print(
                    f"{args.scenario} t={t+interval:.2f}s wall={time.perf_counter()-begin:.1f}s", flush=True
                )
            if i % 500 == 0:
                _, counts = components(s.node_numpy, s.weight_numpy, s.codes.numpy())
                events.append(dict(time_s=t, components=counts.tolist()))
    except Exception as exc:
        status = "failed"
        error = repr(exc)
    wall = time.perf_counter() - begin
    np.savez_compressed(
        out / "states.npz",
        **{k: np.asarray(v) for k, v in saved.items()},
        reference=s.reference,
        forces=np.array(forces),
        energy=np.array(energy),
    )
    _, counts = components(s.node_numpy, s.weight_numpy, s.codes.numpy())
    simulated = s.step_count * cfg.dt
    meta.update(
        status=status,
        error=error,
        wall_s=wall,
        completed_sim_s=simulated,
        realtime_ratio=simulated / wall,
        physics_wall_s=physics,
        physics_wall_per_sim=physics / simulated if simulated else None,
        graph_build_s=s.graph_build_s,
        components=counts.tolist(),
        events=events,
        min_det_F=float(minimum),
        surface_wall_s=surface.wall_s if surface else 0.0,
    )
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    if error:
        raise RuntimeError(error)


SCENARIOS = ["parallel", "cross-shallow", "slice-lift", "fig4", *FIG7_DEPTHS, "fig8"]

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--scenario", choices=SCENARIOS, default="parallel")
    p.add_argument("--dt", type=float, default=None, help="Material step; default per scene")
    p.add_argument(
        "--single-field",
        action="store_true",
        help="Paper Fig. 4 baseline: grid transfers ignore cut-side codes while the knife is not cutting",
    )
    p.add_argument("--normal-scale", type=float, default=None, help="Per-step contact penalty factor s_n")
    p.add_argument(
        "--board-normal-scale",
        type=float,
        default=None,
        help="Per-step board penalty factor in [0, 1]; 0 uses the particle contact factor",
    )
    p.add_argument("--surface", action="store_true")
    p.add_argument(
        "--surface-embedding",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Nodal field evaluation only; topology remains the CPU reference",
    )
    p.add_argument(
        "--surface-ccd",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Finite moving-segment intersection backend; Hermite/QEF/triangles remain CPU",
    )
    p.add_argument(
        "--surface-spacing",
        type=float,
        default=None,
        help="Surface grid in m; default half the MPM spacing, no effect on physics",
    )
    run(p.parse_args())
