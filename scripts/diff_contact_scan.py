"""One-sided derivatives at predictive contact and Coulomb friction switches."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp

from blademaster_warp.contact import blade_query, penalty_impulse
from blademaster_warp.differentiable_cutting import contact_mode as cutting_contact_mode
from blademaster_warp.provenance import git_revision
from scripts.diff_validate import configuration, relative_error


@wp.func
def contact_mode(gap: float, relative: wp.vec3, normal: wp.vec3, history: wp.vec3,
                 dt: float, sn: float, mus: float, gk: float, gc: float):
    return cutting_contact_mode(gap, relative, normal, history, 1.0, dt, sn, mus, gk, gc, 0, 0) % 10


@wp.kernel
def trajectory_modes(
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), history: wp.array2d(dtype=wp.vec3),
    position: wp.array(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3),
    half: wp.vec3, bevel: float, radius: float, blade_offset: float,
    dt: float, sn: float, board_sn: float, mus: float, gk: float, gc: float,
    step: int, modes: wp.array2d(dtype=int),
):
    p = wp.tid()
    board = contact_mode(x[p][2] - radius, v[p], wp.vec3(0.0, 0.0, 1.0), history[0, p],
                         dt, board_sn, mus, gk, gc)
    closest, normal, distance = blade_query(x[p] - position[0], half, bevel)
    blade = contact_mode(distance - blade_offset, v[p] - velocity[0], normal, history[1, p],
                         dt, sn, mus, gk, gc)
    modes[step, p] = board + 3 * blade


@wp.kernel
def probe(
    parameters: wp.array(dtype=float), normal_velocity: float, mass: float,
    dt: float, sn: float, mus: float, mud: float, gk: float, gc: float,
    impulse: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float), mode: wp.array(dtype=int),
):
    gap = parameters[0]
    speed = parameters[1]
    relative = wp.vec3(normal_velocity, speed, 0.0)
    result, history = penalty_impulse(
        gap, relative, wp.vec3(1.0, 0.0, 0.0), mass, wp.vec3(0.0),
        dt, sn, mus, mud, gk, gc,
    )
    impulse[0] = result
    loss[0] = wp.dot(result, wp.vec3(0.3, 0.7, 0.2)) / dt
    mode[0] = contact_mode(gap, relative, wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0),
                           dt, sn, mus, gk, gc)


def evaluate(cfg, gap, speed, normal_velocity, derivative=False):
    parameters = wp.array([gap, speed], dtype=float, device="cuda:0", requires_grad=derivative)
    impulse = wp.zeros(1, dtype=wp.vec3, device="cuda:0", requires_grad=derivative)
    loss = wp.zeros(1, device="cuda:0", requires_grad=derivative)
    mode = wp.zeros(1, dtype=int, device="cuda:0")
    sn, gk = cfg.contact_scales()
    arguments = [parameters, normal_velocity, cfg.density * (cfg.grid_spacing / 2)**3,
                 cfg.dt, sn, cfg.friction_static, cfg.friction_dynamic, gk, cfg.tangent_damping]
    with wp.Tape() as tape:
        wp.launch(probe, 1, arguments, [impulse, loss, mode], device="cuda:0")
    gradient = None
    if derivative:
        tape.backward(loss)
        gradient = parameters.grad.numpy().astype(float)
    return float(loss.numpy()[0]), int(mode.numpy()[0]), gradient, impulse.numpy()[0] / cfg.dt


def scan_case(cfg, name, gap, speed, normal_velocity, variable, epsilons):
    value, branch, gradient, force = evaluate(cfg, gap, speed, normal_velocity, True)
    rows = []
    for epsilon in epsilons:
        shifted = []
        modes = []
        for sign in (-1, 1):
            p = [gap, speed]
            p[variable] += sign * epsilon
            result, mode, _, _ = evaluate(cfg, *p, normal_velocity)
            shifted.append(result)
            modes.append(mode)
        left = (value - shifted[0]) / epsilon
        right = (shifted[1] - value) / epsilon
        central = (shifted[1] - shifted[0]) / (2 * epsilon)
        rows.append(dict(epsilon=epsilon, left=left, right=right, central=central,
                         modes=[modes[0], branch, modes[1]], same_branch=modes[0] == branch == modes[1],
                         relative_error=relative_error(float(gradient[variable]), central)))
    return dict(name=name, gap_m=gap, tangential_speed_m_s=speed, normal_velocity_m_s=normal_velocity,
                variable=["gap_m", "speed_m_s"][variable], mode=branch, force_n=force.tolist(),
                gradient=float(gradient[variable]), finite_difference=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wp.config.verify_autograd_array_access = True
    report = dict(revision=git_revision(), command=sys.argv, warp_version=wp.__version__,
                  modes={0: "separated", 1: "stick", 2: "slip"}, cases=[])
    for spacing in (0.006, 0.004):
        cfg = configuration(spacing)
        cases = []
        for velocity in (-0.02, 0.0, 0.02):
            threshold = -cfg.dt * min(velocity, 0)
            for offset in (-3e-4, -4e-6, -1e-6, 0.0, 1e-6, 4e-6, 3e-4):
                cases.append(scan_case(cfg, "normal_switch", threshold + offset, 0.01, velocity,
                                       0, (1e-6, 3e-7)))
        gap = -0.0003
        sn, gk = cfg.contact_scales()
        critical_speed = cfg.friction_static * sn * (-gap) / ((gk + 2 * cfg.tangent_damping * np.sqrt(gk)) * cfg.dt)
        for factor in (0.0, 0.5, 0.99, 0.9999, 1.0, 1.0001, 1.01, 2.0):
            cases.append(scan_case(cfg, "friction_switch", gap, critical_speed * factor, 0.0,
                                   1, (critical_speed * 1e-3, critical_speed * 1e-4)))
        report["cases"].append(dict(config=asdict(cfg), critical_speed_m_s=critical_speed, rows=cases))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    all_rows = [fd for case in report["cases"] for row in case["rows"] for fd in row["finite_difference"]]
    print(f"{len(all_rows)} differences; {sum(r['same_branch'] for r in all_rows)} within one branch")


if __name__ == "__main__":
    main()
