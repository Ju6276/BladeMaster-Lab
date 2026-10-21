"""Compare AD, finite differences and CMA-ES on the same bounded pushing task."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import warp as wp

from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.differentiable import ElasticTrajectory
from blademaster_warp.differentiable_contact import PrescribedContact, Tool
from blademaster_warp.provenance import git_revision
from blademaster_warp.solver import Solver
from scripts.diff_validate import configuration
from scripts.diff_contact_scan import trajectory_modes


@wp.kernel
def prescribed_motion(
    controls: wp.array(dtype=float), step: int, segment: int, dt: float,
    maximum_speed: float, origin: wp.vec3,
    position: wp.array(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3),
):
    displacement = float(0.0)
    for i in range(controls.shape[0]):
        elapsed = wp.clamp(step - i * segment, 0, segment)
        displacement += float(elapsed) * dt * maximum_speed * controls[i]
    position[0] = origin + wp.vec3(displacement, 0.0, 0.0)
    velocity[0] = wp.vec3(maximum_speed * controls[step // segment], 0.0, 0.0)


@wp.kernel
def target_objective(
    x: wp.array(dtype=wp.vec3), target: wp.array(dtype=wp.vec3), selected: wp.array(dtype=int),
    count: int, scale: float, loss: wp.array(dtype=float),
):
    p = wp.tid()
    if selected[p] != 0:
        wp.atomic_add(loss, 0, wp.length_sq(x[p] - target[p]) / (float(count) * scale * scale))


@wp.kernel
def force_objective(
    force: wp.array(dtype=wp.vec3), limit: float, steps: int, step: int,
    norms: wp.array(dtype=float), loss: wp.array(dtype=float),
):
    magnitude = wp.length(force[0])
    norms[step] = magnitude
    excess = wp.max(magnitude / limit - 1.0, 0.0)
    wp.atomic_add(loss, 0, 10.0 * excess * excess / float(steps))


class PushTask:
    def __init__(self, spacing=0.006, steps=128, controls=4, maximum_speed=0.2, force_limit=2.0):
        if steps < controls or controls < 2 or steps % controls:
            raise ValueError("steps must be a positive multiple of the control count, with at least two controls")
        self.config = replace(configuration(spacing), external_tool=True)
        self.steps = steps
        self.controls = controls
        self.maximum_speed = maximum_speed
        self.force_limit = force_limit
        self.scale = 0.001
        self.models = {flag: ElasticTrajectory(self.config, requires_grad=flag) for flag in (False, True)}
        model = self.models[False]
        self.origin = np.array([model.reference[:, 0].min() - self.config.blade_half_size[0] - spacing / 4 - 5e-5,
                                0.0, self.config.size[2] / 2])
        self.selected = model.reference[:, 0] <= np.quantile(model.reference[:, 0], 0.25)
        self.target = model.reference.copy()
        self.target_controls = np.resize(np.array([0.7, 0.5, 0.8, 0.4]), controls)
        self.target = self.evaluate(self.target_controls)[1]["positions"]

    def evaluate(self, values, gradient=False, details=False, audit=False):
        started = perf_counter()
        model = self.models[gradient]
        contact = PrescribedContact(model)
        parameters = model.array(values, float)
        target = model.array(self.target, wp.vec3, False)
        selected = model.array(self.selected.astype(np.int32), int, False)
        state = model.state()
        history = contact.history()
        loss = model.zeros(1, float)
        norms = model.zeros(self.steps, float)
        rotation = model.array([[0, 0, 0, 1]], wp.quat, False)
        omega = model.zeros(1, wp.vec3, False)
        positions, velocities = [], []
        modes = model.zeros((self.steps, model.n), int, False) if audit else None
        with wp.Tape() if gradient else nullcontext() as tape:
            for step in range(self.steps):
                position = model.zeros(1, wp.vec3)
                velocity = model.zeros(1, wp.vec3)
                wp.launch(prescribed_motion, 1,
                          [parameters, step, self.steps // self.controls, self.config.dt,
                           self.maximum_speed, wp.vec3(*self.origin)], [position, velocity], device=model.device)
                if audit:
                    cfg = self.config
                    sn, gk = cfg.contact_scales()
                    radius = cfg.grid_spacing / 4
                    offset = max(cfg.blade_contact_radius or radius, radius - cfg.blade_half_size[0])
                    wp.launch(trajectory_modes, model.n,
                              [state.x, state.v, history, position, velocity, wp.vec3(*cfg.blade_half_size),
                               cfg.blade_bevel_height, radius, offset, cfg.dt, sn, cfg.board_normal_scale or sn,
                               cfg.friction_static, gk, cfg.tangent_damping, step], [modes], device=model.device)
                state, history, reaction = contact.step(state, history, Tool(position, velocity, rotation, omega))
                wp.launch(force_objective, 1, [reaction, self.force_limit, self.steps, step],
                          [norms, loss], device=model.device)
                if details:
                    positions.append(position)
                    velocities.append(velocity)
            wp.launch(target_objective, model.n, [state.x, target, selected, int(self.selected.sum()), self.scale],
                      [loss], device=model.device)
        wp.synchronize_device(model.device)
        forward_end = perf_counter()
        derivative = None
        if gradient:
            tape.backward(loss)
            derivative = parameters.grad.numpy().astype(float)
        backward_end = perf_counter()
        x = state.x.numpy()
        value = float(loss.numpy()[0])
        result = dict(loss=value, position_rmse_m=float(np.sqrt(np.mean(np.sum(
            (x[self.selected].astype(float) - self.target[self.selected])**2, axis=1)))),
            peak_force_n=float(norms.numpy().max()), positions=x)
        if details:
            result.update(tool_positions=np.array([p.numpy()[0] for p in positions]),
                          tool_velocities=np.array([v.numpy()[0] for v in velocities]))
        if audit:
            result["contact_modes"] = modes.numpy()
        if not np.isfinite(value) or (derivative is not None and not np.isfinite(derivative).all()):
            raise FloatingPointError("Non-finite pushing objective or gradient")
        if gradient:
            tape.zero()
        result["timings"] = dict(forward_s=forward_end - started, backward_s=backward_end - forward_end,
                                 readback_cleanup_s=perf_counter() - backward_end)
        return derivative, result

    def gradient_check(self, values, epsilons=(0.003, 0.001, 0.0003, 0.0001)):
        derivative, base = self.evaluate(values, True)
        _, audited = self.evaluate(values, audit=True)
        rows = []
        for epsilon in epsilons:
            for i in range(self.controls):
                lo, hi = np.array(values), np.array(values)
                lo[i] -= epsilon
                hi[i] += epsilon
                _, low = self.evaluate(lo, audit=True)
                _, high = self.evaluate(hi, audit=True)
                fd = (high["loss"] - low["loss"]) / (2 * epsilon)
                rows.append(dict(control=i, epsilon=epsilon, ad=float(derivative[i]), fd=fd,
                                 left=(base["loss"] - low["loss"]) / epsilon,
                                 right=(high["loss"] - base["loss"]) / epsilon,
                                 changed_modes_low=int(np.count_nonzero(low["contact_modes"] != audited["contact_modes"])),
                                 changed_modes_high=int(np.count_nonzero(high["contact_modes"] != audited["contact_modes"]))))
        return rows

    def replay(self, values):
        _, expected = self.evaluate(values, details=True)
        solver = Solver(self.config)
        port = ExternalToolPort(solver)
        forces = []
        for position, velocity in zip(expected["tool_positions"], expected["tool_velocities"]):
            wrench = port.advance(ToolState(position, [0, 0, 0, 1], velocity, [0, 0, 0]),
                                  self.config.dt, cutting=False, graph=False)
            forces.append(np.linalg.norm(wrench.force_n))
        x = solver.x.numpy()
        return dict(position_max_abs_m=float(np.max(np.abs(x - expected["positions"]))),
                    position_rmse_m=float(np.sqrt(np.mean(np.sum(
                        (x[self.selected].astype(float) - self.target[self.selected])**2, axis=1)))),
                    peak_force_n=float(max(forces)), solver_errors=solver.errors.numpy().tolist())


def optimize(task, method, seed, budget, learning_rate, fd_step, tolerance):
    import cma

    initial = np.random.default_rng(seed).uniform(0.08, 0.35, task.controls)
    task.evaluate(initial, gradient=method == "ad")
    task.evaluate(initial)
    started = perf_counter()
    records = []
    best = None
    evaluations = 0
    backward_calls = 0
    gradient_seconds = 0.0
    phase_totals = dict(forward_s=0.0, backward_s=0.0, readback_cleanup_s=0.0)

    def evaluate(parameters, gradient=False):
        nonlocal evaluations, backward_calls, best
        derivative, result = task.evaluate(parameters, gradient)
        evaluations += 1
        backward_calls += int(gradient)
        for name, seconds in result["timings"].items():
            phase_totals[name] += seconds
        elapsed = perf_counter() - started
        row = dict(elapsed_s=elapsed, loss=result["loss"], position_rmse_m=result["position_rmse_m"],
                   peak_force_n=result["peak_force_n"], controls=np.asarray(parameters).tolist())
        if elapsed <= budget:
            records.append(row)
            if best is None or row["loss"] < best["loss"]:
                best = row
        return derivative, result["loss"]

    q = initial.copy()
    if method == "cma":
        strategy = cma.CMAEvolutionStrategy(q, 0.2, dict(
            bounds=[0.0, 1.0], seed=seed + 1, verbose=-9, verb_log=0,
        ))
        evaluate(q)
        while perf_counter() - started < budget and not strategy.stop():
            population = strategy.ask()
            scores = []
            for candidate in population:
                if perf_counter() - started >= budget:
                    break
                scores.append(evaluate(candidate)[1])
            if len(scores) == len(population):
                strategy.tell(population, scores)
    else:
        first = np.zeros_like(q)
        second = np.zeros_like(q)
        iteration = 0
        while perf_counter() - started < budget:
            gradient_start = perf_counter()
            if method == "ad":
                derivative, _ = evaluate(q, True)
            else:
                evaluate(q)
                derivative = np.zeros_like(q)
                completed = True
                for index in range(len(q)):
                    if perf_counter() - started >= budget:
                        completed = False
                        break
                    lo, hi = q.copy(), q.copy()
                    lo[index] = max(0.0, q[index] - fd_step)
                    hi[index] = min(1.0, q[index] + fd_step)
                    low = evaluate(lo)[1]
                    if perf_counter() - started >= budget:
                        completed = False
                        break
                    high = evaluate(hi)[1]
                    derivative[index] = (high - low) / (hi[index] - lo[index])
                if not completed:
                    gradient_seconds += perf_counter() - gradient_start
                    break
            gradient_seconds += perf_counter() - gradient_start
            iteration += 1
            first = 0.9 * first + 0.1 * derivative
            second = 0.999 * second + 0.001 * derivative**2
            q = np.clip(q - learning_rate * (first / (1 - 0.9**iteration)) /
                        (np.sqrt(second / (1 - 0.999**iteration)) + 1e-8), 0.0, 1.0)
    elapsed = perf_counter() - started
    if best is None:
        raise ValueError("Budget is shorter than one objective evaluation")
    successes = [row for row in records if row["position_rmse_m"] <= tolerance and row["peak_force_n"] <= task.force_limit]
    pool_peak = wp.get_mempool_used_mem_high("cuda:0")
    gradient_check = task.gradient_check(initial)
    replay = task.replay(best["controls"])
    if replay["position_max_abs_m"] > 1e-6 or any(replay["solver_errors"]):
        raise AssertionError(f"Original solver replay failed: {replay}")
    return dict(method=method, seed=seed, initial_controls=initial.tolist(), budget_s=budget,
                elapsed_s=elapsed, forward_evaluations=evaluations, backward_calls=backward_calls,
                gradient_evaluation_s=gradient_seconds, first_success_s=successes[0]["elapsed_s"] if successes else None,
                best=best, replay=replay, trace=records, peak_live_cuda_pool_bytes=pool_peak,
                cma_version=cma.__version__, initial_gradient_check=gradient_check, phase_totals=phase_totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=0.006)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--controls", type=int, default=4)
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+", choices=["ad", "fd", "cma"], default=["ad", "fd", "cma"])
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--fd-step", type=float, default=0.001)
    parser.add_argument("--tolerance", type=float, default=5e-5)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if min(args.budget, args.learning_rate, args.fd_step, args.tolerance) <= 0:
        parser.error("budget, learning rate, difference step and tolerance must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        wp.config.verify_autograd_array_access = True
        task = PushTask(args.spacing, args.steps, args.controls)
        report = optimize(task, args.methods[0], args.seeds[0], args.budget,
                          args.learning_rate, args.fd_step, args.tolerance)
        report.update(revision=git_revision(), command=sys.argv)
        report["config"] = asdict(task.config)
        report["task"] = dict(steps=task.steps, controls=task.controls, maximum_speed_m_s=task.maximum_speed,
                              force_limit_n=task.force_limit, target_controls=task.target_controls.tolist(),
                              selected_particles=int(task.selected.sum()), position_tolerance_m=args.tolerance,
                              objective_scale_m=task.scale, force_penalty_weight=10.0)
        np.save(args.output / "target.npy", task.target)
    else:
        report = dict(revision=git_revision(), command=sys.argv, warp_version=wp.__version__,
                      optimizer=dict(learning_rate=args.learning_rate, beta1=0.9, beta2=0.999,
                                     epsilon=1e-8, fd_step=args.fd_step, cma_sigma=0.2), runs=[])
        for seed in args.seeds:
            for method in args.methods:
                output = args.output / f"{method}_{seed}"
                output.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-m", "scripts.diff_optimize", "--worker", "--output", str(output),
                           "--spacing", str(args.spacing), "--steps", str(args.steps), "--controls", str(args.controls),
                           "--budget", str(args.budget), "--seeds", str(seed), "--methods", method,
                           "--learning-rate", str(args.learning_rate), "--fd-step", str(args.fd_step),
                           "--tolerance", str(args.tolerance)]
                with (output / "worker.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                row = json.loads((output / "report.json").read_text())
                report["runs"].append(row)
                (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(f"{method} seed={seed}: error={row['best']['position_rmse_m']*1e3:.5f} mm "
                      f"success={row['first_success_s']} evaluations={row['forward_evaluations']}", flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
