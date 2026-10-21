"""Replay cut windows from the original solver and check fixed-branch gradients."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from functools import partial
import sys
from time import perf_counter

import numpy as np
import warp as wp

from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.differentiable_contact import Tool
from blademaster_warp.differentiable_cutting import FixedCutContact, pair_snapshot
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.presets import banana_config
from blademaster_warp.provenance import git_revision
from blademaster_warp.solver import Solver


@wp.kernel
def knife_motion(
    controls: wp.array(dtype=float), origin: wp.vec3, elapsed: float,
    position: wp.array(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3),
):
    v = wp.vec3(0.0, 0.03 * controls[2], -0.12 * controls[1])
    position[0] = origin + wp.vec3(0.001 * controls[0], 0.0, 0.0) + elapsed * v
    velocity[0] = v


@wp.kernel
def reaction_loss(force: wp.array(dtype=wp.vec3), scale: float, loss: wp.array(dtype=float)):
    wp.atomic_add(loss, 0, scale * wp.dot(force[0], wp.vec3(0.3, 0.2, -1.0)))


@wp.kernel
def terminal_loss(v: wp.array(dtype=wp.vec3), Fp: wp.array(dtype=wp.mat33), loss: wp.array(dtype=float)):
    p = wp.tid()
    wp.atomic_add(loss, 0, (20.0 * wp.length_sq(v[p]) + 0.1 * wp.ddot(Fp[p], Fp[p])) / float(v.shape[0]))


def configuration(spacing, material):
    cfg = replace(banana_config(), grid_spacing=spacing, external_tool=True,
                  blade_grid_constraint=False, grid_storage="compact", cache_topology=True)
    if material in ("soft", "plastic"):
        cfg = replace(cfg, young_modulus=20000.0, poisson_ratio=0.4, density=1200.0,
                      yield_stress=2000.0, cutting_energy=10.0)
        if material == "plastic":
            cfg = replace(cfg, yield_stress=200.0)
    return cfg


def numpy_tool(origin, elapsed, controls):
    velocity = np.array([0.0, 0.03 * controls[2], -0.12 * controls[1]])
    position = origin + [0.001 * controls[0], 0.0, 0.0] + elapsed * velocity
    return ToolState(position, [0, 0, 0, 1], velocity, [0, 0, 0])


def differentiable_tool(model, controls, origin, elapsed):
    position, velocity = model.zeros(1, wp.vec3), model.zeros(1, wp.vec3)
    model.launch(knife_motion, 1, inputs=[controls, wp.vec3(*origin), elapsed],
              outputs=[position, velocity], device=model.device)
    return Tool(position, velocity, model.array([[0, 0, 0, 1]], wp.quat, False),
                model.zeros(1, wp.vec3, False))


def capture_window(cfg, prefix, steps, controls):
    original = Solver(cfg)
    port = ExternalToolPort(original)
    origin = np.array([0.0004, 0.0, 0.027 + cfg.blade_half_size[2]])
    for i in range(prefix):
        port.advance(numpy_tool(origin, i * cfg.dt, controls), cfg.dt, graph=False)
    values = {name: getattr(original, name).numpy() for name in ("x", "v", "F", "Fp", "stress")}
    keys, pair_history = pair_snapshot(original)
    history = dict(rigid=original.histories.numpy(), pairs=pair_history, damage=original.damage.numpy())
    if original.blade_constraint is not None:
        history["side"] = original.blade_constraint.side.numpy()
    rows = []
    for i in range(steps):
        codes = original.codes.numpy()
        released = original.damage.numpy() >= 1.0
        previous = keys
        reaction = port.advance(numpy_tool(origin, (prefix + i) * cfg.dt, controls), cfg.dt, graph=False)
        keys, pairs = pair_snapshot(original)
        batches = None
        if original.blade_constraint is not None:
            b = original.blade_constraint
            batches = int(b.batches.numpy()[0]) if b.needs_solve.numpy()[0] else 0
        rows.append(dict(grid_batches=batches, codes=codes, released=released, previous=previous, keys=keys,
                         force=reaction.force_n.copy(), pairs=pairs,
                         next_codes=original.codes.numpy(),
                         rigid=original.histories.numpy(), damage=original.damage.numpy(),
                         state={name: getattr(original, name).numpy() for name in values}))
    errors = original.errors.numpy().tolist()
    if any(errors):
        raise RuntimeError(f"Original solver errors: {errors}")
    return origin, values, history, rows, errors


def evaluate(cfg, origin, prefix, values, history, rows, controls, backward=False, friction=None,
             precision="float32", ordered_transfers=False):
    model = MaterialTrajectory(cfg, requires_grad=backward, precision=precision,
                               ordered_transfers=ordered_transfers)
    contact = FixedCutContact(model)
    if friction is not None:
        contact.friction.assign(friction)
    frozen = [contact.freeze(row["codes"], row["keys"], row["previous"], row["released"],
                             cutting=row.get("cutting", True), rigid_modes=row.get("rigid_modes"),
                             pair_modes=row.get("pair_modes"), yielded=row.get("yielded"),
                             grid_batches=row.get("grid_batches")) for row in rows]
    q = model.array(controls, float)
    loss = model.zeros(1, float)
    initial = model.state(**values)
    initial_history = contact.history(**history)
    tape = wp.Tape()
    trace = []
    started = perf_counter()
    with tape:
        state, h = initial, initial_history
        for i, branch in enumerate(frozen):
            tool = differentiable_tool(model, q, origin, (prefix + i) * cfg.dt)
            state, h, force = contact.step(state, h, tool, branch)
            model.launch(reaction_loss, 1, inputs=[force, 1 / len(rows)], outputs=[loss], device=model.device)
            trace.append((state, h, force))
        model.launch(terminal_loss, model.n, inputs=[state.v, state.Fp], outputs=[loss], device=model.device)
    wp.synchronize_device(model.device)
    forward_s = perf_counter() - started
    gradients = None
    backward_s = 0.0
    if backward:
        started = perf_counter()
        tape.backward(loss)
        wp.synchronize_device(model.device)
        backward_s = perf_counter() - started
        gradients = dict(young_modulus=float(model.young.grad.numpy()[0]),
                         yield_stress=float(model.yield_stress.grad.numpy()[0]),
                         friction=contact.friction.grad.numpy().tolist(), controls=q.grad.numpy().tolist())
        if not all(np.isfinite(value).all() for value in gradients.values()):
            raise FloatingPointError(f"Nonfinite fixed-cut gradient: {gradients}")
    maximum = dict.fromkeys((*values, "rigid_history", "pair_history", "damage", "force"), 0.0)
    branch_modes = []
    contact_modes = []
    precise_loss = 0.0
    for (state, h, force), row in zip(trace, rows):
        precise_loss += float(force.numpy()[0].astype(float) @ np.array([0.3, 0.2, -1.0])) / len(rows)
        branch_modes.extend((h.rigid_modes.numpy().ravel(), h.pair_modes.numpy(), state.yielded.numpy()))
        contact_modes.append((h.rigid_modes.numpy(), h.pair_modes.numpy(), state.yielded.numpy()))
        for name in values:
            error = np.max(np.abs(getattr(state, name).numpy() - row["state"][name]))
            maximum[name] = max(maximum[name], float(error))
        for name, actual, expected in (("rigid_history", h.rigid, row["rigid"]),
                                       ("pair_history", h.pairs, row["pairs"]),
                                       ("damage", h.damage, row["damage"]), ("force", force, row["force"])):
            difference = np.abs(actual.numpy() - expected)
            maximum[name] = max(maximum[name], float(difference.max()) if difference.size else 0.0)
    precise_loss += float(np.mean(20 * np.sum(state.v.numpy().astype(float) ** 2, axis=1)
                                 + 0.1 * np.sum(state.Fp.numpy().astype(float) ** 2, axis=(1, 2))))
    return dict(loss=precise_loss, gradients=gradients, forward_max_abs=maximum,
                forward_s=forward_s, backward_s=backward_s, particles=model.n,
                branch_modes=np.concatenate(branch_modes), contact_modes=contact_modes,
                grid_batches=[h.grid_batches for _, h, _ in trace],
                grid_residual=[h.grid_residual for _, h, _ in trace])


def run_case(spacing, material, prefix, steps, natural_branches=False,
             control_epsilons=(0.01, 0.003, 0.001), parameters=None, repeats=1,
             grid_constraint=False, capture_path=None, precision="float32", ordered_transfers=False):
    evaluate_case = partial(evaluate, precision=precision, ordered_transfers=ordered_transfers)
    cfg = replace(configuration(spacing, material), blade_grid_constraint=grid_constraint)
    controls = np.array([0.0, 1.0, 0.4])
    origin, values, history, rows, errors = capture_window(cfg, prefix, steps, controls)
    recorded = evaluate_case(cfg, origin, prefix, values, history, rows, controls)
    if not natural_branches:
        for row, (rigid_modes, pair_modes, yielded) in zip(rows, recorded["contact_modes"]):
            row.update(rigid_modes=rigid_modes, pair_modes=pair_modes, yielded=yielded)
    if capture_path is not None:
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        saved = {f"initial_{name}": value for name, value in values.items()}
        saved.update({f"history_{name}": value for name, value in history.items()})
        saved['metadata'] = json.dumps(dict(config=asdict(cfg), origin=origin.tolist(), prefix=prefix,
                                            controls=controls.tolist(), steps=steps))
        for i, row in enumerate(rows):
            for name, value in row.items():
                if name == 'state':
                    saved.update({f"step_{i}_state_{key}": array for key, array in value.items()})
                elif value is not None:
                    saved[f"step_{i}_{name}"] = value
        np.savez_compressed(capture_path, **saved)
    base = evaluate_case(cfg, origin, prefix, values, history, rows, controls, backward=True)
    repeat_losses = [base["loss"]]
    repeat_gradients = [base["gradients"]]
    for _ in range(repeats - 1):
        repeated = evaluate_case(cfg, origin, prefix, values, history, rows, controls, backward=True)
        repeat_losses.append(repeated["loss"])
        repeat_gradients.append(repeated["gradients"])
    direction = np.array([0.7, 0.3, 0.4])
    friction = np.array([cfg.friction_static, cfg.friction_dynamic])
    checks = []
    for parameter in parameters or ("young_modulus", "yield_stress", "friction", "controls"):
        gradient = base["gradients"][parameter]
        if parameter == "controls":
            ad = float(np.dot(gradient, direction))
        elif parameter == "friction":
            ad = float(np.sum(gradient))
        else:
            ad = gradient * getattr(cfg, parameter)
        epsilons = control_epsilons if parameter == "controls" else (0.01, 0.003, 0.001)
        for epsilon in epsilons:
            losses = []
            changed = []
            changes_by_kind = []
            for sign in (-1, 1):
                other_cfg, other_q, other_friction = cfg, controls, friction
                if parameter in ("young_modulus", "yield_stress"):
                    other_cfg = replace(cfg, **{parameter: getattr(cfg, parameter) * (1 + sign * epsilon)})
                elif parameter == "controls":
                    other_q = controls + sign * epsilon * direction
                else:
                    other_friction = friction + sign * epsilon
                result = evaluate_case(other_cfg, origin, prefix, values, history, rows, other_q,
                                  friction=other_friction)
                losses.append(result["loss"])
                changed.append(int(np.count_nonzero(result["branch_modes"] != base["branch_modes"])))
                counts = dict(board=0, blade=0, active_blade_feature=0, cut_hold=0, cut_work=0,
                              pairs=0, yielding=0)
                for before, after in zip(base["contact_modes"], result["contact_modes"]):
                    for index, name in enumerate(("board", "blade", "active_blade_feature", "cut_hold", "cut_work")):
                        mask = before[0][index] != after[0][index]
                        if index == 2:
                            mask &= before[0][1] != 0
                        counts[name] += int(np.count_nonzero(mask))
                    counts["pairs"] += int(np.count_nonzero(before[1] != after[1]))
                    counts["yielding"] += int(np.count_nonzero(before[2] != after[2]))
                changes_by_kind.append(counts)
            fd = (losses[1] - losses[0]) / (2 * epsilon)
            checks.append(dict(parameter=parameter, epsilon=epsilon, ad=ad, fd=fd,
                               abs_error=abs(ad - fd), relative_error=abs(ad - fd) / max(abs(ad), abs(fd), 1e-12),
                               changed_branches=changed, changes_by_kind=changes_by_kind))
    del base["branch_modes"]
    del base["contact_modes"]
    return dict(config=asdict(cfg), material=material, prefix_steps=prefix, steps=steps,
                precision=precision, ordered_transfers=ordered_transfers,
                controls=controls.tolist(), local_branch_policy="natural" if natural_branches else "recorded",
                repeat_losses=repeat_losses, repeat_gradients=repeat_gradients,
                original_errors=errors, initial_coded_particles=int(np.count_nonzero(rows[0]["codes"])),
                side_changes=sum(int(np.count_nonzero(row["codes"] != row["next_codes"])) for row in rows),
                yielding_particles=[int(np.count_nonzero(modes[2])) for modes in recorded["contact_modes"]],
                active_pairs=[len(row["keys"]) for row in rows],
                released_particles=[int(np.count_nonzero(row["released"])) for row in rows],
                plastic_change=float(np.max(np.abs(rows[-1]["state"]["Fp"] - values["Fp"]))),
                **base, finite_difference=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacings", type=float, nargs="+", default=[0.006, 0.004])
    parser.add_argument("--materials", nargs="+", choices=["banana", "soft", "plastic"],
                        default=["banana", "soft", "plastic"])
    parser.add_argument("--prefixes", type=int, nargs="+", default=[1200, 2400])
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--natural-branches", action="store_true")
    parser.add_argument("--grid-constraint", action="store_true")
    parser.add_argument("--ordered-transfers", action="store_true")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float32")
    parser.add_argument("--control-epsilons", type=float, nargs="+", default=[0.01, 0.003, 0.001])
    parser.add_argument("--parameters", nargs="+", choices=["young_modulus", "yield_stress", "friction", "controls"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or min(args.prefixes) < 0:
        parser.error("Steps must be positive and prefixes nonnegative")
    if args.repeats < 1 or min(args.control_epsilons) <= 0:
        parser.error("Repeats and control epsilons must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(provenance=git_revision(), command=sys.argv, device="cuda:0", cases=[])
    for spacing in args.spacings:
        for material in args.materials:
            for prefix in args.prefixes:
                row = run_case(spacing, material, prefix, args.steps, args.natural_branches,
                               args.control_epsilons, args.parameters, args.repeats, args.grid_constraint,
                               args.output / "captures" / f"{spacing}_{material}_{prefix}.npz",
                               args.precision, args.ordered_transfers)
                report["cases"].append(row)
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(dict(spacing=spacing, material=material, prefix=prefix,
                                      forward=row["forward_max_abs"], gradients=row["gradients"])), flush=True)


if __name__ == "__main__":
    main()
