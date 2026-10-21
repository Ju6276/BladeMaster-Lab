"""CUDA forward/gradient checks for elastic TLMPM and prescribed rigid contact."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import numpy as np
import warp as wp

from blademaster_warp import material as m
from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.differentiable import ElasticTrajectory, State
from blademaster_warp.differentiable_contact import PrescribedContact
from blademaster_warp.provenance import git_revision
from blademaster_warp.solver import Config, Solver


@wp.kernel
def kinetic_objective(v: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):
    wp.atomic_add(loss, 0, wp.length_sq(v[wp.tid()]) / float(v.shape[0]))


@wp.kernel
def contact_objective(force: wp.array(dtype=wp.vec3), dt: float, loss: wp.array(dtype=float)):
    wp.atomic_add(loss, 0, dt * wp.dot(force[0], wp.vec3(0.3, 0.7, 0.2)))


def configuration(spacing):
    return Config(
        size=(0.024, 0.018, 0.018), grid_spacing=spacing, dt=0.0002,
        blade_grid_constraint=False,
    )


def initial_values(model):
    F = np.array([[1.04, 0.025, 0.0], [0.0, 0.98, 0.01], [0.0, 0.0, 1.01]])
    return (
        model.reference @ F.T + [0.0, 0.0, 0.06],
        model.reference @ np.diag([1.0, -0.7, 0.2]) + [0.02, -0.01, 0.0],
        np.tile(F, (model.n, 1, 1)),
    )


def relative_error(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def state_error(state, original):
    return {
        name: float(np.max(np.abs(getattr(state, name).numpy() - getattr(original, name).numpy())))
        for name in ("x", "v", "F")
    }


def elastic_case(spacing, steps):
    cfg = configuration(spacing)
    model = ElasticTrajectory(cfg)
    values = initial_values(model)
    initial = model.state(*values)
    loss = model.zeros(1, float)
    with wp.Tape() as tape:
        state = initial
        for _ in range(steps):
            state = model.step(state)
        wp.launch(kinetic_objective, model.n, inputs=[state.v], outputs=[loss], device=model.device)
    tape.backward(loss)
    ad = float(model.young.grad.numpy()[0])
    loss_value = float(loss.numpy()[0])
    terminal_velocity = state.v.numpy().astype(float)
    del tape

    original = Solver(cfg)
    for name, value in zip(("x", "v", "F"), values):
        getattr(original, name).assign(value)
    original.tool_pos.assign([[1.0, 1.0, 1.0]])
    original.launch(m.compute_stress, original.n, [
        original.F, original.Fp, original.mu, original.lam, original.stress, original.errors,
    ])
    for _ in range(steps):
        original.step(target_edge=(1.0, 1.0, 1.0), cutting=False)
    forward = state_error(state, original)
    fd_rows = []
    for fraction in (1e-2, 3e-3, 1e-3):
        epsilon = cfg.young_modulus * fraction
        losses = []
        for sign in (-1, 1):
            other = ElasticTrajectory(
                replace(cfg, young_modulus=cfg.young_modulus + sign * epsilon), requires_grad=False,
            )
            end = other.state(*values)
            for _ in range(steps):
                end = other.step(end)
            losses.append(float(np.mean(np.sum(end.v.numpy().astype(float) ** 2, axis=1))))
        fd = (losses[1] - losses[0]) / (2 * epsilon)
        fd_rows.append(dict(relative_step=fraction, gradient=fd, relative_error=relative_error(ad, fd)))
    seed = State(
        model.zeros(model.n, wp.vec3, requires_grad=False),
        model.array(2 * terminal_velocity / model.n, wp.vec3, requires_grad=False),
        model.zeros(model.n, wp.mat33, requires_grad=False),
    )
    segment_steps = min(7, steps)
    _, checkpoint = model.checkpoint_vjp(initial, steps, segment_steps, seed)
    result = dict(
        config=asdict(cfg), particles=model.n, steps=steps, loss=loss_value,
        gradient_E=ad, finite_difference=fd_rows, forward_max_abs=forward,
        checkpoint=dict(segment_steps=segment_steps, gradient_E=checkpoint,
                        relative_error=relative_error(ad, checkpoint)),
    )
    assert forward["x"] < 1e-6 and forward["v"] < 1e-4 and forward["F"] < 1e-5, result
    assert max(row["relative_error"] for row in fd_rows) < 0.02, result
    assert result["checkpoint"]["relative_error"] < 0.001, result
    return result


def contact_case(spacing, steps, speed):
    cfg = replace(configuration(spacing), external_tool=True)
    model = ElasticTrajectory(cfg)
    contact = PrescribedContact(model)
    position = np.array([
        model.reference[:, 0].min() - cfg.blade_half_size[0] - spacing / 4 + 0.0003,
        0.0, cfg.size[2] / 2,
    ])
    velocity = [0.02, speed, 0.0]
    tool = contact.tool(position, velocity)
    state, history = model.state(), contact.history()
    loss = model.zeros(1, float)
    force_trace = []
    with wp.Tape() as tape:
        for _ in range(steps):
            state, history, force = contact.step(state, history, tool)
            force_trace.append(force)
            wp.launch(contact_objective, 1, inputs=[force, cfg.dt], outputs=[loss], device=model.device)
    tape.backward(loss)
    ad = float(tool.position.grad.numpy()[0, 0])
    fd_rows = []
    for epsilon in (1e-5, 3e-6):
        losses = []
        for sign in (-1, 1):
            other = ElasticTrajectory(cfg, requires_grad=False)
            other_contact = PrescribedContact(other)
            shifted = position.copy()
            shifted[0] += sign * epsilon
            other_tool = other_contact.tool(shifted, velocity)
            end, h = other.state(), other_contact.history()
            total = 0.0
            for _ in range(steps):
                end, h, reaction = other_contact.step(end, h, other_tool)
                total += cfg.dt * float(reaction.numpy()[0] @ np.array([0.3, 0.7, 0.2]))
            losses.append(total)
        fd = (losses[1] - losses[0]) / (2 * epsilon)
        fd_rows.append(dict(step_m=epsilon, gradient=fd, relative_error=relative_error(ad, fd)))
    original = Solver(cfg)
    port = ExternalToolPort(original)
    force_error = 0.0
    for force in force_trace:
        expected = port.advance(
            ToolState(position, [0, 0, 0, 1], velocity, [0, 0, 0]),
            cfg.dt, cutting=False, graph=False,
        )
        force_error = max(force_error, float(np.max(np.abs(force.numpy()[0] - expected.force_n))))
    result = dict(
        config=asdict(cfg), particles=model.n, steps=steps, tangential_speed_m_s=speed,
        tool_position_m=position.tolist(), tool_velocity_m_s=velocity,
        loss=float(loss.numpy()[0]), gradient_tool_x=ad, finite_difference=fd_rows,
        forward_max_abs=state_error(state, original), force_max_abs_n=force_error,
    )
    assert force_error < 1e-4, result
    assert max(row["relative_error"] for row in fd_rows) < 0.02, result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacings", type=float, nargs="+", default=[0.006, 0.004])
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 12, 32])
    args = parser.parse_args()
    if min(args.steps) < 1:
        parser.error("steps must be positive")
    wp.config.verify_autograd_array_access = True
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(
        revision=git_revision(), command=sys.argv, warp_version=wp.__version__,
        device=wp.get_device("cuda:0").name, elastic=[], contact=[],
    )
    for spacing in args.spacings:
        for steps in args.steps:
            row = elastic_case(spacing, steps)
            report["elastic"].append(row)
            print(f"elastic h={spacing:g} steps={steps}: E gradient={row['gradient_E']:.6g}", flush=True)
        for speed in (0.01, 2.0):
            row = contact_case(spacing, 8, speed)
            report["contact"].append(row)
            print(f"contact h={spacing:g} speed={speed:g}: x gradient={row['gradient_tool_x']:.6g}", flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
