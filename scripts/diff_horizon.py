"""Measure elastic gradients, synchronized timings and live CUDA pool memory."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import warp as wp

from blademaster_warp.differentiable import ElasticTrajectory, State
from blademaster_warp.provenance import git_revision
from scripts.diff_validate import configuration, initial_values, kinetic_objective, relative_error


def timestamp():
    wp.synchronize_device("cuda:0")
    return perf_counter()


def forward(model, values, steps):
    state = model.state(*values)
    for _ in range(steps):
        state = model.step(state, gravity=(0, 0, 0))
    return state


def finite_difference(cfg, values, steps, gradients):
    rows = []
    for variable, index, direction, epsilons in (
        ("E", None, None, (cfg.young_modulus * 0.01, cfg.young_modulus * 0.003)),
        ("v", 1, np.tile([0.4, -0.2, 0.1], (len(values[0]), 1)), (1e-3, 3e-4)),
        ("F", 2, np.tile(np.diag([0.4, -0.2, 0.1]), (len(values[0]), 1, 1)), (1e-3, 3e-4)),
    ):
        ad = float(gradients["E"]) if index is None else float(np.sum(gradients[variable] * direction))
        for epsilon in epsilons:
            losses = []
            for sign in (-1, 1):
                perturbed = [value.copy() for value in values]
                other_cfg = cfg
                if index is None:
                    other_cfg = replace(cfg, young_modulus=cfg.young_modulus + sign * epsilon)
                else:
                    perturbed[index] += sign * epsilon * direction
                model = ElasticTrajectory(other_cfg, requires_grad=False)
                end = forward(model, perturbed, steps)
                losses.append(float(np.mean(np.sum(end.v.numpy().astype(float) ** 2, axis=1))))
            fd = (losses[1] - losses[0]) / (2 * epsilon)
            rows.append(dict(variable=variable, epsilon=epsilon, ad=ad, fd=fd,
                             absolute_error=abs(ad - fd), relative_error=relative_error(ad, fd)))
    return rows


def worker(args):
    wp.config.verify_autograd_array_access = True
    cfg = configuration(args.spacing)
    model = ElasticTrajectory(cfg)
    values = initial_values(model)
    warm_start = model.state(*values)
    with wp.Tape() as warm_tape:
        warm_end = model.step(warm_start, gravity=(0, 0, 0))
        warm_loss = model.zeros(1, float)
        wp.launch(kinetic_objective, model.n, [warm_end.v], [warm_loss], device=model.device)
    warm_tape.backward(warm_loss)
    warm_tape.zero()
    del warm_tape, warm_start, warm_end, warm_loss
    start = model.state(*values)
    timings = {}
    if args.segment == 0:
        begun = timestamp()
        with wp.Tape() as tape:
            end = start
            for _ in range(args.steps):
                end = model.step(end, gravity=(0, 0, 0))
            loss = model.zeros(1, float)
            wp.launch(kinetic_objective, model.n, [end.v], [loss], device=model.device)
        now = timestamp()
        timings["forward_s"] = now - begun
        tape.backward(loss)
        timings["backward_s"] = timestamp() - now
        timings["recompute_s"] = 0.0
        gradients = {name: getattr(start, name).grad.numpy() for name in ("x", "v", "F")}
        gradients["E"] = float(model.young.grad.numpy()[0])
        seed_values = 2 * end.v.numpy() / model.n
        np.save(args.output / "seed.npy", seed_values)
        loss_value = float(loss.numpy()[0])
        minimum_j = float(np.linalg.det(end.F.numpy()).min())
    else:
        seed_values = np.load(args.reference / "seed.npy")
        seed = State(model.zeros(model.n, wp.vec3, False), model.array(seed_values, wp.vec3, False),
                     model.zeros(model.n, wp.mat33, False))
        gradient, young = model.checkpoint_vjp(
            start, args.steps, args.segment, seed, gravity=(0, 0, 0), timings=timings,
        )
        gradients = {name: getattr(gradient, name).numpy() for name in ("x", "v", "F")}
        gradients["E"] = young
        loss_value = minimum_j = None
    peak = wp.get_mempool_used_mem_high(model.device)
    np.savez(args.output / "gradients.npz", **gradients)
    report = dict(revision=git_revision(), command=sys.argv, config=asdict(cfg),
                  gravity=[0, 0, 0], steps=args.steps, segment_steps=args.segment,
                  particles=model.n, timings=timings, peak_live_cuda_pool_bytes=peak,
                  loss=loss_value, terminal_minimum_J=minimum_j)
    if args.segment == 0:
        del tape
        report["finite_difference"] = finite_difference(cfg, values, args.steps, gradients)
    else:
        expected = np.load(args.reference / "gradients.npz")
        report["gradient_error"] = {
            name: dict(max_abs=float(np.max(np.abs(gradients[name] - expected[name]))),
                       relative_l2=float(np.linalg.norm(gradients[name] - expected[name]) /
                                         max(np.linalg.norm(expected[name]), 1e-30)))
            for name in ("x", "v", "F", "E")
        }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacings", type=float, nargs="+", default=[0.006, 0.004])
    parser.add_argument("--horizons", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--segments", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--spacing", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--segment", type=int, default=0)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    if min(args.horizons + args.segments) < 1 or min(args.spacings) <= 0:
        parser.error("spacings, horizons and segment lengths must be positive")
    report = dict(revision=git_revision(), command=sys.argv, warp_version=wp.__version__,
                  memory_metric="CUDA memory pool high-water used bytes in a fresh worker", cases=[])
    for spacing in args.spacings:
        for steps in args.horizons:
            reference = args.output / f"h{spacing:g}_n{steps}_full"
            for segment in [0] + sorted({min(s, steps) for s in args.segments}):
                destination = reference if segment == 0 else args.output / f"h{spacing:g}_n{steps}_s{segment}"
                command = [sys.executable, "-m", "scripts.diff_horizon", "--worker", "--spacing", str(spacing),
                           "--steps", str(steps), "--segment", str(segment), "--output", str(destination),
                           "--reference", str(reference)]
                destination.mkdir(parents=True, exist_ok=True)
                with (destination / "worker.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                row = json.loads((destination / "report.json").read_text())
                report["cases"].append(row)
                (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(f"h={spacing:g} steps={steps} segment={segment}: {row['timings']} "
                      f"pool={row['peak_live_cuda_pool_bytes'] / 2**20:.2f} MiB", flush=True)


if __name__ == "__main__":
    main()
