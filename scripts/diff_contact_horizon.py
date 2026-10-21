"""Compare full-tape and checkpointed pushing gradients in fresh CUDA processes."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import warp as wp

from blademaster_warp.provenance import git_revision
from scripts.diff_optimize import PushTask


def worker(args):
    wp.config.verify_autograd_array_access = True
    task = PushTask(args.spacings[0], args.steps[0], args.controls[0],
                    segment_steps=args.segment, tool_offset=args.tool_offset)
    if args.reference:
        task.target = np.load(args.reference / "target.npy")
    else:
        np.save(args.output / "target.npy", task.target)
    controls = np.resize([0.22, 0.28, 0.25, 0.2], task.controls)
    task.evaluate(controls, True)
    rows, gradients = [], []
    for _ in range(args.repeats):
        started = perf_counter()
        gradient, result = task.evaluate(controls, True)
        elapsed = perf_counter() - started
        gradients.append(gradient)
        rows.append(dict(loss=result["loss"], position_rmse_m=result["position_rmse_m"],
                         peak_force_n=result["peak_force_n"], timings=result["timings"], elapsed_s=elapsed))
    memory = wp.get_mempool_used_mem_high("cuda:0")
    np.savez(args.output / "values.npz", gradients=np.array(gradients), positions=result["positions"])
    reference = np.load((args.reference or args.output) / "values.npz")
    norm = max(float(np.linalg.norm(reference["gradients"][0])), 1e-10)
    differences = np.linalg.norm(np.array(gradients) - reference["gradients"][0], axis=1) / norm
    return dict(revision=git_revision(), command=sys.argv, warp_version=wp.__version__,
                config=asdict(task.config), particles=task.models[True].n, steps=task.steps,
                controls=task.controls, segment_steps=task.segment_steps,
                tool_offset_m=task.tool_offset.tolist(), control_values=controls.tolist(),
                peak_live_cuda_pool_bytes=memory, gradient_relative_l2=differences.tolist(),
                position_max_abs_m=float(np.max(np.abs(result["positions"] - reference["positions"]))),
                measurements=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spacings", type=float, nargs="+", default=[0.006, 0.004])
    parser.add_argument("--steps", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--controls", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--segments", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--tool-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--segment", type=int, default=0)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(segment < 1 for segment in args.segments):
        parser.error("repeats and checkpoint segment lengths must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        report = worker(args)
    else:
        report = dict(revision=git_revision(), command=sys.argv, cases=[])
        for spacing in args.spacings:
            for steps in args.steps:
                for controls in args.controls:
                    case = args.output / f"h{spacing:g}_n{steps}_q{controls}"
                    full = case / "full"
                    for segment in [0, *sorted(set(min(steps, count) for count in args.segments))]:
                        output = full if segment == 0 else case / f"segment{segment}"
                        output.mkdir(parents=True, exist_ok=True)
                        command = [sys.executable, "-m", "scripts.diff_contact_horizon", "--worker",
                                   "--output", str(output), "--spacings", str(spacing), "--steps", str(steps),
                                   "--controls", str(controls), "--segment", str(segment),
                                   "--repeats", str(args.repeats), "--tool-offset", *map(str, args.tool_offset)]
                        if segment:
                            command.extend(["--reference", str(full)])
                        with (output / "worker.log").open("w") as log:
                            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                        row = json.loads((output / "report.json").read_text())
                        report["cases"].append(row)
                        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                        print(f"h={spacing:g} steps={steps} controls={controls} segment={segment}: "
                              f"gradient={max(row['gradient_relative_l2']):.3g} "
                              f"memory={row['peak_live_cuda_pool_bytes']/2**20:.2f} MiB", flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
