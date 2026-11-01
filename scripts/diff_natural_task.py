"""Run a natural knife trajectory from a complete JSON task specification."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np

from blademaster_warp.cohesive_rigid_contact import RigidContacts
from blademaster_warp.cohesive_stages import GAMMA
from blademaster_warp.natural_checkpoint import snapshot
from blademaster_warp.natural_energy import leaves
from blademaster_warp.natural_task import NaturalTask
from blademaster_warp.provenance import git_revision
from blademaster_warp.solver import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--gradient", action="store_true")
    parser.add_argument("--checkpoint-steps", type=int, default=0)
    args = parser.parse_args()
    if args.checkpoint_steps < 0:
        parser.error("Checkpoint length must be nonnegative")
    specification = json.loads(args.case.read_text())
    config = dict(specification["simulation"])
    if args.dt is not None:
        config["dt"] = args.dt
    task = NaturalTask(Config(**config), **specification["task"])
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(
        provenance=git_revision(),
        command=sys.argv,
        config=asdict(task.model.config),
        task=specification["task"],
        particles=task.model.n,
        steps=len(task.path.steps),
        trajectory=[],
    )
    started = perf_counter()

    def observe(index, state, history, tool, result):
        energy = task.energy(state, history, tool, result)
        impulse = np.zeros(3)
        peak = 0.0
        for segment in result.segments:
            for weight, stage in zip((1 - GAMMA, GAMMA), segment.step.material.contact_stages, strict=True):
                first = 0
                for batch in leaves(stage.batch):
                    if isinstance(batch, RigidContacts):
                        force = -stage.force.cpu()[first : first + batch.count].numpy().sum(axis=0)
                        impulse += weight * float(segment.duration.numpy()[0]) * force
                        peak = max(peak, float(np.linalg.norm(force)))
                    first += batch.count
        retained = result.history.cohesive
        maximum = retained.history.numpy()
        material = retained.material.numpy()
        failure = 2 * material[:, 1] / material[:, 0]
        area = retained.quadrature.area.numpy()
        row = dict(
            step=index + 1,
            phase=task.path.steps[index].phase,
            elapsed_s=perf_counter() - started,
            tool_position=result.tool.position.numpy()[0].tolist(),
            area_m2=float(area.sum()),
            fractured_area_m2=float(area[maximum >= failure].sum()),
            max_history_m=float(maximum.max(initial=0.0)),
            knife_impulse_ns=impulse.tolist(),
            knife_peak_force_n=peak,
            geometry_events=[(int(p), float(t.numpy()[0])) for p, t in result.geometry_events],
            contact_events=[(float(e.time.numpy()[0]), e.changes) for e in result.contact_events],
            energy=energy,
        )
        report["trajectory"].append(row)
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        with (args.output / "checkpoint.pkl").open("wb") as file:
            pickle.dump(snapshot(result.state, result.history, result.tool), file)
        print(json.dumps({k: v for k, v in row.items() if k != "energy"}), flush=True)

    gradient = args.gradient or args.checkpoint_steps > 0
    result = task.evaluate(
        gradient=gradient, segment_steps=args.checkpoint_steps, observer=None if gradient else observe
    )
    report.update(loss=result.loss, timings=result.timings)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    with (args.output / "terminal.pkl").open("wb") as file:
        pickle.dump(result.terminal, file)
    if gradient:
        np.savez(
            args.output / "gradients.npz",
            **result.parameter_gradients,
            **{"initial_" + k: v for k, v in result.initial_gradients.items()}
        )


if __name__ == "__main__":
    main()
