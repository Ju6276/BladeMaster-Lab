"""Run the Fig. 4 slice-lift and record blade-body intrusion at every robot step.

The saved 3 ms replay holds the knife still. This run starts from the normal
initial state, so it also counts crossings created when the measured pose
changes at the 1 ms coupling interval.

    python -m scripts.check_support_path --output runs/support-path
"""

import argparse
import json
from pathlib import Path

import numpy as np

from blademaster_warp.franka_run import FrankaCuttingRun, parser as run_parser


def frame(x, position, rotation, half, bevel):
    local = (np.asarray(x, float) - position) @ rotation
    _, hy, hz = half
    margin = np.minimum.reduce(
        [hy - np.abs(local[:, 1]), local[:, 2] + hz - bevel, hz - local[:, 2]]
    )
    return local[:, 0], margin


class SupportPathRun(FrankaCuttingRun):
    def __init__(self, args):
        super().__init__(args)
        self.samples = []
        self._sign = None
        self._inside = None
        self.pose_crossings = 0
        self.material_crossings = 0
        self.unique_pose = set()
        self.unique_material = set()
        self.embedded_steps = 0
        self.pose_embedded_steps = 0
        self.max_depth_m = 0.0
        self.max_pose_depth_m = 0.0
        self.max_embedded_run = 0
        self._embedded_run = 0
        self.max_kinetic_j = 0.0
        self.final_energy = None

    def _observe(self, state, phase):
        solver = self.solver
        rotation = np.asarray(state.orientation_xyzw, float)
        # scipy Rotation is not required: the port stores xyzw and MuJoCo's
        # state quaternion is already the blade orientation used by the solver.
        from scipy.spatial.transform import Rotation

        matrix = Rotation.from_quat(rotation).as_matrix()
        signed, margin = frame(
            solver.x.numpy(),
            state.position,
            matrix,
            solver.config.blade_half_size,
            solver.config.blade_bevel_height,
        )
        interior = margin > solver.radius
        sign = np.sign(signed).astype(np.int8)
        sign[~interior] = 0
        depth = solver.config.blade_half_size[0] - np.abs(signed)
        embedded = interior & (depth > 1.0e-6)
        if self._sign is not None:
            crossed = interior & (self._sign != 0) & (sign != 0) & (sign != self._sign)
            ids = np.flatnonzero(crossed)
            if phase == "pose":
                self.pose_crossings += int(ids.size)
                self.unique_pose.update(int(i) for i in ids)
            else:
                self.material_crossings += int(ids.size)
                self.unique_material.update(int(i) for i in ids)
        self._sign = sign
        self._inside = embedded
        if phase == "pose" and embedded.any():
            self.pose_embedded_steps += 1
            self.max_pose_depth_m = max(self.max_pose_depth_m, float(depth[embedded].max()))
        if phase == "material":
            if embedded.any():
                self.embedded_steps += 1
                self.max_depth_m = max(self.max_depth_m, float(depth[embedded].max()))
                self._embedded_run += 1
                self.max_embedded_run = max(self.max_embedded_run, self._embedded_run)
            else:
                self._embedded_run = 0
            if self.meter is not None and self.solver is not None:
                kinetic, gravity, elastic, speed, max_z, min_det = self.meter.read()
                self.max_kinetic_j = max(self.max_kinetic_j, float(kinetic))
                self.final_energy = dict(
                    time_s=float(self.robot.data.time),
                    kinetic_J=float(kinetic),
                    gravity_J=float(gravity),
                    elastic_diagnostic_J=float(elastic),
                    max_speed_m_s=float(speed),
                    directional_contact_work_J=float(self.solver.work.numpy()[0]),
                )

    def _exchange_wrench(self):
        if self.solver is None:
            return super()._exchange_wrench()
        state = self.robot.state().checked()
        self._observe(state, "pose")
        wrench = super()._exchange_wrench()
        self._observe(state, "material")
        return wrench

    def summary(self, report):
        task = report.get("task_evaluation") or {}
        return dict(
            status=report.get("status"),
            final_phase=report.get("final_phase"),
            task_success=task.get("success"),
            sim_s=report.get("sim_s"),
            wall_s=report.get("wall_s"),
            pose_update_crossing_events=self.pose_crossings,
            pose_update_particles=len(self.unique_pose),
            material_step_crossing_events=self.material_crossings,
            material_step_particles=len(self.unique_material),
            embedded_steps=self.embedded_steps,
            pose_embedded_steps=self.pose_embedded_steps,
            max_embedded_run_steps=self.max_embedded_run,
            max_intrusion_depth_mm=self.max_depth_m * 1000.0,
            max_pose_intrusion_depth_mm=self.max_pose_depth_m * 1000.0,
            peak_material_force_n=report.get("peak_material_force_n"),
            max_kinetic_J=self.max_kinetic_j,
            final_energy=self.final_energy,
            gpu_errors=report.get("gpu_errors"),
            task=task,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = SupportPathRun(
        run_parser().parse_args(
            ["--motion", "slice-lift", "--output", str(args.output)]
        )
    )
    report = run.execute()
    summary = run.summary(report)
    (args.output / "support_path.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    failed = (
        summary["status"] != "completed"
        or not summary["task_success"]
        or summary["pose_update_particles"]
        or summary["material_step_particles"]
        or summary["embedded_steps"]
        or any(summary["gpu_errors"] or [])
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
