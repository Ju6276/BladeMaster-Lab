"""Franka + TLMPM cutting run (offline or live browser display)."""

import argparse
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import time
import numpy as np
from .franka import Franka, HALF, BEVEL, SOURCE, BOARD_HALF_EXTENT
from .geometry import BANANA_DEMO_SIZE_M
from .provenance import git_revision
from .topology import components
from .cut_cycle import CutCycle, CycleSettings, evaluate_cut


def target_at(t, initial):
    z = np.interp(t, [0, 0.3, 2.3, 2.6, 4.6], [0, 0, -0.04, -0.04, 0.01])
    return initial + np.array([0.0, 0.0, z])


def material_config(args):
    from .solver import Config
    from .presets import banana_config

    # Shared banana material and assets. Only the rigid-body integration owner
    # changes; material/contact kernels and their numerical settings do not.
    cfg = (
        banana_config()
        if args.material == "paper-banana"
        else Config(
            size=(0.06, 0.03, 0.03),
            grid_spacing=0.0075,
            dt=0.00025,
            contact_reference_dt=0.00025,
            density=1100.0,
            young_modulus=6000.0,
            poisson_ratio=0.4,
            blade_mass=0.1,
            blade_half_size=HALF,
            blade_bevel_height=BEVEL,
        )
    )
    overrides = {
        name: getattr(args, name) for name in ("dt", "cutting_energy") if getattr(args, name) is not None
    }
    if args.spacing is not None:
        overrides["grid_spacing"] = args.spacing
    if getattr(args, "contact_scaling", "physical") == "per-step":
        overrides["contact_reference_dt"] = 0.0
    storage = getattr(args, "grid_storage", cfg.grid_storage)
    return replace(
        cfg,
        external_tool=True,
        board_half_extent=BOARD_HALF_EXTENT,
        grid_storage=storage,
        cache_topology=storage == "compact",
        pair_search=getattr(args, "pair_search", "auto"),
        tool_substep=getattr(args, "tool_substep", "hold"),
        **overrides,
    )


SERL_CONTROLLER_PARAMETERS = {
    "pos_gains": [100.0, 100.0, 100.0],
    "ori_gains": [2000.0, 2000.0, 200.0],
    "damping_ratio": 1.0,
    "nullspace_stiffness": 0.5,
    "max_pos_acceleration": None,
    "max_ori_acceleration": None,
}

CYCLE_TRACE_COLUMNS = [
    "time_s",
    "phase_index",
    "tool_target_x",
    "tool_target_y",
    "tool_target_z",
    "edge_end_1_height_m",
    "edge_end_2_height_m",
    "board_normal_force_n",
]


def run_manifest(args, robot, cfg, cycle):
    """Configuration, asset sources and code revision of one run."""
    import mujoco
    import warp as wp

    banana = args.material == "paper-banana"
    controller_source = (
        "libfranka/SOURCE.json" if args.controller == "libfranka" else "serl_controller/manifest.json"
    )
    if args.controller == "libfranka":
        controller_parameters = {"stiffness": robot.k.tolist(), "damping": robot.damping.tolist()}
    else:
        controller_parameters = SERL_CONTROLLER_PARAMETERS
    return {
        "revision": git_revision(),
        "robot": json.loads((SOURCE.parent.parent / "SOURCE.json").read_text()),
        "controller": args.controller,
        "controller_source": json.loads((SOURCE.parents[2] / controller_source).read_text()),
        "controller_parameters": controller_parameters,
        "mount": robot.attachment,
        "knife_geometry": {"half_size_m": list(HALF), "bevel_height_m": BEVEL},
        "grip_force_per_finger_n": robot.grip_force_per_finger if args.attachment == "grasp" else None,
        "geometry": "banana" if banana else "block",
        "domain_size_m": list(BANANA_DEMO_SIZE_M) if banana else list(cfg.size),
        "cycle_settings": asdict(cycle.settings) if cycle else None,
        "material_origin_world_m": robot.material_origin.tolist(),
        "config": asdict(cfg),
        "contact_scales_at_dt": list(cfg.contact_scales()),
        "args": vars(args),
        "versions": {"mujoco": mujoco.__version__, "warp": wp.__version__},
        "surface_embedding_backend": getattr(args, "surface_embedding", "cpu") if args.surface else None,
        "surface_ccd_backend": getattr(args, "surface_ccd", "cpu") if args.surface else None,
    }


@dataclass
class Recording:
    """Per-step logs, plus per-frame snapshots (kept only with --record)."""

    logs: list = field(default_factory=list)
    edge_heights: list = field(default_factory=list)
    cycle_trace: list = field(default_factory=list)
    task_observations: list = field(default_factory=list)
    energies: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    controls: list = field(default_factory=list)
    particles: list = field(default_factory=list)
    codes: list = field(default_factory=list)
    deformations: list = field(default_factory=list)
    velocities: list = field(default_factory=list)
    plastic_deformations: list = field(default_factory=list)

    def drop_snapshots(self):
        for name in ("frames", "controls", "particles", "codes", "deformations"):
            getattr(self, name).clear()


class FrankaCuttingRun:
    """Franka arm in MuJoCo, banana/block in TLMPM, exchanging wrench every robot step."""

    def __init__(self, args, observer=None):
        self.args = args
        self.observer = observer
        self.record = bool(getattr(args, "record", False))
        if (self.record or args.video) and not args.output:
            raise ValueError("--record/--video requires --output")
        if args.video and not self.record:
            raise ValueError("--video requires --record; live display does not use a video file")
        if not np.isfinite([args.duration, args.interval]).all() or min(args.duration, args.interval) <= 0:
            raise ValueError("Duration and timesteps must be finite and positive")
        self.out = Path(args.output) if args.output else None
        if self.out is not None:
            self.out.mkdir(parents=True, exist_ok=False)

        self.robot = Franka(
            args.interval,
            args.stiffness,
            controller=args.controller,
            attachment=args.attachment,
            grip_force_per_finger=args.grip_force_per_finger,
        )
        self.cycle = self._make_cycle()
        self.cfg = material_config(args)
        if self.out is not None:
            manifest = run_manifest(args, self.robot, self.cfg, self.cycle)
            (self.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
            if self.record:
                (self.out / "scene.xml").write_text(self.robot.xml)

        warm = time.perf_counter()
        self.solver, self.port = self._make_solver()
        self.warmup_s = time.perf_counter() - warm
        self.meter = None
        self.surface = None
        if self.solver:
            from .diagnostics import Metrics

            self.meter = Metrics(self.solver)
            if args.surface:
                self.surface = self._make_surface()

        self.data = Recording()
        self.phase_time = dict(robot_state=0.0, material=0.0, robot_step=0.0, observation=0.0)
        self.paused_s = 0.0
        self.cutting = False
        self._last_codes = None
        self._last_component_sizes = None

    def _make_cycle(self):
        if self.args.motion != "feedback":
            return None
        return CutCycle(
            self.robot.position,
            self.robot.material_origin[2],
            HALF[2],
            CycleSettings(
                descent_speed_m_s=self.args.descent_speed,
                withdrawal_speed_m_s=self.args.withdrawal_speed,
                position_tolerance_m=self.args.goal_tolerance,
            ),
        )

    def _make_solver(self):
        if self.args.no_material:
            return None, None
        from .solver import Solver
        from .coupling import ExternalToolPort

        # Compile every kernel on a throwaway solver so the measured run starts from fresh state.
        disposable = Solver(self.cfg)
        ExternalToolPort(disposable).advance(self.robot.state(), self.args.interval)
        del disposable
        solver = Solver(self.cfg)
        return solver, ExternalToolPort(solver)

    def _make_surface(self):
        from .surface_runtime import SurfaceRecorder

        return SurfaceRecorder(
            self.solver,
            self.out / "surface" if self.record else None,
            self.args.surface_spacing,
            getattr(self.args, "surface_embedding", "cpu"),
            getattr(self.args, "surface_ccd", "cpu"),
        )

    def _timed(self, phase, start):
        self.phase_time[phase] += time.perf_counter() - start

    def _targets(self, t):
        """Tool target from the cut cycle (or timed script) and the matching TCP target."""
        robot = self.robot
        if self.cycle is None:
            return None, target_at(t, robot.initial_position)
        tool_target = self.cycle.update(t, robot.position, robot.edge_points)
        # Grasp mode uses the measured tool-to-TCP offset; fixed mounting has zero offset.
        return tool_target, tool_target + robot.control_position - robot.position

    def _exchange_wrench(self):
        """Advance the material over one robot step and return the wrench it applies to the knife."""
        if not self.port:
            return np.zeros(6)
        tick = time.perf_counter()
        state = self.robot.state()
        self._timed("robot_state", tick)
        tick = time.perf_counter()
        self.cutting = not self.args.no_cutting and (self.cycle is None or self.cycle.phase == "descend")
        reaction = self.port.advance(state, self.args.interval, cutting=self.cutting)
        self._timed("material", tick)
        return np.r_[reaction.force_n, reaction.torque_nm]

    def _log_step(self, target, tool_target, wrench, requested, applied):
        robot, data = self.robot, self.data
        data.logs.append(
            np.r_[
                robot.data.time,
                target,
                robot.position,
                wrench,
                robot.data.qpos,
                robot.data.qvel,
                requested,
                robot.data.ctrl[:7],
                applied,
            ]
        )
        heights = robot.edge_points[:, 2] - robot.material_origin[2]
        data.edge_heights.append(heights)
        if self.cycle:
            data.cycle_trace.append(
                [
                    robot.data.time,
                    self.cycle.phases.index(self.cycle.phase),
                    *tool_target,
                    *heights,
                    robot.board_normal_force(),
                ]
            )
        return heights

    def _component_sizes(self, codes):
        """Connected material pieces; recomputed only when side codes changed."""
        if self._last_codes is None or not np.array_equal(codes, self._last_codes):
            _, sizes = components(self.solver.node_numpy, self.solver.weight_numpy, codes)
            self._last_codes = codes
            self._last_component_sizes = sizes.tolist()
        return self._last_component_sizes

    def _snapshot(self, heights, wrench, begin):
        """Frame sample: material readback, task observation, surface, live display."""
        robot, s, data = self.robot, self.solver, self.data
        data.frames.append(robot.data.qpos.copy())
        data.controls.append(
            np.r_[robot.data.time, robot.control_position, robot.position - robot.control_position]
        )
        if s:
            data.particles.append(s.x.numpy().copy())
            data.codes.append(s.codes.numpy().copy())
            data.deformations.append(s.F.numpy().copy())
            if self.record:
                data.velocities.append(s.v.numpy().copy())
                data.plastic_deformations.append(s.Fp.numpy().copy())
                data.energies.append(np.r_[robot.data.time, self.meter.read(), s.work.numpy()[0]])
            if self.cycle:
                data.task_observations.append(
                    dict(
                        time_s=float(robot.data.time),
                        phase=self.cycle.phase,
                        component_sizes=self._component_sizes(data.codes[-1]),
                        knife_clear_of_material=bool(
                            min(heights) > data.particles[-1][:, 2].max() + s.radius
                        ),
                    )
                )
            if self.surface:
                self.surface.update(
                    robot.data.time,
                    data.particles[-1],
                    data.deformations[-1],
                    data.codes[-1],
                    robot.state(),
                    self.cutting,
                )
        if self.observer:
            self.observer.publish(
                robot,
                s,
                self.surface,
                data.particles[-1] if s else None,
                data.codes[-1] if s else None,
                time.perf_counter() - begin - self.paused_s,
                self.cycle.phase if self.cycle else "timed",
                wrench,
            )
        if not self.record:
            # Live mode keeps only the current state, no history.
            data.drop_snapshots()

    def execute(self):
        args, robot = self.args, self.robot
        if self.observer:
            self.observer.initialize(robot, self.solver, self.surface)
            if self.surface:
                self.surface.wall_s = 0.0
                self.surface.timing_s = dict.fromkeys(self.surface.timing_s, 0.0)
        status, exception = "completed", None
        count = int(round(args.duration / args.interval))
        frame_stride = max(1, round(0.04 / args.interval))
        print_stride = max(1, round(0.5 / args.interval))
        begin = time.perf_counter()
        try:
            for i in range(count):
                if self.observer:
                    self.paused_s += self.observer.before_step()
                    if self.observer.stop_requested:
                        status = "stopped"
                        break
                tool_target, target = self._targets(i * args.interval)
                if self.cycle and self.cycle.done:
                    break
                wrench = self._exchange_wrench()
                tick = time.perf_counter()
                requested, applied = robot.step(target, wrench)
                self._timed("robot_step", tick)
                tick = time.perf_counter()
                heights = self._log_step(target, tool_target, wrench, requested, applied)
                if i % frame_stride == 0 or i == count - 1:
                    self._snapshot(heights, wrench, begin)
                self._timed("observation", tick)
                if (i + 1) % print_stride == 0:
                    phase = f" phase={self.cycle.phase} edge_mm={min(heights)*1000:.3f}" if self.cycle else ""
                    print(
                        f"t={robot.data.time:.3f}s wall={time.perf_counter()-begin:.2f}s{phase}", flush=True
                    )
        except Exception as error:
            status, exception = "failed", repr(error)
        wall_s = time.perf_counter() - begin - self.paused_s

        if self.record:
            self._save_recording(frame_stride)
        report = self._report(status, exception, wall_s)
        if self.out is not None:
            (self.out / "report.json").write_text(json.dumps(report, indent=2))
        if self.observer:
            self.observer.finish(report)
        print(json.dumps(report, indent=2), flush=True)
        if args.video and self.record:
            if self.surface:
                from .render_franka import render as render_surface_robot

                render_surface_robot(self.out)
            else:
                render(self.out)
        if status == "failed":
            raise RuntimeError(exception)
        return report

    def _log_columns(self):
        nq, nv = self.robot.model.nq, self.robot.model.nv
        qend = 13 + nq
        vend = qend + nv
        return dict(
            time=0,
            target=[1, 4],
            tool_position=[4, 7],
            wrench=[7, 13],
            qpos=[13, qend],
            qvel=[qend, vend],
            requested_arm_torque=[vend, vend + 7],
            commanded_arm_torque=[vend + 7, vend + 14],
            material_generalized_force=[vend + 14, vend + 14 + nv],
        )

    def _save_recording(self, frame_stride):
        from .diagnostics import Metrics

        out, data, s = self.out, self.data, self.solver
        np.savez_compressed(
            out / "trajectory.npz",
            log=np.array(data.logs),
            qpos=np.array(data.frames),
            particles=np.array(data.particles),
            codes=np.array(data.codes),
            origin=self.robot.material_origin,
            deformation=np.array(data.deformations),
            plastic_deformation=np.array(data.plastic_deformations),
            velocity=np.array(data.velocities),
            energy=np.array(data.energies),
            control_tcp=np.array(data.controls),
            cycle_trace=np.array(data.cycle_trace),
            edge_heights=np.array(data.edge_heights),
            radius=s.radius if s else 0.001,
            interval=self.args.interval,
            frame_stride=frame_stride,
        )
        if self.cycle:
            cycle_summary = dict(
                events=self.cycle.events,
                observations=data.task_observations,
                trace_columns=CYCLE_TRACE_COLUMNS,
                phases=self.cycle.phases,
            )
            (out / "cycle.json").write_text(json.dumps(cycle_summary, indent=2))
        energy_columns = ["time_s"] + Metrics.columns + ["directional_contact_work_J"]
        (out / "energy_columns.json").write_text(json.dumps(energy_columns))
        (out / "columns.json").write_text(json.dumps(self._log_columns(), indent=2))

    def _report(self, status, exception, wall_s):
        robot, s, cycle, surface = self.robot, self.solver, self.cycle, self.surface
        report = dict(
            status=status,
            exception=exception,
            sim_s=robot.data.time,
            wall_s=wall_s,
            realtime_ratio=robot.data.time / wall_s,
            warmup_s=self.warmup_s,
            recording=self.record,
            display="live WebGL" if self.observer else "none",
            pause_wall_s=self.paused_s,
            live_setup_s=getattr(self.observer, "setup_s", 0.0),
            timing_scope="physics, observation and state publication; excludes JIT, viewer initialization, user pause and client rendering",
            time_s=self.phase_time,
            material_present=s is not None,
            warnings=robot.data.warning.number.tolist(),
            surface_wall_s=surface.wall_s if surface else 0.0,
            surface_phase_s=surface.timing_s if surface else {},
        )
        if cycle:
            completed = status == "completed"
            report.update(
                motion_status="completed" if cycle.done else "timeout" if completed else "failed",
                final_phase=cycle.phase,
                task_evaluation=evaluate_cut(self.data.task_observations, cycle.done and completed),
            )
        log = np.array(self.data.logs)
        if len(log):
            columns = self._log_columns()
            requested = log[:, slice(*columns["requested_arm_torque"])]
            commanded = log[:, slice(*columns["commanded_arm_torque"])]
            report.update(
                peak_material_force_n=float(np.linalg.norm(log[:, 7:10], axis=1).max()),
                peak_material_torque_nm=float(np.linalg.norm(log[:, 10:13], axis=1).max()),
                max_tracking_error_m=float(np.linalg.norm(log[:, 1:4] - log[:, 4:7], axis=1).max()),
                saturated_steps=int(np.any(np.abs(requested - commanded) > 1.0e-10, axis=1).sum()),
                min_edge_height_m=float(np.min(self.data.edge_heights)),
                arm_joint_motion_rad=(np.ptp(log[:, 13:20], axis=0)).tolist(),
            )
        if s:
            labels, counts = components(s.node_numpy, s.weight_numpy, s.codes.numpy())
            report.update(
                components=counts.tolist(),
                min_det_F=float(np.linalg.det(s.F.numpy()).min()),
                gpu_errors=s.errors.numpy().tolist(),
                particle_count=s.n,
            )
            if self.record:
                np.save(self.out / "component_labels.npy", labels)
            if s.compact:
                report["active_grid_nodes"] = int(s.active_count.numpy()[0])
                report["lookup_capacity"] = s.capacity
        return report


def run(args, observer=None):
    return FrankaCuttingRun(args, observer).execute()


def render(out):
    import mujoco as mj
    import imageio.v2 as imageio
    import xml.etree.ElementTree as ET
    from PIL import Image, ImageDraw

    out = Path(out)
    z = np.load(out / "trajectory.npz")
    report = json.loads((out / "report.json").read_text())
    outcome = report.get("task_evaluation", {}).get("success")
    result_label = "timed" if outcome is None else "cycle complete" if outcome else "cycle incomplete"
    audit_path = out / "material_geometry_validation.json"
    folded = (
        audit_path.exists() and json.loads(audit_path.read_text())["status"] == "sampled_material_map_folded"
    )
    quality = "MATERIAL GEOMETRY FOLDED" if folded else ""
    root = ET.parse(out / "scene.xml").getroot()
    root.find("compiler").set("meshdir", str(SOURCE.parent / "assets"))
    m = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mj.MjData(m)
    renderer = mj.Renderer(m, height=540, width=720, max_geom=20000)
    camera = mj.MjvCamera()
    camera.lookat[:] = [0.27, 0, 0.4]
    camera.distance, camera.azimuth, camera.elevation = 1.6, 135, -22
    close = mj.MjvCamera()
    close.lookat[:] = z["origin"] + [0, 0, 0.035]
    close.distance, close.azimuth, close.elevation = 0.27, 135, -20
    option = mj.MjvOption()
    option.geomgroup[3] = 0
    palette = np.array([[0.96, 0.72, 0.22, 1], [0.35, 0.7, 0.9, 1], [0.7, 0.45, 0.8, 1]])
    fps = 1 / (float(z["interval"]) * int(z["frame_stride"]))
    with imageio.get_writer(out / "replay.mp4", fps=fps, macro_block_size=2) as writer:
        for i, q in enumerate(z["qpos"]):
            d.qpos[:] = q
            mj.mj_forward(m, d)
            panels = []
            for cam in (camera, close):
                renderer.update_scene(d, camera=cam, scene_option=option)
                if len(z["particles"]):
                    for p, code in zip(z["particles"][i] + z["origin"], z["codes"][i]):
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        mj.mjv_initGeom(
                            geom,
                            mj.mjtGeom.mjGEOM_SPHERE,
                            np.full(3, float(z["radius"]) * 0.85),
                            p,
                            np.eye(3).ravel(),
                            palette[int(code) % 3],
                        )
                        renderer.scene.ngeom += 1
                panels.append(renderer.render().copy())
            frame = Image.fromarray(np.concatenate(panels, axis=1))
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 1440, 42), fill=(18, 24, 30))
            time_s = float(z["control_tcp"][i, 0]) if "control_tcp" in z else i / fps
            phase = ""
            if "cycle_trace" in z and z["cycle_trace"].size:
                from .cut_cycle import CutCycle

                idx = np.argmin(abs(z["cycle_trace"][:, 0] - time_s))
                phase = " | " + CutCycle.phases[int(z["cycle_trace"][idx, 1])]
            draw.text(
                (14, 8),
                f"Franka torque dynamics + GPU TLMPM | t={time_s:.2f}s{phase} | final: {result_label}",
                fill="white",
            )
            draw.text(
                (14, 23),
                f"{quality} Particle colors = cut-side codes",
                fill=(255, 150, 130) if folded else "white",
            )
            writer.append_data(np.asarray(frame))
            if i in (0, len(z["qpos"]) // 2, len(z["qpos"]) - 1):
                frame.save(out / f"frame_{i:03d}.png")
    renderer.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", help="Optional new directory for configuration and summary")
    p.add_argument(
        "--record",
        action="store_true",
        help="Also persist raw trajectory and surface snapshots for offline analysis",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Display current physics in the browser; no offscreen renderer or replay",
    )
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--grid-storage",
        choices=["compact", "hash"],
        default="compact",
        help="Dense active grid with unchanged-topology cache, or original hash baseline",
    )
    p.add_argument("--interval", type=float, default=0.001)
    p.add_argument(
        "--material",
        choices=["elastic-block", "paper-banana"],
        default="elastic-block",
        help="paper-banana: paper §IV-C material on the 120 mm synthetic banana",
    )
    p.add_argument(
        "--dt", type=float, default=None, help="Defaults: elastic block 250 us; paper banana 62.5 us"
    )
    p.add_argument(
        "--contact-scaling",
        choices=["physical", "per-step"],
        default="physical",
        help="physical: contact stiffness fixed at the preset dt; per-step: fixed s_n, gamma_k (stiffness ~ 1/dt^2)",
    )
    p.add_argument("--pair-search", choices=["auto", "hash", "all"], default="auto")
    p.add_argument(
        "--tool-substep",
        choices=["hold", "predict"],
        default="hold",
        help="Tool state within a coupling interval: held (paper) or mobility-predicted",
    )
    p.add_argument("--spacing", type=float, default=None)
    p.add_argument("--duration", type=float, default=4.6)
    p.add_argument(
        "--motion",
        choices=["timed", "feedback"],
        default="timed",
        help="Feedback gates phases using actual knife pose; independent robot adapter",
    )
    p.add_argument("--descent-speed", type=float, default=0.02, help="Feedback task reference speed, m/s")
    p.add_argument("--withdrawal-speed", type=float, default=0.025, help="Feedback task reference speed, m/s")
    p.add_argument(
        "--goal-tolerance",
        type=float,
        default=0.0005,
        help="Feedback endpoint tolerance, m; applies to both blade-edge ends",
    )
    p.add_argument("--stiffness", type=float, default=150.0)
    p.add_argument("--controller", choices=["libfranka", "serl"], default="libfranka")
    p.add_argument("--attachment", choices=["fixed", "grasp"], default="fixed")
    p.add_argument(
        "--grip-force-per-finger", type=float, default=10.0, help="Grasp actuator command, N per finger"
    )
    p.add_argument("--cutting-energy", type=float, default=None)
    p.add_argument("--no-material", action="store_true")
    p.add_argument("--no-cutting", action="store_true")
    p.add_argument("--video", action="store_true")
    p.add_argument("--surface", action="store_true", help="Online reconstruction, included in elapsed time")
    p.add_argument(
        "--surface-embedding",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Compatible nodal field backend only; topology/QEF/triangles remain CPU",
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
        help="Surface-only reference spacing in m; default half the MPM spacing. Does not change physics",
    )
    return p


def main():
    args = parser().parse_args()
    if args.live:
        if not args.surface:
            raise ValueError("--live currently requires --surface")
        from .live import serve

        serve(args, run)
    else:
        run(args)


if __name__ == "__main__":
    main()
