"""Deterministic six-axis input replay, without a device or physical robot.

python -m scripts.stress_teleop --seed 17 --output runs/stress-teleop
python -m scripts.stress_teleop --device cuda:0 --seed 17 --output runs/stress-teleop-gpu
Replay an exported stream with --replay runs/stress-teleop/commands.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from blademaster_warp.live import LiveState
from blademaster_warp.provenance import git_revision
from blademaster_warp.teleop_input import Command, LatestInput, PoseTarget, hid_twist

CONTROL_DT = 0.01
INITIAL_EDGE = (-0.01, 0.0, 0.036)


def seeded_replay(seed=17, extra_cuts=0, restart_cycles=0):
    from scripts.stress_multi_tool import cut_code_capacity

    last = cut_code_capacity()["max_cut_index"]
    if not isinstance(extra_cuts, int) or isinstance(extra_cuts, bool) or not 0 <= extra_cuts <= last:
        raise ValueError(f"extra_cuts must be an integer within [0, {last}]")
    if not isinstance(restart_cycles, int) or isinstance(restart_cycles, bool) or restart_cycles < 0:
        raise ValueError("restart_cycles must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    rotating = rng.uniform(0.3, 0.7, 6) * rng.choice([-1, 1], 6)
    rows, seq = [], 0
    for tick in range(330):
        raw, event, action, phase = np.zeros(6), "", "", "idle"
        if tick < 24 or 100 <= tick < 120 or 250 <= tick < 274:
            raw[2], phase = 1.0, "fast_in"
        elif 35 <= tick < 65 or 130 <= tick < 150 or 290 <= tick < 320:
            raw[2], phase = -1.0, "fast_out"
        elif 65 <= tick < 95 or 167 <= tick < 170 or 207 <= tick < 225:
            raw, phase = rotating.copy(), "six_axis"
            raw[:3] *= 0.3
        if tick in (0, 100, 207, 250):
            event = "begin_cut"
        elif tick in (35, 130, 225, 290):
            event = "end_cut"
        elif tick in (24, 120, 150, 274, 320):
            # A hold must suppress even the nonzero twist carried by its packet.
            event, raw, phase = "hold", rotating.copy(), "hold"
        if tick in (0, 166, 249):
            action = "start"
        elif tick == 150:
            action = "pause"
        elif tick == 240:
            action = "stop"
        disconnected = 170 <= tick < 205
        paused = 151 <= tick < 166
        holding = tick in (25, 26, 121, 122, 275, 276, 321, 322)
        row = dict(tick=tick, phase="disconnect" if disconnected else phase, action=action)
        if not (disconnected or paused or holding or tick == 249):
            velocity, omega = hid_twist(raw, speed=0.15, angular_speed=0.9)
            if tick == 245:
                event = "begin_cut"  # This stopped-session event must be discarded on restart.
            row["packet"] = Command(seq, tuple(velocity), tuple(omega), event).packet()
            seq += 1
        if tick == 206:
            row["packet"]["seq"] -= 2
            row["reject"] = True
        rows.append(row)
    for _ in range(extra_cuts):
        for phase_tick in range(80):
            raw, event = np.zeros(6), ""
            if phase_tick < 30:
                raw[2], phase = 1.0, "fast_in"
                event = "begin_cut" if phase_tick == 0 else ""
            elif phase_tick < 38:
                phase, event = "hold", "hold"
            elif phase_tick < 68:
                raw[2], phase = -1.0, "fast_out"
                event = "end_cut" if phase_tick == 38 else ""
            else:
                raw[1], phase = -0.2, "shift"
            velocity, omega = hid_twist(raw, speed=0.15, angular_speed=0.9)
            rows.append(
                dict(
                    tick=len(rows),
                    phase=phase,
                    action="",
                    packet=Command(seq, tuple(velocity), tuple(omega), event).packet(),
                )
            )
            seq += 1
    # Repeat the long second session after stop/start, retaining the stream's
    # sequence watermark while discarding queued events and the old target.
    template = rows[250:]
    for _ in range(restart_cycles):
        for action, event in (("stop", ""), ("", "begin_cut")):
            rows.append(
                dict(
                    tick=len(rows),
                    phase="stopped",
                    action=action,
                    packet=Command(seq, (0.1, 0, 0), (0, 0.2, 0), event).packet(),
                )
            )
            seq += 1
        rows.append(dict(tick=len(rows), phase="restart", action="start"))
        rows.append(
            dict(
                tick=len(rows),
                phase="duplicate",
                action="",
                reject=True,
                packet=Command(seq - 1, (0.1, 0, 0), (0, 0.2, 0), "begin_cut").packet(),
            )
        )
        for source in template:
            row = dict(source, tick=len(rows))
            if "packet" in row:
                row["packet"] = dict(row["packet"], seq=seq)
                seq += 1
            rows.append(row)
    return rows


def replay(rows, on_frame=None):
    live = LiveState()
    live.teleop = LatestInput(timeout=0.25)
    live._set(dict(state="ready"))

    def new_target():
        return PoseTarget(
            INITIAL_EDGE, workspace=((-0.025, -0.015, 0.002), (0.025, 0.015, 0.09)), max_tilt=0.6
        )

    target, session = new_target(), 0
    trace, rejected, stale_frames = [], 0, 0
    for expected_tick, row in enumerate(rows):
        if row["tick"] != expected_tick:
            raise ValueError("Replay ticks must be consecutive from zero")
        now = expected_tick * CONTROL_DT
        if row.get("packet") is not None:
            previous = live.teleop.latest
            try:
                live.teleop.feed(row["packet"], now=now)
            except ValueError:
                if not row.get("reject"):
                    raise
                assert live.teleop.latest == previous
                rejected += 1
            else:
                assert not row.get("reject"), "An out-of-order packet was accepted"
        action = row.get("action", "")
        if action:
            live.command(action)
            if action == "stop":
                assert live.stop_requested and live.play.is_set()
                live.finish(dict(status="stopped"))
            elif live.restart.is_set():
                assert live.autostart and not live.stop_requested
                command, events, stale = live.teleop.sample(now=now)
                assert stale and not events and not any(command.velocity) and not any(command.omega)
                target, session = new_target(), session + 1
                # Scene creation is outside this input replay; the material callback
                # creates a fresh solver on the first frame of the new session.
                live.restart.clear()
                live.autostart = False
                live.play.set()
                live._set(dict(state="running"))
        before_position, before_orientation = target.position.copy(), target.orientation.copy()
        active = live.play.is_set() and not live.stop_requested
        stale, events = False, []
        if active:
            command, events, stale = live.teleop.sample(now=now)
            stale_frames += stale
            positions, rotations = target.integrate(command, events, 0.0005, 20)
            np.testing.assert_allclose(np.linalg.norm(rotations, axis=1), 1, atol=1e-12)
            assert np.isfinite(positions).all() and np.isfinite(rotations).all()
            if stale or command.event == "hold":
                np.testing.assert_array_equal(target.position, before_position)
                np.testing.assert_array_equal(target.orientation, before_orientation)
            if on_frame:
                on_frame(session, positions, rotations, target.cutting, max(0, target.cut_index))
        else:
            np.testing.assert_array_equal(target.position, before_position)
        trace.append(
            dict(
                tick=expected_tick,
                session=session,
                phase=row["phase"],
                active=active,
                stale=stale,
                seq=-1 if live.teleop.latest is None else live.teleop.latest.seq,
                events=events,
                cutting=target.cutting,
                cut_index=target.cut_index,
                position=target.position.tolist(),
                orientation=target.orientation.tolist(),
            )
        )
    return trace, dict(
        frames=len(rows),
        sessions=session + 1,
        rejected_packets=rejected,
        stale_frames=int(stale_frames),
        cuts_per_session=[
            sum(e == "begin_cut" for r in trace if r["session"] == i for _, e in r["events"])
            for i in range(session + 1)
        ],
    )


class MaterialReplay:
    def __init__(self, device, surface=False):
        self.device, self.session = device, -1
        self.surface_enabled = surface
        self.surface = None
        self.frames = self.entered = self.exited = 0
        self.was_embedded = False
        self.peak_reaction = 0.0
        self.cut_particles = 0
        self.trace = []
        self.max_components = self.max_broken_links = 0
        self.cut_history = {}

    def __call__(self, session, positions, rotations, cutting, cut_index):
        from blademaster_warp.solver import Config, Solver
        from blademaster_warp.coupling import ToolState
        from blademaster_warp.topology import components

        def tool_state(state):
            return ToolState(
                state["blade_position"][0],
                state["blade_orientation"][0],
                state["blade_velocity"][0],
                state["blade_omega"][0],
            )

        if session != self.session:
            from scripts.stress_multi_tool import cut_code_capacity

            self.solver = Solver(
                Config(pose_servo=True, cache_topology=True, grid_storage="compact"), device=self.device
            )
            self.capacity = cut_code_capacity(solver=self.solver)
            self.solver.tool_pos.assign(
                [np.asarray(INITIAL_EDGE) + [0, 0, self.solver.config.blade_half_size[2]]]
            )
            self.old_codes = np.zeros(self.solver.n, dtype=np.int64)
            self.old_labels = np.zeros(self.solver.n, dtype=int)
            self.was_embedded, self.session = False, session
            if self.surface_enabled:
                from blademaster_warp.surface_runtime import SurfaceRecorder

                self.surface = SurfaceRecorder(
                    self.solver, None, embedding_backend="cuda", ccd_backend="cuda"
                )
                state = self.solver.state()
                self.surface.update(
                    0,
                    state["position"],
                    state["deformation"],
                    state["sides"],
                    tool_state(state),
                    cutting=False,
                    record_frame=False,
                )
                self.old_broken = self.surface.surface.broken.copy()
        s = self.solver
        if self.surface:
            start_pose = np.r_[s.tool_pos.numpy()[0], s.tool_rot.numpy()[0]]
            self.surface.record_trajectory([s.step_count * s.config.dt], start_pose[None], cutting, cut_index)
            times, path = [], []
            for p, q in zip(positions, rotations):
                s.advance(p[None], cutting=cutting, cut_index=cut_index, orientations=q[None], graph=False)
                times.append(s.step_count * s.config.dt)
                path.append(np.r_[s.tool_pos.numpy()[0], s.tool_rot.numpy()[0]])
            self.surface.record_trajectory(times, np.array(path), cutting, cut_index)
        else:
            s.advance(positions, cutting=cutting, cut_index=cut_index, orientations=rotations)
        state = s.state()
        for digit in range(cut_index + 1):
            old, new = (self.old_codes // 3**digit) % 3, (state["sides"] // 3**digit) % 3
            np.testing.assert_array_equal(new[old != 0], old[old != 0])
        self.old_codes = state["sides"].copy()
        if cutting:
            key = (session, cut_index)
            labeled = int(np.count_nonzero((state["sides"] // 3**cut_index) % 3))
            self.cut_history[key] = max(self.cut_history.get(key, 0), labeled)
        labels, counts = components(s.node_numpy, s.weight_numpy, state["sides"])
        assert sum(counts) == s.n
        for label in range(len(counts)):
            assert len(np.unique(self.old_labels[labels == label])) == 1, "Cut components merged"
        self.old_labels = labels
        self.max_components = max(self.max_components, len(counts))
        if self.surface:
            self.surface.update(
                state["time"],
                state["position"],
                state["deformation"],
                state["sides"],
                tool_state(state),
                cutting=cutting,
                cut_index=cut_index,
                record_frame=False,
            )
            broken = self.surface.surface.broken
            assert np.all(broken[self.old_broken]), "Surface cut healed"
            self.old_broken = broken.copy()
            self.max_broken_links = max(self.max_broken_links, int(broken.sum()))
            mesh = self.surface.current
            assert np.isfinite(mesh["vertices"]).all()
            assert mesh["triangles"].size and mesh["triangles"].min() >= 0
            assert mesh["triangles"].max() < len(mesh["vertices"])
        embedded = s.embedded_particles() > 0
        self.entered += embedded and not self.was_embedded
        self.exited += self.was_embedded and not embedded
        self.was_embedded = embedded
        self.frames += 1
        self.cut_particles = max(self.cut_particles, int(np.count_nonzero(state["sides"])))
        self.peak_reaction = max(self.peak_reaction, float(np.linalg.norm(s.reaction.numpy())))
        self.trace.append(
            dict(
                session=session,
                time=state["time"],
                embedded=embedded,
                cut_index=cut_index,
                cutting=cutting,
                component_sizes=counts.tolist(),
                blade_position=state["blade_position"].tolist(),
            )
        )

    def report(self):
        assert self.entered >= 2 and self.exited >= 1, "Replay must enter and leave actual material"
        assert self.cut_particles > 0 and self.peak_reaction > 0
        if self.surface_enabled:
            assert self.max_broken_links > 0
        return dict(
            frames=self.frames,
            capacity=self.capacity,
            material_entries=int(self.entered),
            material_exits=int(self.exited),
            cut_particles=self.cut_particles,
            peak_reaction_impulse_Ns=self.peak_reaction,
            max_components=self.max_components,
            surface=self.surface_enabled,
            max_broken_surface_links=self.max_broken_links,
            cut_history=[
                dict(session=session, cut_index=index, labeled_particles=count)
                for (session, index), count in self.cut_history.items()
            ],
        )


def franka_stress():
    """The production MuJoCo Franka loop with the six-axis mock and real material."""
    from scipy.spatial.transform import Rotation
    from blademaster_warp.franka_run import FrankaCuttingRun, parser

    args = parser().parse_args(
        [
            "--motion",
            "teleop",
            "--teleop-rotation",
            "full",
            "--cut-events",
            "manual",
            "--duration",
            "4.0",
            "--surface",
            "--surface-embedding",
            "cuda",
            "--surface-ccd",
            "cuda",
        ]
    )
    run = FrankaCuttingRun(args)
    initial = run.robot.rotation.copy()
    target_tilts, measured_tilts = [], []
    exchange = run._exchange_wrench

    def observe_rotation():
        result = exchange()
        target = Rotation.from_quat(run.teleop.orientation).as_matrix()
        for values, rotation in ((target_tilts, target), (measured_tilts, run.robot.rotation)):
            values.append(float(np.arccos(np.clip(initial[:, 2] @ rotation[:, 2], -1, 1))))
        return result

    run._exchange_wrench = observe_rotation
    report = run.execute()
    assert report["status"] == "completed"
    assert report["teleop"]["cut_strokes"] == 2
    assert report["gpu_errors"] == [0, 0, 0] and report["min_det_F"] > 0
    # The fixture tilts and then counter-rotates; inspect the entire motion.
    assert max(target_tilts) > 0.005, "Full rotation replay did not tilt the tool target"
    assert max(measured_tilts) > 0.001, "The simulated Franka did not follow the tilt"
    assert np.isfinite(run.robot.rotation).all()
    report["max_target_tilt_rad"] = max(target_tilts)
    report["max_measured_tilt_rad"] = max(measured_tilts)
    return report


def run_replay(rows, device="cpu", surface=False):
    if surface and device == "cpu":
        raise ValueError("Surface replay requires CUDA")
    material = MaterialReplay(device, surface) if device != "cpu" else None
    trace, report = replay(rows, material)
    again, repeated = replay(rows)
    assert trace == again and report == repeated, "Input replay is not deterministic"
    if material:
        report["material"] = material.report()
    return trace, report, material


def main(argv=None):
    from scripts.stress_multi_tool import cut_code_capacity

    parser = argparse.ArgumentParser(description=__doc__)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, default=17)
    seeds.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", default="cpu", help="cpu: input replay; cuda:0: also evolve material")
    parser.add_argument(
        "--surface", action="store_true", help="Check surface history using every actual tool substep"
    )
    history = parser.add_mutually_exclusive_group()
    history.add_argument("--extra-cuts", type=int, default=0)
    history.add_argument("--near-capacity", action="store_true")
    parser.add_argument("--restart-cycles", type=int, default=0)
    parser.add_argument(
        "--franka", action="store_true", help="Also run the MuJoCo Franka full-rotation fixture"
    )
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (args.surface or args.franka) and args.device == "cpu":
        parser.error("--surface and --franka require a CUDA material replay")
    if args.replay and (
        args.seeds is not None or args.extra_cuts or args.near_capacity or args.restart_cycles
    ):
        parser.error("--replay cannot be combined with generated history or seed batches")
    capacity = cut_code_capacity()
    extra_cuts = capacity["max_cut_index"] if args.near_capacity else args.extra_cuts
    selected_seeds = args.seeds if args.seeds is not None else [args.seed]
    if len(set(selected_seeds)) != len(selected_seeds):
        parser.error("Seeds must be distinct")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for seed in selected_seeds:
        rows = (
            json.loads(args.replay.read_text())
            if args.replay
            else seeded_replay(seed, extra_cuts, args.restart_cycles)
        )
        trace, report, material = run_replay(rows, args.device, args.surface)
        report.update(
            revision=git_revision(),
            seed=None if args.replay else seed,
            device=args.device,
            control_dt_s=CONTROL_DT,
            capacity=capacity,
            status="passed",
        )
        if material and not args.replay:
            entries = 2 + extra_cuts + args.restart_cycles * (1 + extra_cuts)
            assert material.entered >= entries, "Repeated strokes missed the material"
        reports.append(report)
        if args.output:
            output = args.output if args.seeds is None else args.output / f"seed-{seed}"
            output.mkdir(exist_ok=True)
            for name, data in (("commands", rows), ("trace", trace), ("report", report)):
                (output / f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
            if material:
                (output / "material.json").write_text(
                    json.dumps(material.trace, indent=2, allow_nan=False) + "\n"
                )
    report = reports[0] if args.seeds is None else dict(status="passed", runs=reports)
    if args.franka:
        report["franka"] = franka_stress()
    if args.output:
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
