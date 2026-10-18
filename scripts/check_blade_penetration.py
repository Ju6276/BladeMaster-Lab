"""Replay the saved blade-support failure for 3 ms with the imported Solver.

python -m scripts.check_blade_penetration \
    --state runs/franka_penetration_20261018/penetration-first-cross-state.npz \
    --output runs/penetration-check

Reuse that identical reconstructed initial state with another checkout:
python -m scripts.check_blade_penetration --prepared runs/penetration-check \
    --output runs/penetration-new-backend

The diagnostic NPZ mixes pre-step x/v/stress with post-step F. Its archived
solver reconstructs pre-step F, Fp and friction history from the recording.
Subsequent checks use the current Solver, a held tool pose/twist/mobility, and
fixed physical contact stiffness. Positive height is the captured entry side;
the zero threshold is the blade midplane, not its outer face or proxy offset.
"""

import argparse
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
from scipy.spatial.transform import Rotation

from blademaster_warp import geometry, tool
from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.provenance import git_revision
from blademaster_warp.solver import Config, Solver


def blade_coordinates(x, position, rotation, half_size, bevel_height, entry_sign):
    local = (np.asarray(x, dtype=float) - position) @ rotation
    _, hy, hz = half_size
    margin = np.minimum.reduce([hy - abs(local[:, 1]), local[:, 2] + hz - bevel_height, hz - local[:, 2]])
    return entry_sign * local[:, 0], margin


def interior_crossings(before, after, radius):
    height, margin = before
    next_height, next_margin = after
    return (height > 0) & (next_height <= 0) & (margin > radius) & (next_margin > radius)


def fixed_stiffness_config(config, dt):
    reference_dt = config.contact_reference_dt or config.dt
    # The explicit board factor normally stays per-step; scale it as well here.
    board_scale = config.board_normal_scale * (dt / config.dt) ** 2
    return replace(config, dt=dt, contact_reference_dt=reference_dt, board_normal_scale=board_scale)


def archived_solver(source):
    """Keep state reconstruction independent of changes to the tested backend."""
    for name in ("solver", "material", "contact"):
        if not (source / f"{name}.py").is_file():
            raise FileNotFoundError(source / f"{name}.py")
    package_name = "_blade_penetration_archive"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source)]
    sys.modules[package_name] = package
    # These two helpers were not included in the archive. The reference stencils
    # are checked against the NPZ before any replay is accepted.
    sys.modules[f"{package_name}.geometry"] = geometry
    sys.modules[f"{package_name}.tool"] = tool
    return importlib.import_module(f"{package_name}.solver")


class RecordedTool:
    def __init__(self, recording, origin, columns):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(recording / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.body = self.model.body("knife").id
        self.origin = origin
        self.columns = columns

    def state(self, row):
        mj, model, data = self.mj, self.model, self.data
        data.qpos[:] = row[slice(*self.columns["qpos"])]
        data.qvel[:] = row[slice(*self.columns["qvel"])]
        mj.mj_forward(model, data)
        jacobian = np.zeros((6, model.nv))
        mj.mj_jacBody(model, data, jacobian[:3], jacobian[3:], self.body)
        inverse_mass_jacobian = np.empty_like(jacobian)
        mj.mj_solveM(model, data, inverse_mass_jacobian, jacobian)
        rotation = data.xmat[self.body].reshape(3, 3).copy()
        twist = jacobian @ data.qvel
        return ToolState(
            data.xpos[self.body] - self.origin,
            Rotation.from_matrix(rotation).as_quat(),
            twist[:3],
            twist[3:],
            jacobian @ inverse_mass_jacobian.T,
        ).checked()


def set_tool(solver, state):
    for name, value in (
        ("tool_pos", state.position),
        ("tool_rot", state.orientation_xyzw),
        ("tool_vel", state.linear_velocity),
        ("tool_omega", state.angular_velocity),
    ):
        getattr(solver, name).assign(np.asarray([value], dtype=np.float32))
    solver.tool_mobility.assign(np.asarray(state.mobility, dtype=np.float32))


def reconstruct(state_path, recording, device):
    folder = state_path.parent
    metadata = json.loads((folder / "force_balance.json").read_text())
    manifest = json.loads((recording / "manifest.json").read_text())
    config = Config(**manifest["config"])
    if not config.external_tool or config.tool_substep != "hold" or config.tool_count != 1:
        raise ValueError("This saved-state protocol requires one externally held tool")
    if config.maxwell_tau and config.maxwell_fraction:
        raise ValueError("The source recording lacks Maxwell history")
    with np.load(state_path) as archive:
        saved = dict(archive)
    baseline = archived_solver(folder / "source")
    solver = baseline.Solver(baseline.Config(**manifest["config"]), device=device)
    for name, array in (("weights", solver.weight_numpy), ("grad", solver.gradients.numpy())):
        np.testing.assert_allclose(array, saved[name], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(solver.mass, saved["mass"], rtol=1e-12)
    start = metadata["snapshot_restart_time_s"]
    end = metadata["replay_crossing_time_s"]
    with np.load(recording / "trajectory.npz") as trace:
        times = trace["control_tcp"][:, 0]
        frame = int(np.argmin(abs(times - start)))
        np.testing.assert_allclose(times[frame], start, rtol=0, atol=1e-8)
        for name, key in (
            ("x", "particles"),
            ("v", "velocity"),
            ("F", "deformation"),
            ("Fp", "plastic_deformation"),
            ("codes", "codes"),
        ):
            getattr(solver, name).assign(trace[key][frame])
        logs = trace["log"]
        robot = RecordedTool(recording, trace["origin"], json.loads((recording / "columns.json").read_text()))
    solver.refresh_stress()
    solver._set_cut(1)  # The archived probe restarts after the second radial cut.
    entry_sign = float(np.dot(saved["normal"], saved["R"][:, 0]))
    if not np.isclose(abs(entry_sign), 1.0, atol=1e-8):
        raise ValueError("Saved entry normal is not a blade-face normal")
    entry_sign = float(np.sign(entry_sign))
    crossed_before = np.zeros(solver.n, dtype=bool)
    rows = logs[(logs[:, 0] >= start - 1e-8) & (logs[:, 0] < end - 1e-8)]
    np.testing.assert_allclose(len(rows) * config.dt, end - start, rtol=0, atol=1e-8)
    for row in rows:
        state = robot.state(row)
        set_tool(solver, state)
        rotation = Rotation.from_quat(state.orientation_xyzw).as_matrix()
        before = blade_coordinates(
            solver.x.numpy(),
            state.position,
            rotation,
            config.blade_half_size,
            config.blade_bevel_height,
            entry_sign,
        )
        solver.launch(baseline.c.clear_step, 1, [solver.reaction, solver.torque, solver.board, solver.speed])
        solver.rigid_contacts(False)
        solver.particle_contacts()
        solver.material_step()
        after = blade_coordinates(
            solver.x.numpy(),
            state.position,
            rotation,
            config.blade_half_size,
            config.blade_bevel_height,
            entry_sign,
        )
        crossed_before |= interior_crossings(before, after, solver.radius)
        solver.step_count += 1

    row = logs[int(np.argmin(abs(logs[:, 0] - end)))]
    np.testing.assert_allclose(row[0], end, rtol=0, atol=1e-8)
    captured_tool = robot.state(row)
    np.testing.assert_allclose(captured_tool.position, saved["pos"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        Rotation.from_quat(captured_tool.orientation_xyzw).as_matrix(), saved["R"], rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        np.r_[captured_tool.linear_velocity, captured_tool.angular_velocity],
        saved["twist"],
        rtol=0,
        atol=1e-10,
    )
    differences = {}
    # The archive was reduced with GPU atomics; tolerate their accumulation order,
    # while keeping the error well below the target's 16.8 um initial height.
    for name, key, tolerance in (("x", "x", 2e-6), ("v", "v", 5e-4), ("stress", "stress", 1.0)):
        actual = getattr(solver, name).numpy()
        differences[name] = float(np.max(abs(actual - saved[key])))
        if differences[name] > tolerance:
            raise ValueError(f"Reconstructed {name} differs from capture: {differences[name]}")
    np.testing.assert_array_equal(solver.codes.numpy(), saved["code"])
    arrays = {
        name: getattr(solver, name).numpy()
        for name in ("x", "v", "F", "Fp", "viscous", "stress", "codes", "damage", "histories")
    }
    for name in ("pair_keys", "pair_history", "pair_used", "pair_count"):
        for index, value in enumerate(getattr(solver, name)):
            arrays[f"{name}_{index}"] = value.numpy()
    # Use the exact captured pre-step values; never restore its post-step F.
    for name, key in (("x", "x"), ("v", "v"), ("stress", "stress"), ("codes", "code")):
        arrays[name] = saved[key]
    arrays["previous_crossings"] = crossed_before
    details = dict(
        recording=str(recording),
        state=str(state_path),
        start_time_s=start,
        capture_time_s=end,
        replay_steps=len(rows),
        cut_index=solver.cut_index,
        step_count=solver.step_count,
        entry_sign=entry_sign,
        reconstructed_max_abs_error=differences,
        restored_from_capture=["x", "v", "stress", "codes", "tool pose", "tool twist"],
        reconstructed=[
            "pre-step F",
            "Fp",
            "blade/board friction history",
            "pair friction history",
            "tool mobility",
        ],
        history_origin=f"Zero histories and damage at the original probe restart ({start} s)",
        saved_F_phase="post-step; excluded from restored initial state",
    )
    return config, captured_tool, arrays, details


def restore(config, arrays, details, device):
    solver = Solver(config, device=device)
    solver._set_cut(details["cut_index"])
    solver.step_count = details["step_count"]
    for name in ("x", "v", "F", "Fp", "viscous", "stress", "codes", "damage", "histories"):
        getattr(solver, name).assign(arrays[name])
    for name in ("pair_keys", "pair_history", "pair_used", "pair_count"):
        for index, value in enumerate(getattr(solver, name)):
            value.assign(arrays[f"{name}_{index}"])
    # Topology, grid mass and neighbor candidates are derived from the restored state.
    solver.topology_dirty.fill_(1)
    if solver.pair_list:
        solver.pair_list_dirty.fill_(1)
    return solver


def read_prepared(folder):
    report = json.loads((folder / "report.json").read_text())
    with np.load(folder / "restored_state.npz") as archive:
        arrays = dict(archive)
    state = ToolState(
        arrays["tool_pos"], arrays["tool_rot"], arrays["tool_vel"], arrays["tool_omega"], arrays["mobility"]
    ).checked()
    return Config(**report["config"]), state, arrays, report["restoration"]


def evaluate(config, state, arrays, details, device, dt, duration, output, target, graph=False):
    solver = restore(fixed_stiffness_config(config, dt), arrays, details, device)
    port = ExternalToolPort(solver)
    rotation = Rotation.from_quat(state.orientation_xyzw).as_matrix()

    def coordinates(x):
        return blade_coordinates(
            x,
            state.position,
            rotation,
            config.blade_half_size,
            config.blade_bevel_height,
            details["entry_sign"],
        )

    steps = int(round(duration / dt))
    if steps < 1 or not np.isclose(steps * dt, duration, rtol=0, atol=1e-12):
        raise ValueError("Duration must be an integer number of material steps")
    positions = [solver.x.numpy()]
    heights, margins = coordinates(positions[0])
    initial_positive = heights > 0
    initial_interior = margins > solver.radius
    height_trace, margin_trace = [heights], [margins]
    crossed = np.zeros(solver.n, dtype=bool)
    events, reactions = [], []
    worst = None
    tracked = initial_positive & initial_interior
    for step in range(steps + 1):
        if step:
            reaction = port.advance(state, dt, cutting=False, cut_index=details["cut_index"], graph=graph)
            reactions.append(np.r_[reaction.force_n, reaction.torque_nm])
            positions.append(solver.x.numpy())
            before = heights, margins
            heights, margins = coordinates(positions[-1])
            crossing = interior_crossings(before, (heights, margins), solver.radius)
            reverse = interior_crossings((-before[0], before[1]), (-heights, margins), solver.radius)
            crossing |= reverse
            for particle in np.flatnonzero(crossing):
                events.append(
                    dict(
                        particle=int(particle),
                        time_s=step * dt,
                        before_midplane_m=float(before[0][particle]),
                        after_midplane_m=float(heights[particle]),
                        minimum_edge_margin_m=float(min(before[1][particle], margins[particle])),
                        initially_positive=bool(initial_positive[particle]),
                        direction="negative_to_positive" if reverse[particle] else "positive_to_negative",
                    )
                )
            crossed |= crossing
            tracked |= (heights > 0) & (margins > solver.radius)
            height_trace.append(heights)
            margin_trace.append(margins)
        relevant = tracked & (margins > solver.radius)
        if relevant.any():
            particle = np.flatnonzero(relevant)[np.argmin(heights[relevant])]
            if worst is None or heights[particle] < worst["signed_midplane_clearance_m"]:
                worst = dict(
                    particle=int(particle),
                    time_s=step * dt,
                    signed_midplane_clearance_m=float(heights[particle]),
                    signed_face_clearance_m=float(heights[particle] - config.blade_half_size[0]),
                    proxy_gap_m=float(heights[particle] - config.blade_half_size[0] - solver.blade_offset),
                    edge_margin_m=float(margins[particle]),
                    position_m=positions[-1][particle].tolist(),
                    velocity_m_s=solver.v.numpy()[particle].tolist(),
                )
    gpu_errors = solver.errors.numpy().tolist()
    solver.state()  # Check finite state and positive det(F), not only tracked particles.
    initial_nonpositive = initial_interior & ~initial_positive
    result = dict(
        dt_s=dt,
        duration_s=duration,
        steps=steps,
        particle_count=solver.n,
        normal_scale=solver.normal_scale,
        tangent_stiffness=solver.tangent_stiffness,
        board_scale=solver.board_scale,
        initial_positive_interior_count=int((initial_positive & initial_interior).sum()),
        initial_nonpositive_interior_ids=np.flatnonzero(initial_nonpositive).tolist(),
        known_preexisting_crossing_ids=np.flatnonzero(
            initial_nonpositive & arrays["previous_crossings"]
        ).tolist(),
        new_crossing_count=int(crossed.sum()),
        new_crossing_ids=np.flatnonzero(crossed).tolist(),
        initially_positive_crossing_ids=np.flatnonzero(crossed & initial_positive).tolist(),
        initially_nonpositive_recrossing_ids=np.flatnonzero(crossed & ~initial_positive).tolist(),
        worst_state=worst,
        events=events,
        gpu_errors=gpu_errors,
        grid_contact_max_inward_speed_m_s=(
            float(solver.blade_constraint.max_violation.numpy()[0])
            if getattr(solver, "blade_constraint", None) is not None
            else None
        ),
        passed=not crossed.any() and not any(gpu_errors),
    )
    if 0 <= target < solver.n:
        result["target"] = dict(
            particle=target,
            initial_height_m=float(height_trace[0][target]),
            heights_m=[float(row[target]) for row in height_trace],
            crossed=bool(crossed[target]),
        )
    np.savez_compressed(
        output / f"trace_{dt * 1e6:g}us.npz",
        time=np.arange(steps + 1) * dt,
        x=np.asarray(positions),
        height=np.asarray(height_trace),
        margin=np.asarray(margin_trace),
        reaction=np.asarray(reactions),
        initial_positive=initial_positive,
        initial_interior=initial_interior,
        previous_crossings=arrays["previous_crossings"],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--state", type=Path)
    source.add_argument(
        "--prepared", type=Path, help="Previous output containing restored_state.npz and report.json"
    )
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graph", action="store_true", help="Use the captured CUDA scheduling path")
    parser.add_argument("--dts", type=float, nargs="+", default=[0.001, 0.0005, 0.00025])
    parser.add_argument("--duration", type=float, default=0.003)
    parser.add_argument(
        "--target", type=int, default=8096, help="Extra trace; all interior particles are checked"
    )
    args = parser.parse_args()
    if any(not np.isfinite(value) or value <= 0 for value in [args.duration, *args.dts]):
        parser.error("Duration and steps must be positive and finite")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.prepared:
        if args.recording:
            parser.error("--recording is only used with --state")
        config, state, arrays, details = read_prepared(args.prepared.resolve())
    else:
        args.state = args.state.resolve()
        recording = args.recording
        if recording is None:
            provenance = json.loads((args.state.parent / "provenance.json").read_text())
            recording = Path(provenance["original_recording"])
        config, state, arrays, details = reconstruct(args.state, recording.resolve(), args.device)
    if args.target < 0 or args.target >= len(arrays["x"]):
        parser.error("--target must be a particle index in the captured state")
    # Persist only physical restart arrays; read_prepared also loads tool fields.
    material_arrays = {
        name: value
        for name, value in arrays.items()
        if name not in ("tool_pos", "tool_rot", "tool_vel", "tool_omega", "mobility")
    }
    np.savez_compressed(
        args.output / "restored_state.npz",
        **material_arrays,
        tool_pos=state.position,
        tool_rot=state.orientation_xyzw,
        tool_vel=state.linear_velocity,
        tool_omega=state.angular_velocity,
        mobility=state.mobility,
    )
    report = dict(
        revision=git_revision(), config=asdict(config), restoration=details, graph=args.graph, runs=[]
    )
    report_path = args.output / "report.json"
    for dt in args.dts:
        result = evaluate(
            config,
            state,
            arrays,
            details,
            args.device,
            dt,
            args.duration,
            args.output,
            args.target,
            args.graph,
        )
        report["runs"].append(result)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "dt_s",
                        "new_crossing_count",
                        "new_crossing_ids",
                        "worst_state",
                        "target",
                        "passed",
                    )
                }
            ),
            flush=True,
        )
    return 0 if all(run["passed"] for run in report["runs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
