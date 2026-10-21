"""External rigid-tool interface (paper III-G).

The tool state is set at the start of each coupling interval; contact impulses
are accumulated over its substeps and returned as an interval-average wrench.
All quantities are SI in the material frame."""

from dataclasses import dataclass
import time
import numpy as np
import warp as wp
from . import tool
from .material import validate_interval
from .blade_constraint import store_residual


def vector(value, length, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return result


@dataclass(frozen=True)
class ToolState:
    position: object  # Tool origin; must match the collision geometry origin.
    orientation_xyzw: object
    linear_velocity: object  # Velocity at that same origin.
    angular_velocity: object
    mobility: object = None  # World [linear, angular] at the tool origin, 6x6.

    def checked(self):
        position = vector(self.position, 3, "position")
        q = vector(self.orientation_xyzw, 4, "orientation_xyzw")
        if abs(np.linalg.norm(q) - 1.0) > 1.0e-5:
            raise ValueError("Tool orientation must be a unit xyzw quaternion")
        mobility = np.zeros((6, 6)) if self.mobility is None else np.asarray(self.mobility, dtype=float)
        if (
            mobility.shape != (6, 6)
            or not np.isfinite(mobility).all()
            or np.abs(mobility - mobility.T).max() > 1.0e-10
            or np.linalg.eigvalsh(mobility).min() < -1.0e-10
        ):
            raise ValueError("Mobility must be a finite symmetric positive-semidefinite 6x6 matrix")
        return ToolState(
            position,
            q / np.linalg.norm(q),
            vector(self.linear_velocity, 3, "linear_velocity"),
            vector(self.angular_velocity, 3, "angular_velocity"),
            mobility.copy(),
        )


@dataclass(frozen=True)
class Reaction:
    duration_s: float
    impulse_ns: np.ndarray
    angular_impulse_nms: np.ndarray
    origin_world_m: np.ndarray

    @property
    def force_n(self):
        return self.impulse_ns / self.duration_s

    @property
    def torque_nm(self):
        return self.angular_impulse_nms / self.duration_s

    def about(self, origin_world_m):
        """Shift the moment reference, preserving the same physical wrench."""
        origin = vector(origin_world_m, 3, "moment origin")
        angular = self.angular_impulse_nms + np.cross(self.origin_world_m - origin, self.impulse_ns)
        return Reaction(self.duration_s, self.impulse_ns.copy(), angular, origin)


class ExternalToolPort:
    def __init__(self, solver, record_trajectory=False):
        if not solver.config.external_tool:
            raise ValueError("ExternalToolPort requires Config(external_tool=True)")
        self.solver = solver
        scalar = solver.x.dtype._wp_scalar_type_
        self.packed = wp.zeros(49, dtype=scalar, device=solver.device)
        self.result = wp.zeros(10, dtype=scalar, device=solver.device)
        self.record_trajectory = record_trajectory
        self.trajectory = None
        self.trajectory_times = None
        self._results = {}
        self._replays = {}
        self._pack = np.empty(49, np.float64 if scalar == wp.float64 else np.float32)
        self._pending = None
        self.timing_s = dict(submission=0.0, completion_and_readback=0.0)

    def advance(self, state, duration_s, *, cutting=True, cut_index=0, graph=True):
        self.begin(state, duration_s, cutting=cutting, cut_index=cut_index, graph=graph)
        return self.finish()

    def begin(self, state, duration_s, *, cutting=True, cut_index=0, graph=True):
        """Queue one coupling interval without waiting for the device.

        The tool state is fixed here. ``finish`` reads the wrench. Host work
        that does not need that wrench can run between the two calls, while
        the already queued substeps execute.
        """
        if self._pending is not None:
            raise RuntimeError("The previous coupling interval has not been read back")
        submission_begin = time.perf_counter()
        state = state.checked()
        s = self.solver
        if not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Coupling duration must be positive and finite")
        steps = int(round(duration_s / s.config.dt))
        if steps < 1 or abs(steps * s.config.dt - duration_s) > 1.0e-12 + 1.0e-10 * abs(duration_s):
            raise ValueError("Coupling duration must be an integer number of material substeps")
        # Validate before mutating state. No interpolation, trajectory limits,
        # local PD controller, or table clamp is applied to the external tool.
        cut_index = s._set_cut(cut_index)
        held_pose = s.config.tool_substep == "hold"
        samples = 2 if held_pose else steps + 1
        if self.record_trajectory:
            if steps not in self._results:
                self._results[steps] = wp.zeros(10 + 7 * samples, dtype=self.packed.dtype, device=s.device)
            self.result = self._results[steps]
        pack = self._pack
        pack[0:3] = state.position
        pack[3:7] = state.orientation_xyzw
        pack[7:10] = state.linear_velocity
        pack[10:13] = state.angular_velocity
        pack[13:49] = state.mobility.ravel()
        self.packed.assign(pack)
        if graph:
            # Friction history swaps buffers each substep, so a captured interval
            # is valid only for the parity it was recorded at. Replaying it
            # launches that same graph; the captured arrays keep their identity.
            key = (
                steps,
                bool(cutting),
                int(cut_index),
                s.step_count % 2,
                bool(s.single_field),
                bool(self.record_trajectory),
            )
            replay = self._replays.get(key)
            if replay is None:
                replay = self._capture(steps, cutting, cut_index, held_pose, samples)
                self._replays[key] = replay
            else:
                self._launch_graph(replay)
            s.step_count += steps
        else:
            before, after, sample = self._boundary(held_pose, steps, samples)
            for launch in before:
                s.launch(*launch)
            for i in range(steps):
                s._step_kernels((0.0, 0.0, 0.0), cutting, cut_index)
                if sample is not None:
                    sample(i)
            for launch in after:
                s.launch(*launch)
        self.timing_s["submission"] += time.perf_counter() - submission_begin
        self._pending = (state, float(duration_s), samples)

    def finish(self):
        """Wait for the queued interval and return its wrench."""
        if self._pending is None:
            raise RuntimeError("No coupling interval is waiting for readback")
        state, duration_s, samples = self._pending
        self._pending = None
        readback_begin = time.perf_counter()
        # This wait includes any kernels still running when the host reaches
        # the readback; it is not a standalone PCIe transfer measurement.
        out = self.result.numpy().astype(np.float64)
        self.timing_s["completion_and_readback"] += time.perf_counter() - readback_begin
        impulse, angular, errors = out[:3], out[3:6], out[6:9]
        if self.record_trajectory:
            self.trajectory = out[9 : 9 + 7 * samples].reshape(samples, 7)
            self.trajectory_times = np.linspace(0.0, duration_s, samples)
        (px, py, pz), (ix, iy, iz) = state.position, impulse
        angular -= (py * iz - pz * iy, pz * ix - px * iz, px * iy - py * ix)
        if errors.any() or not np.isfinite([impulse, angular]).all():
            raise RuntimeError(
                f"Invalid material coupling interval: GPU errors {errors.astype(int).tolist()}"
            )
        if not np.isfinite(out[-1]) or out[-1] > 1.0e-6:
            raise RuntimeError(f"Blade grid contact did not converge: inward residual {out[-1]:.6g} m/s")
        return Reaction(duration_s, impulse, angular, state.position.copy())

    def _boundary(self, held_pose, steps, samples):
        s = self.solver
        load = (
            tool.load_external_state,
            1,
            [
                self.packed,
                s.tool_pos,
                s.tool_rot,
                s.tool_vel,
                s.tool_omega,
                s.tool_mobility,
                s.interval_impulse,
                s.interval_angular_impulse,
            ],
        )
        after = [
            (validate_interval, s.n, [s.x, s.v, s.F, s.errors]),
            (tool.store_interval, 1, [s.interval_impulse, s.interval_angular_impulse, s.errors, self.result]),
        ]
        if s.blade_constraint is not None:
            after.append((store_residual, 1, [s.blade_constraint.max_violation, self.result]))
        before = [load]
        sample = None
        if self.record_trajectory:

            def sample(i, _held=held_pose, _steps=steps):
                # In hold mode every intermediate pose is bitwise identical.
                if not _held or i == _steps - 1:
                    index = 1 if _held else i + 1
                    s.launch(tool.record_pose, 1, [s.tool_pos, s.tool_rot, index, self.result])

            before.append((tool.record_pose, 1, [s.tool_pos, s.tool_rot, 0, self.result]))
        return before, after, sample

    def _capture(self, steps, cutting, cut_index, held_pose, samples):
        before, after, sample = self._boundary(held_pose, steps, samples)
        # One replay per interval: loading the tool state and storing the result are in the graph.
        unused_targets = np.zeros((steps, 3), dtype=np.float32)
        _, _, captured = self.solver.prepare(
            unused_targets,
            cutting,
            cut_index,
            before=before,
            after=after,
            tag=("external", id(self)),
            after_step=sample,
        )
        wp.capture_launch(captured)
        stream = captured.device.stream.cuda_stream if captured.device.is_cuda else None
        return captured, captured.graph_exec, stream

    def _launch_graph(self, replay):
        captured, graph_exec, cuda_stream = replay
        if not captured.device.is_cuda:
            wp.capture_launch(captured)
            return
        from warp._src.context import runtime

        if not runtime.core.wp_cuda_graph_launch(graph_exec, cuda_stream):
            raise RuntimeError(runtime.get_error_string() or "CUDA graph launch failed")
