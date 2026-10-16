"""BladeMaster III-G external rigid-tool interface.

The external engine owns tool dynamics. A tool pose/twist is held constant
through an interval of TLMPM substeps, and equal/opposite contact impulses are
returned. Optional spatial mobility J M^-1 J.T supplies articulated effective
mass in contact; omitting it selects the prescribed-tool case (w_B=0).
All quantities are SI and expressed in the material solver's world frame.
"""
from dataclasses import dataclass
import numpy as np
import warp as wp
from .material import validate_interval


def vector(value, length, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain {length} finite values')
    return result


@dataclass(frozen=True)
class ToolState:
    position: object  # Tool origin; must match the collision geometry origin.
    orientation_xyzw: object
    linear_velocity: object  # Velocity at that same origin.
    angular_velocity: object
    mobility: object = None  # World [linear, angular] at the tool origin, 6x6.

    def checked(self):
        position = vector(self.position, 3, 'position')
        q = vector(self.orientation_xyzw, 4, 'orientation_xyzw')
        if not np.isclose(np.linalg.norm(q), 1., rtol=0., atol=1.e-5):
            raise ValueError('Tool orientation must be a unit xyzw quaternion')
        mobility = np.zeros((6, 6)) if self.mobility is None else np.asarray(self.mobility, dtype=float)
        if (mobility.shape != (6, 6) or not np.isfinite(mobility).all()
                or not np.allclose(mobility, mobility.T, atol=1.e-10)
                or np.linalg.eigvalsh(mobility).min() < -1.e-10):
            raise ValueError('Mobility must be a finite symmetric positive-semidefinite 6x6 matrix')
        return ToolState(position, q/np.linalg.norm(q),
                         vector(self.linear_velocity, 3, 'linear_velocity'),
                         vector(self.angular_velocity, 3, 'angular_velocity'), mobility.copy())


@dataclass(frozen=True)
class Reaction:
    duration_s: float
    impulse_ns: np.ndarray
    angular_impulse_nms: np.ndarray
    origin_world_m: np.ndarray

    @property
    def force_n(self):
        return self.impulse_ns/self.duration_s

    @property
    def torque_nm(self):
        return self.angular_impulse_nms/self.duration_s

    def about(self, origin_world_m):
        """Shift the moment reference, preserving the same physical wrench."""
        origin = vector(origin_world_m, 3, 'moment origin')
        angular = self.angular_impulse_nms + np.cross(
            self.origin_world_m-origin, self.impulse_ns)
        return Reaction(self.duration_s, self.impulse_ns.copy(), angular, origin)


class ExternalToolPort:
    def __init__(self, solver):
        if not solver.config.external_tool:
            raise ValueError('ExternalToolPort requires Config(external_tool=True)')
        self.solver = solver

    def advance(self, state, duration_s, *, cutting=True, cut_index=0, graph=True):
        state = state.checked()
        s = self.solver
        if not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError('Coupling duration must be positive and finite')
        steps = int(round(duration_s/s.config.dt))
        if steps < 1 or not np.isclose(steps*s.config.dt, duration_s, rtol=1.e-10, atol=1.e-12):
            raise ValueError('Coupling duration must be an integer number of material substeps')
        # Validate before mutating state. No interpolation, trajectory limits,
        # local PD controller, or table clamp is applied to the external tool.
        s._set_cut(cut_index)
        s.bx.assign([state.position])
        s.bq.assign([state.orientation_xyzw])
        s.bv.assign([state.linear_velocity])
        s.bw.assign([state.angular_velocity])
        s.tool_mobility.assign(state.mobility.astype(np.float32))
        s.interval_impulse.zero_()
        s.interval_angular_impulse.zero_()
        if graph:
            unused_targets = np.zeros((steps, 3), dtype=np.float32)
            _, _, captured = s.prepare(unused_targets, cutting, cut_index)
            wp.capture_launch(captured)
            s.step_count += steps
        else:
            for _ in range(steps):
                s._step_kernels((0., 0., 0.), cutting, cut_index)
        s.launch(validate_interval,s.n,[s.x,s.v,s.F,s.errors])
        impulse = s.interval_impulse.numpy()[0].astype(np.float64)
        angular = s.interval_angular_impulse.numpy()[0].astype(np.float64)
        angular -= np.cross(state.position, impulse)
        errors = s.errors.numpy()
        if errors.any() or not np.isfinite([impulse, angular]).all():
            raise RuntimeError(f'Invalid material coupling interval: GPU errors {errors.tolist()}')
        return Reaction(float(duration_s), impulse, angular, state.position.copy())
