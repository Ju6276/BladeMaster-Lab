"""Piecewise constant world velocities with continuous tool pose."""

from dataclasses import dataclass

import numpy as np
import warp as wp

from blademaster_warp.differentiable_contact import Tool


@wp.kernel
def _velocity(
    controls: wp.array2d(dtype=wp.float64),
    phase: int,
    linear: wp.array(dtype=wp.vec3d),
    angular: wp.array(dtype=wp.vec3d),
):
    linear[0] = wp.vec3d(controls[phase, 0], controls[phase, 1], controls[phase, 2])
    angular[0] = wp.vec3d(controls[phase, 3], controls[phase, 4], controls[phase, 5])


@dataclass(frozen=True)
class MotionStep:
    phase: int
    duration: float


class PrescribedPath:
    """Durations are seconds; controls are world linear m/s and angular rad/s.

    Phase durations and maximum integration step define a fixed time partition.
    Initial pose and six velocity controls per phase remain differentiable.
    """

    def __init__(self, durations, maximum_step):
        durations = np.asarray(durations, dtype=float)
        if (
            durations.ndim != 1
            or not len(durations)
            or not np.isfinite(durations).all()
            or np.any(durations <= 0)
        ):
            raise ValueError("Phase durations must be finite positive seconds")
        if not np.isfinite(maximum_step) or maximum_step <= 0:
            raise ValueError("Maximum integration step must be finite and positive")
        self.durations = durations.copy()
        self.steps = []
        for phase, duration in enumerate(durations):
            count = int(np.ceil(duration / maximum_step))
            edges = np.linspace(0.0, duration, count + 1)
            self.steps.extend(MotionStep(phase, float(dt)) for dt in np.diff(edges))
        self.steps = tuple(self.steps)

    def inputs(self, index, tool, controls):
        if controls.dtype != wp.float64 or controls.shape != (len(self.durations), 6):
            raise ValueError("Each phase requires six float64 velocity controls")
        if controls.device != tool.position.device:
            raise ValueError("Tool and controls must share a device")
        step = self.steps[index]
        linear = wp.zeros(1, dtype=wp.vec3d, device=controls.device, requires_grad=True)
        angular = wp.zeros(1, dtype=wp.vec3d, device=controls.device, requires_grad=True)
        wp.launch(
            _velocity, 1, inputs=[controls, step.phase], outputs=[linear, angular], device=controls.device
        )
        duration = wp.array([step.duration], dtype=wp.float64, device=controls.device)
        return Tool(tool.position, linear, tool.rotation, angular), duration

    def advance(self, driver, index, state, history, tool, controls, *, cutting=True):
        controlled, duration = self.inputs(index, tool, controls)
        return driver.step(state, history, controlled, duration=duration, cutting=cutting)
