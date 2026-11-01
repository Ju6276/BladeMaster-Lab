"""A prescribed knife, stationary board and natural cohesive material trajectory."""

from time import perf_counter

import numpy as np
import warp as wp

from .cohesive_contact import predict_contacts
from .cohesive_contact_batch import CombinedContacts, split_state
from .cohesive_natural import NaturalCohesive
from .cohesive_rigid_contact import RigidContacts, predict_rigid_contacts
from .cohesive_stages import AffineState, GAMMA
from .differentiable_contact import Tool
from .differentiable_material import MaterialTrajectory
from .natural_checkpoint import checkpoint_vjp, gradient_value, snapshot, state_arrays
from .natural_energy import leaves, measure_step
from .prescribed_path import PrescribedPath


@wp.func
def _position_sums(
    x: wp.array(dtype=wp.vec3d), target: wp.array(dtype=wp.vec3d), weights: wp.array(dtype=wp.float64)
):
    total = wp.float64(0.0)
    weight = wp.float64(0.0)
    for i in range(x.shape[0]):
        delta = x[i] - target[i]
        total += weights[i] * wp.dot(delta, delta)
        weight += weights[i]
    return wp.vec2d(total, weight)


@wp.kernel
def _position_loss(
    x: wp.array(dtype=wp.vec3d),
    target: wp.array(dtype=wp.vec3d),
    weights: wp.array(dtype=wp.float64),
    scale: wp.float64,
    loss: wp.array(dtype=wp.float64),
):
    # Function replay supplies final sums to the nonlinear quotient's adjoint.
    sums = _position_sums(x, target, weights)
    loss[0] += sums[0] / (sums[1] * scale * scale)


@wp.func
def _net_force(force: wp.array(dtype=wp.vec3d), first: int, count: int):
    total = wp.vec3d(0.0, 0.0, 0.0)
    for i in range(count):
        total += force[first + i]
    return total


@wp.kernel
def _force_loss(
    force: wp.array(dtype=wp.vec3d),
    first: int,
    count: int,
    duration: wp.array(dtype=wp.float64),
    coefficient: wp.float64,
    loss: wp.array(dtype=wp.float64),
):
    total = _net_force(force, first, count)
    loss[0] += coefficient * duration[0] * wp.dot(total, total)


class NaturalTask:
    def __init__(
        self,
        config,
        *,
        durations,
        controls,
        position,
        rotation,
        strength,
        fracture_energy,
        contact_parameters,
        diameter,
        offset,
        gravity=(0.0, 0.0, -9.81),
        cutting=True,
        target=None,
        target_weights=None,
        position_scale=0.001,
        force_scale=1.0,
        force_weight=0.0
    ):
        if not config.external_tool or config.blade_grid_constraint or config.label_board_layer:
            raise ValueError("Natural tasks require an external knife and natural geometric labels")
        if any(
            not np.isfinite(v) or v <= 0
            for v in [strength, fracture_energy, diameter, position_scale, force_scale]
        ):
            raise ValueError("Material, length and force scales must be finite and positive")
        if not np.isfinite(force_weight) or force_weight < 0:
            raise ValueError("Force weight must be finite and nonnegative")
        self.path = PrescribedPath(durations, config.dt)
        values = np.asarray(controls, dtype=float)
        if values.shape != (len(durations), 6) or not np.isfinite(values).all():
            raise ValueError("Controls require six finite world velocities per phase")
        m = self.model = MaterialTrajectory(config, precision="float64", ordered_transfers=True)
        self.controls = m.array(values, float)
        self.initial_position = m.array([position], wp.vec3)
        self.initial_rotation = m.array([rotation], wp.quat)
        self.parameters = m.array([contact_parameters], float)
        self.half = m.array([config.blade_half_size], wp.vec3)
        self.bevel = m.array([config.blade_bevel_height], float)
        self.offset = m.array([offset], float)
        self.diameter = m.array([diameter], float)
        self.plane = m.array([[0.0, 0.0, config.grid_spacing / 4]], wp.vec3, False)
        self.normal = m.array(np.tile([0.0, 0.0, 1.0], (m.n, 1)), wp.vec3, False)
        self.board_velocity = m.zeros(1, wp.vec3, False)
        self.point_map = m.array(np.eye(m.n), float, False)
        self.driver = NaturalCohesive(
            m,
            material=m.array(np.tile([strength, fracture_energy, 0.0], (m.n, 1)), wp.vec3),
            contact_provider=self.contacts,
            pair_parameters=self.parameters,
            diameter=self.diameter,
            gravity=gravity,
        )
        self.cutting = cutting
        self.position_scale = position_scale
        self.force_coefficient = force_weight / (force_scale**2 * np.sum(self.path.durations))
        target = m.reference if target is None else np.asarray(target, dtype=float)
        weights = np.ones(m.n) if target_weights is None else np.asarray(target_weights, dtype=float)
        if target.shape != (m.n, 3) or not np.isfinite(target).all():
            raise ValueError("Target positions require one finite world point per particle")
        if (
            weights.shape != (m.n,)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or weights.sum() <= 0
        ):
            raise ValueError("Target weights must be finite, nonnegative and have positive sum")
        self.target = m.array(target, wp.vec3, False)
        self.weights = m.array(weights, float, False)

    def contacts(self, transfer, tool, x, moments, tau, elapsed, history):
        m = self.model
        board_history, knife_history = split_state(history, [m.n, m.n])
        pose = self.driver.pose(tool, elapsed)
        board_parameters = wp.empty((m.n, 5), dtype=wp.float64, device=m.device, requires_grad=True)
        wp.launch(
            _broadcast_parameters, m.n, inputs=[self.parameters], outputs=[board_parameters], device=m.device
        )
        board = predict_contacts(
            x,
            moments,
            tau,
            board_history,
            self.point_map,
            transfer.flow_map,
            self.plane,
            self.board_velocity,
            self.normal,
            board_parameters,
        )
        knife = predict_rigid_contacts(
            x,
            moments,
            tau,
            knife_history,
            self.point_map,
            transfer.flow_map,
            parameters=self.parameters,
            position=pose.position,
            rotation=pose.rotation,
            tool_velocity=tool.velocity,
            omega=tool.omega,
            half=self.half,
            bevel=self.bevel,
            offset=self.offset,
        )
        return CombinedContacts([board, knife])

    def initial(self):
        m = self.model
        state = AffineState.from_material(m.state(), m.zeros(m.n, wp.mat33))
        history = self.driver.history(m.zeros(2 * m.n, wp.vec3))
        tool = Tool(self.initial_position, m.zeros(1, wp.vec3), self.initial_rotation, m.zeros(1, wp.vec3))
        return state, history, tool

    def advance(self, index, state, history, tool):
        return self.path.advance(
            self.driver, index, state, history, tool, self.controls, cutting=self.cutting
        )

    def objective(self, index, result, loss):
        if index == len(self.path.steps) - 1:
            wp.launch(
                _position_loss,
                1,
                inputs=[result.state.x, self.target, self.weights, self.position_scale],
                outputs=[loss],
                device=self.model.device,
            )
        if self.force_coefficient:
            for segment in result.segments:
                for weight, stage in zip(
                    (1 - GAMMA, GAMMA), segment.step.material.contact_stages, strict=True
                ):
                    first = 0
                    for batch in leaves(stage.batch):
                        if isinstance(batch, RigidContacts):
                            wp.launch(
                                _force_loss,
                                1,
                                inputs=[
                                    stage.force.high,
                                    first,
                                    batch.count,
                                    segment.duration,
                                    self.force_coefficient * weight,
                                ],
                                outputs=[loss],
                                device=self.model.device,
                            )
                        first += batch.count

    def energy(self, state, history, tool, result):
        return measure_step(
            self.driver,
            state,
            history,
            tool,
            result,
            parameters=self.parameters,
            plane=self.plane,
            half=self.half,
            bevel=self.bevel,
            offset=self.offset,
        )

    def evaluate(self, *, gradient=False, segment_steps=0, observer=None):
        if gradient and observer is not None:
            raise ValueError("Record observed trajectories in a separate forward evaluation")
        initial, history, tool = self.initial()
        parameters = dict(
            controls=self.controls, young=self.model.young, cohesive_material=self.driver.material
        )
        for value in [*state_arrays(initial, history, tool).values(), *parameters.values()]:
            if value.grad is not None:
                value.grad.zero_()
        if segment_steps:
            if not gradient or observer is not None:
                raise ValueError(
                    "Checkpoint evaluation requires gradients and a separate observed forward pass"
                )
            return checkpoint_vjp(
                initial,
                history,
                tool,
                advance=self.advance,
                objective=self.objective,
                steps=len(self.path.steps),
                segment_steps=segment_steps,
                parameters=parameters,
            )
        from contextlib import nullcontext
        from .natural_checkpoint import NaturalGradient

        tape = wp.Tape() if gradient else None
        loss = self.model.zeros(1, float)
        started = perf_counter()
        state, retained, current_tool = initial, history, tool
        with tape if tape is not None else nullcontext():
            for index in range(len(self.path.steps)):
                result = self.advance(index, state, retained, current_tool)
                self.objective(index, result, loss)
                if observer is not None:
                    observer(index, state, retained, current_tool, result)
                state, retained, current_tool = result.state, result.history, result.tool
        wp.synchronize_device(self.model.device)
        forward_seconds = perf_counter() - started
        started = perf_counter()
        if tape is not None:
            tape.backward(loss=loss)
            wp.synchronize_device(self.model.device)
        boundary_gradients = {k: gradient_value(v) for k, v in state_arrays(initial, history, tool).items()}
        parameter_gradients = {k: gradient_value(v) for k, v in parameters.items()}
        backward_seconds = perf_counter() - started if tape is not None else 0.0
        if tape is not None:
            tape.zero()
        return NaturalGradient(
            snapshot(state, retained, current_tool),
            float(loss.numpy()[0]),
            boundary_gradients,
            parameter_gradients,
            dict(forward_s=forward_seconds, backward_s=backward_seconds),
        )


@wp.kernel
def _broadcast_parameters(source: wp.array2d(dtype=wp.float64), result: wp.array2d(dtype=wp.float64)):
    i = wp.tid()
    for j in range(5):
        result[i, j] = source[0, j]
