"""Prescribed rigid-tool contact with separate friction history for every step."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import warp as wp

from .contact import blade_query, penalty_impulse
from .differentiable import State


@wp.kernel
def contact_step(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    history: wp.array2d(dtype=wp.vec3),
    position: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
    rotation: wp.array(dtype=wp.quat),
    omega: wp.array(dtype=wp.vec3),
    half: wp.vec3,
    bevel: float,
    radius: float,
    blade_offset: float,
    board_half: wp.vec2,
    mass: float,
    dt: float,
    sn: float,
    board_sn: float,
    mus: float,
    mud: float,
    gk: float,
    gc: float,
    next_history: wp.array2d(dtype=wp.vec3),
    impulse: wp.array(dtype=wp.vec3),
    reaction: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    xp = x[p]
    board_gap = xp[2] - radius
    if board_half[0] > 0.0 and wp.abs(xp[0]) > board_half[0]:
        board_gap = 1.0e30
    if board_half[1] > 0.0 and wp.abs(xp[1]) > board_half[1]:
        board_gap = 1.0e30
    board, board_history = penalty_impulse(
        board_gap, v[p], wp.vec3(0.0, 0.0, 1.0), mass, history[0, p],
        dt, board_sn, mus, mud, gk, gc,
    )
    q = rotation[0]
    local = wp.quat_rotate_inv(q, xp - position[0])
    closest, normal, distance = blade_query(local, half, bevel)
    lever = wp.quat_rotate(q, closest)
    n = wp.quat_rotate(q, normal)
    relative = v[p] - velocity[0] - wp.cross(omega[0], lever)
    blade, blade_history = penalty_impulse(
        distance - blade_offset, relative, n, mass, history[1, p],
        dt, sn, mus, mud, gk, gc,
    )
    next_history[0, p] = board_history
    next_history[1, p] = blade_history
    impulse[p] = wp.vec3(0.0, 0.0, -9.81 * mass * dt) + board + blade
    wp.atomic_add(reaction, 0, -blade / dt)


@dataclass
class Tool:
    position: wp.array
    velocity: wp.array
    rotation: wp.array
    omega: wp.array


@dataclass
class ContactGradient:
    terminal: State
    history: wp.array
    loss: wp.array
    initial_gradient: State
    history_gradient: wp.array
    parameter_gradients: dict


class PrescribedContact:
    """Zero tool mobility, fixed contact coefficients, no cutting or side labels.

    Reuses the forward solver's blade distance, predictive normal penalty and
    spring-dashpot Coulomb friction. Reaction is the force on the tool in N.
    """

    def __init__(self, trajectory, *, allow_grid_constraint=False):
        self.model = trajectory
        cfg = trajectory.config
        if (not cfg.external_tool or cfg.tool_substep != "hold"
                or (cfg.blade_grid_constraint and not allow_grid_constraint)):
            raise ValueError("PrescribedContact requires external hold mode without grid constraint")

    def history(self):
        return self.model.zeros((2, self.model.n), wp.vec3)

    def tool(self, position, velocity=(0, 0, 0), rotation=(0, 0, 0, 1), omega=(0, 0, 0)):
        from .coupling import ToolState

        checked = ToolState(position, rotation, velocity, omega).checked()
        return Tool(
            self.model.array([checked.position], wp.vec3),
            self.model.array([checked.linear_velocity], wp.vec3),
            self.model.array([checked.orientation_xyzw], wp.quat),
            self.model.array([checked.angular_velocity], wp.vec3),
        )

    def step(self, state, history, tool):
        model, cfg = self.model, self.model.config
        sn, gk = cfg.contact_scales()
        radius = cfg.grid_spacing / 4
        blade_offset = max(cfg.blade_contact_radius or radius, radius - cfg.blade_half_size[0])
        next_history = self.history()
        impulse = model.zeros(model.n, wp.vec3)
        reaction = model.zeros(1, wp.vec3)
        wp.launch(
            contact_step, model.n,
            inputs=[state.x, state.v, history, tool.position, tool.velocity, tool.rotation, tool.omega,
                    wp.vec3(*cfg.blade_half_size), cfg.blade_bevel_height, radius, blade_offset,
                    wp.vec2(*cfg.board_half_extent), model.mass, cfg.dt, sn,
                    cfg.board_normal_scale or sn, cfg.friction_static, cfg.friction_dynamic,
                    gk, cfg.tangent_damping],
            outputs=[next_history, impulse, reaction], device=model.device,
        )
        return model.step(state, impulse=impulse), next_history, reaction

    def checkpoint_vjp(
        self, initial, history, steps, segment_steps, tool_at_step, objective,
        parameters=(), timings=None,
    ):
        """Differentiate an additive objective while retaining segment boundaries.

        tool_at_step(step) builds a Tool from the supplied parameter arrays.
        objective(step, state, history, reaction, loss) accumulates into loss;
        callbacks must reproduce their computations during segment recomputation.
        State, friction history and parameter gradient buffers are cleared here.
        Young's modulus is included in parameter_gradients automatically.
        """
        model = self.model
        if type(initial) is not State:
            raise ValueError("This checkpoint routine requires elastic x/v/F states")
        if not model.requires_grad or steps < 1 or segment_steps < 1:
            raise ValueError("Contact checkpoint VJP requires gradients and positive step counts")
        parameters = tuple(dict.fromkeys((*parameters, model.young)))
        arrays = (initial.x, initial.v, initial.F, history, *parameters)
        if any(array.grad is None for array in arrays):
            raise ValueError("Initial state, history and parameters must have gradient buffers")
        parameter_gradients = {array: np.zeros(array.numpy().shape, dtype=float) for array in parameters}

        def timestamp():
            wp.synchronize_device(model.device)
            return perf_counter()

        measured = dict(forward_s=0.0, recompute_s=0.0, backward_s=0.0)
        begun = timestamp()
        loss = model.zeros(1, float)
        checkpoints = [(0, initial, history)]
        current = initial
        current_history = history
        for step in range(steps):
            current, current_history, reaction = self.step(current, current_history, tool_at_step(step))
            objective(step, current, current_history, reaction, loss)
            if (step + 1) % segment_steps == 0 or step + 1 == steps:
                checkpoints.append((step + 1, current, current_history))
        terminal, terminal_history = current, current_history
        measured["forward_s"] = timestamp() - begun
        gradient = State(model.zeros(model.n, wp.vec3, False), model.zeros(model.n, wp.vec3, False),
                         model.zeros(model.n, wp.mat33, False))
        history_gradient = model.zeros(history.shape, wp.vec3, False)
        loss_seed = model.array([1.0], float, False)
        for index in reversed(range(len(checkpoints) - 1)):
            begun = timestamp()
            first, start, start_history = checkpoints[index]
            last = checkpoints[index + 1][0]
            for array in (start.x, start.v, start.F, start_history, *parameters):
                array.grad.zero_()
            local_loss = model.zeros(1, float)
            with wp.Tape() as tape:
                current, current_history = start, start_history
                for step in range(first, last):
                    current, current_history, reaction = self.step(current, current_history, tool_at_step(step))
                    objective(step, current, current_history, reaction, local_loss)
            now = timestamp()
            measured["recompute_s"] += now - begun
            tape.backward(grads={current.x: gradient.x, current.v: gradient.v, current.F: gradient.F,
                                 current_history: history_gradient, local_loss: loss_seed})
            gradient = State(wp.clone(start.x.grad), wp.clone(start.v.grad), wp.clone(start.F.grad))
            history_gradient = wp.clone(start_history.grad)
            for array in parameters:
                parameter_gradients[array] += array.grad.numpy()
            tape.zero()
            measured["backward_s"] += timestamp() - now
            del tape
        if timings is not None:
            timings.update(measured)
        return ContactGradient(terminal, terminal_history, loss, gradient, history_gradient, parameter_gradients)
