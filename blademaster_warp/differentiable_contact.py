"""Prescribed rigid-tool contact with separate friction history for every step."""

from dataclasses import dataclass
import warp as wp

from .contact import blade_query, penalty_impulse


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


class PrescribedContact:
    """Zero tool mobility, fixed contact coefficients, no cutting or side labels.

    Reuses the forward solver's blade distance, predictive normal penalty and
    spring-dashpot Coulomb friction. Reaction is the force on the tool in N.
    """

    def __init__(self, trajectory):
        self.model = trajectory
        cfg = trajectory.config
        if not cfg.external_tool or cfg.tool_substep != "hold" or cfg.blade_grid_constraint:
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
