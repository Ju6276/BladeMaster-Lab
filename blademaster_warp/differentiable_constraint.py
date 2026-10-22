"""Blade grid constraint with independent projected-solve iterates."""

from dataclasses import dataclass

import warp as wp

from . import blade_constraint as constraint


@wp.kernel
def prepare_timed(
    x: wp.array(dtype=wp.vec3), slots: wp.array2d(dtype=int), weights: wp.array2d(dtype=float),
    velocity: wp.array(dtype=wp.vec3), position: wp.array(dtype=wp.vec3), rotation: wp.array(dtype=wp.quat),
    tool_velocity: wp.array(dtype=wp.vec3), tool_omega: wp.array(dtype=wp.vec3), moving_tool: int,
    half: wp.vec3, bevel: float, radius: float, duration: wp.array(dtype=float),
    side: wp.array(dtype=float), normal: wp.array(dtype=wp.vec3), lever: wp.array(dtype=wp.vec3),
    rhs: wp.array(dtype=float), active_weights: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=float), extrapolated: wp.array(dtype=float), needs_solve: wp.array(dtype=int),
):
    translation = wp.vec3(0.0)
    angular = wp.vec3(0.0)
    if moving_tool != 0:
        translation = tool_velocity[0]
        angular = tool_omega[0]
    constraint.prepare_at(wp.tid(), x, slots, weights, velocity, position, rotation, half, bevel,
                          radius, duration[0], translation, angular, side, normal, lever, rhs, active_weights,
                          impulse, extrapolated, needs_solve)


@wp.kernel
def project(
    slots: wp.array2d(dtype=int), weights: wp.array2d(dtype=float),
    mass: wp.array(dtype=float), coupled: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3), rhs: wp.array(dtype=float), bound: wp.array(dtype=float),
    impulse: wp.array(dtype=float), extrapolated: wp.array(dtype=float), acceleration: float,
    next_impulse: wp.array(dtype=float), next_extrapolated: wp.array(dtype=float),
):
    p = wp.tid()
    current = float(0.0)
    momentum = float(0.0)
    if wp.length_sq(normal[p]) > 0.0:
        response = constraint.response_at(p, slots, weights, mass, coupled)
        current = wp.max(0.0, extrapolated[p] + (rhs[p] - wp.dot(normal[p], response))
                         / wp.max(bound[p], 1.0e-12))
        momentum = current + acceleration * (current - impulse[p])
    next_impulse[p] = current
    next_extrapolated[p] = momentum


@wp.kernel
def apply_grid(mass: wp.array(dtype=float), coupled: wp.array(dtype=wp.vec3),
               velocity: wp.array(dtype=wp.vec3), result: wp.array(dtype=wp.vec3)):
    s = wp.tid()
    value = velocity[s]
    if mass[s] > 0.0:
        value += coupled[s] / mass[s]
    result[s] = value


@wp.kernel
def add_reaction(contact: wp.array(dtype=wp.vec3), constraint_impulse: wp.array(dtype=wp.vec3),
                 duration: wp.array(dtype=float), force: wp.array(dtype=wp.vec3)):
    force[0] = contact[0] - constraint_impulse[0] / duration[0]


@dataclass
class GridConstraintResult:
    velocity: wp.array
    side: wp.array
    linear_impulse: wp.array
    batches: int
    residual: float


def solve(model, grid, mass, velocity, x, tool, old_side=None, fixed_batches=None, *, duration=None,
          moving_tool=False):
    """Same 48-iteration batches and residual stopping rule as BladeConstraint.

    Entry sides and the iteration count are discrete. A supplied fixed_batches
    replays the recorded solve schedule while x, tool pose and iterates carry VJPs.
    """
    cfg = model.config
    duration = model.duration if duration is None else duration
    side = model.zeros(model.n, float, False)
    if old_side is not None:
        wp.copy(side, old_side)
    normal = model.zeros(model.n, wp.vec3)
    lever = model.zeros(model.n, wp.vec3)
    rhs = model.zeros(model.n, float)
    active_weights = model.zeros(grid.count, wp.vec3, False)
    impulse = model.zeros(model.n, float)
    extrapolated = model.zeros(model.n, float)
    needs_solve = model.zeros(1, int, False)
    radius = cfg.grid_spacing / 4
    offset = max(cfg.blade_contact_radius or radius, radius - cfg.blade_half_size[0])
    model.launch(prepare_timed, model.n,
              inputs=[x, grid.slots, model.weights, velocity, tool.position, tool.rotation,
                      tool.velocity, tool.omega, int(moving_tool),
                      cfg.blade_half_size, cfg.blade_bevel_height, offset, duration],
              outputs=[side, normal, lever, rhs, active_weights, impulse, extrapolated, needs_solve],
              device=model.device)
    count = int(needs_solve.numpy()[0]) if fixed_batches is None else fixed_batches
    if count == 0:
        return GridConstraintResult(velocity, side, model.zeros(1, wp.vec3), 0, 0.0)
    coupled_count = model.couple(grid, active_weights)
    bound = model.zeros(model.n, float, False)
    model.launch(constraint.row_bound, model.n,
              inputs=[grid.slots, model.weights, mass, coupled_count], outputs=[bound], device=model.device)

    def scatter(values):
        output = model.zeros(grid.count, wp.vec3)
        linear, angular = model.zeros(1, wp.vec3), model.zeros(1, wp.vec3)
        model.launch(constraint.scatter, model.n,
                  inputs=[grid.slots, model.weights, normal, lever, values],
                  outputs=[output, linear, angular], device=model.device)
        return model.couple(grid, output), linear

    batches, residual_value = 0, 0.0
    while True:
        extrapolated = impulse
        for iteration in range(48):
            coupled, _ = scatter(extrapolated)
            next_impulse = model.zeros(model.n, float)
            next_extrapolated = model.zeros(model.n, float)
            model.launch(project, model.n,
                      inputs=[grid.slots, model.weights, mass, coupled, normal, rhs, bound,
                              impulse, extrapolated, float(iteration / (iteration + 3))],
                      outputs=[next_impulse, next_extrapolated], device=model.device)
            impulse, extrapolated = next_impulse, next_extrapolated
        coupled, linear = scatter(impulse)
        residual = model.zeros(1, float, False)
        model.launch(constraint.record_residual, model.n,
                  inputs=[grid.slots, model.weights, mass, coupled, normal, rhs],
                  outputs=[residual], device=model.device)
        residual_value = float(residual.numpy()[0])
        batches += 1
        if fixed_batches is not None:
            if batches >= fixed_batches:
                break
        elif residual_value <= 1.0e-6 or batches >= 8:
            break
    result = model.zeros(grid.count, wp.vec3)
    model.launch(apply_grid, grid.count, inputs=[mass, coupled, velocity], outputs=[result], device=model.device)
    return GridConstraintResult(result, side, linear, batches, residual_value)
