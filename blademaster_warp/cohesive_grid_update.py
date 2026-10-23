"""First-order MUSL with intact grid constraints and moving cohesive interfaces."""

from dataclasses import dataclass

import numpy as np
import warp as wp

from .cohesive_geometry import InterfaceGeometry, solve_geometry
from .differentiable_cohesive import CohesiveStep
from .differentiable_material import PlasticState
from .grid_constraints import project_velocity


@wp.kernel
def divide_mass(
    momentum: wp.array(dtype=wp.vec3d), mass: wp.array(dtype=wp.float64), velocity: wp.array(dtype=wp.vec3d)
):
    i = wp.tid()
    velocity[i] = momentum[i] / mass[i]


@wp.kernel
def multiply_mass(
    velocity: wp.array(dtype=wp.vec3d), mass: wp.array(dtype=wp.float64), momentum: wp.array(dtype=wp.vec3d)
):
    i = wp.tid()
    momentum[i] = mass[i] * velocity[i]


@wp.kernel
def whiten(
    mapping: wp.array2d(dtype=wp.float64),
    gradient: wp.array2d(dtype=wp.vec3d),
    mass: wp.array(dtype=wp.float64),
    result: wp.array2d(dtype=wp.float64),
    result_gradient: wp.array2d(dtype=wp.vec3d),
):
    q, i = wp.tid()
    result[q, i] = mapping[q, i] / wp.sqrt(mass[i])
    result_gradient[q, i] = gradient[q, i] / wp.sqrt(mass[i])


@wp.kernel
def closed_gap(
    previous: wp.array(dtype=wp.vec3d),
    advanced: wp.array(dtype=wp.vec3d),
    intact: wp.array(dtype=int),
    result: wp.array(dtype=wp.vec3d),
):
    q = wp.tid()
    result[q] = advanced[q]
    if intact[q] != 0:
        result[q] = previous[q]


@wp.kernel
def add_velocity(
    free: wp.array(dtype=wp.vec3d),
    correction: wp.array(dtype=wp.vec3d),
    mass: wp.array(dtype=wp.float64),
    duration: wp.array(dtype=wp.float64),
    result: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    result[i] = free[i] + correction[i] / (wp.sqrt(mass[i]) * duration[0])


@dataclass(frozen=True)
class CohesiveMaterialStep:
    state: PlasticState
    geometry: InterfaceGeometry
    interface: CohesiveStep
    old_velocity: wp.array
    new_velocity: wp.array


def advance_material(
    model,
    state,
    geometry,
    history,
    reference_normal,
    material,
    separation_map,
    frame_map,
    area,
    *,
    topology,
    duration=None,
    impulse=None,
    gravity=(0.0, 0.0, -9.81)
):
    """Advance material and interface geometry with the same grid velocity.

    Maps use reference coordinates; area is in m² and history is in metres.
    The zero-history set is intact. Positive history remains active on unloading.
    """
    duration = model.duration if duration is None else duration
    if model.precision != "float64":
        raise ValueError("Cohesive grid updates require float64 material arrays")
    if separation_map.shape != (len(history), topology.count):
        raise ValueError("Interface maps must match the history and topology slots")
    intact = history.numpy() == 0.0
    active = model.array(intact.astype(np.int32), int, False)
    unit_mass = model.array([1.0], float, False)
    data = {}

    def momentum_transform(grid, mass, momentum, phase):
        velocity = model.zeros(grid.count, wp.vec3)
        wp.launch(divide_mass, grid.count, inputs=[momentum, mass], outputs=[velocity], device=model.device)
        if phase == "load":
            data["increment"] = velocity
            return momentum
        mask = intact if phase == "state" else data["response"].history.numpy() == 0.0
        velocity = project_velocity(separation_map, mass, velocity, mask)
        if phase == "state":
            data["old_velocity"] = velocity
        projected = model.zeros(grid.count, wp.vec3)
        wp.launch(
            multiply_mass, grid.count, inputs=[velocity, mass], outputs=[projected], device=model.device
        )
        return projected

    def update_velocity(grid, mass, free_velocity):
        base = geometry.advance(separation_map, frame_map, data["old_velocity"], duration)
        compatible_gap = model.zeros(len(history), wp.vec3)
        # Q times the intact projected velocity is identically zero.
        wp.launch(
            closed_gap,
            len(history),
            inputs=[geometry.gap, base.gap, active],
            outputs=[compatible_gap],
            device=model.device,
        )
        base = InterfaceGeometry(
            compatible_gap,
            base.frame,
            np.where(intact[:, None], geometry.gap_remainder, base.gap_remainder),
            base.frame_remainder,
        )
        free = base.advance(separation_map, frame_map, data["increment"], duration)
        mapping = model.zeros(separation_map.shape, float)
        gradient = model.zeros(frame_map.shape, wp.vec3)
        wp.launch(
            whiten,
            separation_map.shape,
            inputs=[separation_map, frame_map, mass],
            outputs=[mapping, gradient],
            device=model.device,
        )
        response, updated = solve_geometry(
            free, reference_normal, history, material, mapping, gradient, area, unit_mass, duration
        )
        velocity = model.zeros(grid.count, wp.vec3)
        wp.launch(
            add_velocity,
            grid.count,
            inputs=[free_velocity, response.displacement, mass, duration],
            outputs=[velocity],
            device=model.device,
        )
        data.update(response=response, geometry=updated, new_velocity=velocity)
        return velocity

    result = model.step(
        state,
        gravity=gravity,
        impulse=impulse,
        topology=topology,
        duration=duration,
        momentum_transform=momentum_transform,
        grid_transform=update_velocity,
    )
    return CohesiveMaterialStep(
        result, data["geometry"], data["response"], data["old_velocity"], data["new_velocity"]
    )
