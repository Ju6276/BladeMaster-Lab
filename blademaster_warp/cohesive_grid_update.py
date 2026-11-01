"""First-order MUSL with intact grid constraints and moving cohesive interfaces."""

from dataclasses import dataclass

import numpy as np
import warp as wp

from .cohesive_geometry import InterfaceGeometry, solve_geometry
from .differentiable_cohesive import CohesiveForceStep
from .differentiable_material import PlasticState
from .grid_constraints import project_velocity
from ._cohesive_expansion import E, root, divide
from .compensated_state import CompensatedArray, add_gradient


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


def mass_coordinates(mapping, gradient, velocity, mass):
    """Geometry and momentum use the same positive-mass coordinate transform."""
    B, G, v, weights = [E(source.numpy()) for source in [mapping, gradient, velocity, mass]]
    if np.any(weights.high <= 0.0):
        raise ValueError("Mass coordinates require positive active grid masses")
    roots = root(weights)
    mapped = divide(B, roots[None, :])
    derivative = divide(G, roots[None, :, None])
    speed = roots[:, None] * v
    outputs = [
        CompensatedArray.from_cpu(value, dtype=source.dtype, device=source.device)
        for value, source in [(mapped, mapping), (derivative, gradient), (speed, velocity)]
    ]
    tape = wp._src.context.runtime.tape
    if tape is not None:

        def backward():
            b, g, z = [E(value.high.grad.numpy()) for value in outputs]
            add_gradient(mapping, divide(b, roots[None, :]))
            add_gradient(gradient, divide(g, roots[None, :, None]))
            add_gradient(velocity, roots[:, None] * z)
            contraction = (z * speed).sum(axis=1)
            contraction = contraction - (b * mapped).sum(axis=0)
            contraction = contraction - (g * derivative).sum(axis=0).sum(axis=1)
            add_gradient(mass, divide(0.5 * contraction, weights))

        sources = [mapping, gradient, velocity, mass]
        arrays = [value.high for value in outputs] + [a for a in sources if a.requires_grad]
        tape.record_func(backward, [a for a in arrays if a.size])
    return outputs


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
    interface: CohesiveForceStep
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
    Positive-area, zero-history points are intact. Reactions are integrated forces
    in N; storage and fracture outputs remain densities per reference area.
    """
    duration = model.duration if duration is None else duration
    if model.precision != "float64":
        raise ValueError("Cohesive grid updates require float64 material arrays")
    if separation_map.shape != (len(history), topology.count):
        raise ValueError("Interface maps must match the history and topology slots")
    support = area.numpy() > 0.0
    intact = (history.numpy() == 0.0) & support
    active = model.array(intact.astype(np.int32), int, False)
    unit_mass = model.array([1.0], float, False)
    data = {}

    def momentum_transform(grid, mass, momentum, phase):
        velocity = model.zeros(grid.count, wp.vec3)
        wp.launch(divide_mass, grid.count, inputs=[momentum, mass], outputs=[velocity], device=model.device)
        if phase == "load":
            data["increment"] = velocity
            return momentum
        mask = intact if phase == "state" else (data["response"].history.numpy() == 0.0) & support
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
        mapping, gradient, speed = mass_coordinates(separation_map, frame_map, data["increment"], mass)
        free = base.advance(mapping, gradient, speed, duration)
        closed_motion = None
        if (
            np.all(intact[support])
            and np.all(geometry.gap.numpy()[support] == 0.0)
            and np.all(geometry.gap_remainder[support] == 0.0)
        ):
            closed_motion = (
                duration.numpy().astype(np.longdouble)[0]
                * np.sqrt(mass.numpy().astype(np.longdouble))[:, None]
                * data["increment"].numpy().astype(np.longdouble)
            )
        response, updated = solve_geometry(
            free,
            reference_normal,
            history,
            material,
            mapping,
            gradient,
            area,
            unit_mass,
            duration,
            integrated_force=True,
            closed_motion=closed_motion,
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
