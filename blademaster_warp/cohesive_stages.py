"""ARS(2,2,2) with explicit bulk force and joint implicit cohesive/damping stages.

Position rates are transported velocities; particle translation moments and
reference velocity gradients remain separate state arrays.
"""

from dataclasses import dataclass, field, fields, replace
from typing import NamedTuple

import numpy as np
import warp as wp

from .cohesive_geometry import InterfaceGeometry, solve_geometry
from .differentiable_cohesive import CohesiveForceStep
from .differentiable_material import PlasticState, project_plastic
from .reference_inertia import ReferenceInertia
from .compensated_state import CompensatedArray, as_compensated
from ._affine_arithmetic import AffineArithmetic
from ._increment_stress import compute as compute_increment_stress


GAMMA = 1.0 - 1.0 / np.sqrt(2.0)
DELTA = 1.0 - 1.0 / (2.0 * GAMMA)


@wp.kernel
def _body_force(gravity: wp.vec3d, mass: wp.float64, result: wp.array(dtype=wp.vec3d)):
    result[wp.tid()] = mass * gravity


@wp.kernel
def _impulse_force(
    impulse: wp.array(dtype=wp.vec3d),
    duration: wp.array(dtype=wp.float64),
    result: wp.array(dtype=wp.vec3d),
):
    p = wp.tid()
    result[p] = impulse[p] / duration[0]


@wp.kernel
def _path_midpoint(
    start: wp.array(dtype=wp.vec3d),
    end: wp.array(dtype=wp.vec3d),
    start_rate: wp.array(dtype=wp.vec3d),
    end_rate: wp.array(dtype=wp.vec3d),
    duration: wp.array(dtype=wp.float64),
    result: wp.array(dtype=wp.vec3d),
):
    p = wp.tid()
    result[p] = wp.float64(0.5) * (start[p] + end[p]) + wp.float64(0.125) * duration[0] * (
        start_rate[p] - end_rate[p]
    )


@wp.kernel
def _damping_energy(
    z1: wp.array(dtype=wp.vec3d),
    l1: wp.array(dtype=wp.vec3d),
    z2: wp.array(dtype=wp.vec3d),
    l2: wp.array(dtype=wp.vec3d),
    duration: wp.array(dtype=wp.float64),
    rate: wp.array(dtype=wp.float64),
    mass: wp.float64,
    gamma: wp.float64,
    result: wp.array(dtype=wp.float64),
):
    p = wp.tid()
    value = wp.float64(0.0)
    for a in range(4):
        j = 4 * p + a
        value += (wp.float64(1.0) - gamma) * wp.dot(z1[j], l1[j]) + gamma * wp.dot(z2[j], l2[j])
    result[p] = mass * duration[0] * rate[0] * value


@dataclass
class AffineState(PlasticState):
    affine: wp.array | None = None
    rounding_remainders: dict = field(default_factory=dict)

    @classmethod
    def from_material(cls, state, affine):
        if (
            affine.dtype != wp.mat33d
            or affine.shape != state.v.shape
            or affine.device != state.v.device
            or state.v.dtype != wp.vec3d
        ):
            raise ValueError("Affine gradients must match float64 particle velocities")
        return cls(
            **{field.name: getattr(state, field.name) for field in fields(PlasticState)}, affine=affine
        )

    def capture_remainders(self, arithmetic):
        self.rounding_remainders = {
            item.name: arithmetic.state(getattr(self, item.name)).low.copy()
            for item in fields(self)
            if item.name != "rounding_remainders"
            and getattr(self, item.name) is not None
            and getattr(self, item.name).numpy().dtype == np.float64
        }
        return self

    def snapshot(self):
        result = {
            item.name: None if getattr(self, item.name) is None else getattr(self, item.name).numpy().copy()
            for item in fields(self)
            if item.name != "rounding_remainders"
        }
        result["rounding_remainders"] = {
            name: value.copy() for name, value in self.rounding_remainders.items()
        }
        return result

    @classmethod
    def restore(cls, snapshot, *, device, requires_grad=True):
        kinds = dict(
            x=wp.vec3d, v=wp.vec3d, F=wp.mat33d, Fp=wp.mat33d, yielded=int, stress=wp.mat33d, affine=wp.mat33d
        )
        return cls(
            **{
                name: (
                    None
                    if snapshot[name] is None
                    else wp.array(
                        snapshot[name], dtype=kind, device=device, requires_grad=requires_grad and kind != int
                    )
                )
                for name, kind in kinds.items()
            },
            rounding_remainders={
                name: value.copy() for name, value in snapshot.get("rounding_remainders", {}).items()
            },
        )


class ContactStage(NamedTuple):
    batch: object
    force: CompensatedArray
    history: CompensatedArray
    correction: CompensatedArray
    duration: wp.array


@dataclass(frozen=True)
class AffineMaterialStep:
    state: AffineState
    geometry: InterfaceGeometry
    history: wp.array
    interface: CohesiveForceStep | None
    stage: AffineState
    path_midpoint: wp.array
    damping_energy: wp.array
    contact_history: CompensatedArray | None = None
    contact_stages: tuple = ()


def advance_affine(
    transfer,
    state,
    geometry,
    history,
    reference_normal,
    material,
    separation_map,
    frame_map,
    area,
    *,
    duration,
    impulse=None,
    gravity=(0.0, 0.0, -9.81),
    force=None,
    damping_rate=None,
    contacts=None,
    contact_history=None,
    constrain_normal=True,
):
    """Advance material and permanent interface history over one event segment.

    Interface maps act on packed affine moments. A force callback receives
    (stage_state, transported_velocity, elapsed_time) and returns particle forces
    in N; it owns no mutable history. Impulse is a constant-in-time particle load
    over this segment and replaces gravity, matching MaterialTrajectory.step.
    The callback replaces both impulse and gravity. Stage plastic returns use
    the segment's initial Fp; the final return supplies the retained plastic state.
    Transfer damping shares the implicit stages at a fixed rate in s^-1;
    its default is (1 - flip_fraction) / (flip_reference_dt or config.dt).
    contacts(x, filtered_moments, stage_duration, elapsed, history) returns
    MappedContacts for that stage; x, moments and history retain rounding
    remainders. The provider evaluates the tool at the supplied stage end time.
    """
    m = transfer.model
    arithmetic = AffineArithmetic(state)
    if (contacts is None) != (contact_history is None):
        raise ValueError("Contact stages require both a provider and elastic history")
    if not constrain_normal and contacts is None:
        raise ValueError("Separated adhesion requires an independent contact provider")
    initial_contact = None if contacts is None else arithmetic.remember(as_compensated(contact_history))
    contact_outputs = []
    if not isinstance(state, AffineState) or state.affine is None:
        raise ValueError("Affine material stages require independent velocity-gradient state")
    transfer._check(duration, wp.float64, (1,))
    if not np.isfinite(duration.numpy()).all() or duration.numpy()[0] <= 0.0:
        raise ValueError("Affine stage duration must be positive and finite")
    if force is not None and impulse is not None:
        raise ValueError("Specify a force callback or a segment impulse")
    count, size = len(history), 4 * m.n
    transfer._check(as_compensated(separation_map).high, wp.float64, (count, size))
    transfer._check(as_compensated(frame_map).high, wp.vec3d, (count, size))
    for array, kind, shape in [
        (state.x, wp.vec3d, (m.n,)),
        (state.F, wp.mat33d, (m.n,)),
        (state.Fp, wp.mat33d, (m.n,)),
        (history, wp.float64, (count,)),
        (geometry.gap, wp.vec3d, (count,)),
        (geometry.frame, wp.mat33d, (count,)),
        (reference_normal, wp.vec3d, (count,)),
        (material, wp.vec3d, (count,)),
        (area, wp.float64, (count,)),
    ]:
        transfer._check(array, kind, shape)

    def launch(kernel, shape, sources, kind=wp.vec3d):
        result = m.zeros(shape, kind)
        wp.launch(kernel, shape, inputs=sources, outputs=[result], device=m.device)
        return result

    strains = {state.F: arithmetic.shift_identity(state.F, -1.0)}
    free_branch = m.zeros(m.n, int, False)
    free_branch.fill_(-1)

    def material_state(x, moments, F):
        velocity, affine = arithmetic.unpack(transfer, moments)
        result = AffineState(
            x, velocity, F, m.zeros(m.n, wp.mat33), m.zeros(m.n, int, False), m.zeros(m.n, wp.mat33), affine
        )
        errors = m.zeros(1, int, False)
        m.launch(
            project_plastic,
            m.n,
            inputs=[F, state.Fp, m.young, m.yield_stress, m.config.poisson_ratio, free_branch],
            outputs=[result.Fp, result.yielded, result.stress, errors],
            device=m.device,
        )
        if int(errors.numpy()[0]):
            raise FloatingPointError("Affine material-stage return mapping failed")
        result.stress = launch(
            compute_increment_stress, m.n, [strains[F], result.Fp, m.young, m.config.poisson_ratio], wp.mat33d
        )
        return result.capture_remainders(arithmetic)

    constant_force = None
    if force is None:
        if impulse is None:
            constant_force = launch(_body_force, m.n, [wp.vec3d(*gravity), m.mass])
        else:
            transfer._check(impulse, wp.vec3d, (m.n,))
            constant_force = launch(_impulse_force, m.n, [impulse, duration])

    def explicit(value, moments, elapsed):
        physical_velocity, _ = arithmetic.kinematics(transfer, moments)
        load = constant_force if force is None else force(value, physical_velocity, elapsed)
        transfer._check(load, wp.vec3d, (m.n,))
        moment = arithmetic.scale(value.stress, -m.volume)
        return arithmetic.scale(arithmetic.pullback_force(transfer, load, moment), m.mass, inverse=True)

    def move(x, F, moments, dt, factor=1.0):
        velocity, rate = arithmetic.kinematics(transfer, moments)
        H = arithmetic.increment(strains[F], rate, dt, factor)
        next_F = arithmetic.shift_identity(H, 1.0)
        strains[next_F] = H
        return arithmetic.increment(x, velocity, dt, factor), next_F

    tau = arithmetic.scale(duration, GAMMA)
    mass = m.array([m.mass], float, False)
    if damping_rate is None:
        reference_dt = m.config.flip_reference_dt or m.config.dt
        damping_rate = m.array([(1.0 - m.config.flip_fraction) / reference_dt], float, False)
    inertia = ReferenceInertia(transfer, tau, damping_rate)

    def stage(x, F, moments, base_geometry, maximum, contact_base, elapsed):
        original = moments
        moments = arithmetic.inverse(inertia, original)
        if contacts is not None:
            from .cohesive_contact import solve as solve_contact, history_rate
            from .differentiable_cohesive import _evaluate

            free = (
                arithmetic.advance_geometry(inertia, base_geometry, separation_map, frame_map, original)
                if count
                else base_geometry
            )
            batch = contacts(
                arithmetic.state(x),
                arithmetic.state(moments),
                arithmetic.state(tau),
                elapsed,
                arithmetic.state(contact_base),
            )
            inputs = [
                free.gap,
                free.frame,
                reference_normal,
                maximum,
                material,
                as_compensated(separation_map).high,
                as_compensated(frame_map).high,
                area,
                mass,
                tau,
            ]
            low_maps = (as_compensated(separation_map).low, as_compensated(frame_map).low)
            root, displacement, selected, contact_force, updated_history = solve_contact(
                inputs, free, inertia, batch, low_maps, constrain_normal=constrain_normal
            )
            arithmetic.remember(CompensatedArray(displacement, root.displacement.low))
            acceleration = arithmetic.acceleration(displacement, tau)
            moments = arithmetic.increment(moments, acceleration, tau)
            if count:
                raw, _, _ = _evaluate(
                    inputs,
                    displacement,
                    selected,
                    root.compliance,
                    integrated_force=True,
                    inertia=inertia,
                    remainders=(free.gap_remainder, free.frame_remainder, root.displacement.low),
                    map_remainders=low_maps,
                    constrain_normal=constrain_normal,
                )
                response = CohesiveForceStep(*raw, displacement, root.info, root.displacement.low)
                response.gap.assign(root.gap.high)
                updated = InterfaceGeometry(response.gap, response.frame, root.gap.low, root.frame.low)
                maximum = response.history
            else:
                response, updated = None, base_geometry
            contact_rate = arithmetic.remember(
                history_rate(updated_history, arithmetic.state(contact_base), arithmetic.state(tau))
            )
            next_contact = arithmetic.remember(updated_history)
            contact_outputs.append(ContactStage(
                batch, contact_force, updated_history,
                CompensatedArray(displacement, root.displacement.low), tau,
            ))
        elif count:
            free = arithmetic.advance_geometry(inertia, base_geometry, separation_map, frame_map, original)
            response, updated = solve_geometry(
                free,
                reference_normal,
                maximum,
                material,
                separation_map,
                frame_map,
                area,
                mass,
                tau,
                integrated_force=True,
                inertia=inertia,
            )
            arithmetic.remember(response.displacement_state)
            acceleration = arithmetic.acceleration(response.displacement, tau)
            moments = arithmetic.increment(moments, acceleration, tau)
            maximum = response.history
        else:
            response, updated = None, base_geometry
            acceleration = m.zeros(size, wp.vec3)
        generator = arithmetic.generator(inertia, original)
        filtered_rate = arithmetic.inverse(inertia, generator)
        acceleration = arithmetic.increment(acceleration, filtered_rate, damping_rate, -1.0)
        xp, Fp = move(x, F, moments, tau)
        result_state = material_state(xp, moments, Fp)
        if contacts is None:
            next_contact, contact_rate = None, None
        return result_state, moments, updated, maximum, response, acceleration, next_contact, contact_rate

    incoming = arithmetic.pack(transfer, state.v, state.affine)
    z0 = incoming
    initial = state
    if state.stress is None:
        stress = launch(
            compute_increment_stress,
            m.n,
            [strains[state.F], state.Fp, m.young, m.config.poisson_ratio],
            wp.mat33d,
        )
        initial = replace(initial, stress=stress)
    e0 = explicit(initial, z0, m.zeros(1, float, False))
    zbase = arithmetic.increment(z0, e0, duration, GAMMA)
    mid, z1, geometry1, history1, _, a1, _, rate1 = stage(
        state.x, state.F, zbase, geometry, history, initial_contact, tau
    )
    e1 = explicit(mid, z1, tau)
    xb, Fb = move(state.x, state.F, z1, duration, 1.0 - GAMMA)
    gb = geometry.extrapolate(geometry1, (1.0 - GAMMA) / GAMMA) if count else geometry
    zbase = arithmetic.increment(z0, a1, duration, 1.0 - GAMMA)
    zbase = arithmetic.increment(zbase, e0, duration, DELTA)
    zbase = arithmetic.increment(zbase, e1, duration, 1.0 - DELTA)
    contact_base = (
        None if contacts is None else arithmetic.increment(initial_contact, rate1, duration, 1.0 - GAMMA)
    )
    result, zend, updated, maximum, response, _, final_contact, _ = stage(
        xb, Fb, zbase, gb, history1, contact_base, duration
    )
    dissipated = launch(
        _damping_energy,
        m.n,
        [
            z1,
            arithmetic.generator(inertia, z1),
            zend,
            arithmetic.generator(inertia, zend),
            duration,
            damping_rate,
            m.mass,
            GAMMA,
        ],
        wp.float64,
    )
    start_rate, _ = arithmetic.kinematics(transfer, incoming)
    end_rate, _ = arithmetic.kinematics(transfer, zend)
    midpoint = launch(_path_midpoint, m.n, [state.x, result.x, start_rate, end_rate, duration])
    final_history = None if contacts is None else arithmetic.state(final_contact)
    return AffineMaterialStep(
        result, updated, maximum, response, mid, midpoint, dissipated, final_history,
        tuple(contact_outputs)
    )
