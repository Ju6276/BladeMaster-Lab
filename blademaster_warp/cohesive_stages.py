"""ARS(2,2,2) with explicit bulk forces and implicit cohesive motion.

Particle translation and reference velocity gradients remain separate state
arrays. Contact force laws use the transported position rate, not the stored
translation moment. Both material and interface rates use the same transfer.
"""

from dataclasses import dataclass, fields, replace

import numpy as np
import warp as wp

from .cohesive_geometry import InterfaceGeometry, solve_geometry
from .differentiable_cohesive import CohesiveForceStep
from .differentiable_material import PlasticState, compute_stress, project_plastic


GAMMA = 1.0 - 1.0 / np.sqrt(2.0)
DELTA = 1.0 - 1.0 / (2.0 * GAMMA)


@wp.kernel
def _duration(source: wp.array(dtype=wp.float64), factor: wp.float64, result: wp.array(dtype=wp.float64)):
    result[0] = factor * source[0]


@wp.kernel
def _vector_increment(
    base: wp.array(dtype=wp.vec3d),
    rate: wp.array(dtype=wp.vec3d),
    duration: wp.array(dtype=wp.float64),
    factor: wp.float64,
    result: wp.array(dtype=wp.vec3d),
):
    p = wp.tid()
    result[p] = base[p] + factor * duration[0] * rate[p]


@wp.kernel
def _matrix_increment(
    base: wp.array(dtype=wp.mat33d),
    rate: wp.array(dtype=wp.mat33d),
    duration: wp.array(dtype=wp.float64),
    factor: wp.float64,
    result: wp.array(dtype=wp.mat33d),
):
    p = wp.tid()
    result[p] = base[p] + factor * duration[0] * rate[p]


@wp.kernel
def _stress_moment(stress: wp.array(dtype=wp.mat33d), volume: wp.float64, result: wp.array(dtype=wp.mat33d)):
    p = wp.tid()
    result[p] = -volume * stress[p]


@wp.kernel
def _acceleration(force: wp.array(dtype=wp.vec3d), mass: wp.float64, result: wp.array(dtype=wp.vec3d)):
    p = wp.tid()
    result[p] = force[p] / mass


@wp.kernel
def _cohesive_acceleration(
    displacement: wp.array(dtype=wp.vec3d),
    duration: wp.array(dtype=wp.float64),
    result: wp.array(dtype=wp.vec3d),
):
    p = wp.tid()
    result[p] = displacement[p] / (duration[0] * duration[0])


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


@dataclass
class AffineState(PlasticState):
    affine: wp.array | None = None

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

    def snapshot(self):
        return {
            field.name: (
                None if getattr(self, field.name) is None else getattr(self, field.name).numpy().copy()
            )
            for field in fields(self)
        }

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
            }
        )


@dataclass(frozen=True)
class AffineMaterialStep:
    state: AffineState
    geometry: InterfaceGeometry
    history: wp.array
    interface: CohesiveForceStep | None
    stage: AffineState
    path_midpoint: wp.array


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
):
    """Advance material and permanent interface history over one event segment.

    Interface maps act on packed affine moments. A force callback receives
    (stage_state, transported_velocity, elapsed_time) and returns particle forces
    in N; it owns no mutable history. Impulse is a constant-in-time particle load
    over this segment and replaces gravity, matching MaterialTrajectory.step.
    The callback replaces both impulse and gravity. Stage plastic returns use
    the segment's initial Fp; the final return supplies the retained plastic state.
    """
    m = transfer.model
    if not isinstance(state, AffineState) or state.affine is None:
        raise ValueError("Affine material stages require independent velocity-gradient state")
    transfer._check(duration, wp.float64, (1,))
    if not np.isfinite(duration.numpy()).all() or duration.numpy()[0] <= 0.0:
        raise ValueError("Affine stage duration must be positive and finite")
    if force is not None and impulse is not None:
        raise ValueError("Specify a force callback or a segment impulse")
    count, size = len(history), 4 * m.n
    transfer._check(separation_map, wp.float64, (count, size))
    transfer._check(frame_map, wp.vec3d, (count, size))
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

    free_branch = m.zeros(m.n, int, False)
    free_branch.fill_(-1)

    def material_state(x, moments, F):
        velocity, affine = transfer.unpack(moments)
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
        return result

    constant_force = None
    if force is None:
        if impulse is None:
            constant_force = launch(_body_force, m.n, [wp.vec3d(*gravity), m.mass])
        else:
            transfer._check(impulse, wp.vec3d, (m.n,))
            constant_force = launch(_impulse_force, m.n, [impulse, duration])

    def explicit(value, moments, elapsed):
        physical_velocity, _ = transfer.kinematics(moments)
        load = constant_force if force is None else force(value, physical_velocity, elapsed)
        transfer._check(load, wp.vec3d, (m.n,))
        moment = launch(_stress_moment, m.n, [value.stress, m.volume], wp.mat33d)
        return launch(_acceleration, size, [transfer.pullback_force(load, moment), m.mass])

    def move(x, F, moments, dt, factor=1.0):
        velocity, rate = transfer.kinematics(moments)
        return (
            launch(_vector_increment, m.n, [x, velocity, dt, factor]),
            launch(_matrix_increment, m.n, [F, rate, dt, factor], wp.mat33d),
        )

    tau = launch(_duration, 1, [duration, GAMMA], wp.float64)
    mass = m.array([m.mass], float, False)

    def stage(x, F, moments, base_geometry, maximum):
        if count:
            free = base_geometry.advance(separation_map, frame_map, moments, tau)
            motion = None
            if (
                np.all(maximum.numpy() == 0.0)
                and np.all(base_geometry.gap.numpy() == 0.0)
                and np.all(base_geometry.gap_remainder == 0.0)
            ):
                motion = tau.numpy().astype(np.longdouble)[0] * moments.numpy().astype(np.longdouble)
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
                closed_motion=motion,
            )
            acceleration = launch(_cohesive_acceleration, size, [response.displacement, tau])
            moments = launch(_vector_increment, size, [moments, acceleration, tau, 1.0])
            maximum = response.history
        else:
            response, updated = None, base_geometry
            acceleration = m.zeros(size, wp.vec3)
        xp, Fp = move(x, F, moments, tau)
        return material_state(xp, moments, Fp), moments, updated, maximum, response, acceleration

    z0 = transfer.pack(state.v, state.affine)
    initial = state
    if state.stress is None:
        stress = m.zeros(m.n, wp.mat33)
        m.launch(
            compute_stress,
            m.n,
            inputs=[state.F, state.Fp, m.young, m.config.poisson_ratio],
            outputs=[stress],
            device=m.device,
        )
        initial = replace(state, stress=stress)
    e0 = explicit(initial, z0, m.zeros(1, float, False))
    zbase = launch(_vector_increment, size, [z0, e0, duration, GAMMA])
    mid, z1, geometry1, history1, _, a1 = stage(state.x, state.F, zbase, geometry, history)
    e1 = explicit(mid, z1, tau)
    xb, Fb = move(state.x, state.F, z1, duration, 1.0 - GAMMA)
    gb = geometry.extrapolate(geometry1, (1.0 - GAMMA) / GAMMA) if count else geometry
    zbase = launch(_vector_increment, size, [z0, a1, duration, 1.0 - GAMMA])
    zbase = launch(_vector_increment, size, [zbase, e0, duration, DELTA])
    zbase = launch(_vector_increment, size, [zbase, e1, duration, 1.0 - DELTA])
    result, zend, updated, maximum, response, _ = stage(xb, Fb, zbase, gb, history1)
    start_rate, _ = transfer.kinematics(z0)
    end_rate, _ = transfer.kinematics(zend)
    midpoint = launch(_path_midpoint, m.n, [state.x, result.x, start_rate, end_rate, duration])
    return AffineMaterialStep(result, updated, maximum, response, mid, midpoint)
