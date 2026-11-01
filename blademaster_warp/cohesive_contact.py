"""Joint external contact and cohesive equilibrium, with full implicit pullbacks."""

from dataclasses import dataclass

import numpy as np
import warp as wp

from . import differentiable_cohesive as cohesive
from ._cohesive_equilibrium import EquilibriumResult, solve as solve_equilibrium
from ._cohesive_projection import ReactionBranches
from ._cohesive_initial import fixed_normal_initial
from ._cohesive_solve import setup as cohesive_setup
from ._cohesive_expansion import E, divide, stack
from ._contact_law import response as local_response
from .compensated_state import CompensatedArray, as_compensated, add_gradient


@dataclass(frozen=True)
class MappedContacts:
    """Point offsets [m], relative velocity [m/s], history [m], unit normals.

    Mapping sends material displacement moments to contact-point displacement.
    Parameters per row are kn [N/m], kt [N/m], c [N s/m], mu_static, mu_dynamic.
    """

    mapping: CompensatedArray
    point: CompensatedArray
    velocity: CompensatedArray
    history: CompensatedArray
    normal: wp.array
    parameters: wp.array

    def __post_init__(self):
        for name in ("mapping", "point", "velocity", "history"):
            object.__setattr__(self, name, as_compensated(getattr(self, name)))
        if self.mapping.high.ndim != 2:
            raise ValueError("Contact mapping requires contact and material axes")
        count = len(self.point.high)
        kinds = [wp.float64, wp.vec3d, wp.vec3d, wp.vec3d, wp.vec3d, wp.float64]
        arrays = self.arrays
        shapes = [(count, self.mapping.high.shape[1]), *[(count,)] * 4, (count, 5)]
        if any(
            a.dtype != k or a.shape != s or a.device != arrays[0].device
            for a, k, s in zip(arrays, kinds, shapes)
        ):
            raise ValueError("Contact arrays require matching float64 shapes and devices")
        if any(not np.isfinite(a.numpy()).all() for a in arrays):
            raise ValueError("Contact arrays must be finite")
        parameters = self.parameters.numpy()
        if (
            np.any(parameters[:, :2] <= 0)
            or np.any(parameters[:, 2:] < 0)
            or np.any(parameters[:, 4] > parameters[:, 3])
        ):
            raise ValueError(
                "Contact stiffness must be positive; 0 <= mu_dynamic <= mu_static and damping >= 0"
            )
        if not np.allclose(np.linalg.norm(self.normal.numpy(), axis=1), 1.0, rtol=1e-12, atol=0):
            raise ValueError("Contact normals must be unit vectors")

    @property
    def count(self):
        return len(self.point.high)

    @property
    def arrays(self):
        return [
            self.mapping.high,
            self.point.high,
            self.velocity.high,
            self.history.high,
            self.normal,
            self.parameters,
        ]

    @classmethod
    def from_rows(cls, mapping, rows, device):
        values = [E.from_value(mapping)] + [
            (
                stack([E.from_value(r[key]) for r in rows])
                if rows
                else E(np.empty((0, 5 if key == "parameters" else 3)))
            )
            for key in ["point", "velocity", "history", "normal", "parameters"]
        ]
        kinds = [wp.float64, wp.vec3d, wp.vec3d, wp.vec3d, wp.vec3d, wp.float64]
        arrays = [CompensatedArray.from_cpu(v, dtype=k, device=device) for v, k in zip(values, kinds)]
        return cls(*arrays[:4], arrays[4].high, arrays[5].high)

    def clone(self):
        arrays = [
            wp.array(a.numpy(), dtype=a.dtype, device=a.device, requires_grad=True) for a in self.arrays
        ]
        return MappedContacts(
            *[
                CompensatedArray(a, old.low)
                for a, old in zip(arrays[:4], [self.mapping, self.point, self.velocity, self.history])
            ],
            *arrays[4:]
        )

    def rows(self):
        values = [
            self.point.cpu(),
            self.velocity.cpu(),
            self.history.cpu(),
            E(self.normal.numpy()),
            E(self.parameters.numpy()),
        ]
        return [
            dict(zip(["point", "velocity", "history", "normal", "parameters"], [v[i] for v in values]))
            for i in range(len(self.point.high))
        ]

    def length_scale(self, duration):
        values = [self.point.cpu(), duration * self.velocity.cpu(), self.history.cpu()]
        return max([np.max(abs(v.high), initial=0.0) for v in values] + [np.finfo(float).tiny])

    def add_momentum(self, original, u, u_low, mass, duration):
        force, _ = self.response(u, u_low, duration)
        return contact_momentum(original, self.mapping, force.high, mass, duration)

    @staticmethod
    def evaluate(row, point, velocity, duration):
        return local_response(point, velocity, row["history"], row["normal"], row["parameters"], duration)

    def response(self, u, u_low, duration):
        return contact_response(self, u, u_low, duration)


def contact_response(contacts, u, u_low, duration):
    if not len(contacts.point.high):
        return [
            CompensatedArray.from_cpu(E(np.empty((0, 3))), dtype=wp.vec3d, device=u.device) for _ in range(2)
        ]
    points, velocities = contact_kinematics(contacts, u, u_low, duration)
    history, normal, parameters = (
        contacts.history.cpu(),
        E(contacts.normal.numpy()),
        E(contacts.parameters.numpy()),
    )
    dt = E(duration.numpy()[0])
    positions, speeds = points.cpu(), velocities.cpu()
    rows = [
        local_response(positions[i], speeds[i], history[i], normal[i], parameters[i], dt)
        for i in range(len(points.high))
    ]
    outputs = [
        CompensatedArray.from_cpu(
            stack([getattr(row, name) for row in rows]), dtype=wp.vec3d, device=u.device
        )
        for name in ["force", "history"]
    ]
    tape = wp._src.context.runtime.tape
    sources = [
        points.high,
        velocities.high,
        contacts.history.high,
        contacts.normal,
        contacts.parameters,
        duration,
    ]
    if tape is not None:

        def backward():
            pf, ph = [E(value.high.grad.numpy()) for value in outputs]
            gradients = stack(
                [row.force_jacobian.T @ pf[i] + row.history_jacobian.T @ ph[i] for i, row in enumerate(rows)]
            )
            for source, span in zip(
                sources[:-1], [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 17)]
            ):
                add_gradient(source, gradients[:, span])
            add_gradient(duration, gradients[:, 17].sum().reshape(1))

        tape.record_func(backward, [*[a.high for a in outputs], *[a for a in sources if a.requires_grad]])
    return outputs


def contact_kinematics(contacts, u, u_low, duration):
    mapping, point, velocity = contacts.arrays[:3]
    C = contacts.mapping.cpu()
    displacement, dt = E(u.numpy(), u_low), E(duration.numpy()[0])
    motion = C @ displacement
    positions = contacts.point.cpu() + motion
    speed = contacts.velocity.cpu() + divide(motion, dt)
    outputs = [
        CompensatedArray.from_cpu(value, dtype=wp.vec3d, device=u.device) for value in [positions, speed]
    ]
    tape = wp._src.context.runtime.tape
    if tape is not None:

        def backward():
            px, pv = [E(value.high.grad.numpy()) for value in outputs]
            combined = px + divide(pv, dt)
            add_gradient(point, px)
            add_gradient(velocity, pv)
            add_gradient(u, C.T @ combined)
            add_gradient(mapping, combined @ displacement.T)
            add_gradient(duration, -divide((pv * motion).sum(), dt * dt).reshape(1))

        tape.record_func(
            backward,
            [
                *[a.high for a in outputs],
                *[a for a in [mapping, point, velocity, u, duration] if a.requires_grad],
            ],
        )
    return outputs


def contact_momentum(original, contact_map, force, mass, duration):
    if not len(force):
        return original
    mapping = contact_map.high
    C, f = contact_map.cpu(), E(force.numpy())
    m, dt = E(mass.numpy()[0]), E(duration.numpy()[0])
    coefficient = divide(dt * dt, m)
    load = C.T @ f
    result = CompensatedArray.from_cpu(
        E(original.numpy()) - coefficient * load, dtype=wp.vec3d, device=original.device
    ).high
    tape = wp._src.context.runtime.tape
    if tape is not None:

        def backward():
            seed = E(result.grad.numpy())
            add_gradient(original, seed)
            add_gradient(force, -coefficient * (C @ seed))
            add_gradient(mapping, -coefficient * (f @ seed.T))
            power = (seed * load).sum()
            add_gradient(mass, divide(coefficient * power, m).reshape(1))
            add_gradient(duration, -divide(2 * dt * power, m).reshape(1))

        tape.record_func(
            backward, [result, *[a for a in [original, mapping, force, mass, duration] if a.requires_grad]]
        )
    return result


def solve(inputs, geometry, inertia, contacts, map_remainders=None, *, constrain_normal=True):
    """Return the joint root, displacement, selected interface force, and contact response."""
    inertia = None if inertia is None else inertia.frozen()
    inputs, values = cohesive._validated_inputs(
        *inputs, inertia=inertia, integrated_force=True, allow_empty=True
    )
    values = [value.copy() for value in values]
    count, particles = values[5].shape
    effective_contacts = contacts if contacts.count else None
    if contacts.mapping.high.shape[1] != particles:
        raise ValueError("Contact maps must match material degrees of freedom")
    if count:
        compliance = cohesive_setup(values, integrated_force=True, inertia=inertia)[2]
        precise = list(values)
        precise[0] = E(values[0], geometry.gap_remainder).numpy(np.longdouble)
        precise[1] = E(values[1], geometry.frame_remainder).numpy(np.longdouble)
        _, traction, _, info, displacement = fixed_normal_initial(
            precise,
            compliance,
            integrated_force=True,
            max_iterations=1024,
            inertia=inertia,
            return_primal=True,
            constrain_normal=constrain_normal,
        )
    else:
        displacement, traction, compliance, info = np.zeros((particles, 3)), np.empty((0, 3)), 1.0, {}
    seed = displacement, traction, compliance, info, None, None
    root = (
        solve_equilibrium(
            values,
            geometry.gap_remainder,
            geometry.frame_remainder,
            seed,
            integrated_force=True,
            inertia=inertia,
            map_remainders=map_remainders,
            contacts=effective_contacts,
            constrain_normal=constrain_normal,
        )
        if count or effective_contacts is not None
        else EquilibriumResult(
            E(displacement),
            E(traction),
            E(values[0], geometry.gap_remainder),
            E(values[1], geometry.frame_remainder),
            compliance,
            {},
            ReactionBranches(np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)),
        )
    )
    u = wp.array(root.displacement.high, dtype=wp.vec3d, device=inputs[0].device, requires_grad=True)
    traction = wp.array(root.traction.high, dtype=wp.vec3d, device=u.device, requires_grad=True)
    sources = [
        *inputs,
        *([] if effective_contacts is None else contacts.arrays),
        *([] if inertia is None else inertia.parameters),
    ]
    remainders = (geometry.gap_remainder, geometry.frame_remainder, root.displacement.low)
    closed_linear = root.info.get("closed_linear_solution", False)
    if not constrain_normal and effective_contacts is None and count:
        closed_linear = bool(
            np.all(values[7] > 0.0)
            and np.all(root.branches.closed)
            and not np.any(root.branches.compressed)
            and not np.any(root.branches.normal_active)
            and not np.any(root.branches.strength_active)
        )
    tape = wp._src.context.runtime.tape
    if tape is not None and (count or effective_contacts is not None):

        def backward():
            gradients = cohesive._implicit_pullback(
                inputs,
                values,
                root.displacement.high,
                root.traction.high,
                root.compliance,
                u.grad.numpy().ravel(),
                (traction.grad.numpy().ravel() if count else np.empty(0)) / root.compliance,
                closed=closed_linear,
                integrated_force=True,
                inertia=inertia,
                remainders=remainders,
                map_remainders=map_remainders,
                contacts=effective_contacts,
                constrain_normal=constrain_normal,
            )
            for source, gradient in zip(sources, gradients, strict=True):
                add_gradient(source, E.from_value(gradient))

        tape.record_func(backward, [a for a in [u, traction, *sources] if a.requires_grad and a.size])
    if count and not constrain_normal:
        from ._adhesive_reaction import canonical_output

        selected = canonical_output(
            inputs, u, traction, root.branches, remainders=remainders, map_remainders=map_remainders
        )
    else:
        selected = (
            cohesive._canonical_output(
                inputs,
                u,
                traction,
                root.compliance,
                {},
                integrated_force=True,
                remainders=remainders,
                branches=root.branches,
                map_remainders=map_remainders,
            )
            if count
            else traction
        )
    force, history = contacts.response(u, root.displacement.low, inputs[9])
    return root, u, selected, force, history


def history_rate(updated, previous, duration):
    updated, previous, duration = map(as_compensated, (updated, previous, duration))
    difference, time = updated.cpu() - previous.cpu(), duration.cpu()[0]
    result = CompensatedArray.from_cpu(
        divide(difference, time), dtype=updated.high.dtype, device=updated.high.device
    )
    tape = wp._src.context.runtime.tape
    if tape is not None and result.high.size:

        def backward():
            seed = E(result.high.grad.numpy())
            add_gradient(updated.high, divide(seed, time))
            add_gradient(previous.high, -divide(seed, time))
            add_gradient(duration.high, -divide((seed * difference).sum(), time * time).reshape(1))

        tape.record_func(
            backward,
            [
                a
                for a in [result.high, updated.high, previous.high, duration.high]
                if a.requires_grad and a.size
            ],
        )
    return result


def predict_motion(position, moments, duration, point_map, moment_map, origin=None, velocity=None):
    """Predict world points/velocities, or offsets from a prescribed surface.

    point_map interpolates material positions; moment_map interpolates the
    transported velocity. origin and velocity have one row or one per contact.
    """
    position, moments, duration, point_map, moment_map = map(
        as_compensated, (position, moments, duration, point_map, moment_map)
    )
    W, C = point_map.cpu(), moment_map.cpu()
    x, z, time = position.cpu(), moments.cpu(), duration.cpu()[0]
    speed = C @ z
    offsets = W @ x + time * speed
    relative = speed
    if origin is not None:
        offsets = offsets - E(origin.numpy())
    if velocity is not None:
        relative = relative - E(velocity.numpy())
    point = CompensatedArray.from_cpu(offsets, dtype=wp.vec3d, device=position.high.device)
    free_velocity = CompensatedArray.from_cpu(relative, dtype=wp.vec3d, device=position.high.device)
    tape = wp._src.context.runtime.tape
    if tape is not None and point.high.size:

        def backward():
            px, pv = E(point.high.grad.numpy()), E(free_velocity.high.grad.numpy())
            combined = time * px + pv
            add_gradient(position.high, W.T @ px)
            add_gradient(point_map.high, px @ x.T)
            add_gradient(moments.high, C.T @ combined)
            add_gradient(moment_map.high, combined @ z.T)
            add_gradient(duration.high, (px * speed).sum().reshape(1))
            if origin is not None:
                add_gradient(origin, -px.sum(axis=0).reshape(1, 3) if len(origin) == 1 else -px)
            if velocity is not None:
                add_gradient(velocity, -pv.sum(axis=0).reshape(1, 3) if len(velocity) == 1 else -pv)

        sources = [
            position.high,
            moments.high,
            duration.high,
            point_map.high,
            moment_map.high,
            origin,
            velocity,
        ]
        tape.record_func(
            backward,
            [
                point.high,
                free_velocity.high,
                *[a for a in sources if a is not None and a.requires_grad and a.size],
            ],
        )
    return point, free_velocity


def predict_contacts(
    position, moments, duration, history, point_map, moment_map, origin, velocity, normal, parameters
):
    point, free_velocity = predict_motion(
        position, moments, duration, point_map, moment_map, origin, velocity
    )
    return MappedContacts(moment_map, point, free_velocity, history, normal, parameters)
