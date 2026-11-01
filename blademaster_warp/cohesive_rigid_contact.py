"""Finite rigid blades sharing the material/interface equilibrium and its adjoint."""

from dataclasses import dataclass

import numpy as np
import warp as wp

from ._cohesive_expansion import E, stack
from ._rigid_contact_law import response as local_response
from .cohesive_contact import contact_kinematics, contact_momentum, predict_motion
from .compensated_state import CompensatedArray, as_compensated, add_gradient


_FIELDS = (
    ("parameters", wp.float64, 5, slice(27, 32)),
    ("position", wp.vec3d, 3, slice(6, 9)),
    ("rotation", wp.quatd, 4, slice(9, 13)),
    ("tool_velocity", wp.vec3d, 3, slice(13, 16)),
    ("omega", wp.vec3d, 3, slice(16, 19)),
    ("half", wp.vec3d, 3, slice(19, 22)),
    ("bevel", wp.float64, 1, slice(22, 23)),
    ("offset", wp.float64, 1, slice(23, 24)),
)


@dataclass(frozen=True)
class RigidContacts:
    """World point/velocity/history per contact; rigid fields have one row or one per contact.

    Position and half-extents are in m, xyzw rotations are unit quaternions,
    velocity is m/s and omega is rad/s. Parameters are kn, kt, c, mus, mud.
    """

    mapping: CompensatedArray
    point: CompensatedArray
    velocity: CompensatedArray
    history: CompensatedArray
    parameters: wp.array
    position: wp.array
    rotation: wp.array
    tool_velocity: wp.array
    omega: wp.array
    half: wp.array
    bevel: wp.array
    offset: wp.array

    def __post_init__(self):
        for name in ("mapping", "point", "velocity", "history"):
            object.__setattr__(self, name, as_compensated(getattr(self, name)))
        count = len(self.point.high)
        if self.mapping.high.ndim != 2 or self.mapping.high.shape[0] != count:
            raise ValueError("Rigid contact mapping requires contact and material axes")
        for source, kind, shape in zip(
            self.arrays[:4],
            [wp.float64, wp.vec3d, wp.vec3d, wp.vec3d],
            [self.mapping.high.shape, (count,), (count,), (count,)],
        ):
            if source.dtype != kind or source.shape != shape:
                raise ValueError("Rigid contact states require matching float64 shapes")
        for name, kind, width, _ in _FIELDS:
            source = getattr(self, name)
            shapes = [(n, width) if width == 5 else (n,) for n in (1, count)]
            if source.dtype != kind or source.shape not in shapes:
                raise ValueError("Rigid fields require one row or one row per contact")
        if any(a.device != self.point.high.device or not np.isfinite(a.numpy()).all() for a in self.arrays):
            raise ValueError("Rigid contact arrays must be finite and share a device")
        p = self.parameters.numpy()
        if np.any(p[:, :2] <= 0) or np.any(p[:, 2:] < 0) or np.any(p[:, 4] > p[:, 3]):
            raise ValueError("Contact requires positive stiffness and 0 <= mu_dynamic <= mu_static")
        if not np.allclose(np.linalg.norm(self.rotation.numpy(), axis=1), 1.0, rtol=1e-12, atol=0.0):
            raise ValueError("Rigid rotations must be unit quaternions")
        half, bevel = self.half.numpy(), self.bevel.numpy()
        if np.any(half <= 0) or np.any(bevel < 0) or np.any(bevel > 2 * half[:, 2]):
            raise ValueError("Blade half-extents must be positive and bevel within the height")
        if np.any(self.offset.numpy() < 0):
            raise ValueError("Contact offsets must be nonnegative")

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
            *[getattr(self, name) for name, *_ in _FIELDS],
        ]

    @classmethod
    def from_rows(cls, mapping, rows, device):
        values = [E.from_value(mapping)]
        for name in ("point", "velocity", "history"):
            values.append(stack([E.from_value(row[name]) for row in rows]) if rows else E(np.empty((0, 3))))
        arrays = [
            CompensatedArray.from_cpu(v, dtype=kind, device=device)
            for v, kind in zip(values, [wp.float64, wp.vec3d, wp.vec3d, wp.vec3d])
        ]
        for name, kind, width, _ in _FIELDS:
            value = np.asarray([E.from_value(row[name]).numpy().reshape(width) for row in rows])
            value = value.reshape(len(rows)) if width == 1 else value.reshape(len(rows), width)
            arrays.append(wp.array(value, dtype=kind, device=device, requires_grad=True))
        return cls(*arrays)

    def clone(self):
        arrays = [
            wp.array(a.numpy(), dtype=a.dtype, device=a.device, requires_grad=True) for a in self.arrays
        ]
        return RigidContacts(
            *[
                CompensatedArray(a, old.low)
                for a, old in zip(arrays[:4], [self.mapping, self.point, self.velocity, self.history])
            ],
            *arrays[4:],
        )

    def rows(self):
        points, velocities, histories = [a.cpu() for a in [self.point, self.velocity, self.history]]
        fields = {name: E(getattr(self, name).numpy()) for name, *_ in _FIELDS}
        return [
            dict(
                point=points[i],
                velocity=velocities[i],
                history=histories[i],
                **{name: value[0 if len(value.high) == 1 else i] for name, value in fields.items()},
            )
            for i in range(len(self.point.high))
        ]

    @staticmethod
    def evaluate(row, point, velocity, duration):
        return local_response(row, point, velocity, duration)

    def length_scale(self, duration):
        values = [
            self.point.cpu() - E(self.position.numpy()),
            duration * self.velocity.cpu(),
            self.history.cpu(),
            E(self.half.numpy()),
        ]
        return max([np.max(abs(v.high), initial=0.0) for v in values] + [np.finfo(float).tiny])

    def response(self, u, u_low, duration):
        if not len(self.point.high):
            return [
                CompensatedArray.from_cpu(E(np.empty((0, 3))), dtype=wp.vec3d, device=u.device)
                for _ in range(2)
            ]
        points, velocities = contact_kinematics(self, u, u_low, duration)
        x, v, dt = points.cpu(), velocities.cpu(), E(duration.numpy()[0])
        rows = [self.evaluate(row, x[i], v[i], dt) for i, row in enumerate(self.rows())]
        outputs = [
            CompensatedArray.from_cpu(
                stack([getattr(row, name) for row in rows]), dtype=wp.vec3d, device=u.device
            )
            for name in ["force", "history"]
        ]
        sources = [
            (points.high, slice(0, 3)),
            (velocities.high, slice(3, 6)),
            (self.history.high, slice(24, 27)),
        ]
        sources += [(getattr(self, name), span) for name, _, _, span in _FIELDS]
        tape = wp._src.context.runtime.tape
        if tape is not None:

            def backward():
                pf, ph = [E(output.high.grad.numpy()) for output in outputs]
                gradients = stack(
                    [
                        row.force_jacobian.T @ pf[i] + row.history_jacobian.T @ ph[i]
                        for i, row in enumerate(rows)
                    ]
                )
                for source, span in sources:
                    value = gradients[:, span]
                    if len(source) == 1:
                        value = value.sum(axis=0)
                    add_gradient(source, value.reshape(source.numpy().shape))
                add_gradient(duration, gradients[:, 32].sum().reshape(1))

            tape.record_func(
                backward, [*[a.high for a in outputs], *[a for a, _ in sources if a.requires_grad], duration]
            )
        return outputs

    def add_momentum(self, original, u, u_low, mass, duration):
        force, _ = self.response(u, u_low, duration)
        return contact_momentum(original, self.mapping, force.high, mass, duration)


def predict_rigid_contacts(
    material_position, moments, duration, history, point_map, moment_map, **rigid_fields
):
    points, velocities = predict_motion(material_position, moments, duration, point_map, moment_map)
    return RigidContacts(moment_map, points, velocities, history, **rigid_fields)
