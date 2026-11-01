"""Particle-pair contact in the shared material and cohesive momentum equation."""

from dataclasses import dataclass

import numpy as np
import warp as wp

from ._cohesive_expansion import E, stack
from ._pair_contact_law import response as local_response
from .cohesive_contact import contact_kinematics, contact_momentum, predict_motion
from .compensated_state import CompensatedArray, as_compensated, add_gradient


@dataclass(frozen=True)
class PairContacts:
    """Relative centre positions/velocities, history in m, and force coefficients in SI.

    Mapping is the difference of the two particle velocity maps. Parameters
    (kn, kt, c, mus, mud) and diameter have one row or one per pair.
    """

    mapping: CompensatedArray
    point: CompensatedArray
    velocity: CompensatedArray
    history: CompensatedArray
    parameters: wp.array
    diameter: wp.array

    def __post_init__(self):
        for name in ("mapping", "point", "velocity", "history"):
            object.__setattr__(self, name, as_compensated(getattr(self, name)))
        if self.mapping.high.ndim != 2 or self.mapping.high.shape[0] != self.count:
            raise ValueError("Pair maps require pair and material axes")
        for source, kind, shape in zip(
            self.arrays[:4],
            [wp.float64, wp.vec3d, wp.vec3d, wp.vec3d],
            [self.mapping.high.shape, (self.count,), (self.count,), (self.count,)],
        ):
            if source.dtype != kind or source.shape != shape:
                raise ValueError("Pair states require matching float64 shapes")
        if self.parameters.dtype != wp.float64 or self.parameters.shape not in [(1, 5), (self.count, 5)]:
            raise ValueError("Pair parameters require five coefficients per row")
        if self.diameter.dtype != wp.float64 or self.diameter.shape not in [(1,), (self.count,)]:
            raise ValueError("Pair diameters require one row or one per pair")
        if any(a.device != self.point.high.device or not np.isfinite(a.numpy()).all() for a in self.arrays):
            raise ValueError("Pair contact arrays must be finite and share a device")
        parameters = self.parameters.numpy()
        if (
            np.any(parameters[:, :2] <= 0)
            or np.any(parameters[:, 2:] < 0)
            or np.any(parameters[:, 4] > parameters[:, 3])
        ):
            raise ValueError("Pair contact requires positive stiffness and 0 <= mu_dynamic <= mu_static")
        if np.any(self.diameter.numpy() <= 0):
            raise ValueError("Contact diameters must be positive")

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
            self.parameters,
            self.diameter,
        ]

    @classmethod
    def from_rows(cls, mapping, rows, device):
        values = [E.from_value(mapping)] + [
            stack([E.from_value(row[name]) for row in rows]) if rows else E(np.empty((0, 3)))
            for name in ["point", "velocity", "history"]
        ]
        arrays = [
            CompensatedArray.from_cpu(value, dtype=kind, device=device)
            for value, kind in zip(values, [wp.float64, wp.vec3d, wp.vec3d, wp.vec3d])
        ]
        parameters = np.asarray([row["parameters"] for row in rows], dtype=float).reshape(len(rows), 5)
        diameter = np.asarray([row["diameter"] for row in rows], dtype=float).reshape(len(rows))
        return cls(
            *arrays,
            wp.array(parameters, dtype=wp.float64, device=device, requires_grad=True),
            wp.array(diameter, dtype=wp.float64, device=device, requires_grad=True)
        )

    def clone(self):
        arrays = [
            wp.array(a.numpy(), dtype=a.dtype, device=a.device, requires_grad=True) for a in self.arrays
        ]
        return PairContacts(
            *[
                CompensatedArray(a, old.low)
                for a, old in zip(arrays[:4], [self.mapping, self.point, self.velocity, self.history])
            ],
            *arrays[4:]
        )

    def rows(self):
        values = [self.point.cpu(), self.velocity.cpu(), self.history.cpu()]
        parameters, diameter = E(self.parameters.numpy()), E(self.diameter.numpy())
        return [
            dict(
                point=values[0][i],
                velocity=values[1][i],
                history=values[2][i],
                parameters=parameters[0 if len(parameters.high) == 1 else i],
                diameter=diameter[0 if len(diameter.high) == 1 else i],
            )
            for i in range(self.count)
        ]

    @staticmethod
    def evaluate(row, point, velocity, duration):
        return local_response(row, point, velocity, duration)

    def length_scale(self, duration):
        values = [
            self.point.cpu(),
            duration * self.velocity.cpu(),
            self.history.cpu(),
            E(self.diameter.numpy()),
        ]
        return max([np.max(abs(v.high), initial=0.0) for v in values] + [np.finfo(float).tiny])

    def response(self, u, u_low, duration):
        if not self.count:
            return [
                CompensatedArray.from_cpu(E(np.empty((0, 3))), dtype=wp.vec3d, device=u.device)
                for _ in range(2)
            ]
        point, velocity = contact_kinematics(self, u, u_low, duration)
        x, v, dt = point.cpu(), velocity.cpu(), E(duration.numpy()[0])
        rows = [self.evaluate(row, x[i], v[i], dt) for i, row in enumerate(self.rows())]
        outputs = [
            CompensatedArray.from_cpu(
                stack([getattr(row, name) for row in rows]), dtype=wp.vec3d, device=u.device
            )
            for name in ["force", "history"]
        ]
        sources = [
            (point.high, slice(0, 3)),
            (velocity.high, slice(3, 6)),
            (self.history.high, slice(6, 9)),
            (self.parameters, slice(9, 14)),
            (self.diameter, slice(14, 15)),
        ]
        tape = wp._src.context.runtime.tape
        if tape is not None:

            def backward():
                pf, ph = [E(value.high.grad.numpy()) for value in outputs]
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
                add_gradient(duration, gradients[:, 15].sum().reshape(1))

            tape.record_func(
                backward, [*[a.high for a in outputs], *[a for a, _ in sources if a.requires_grad], duration]
            )
        return outputs

    def add_momentum(self, original, u, u_low, mass, duration):
        force, _ = self.response(u, u_low, duration)
        return contact_momentum(original, self.mapping, force.high, mass, duration)


def pair_maps(point_map, moment_map, pairs):
    """Select sorted unique p<q pairs, retaining derivatives of both particle maps."""
    point_map, moment_map = map(as_compensated, [point_map, moment_map])
    pairs = np.asarray(pairs)
    count = point_map.high.shape[0]
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("Particle pairs require an integer (count, 2) array")
    if np.any(pairs < 0) or np.any(pairs >= count) or np.any(pairs[:, 0] >= pairs[:, 1]):
        raise ValueError("Pair indices require 0 <= p < q < particle_count")
    keys = pairs[:, 0].astype(np.int64) * count + pairs[:, 1]
    if np.any(np.diff(keys) <= 0):
        raise ValueError("Particle pairs must be sorted and unique")
    if (
        point_map.high.ndim != 2
        or moment_map.high.ndim != 2
        or moment_map.high.shape[0] != count
        or point_map.high.device != moment_map.high.device
    ):
        raise ValueError("Particle position and velocity maps must share a particle axis and device")
    incidence = np.zeros((len(pairs), count))
    incidence[np.arange(len(pairs)), pairs[:, 0]] = 1.0
    incidence[np.arange(len(pairs)), pairs[:, 1]] = -1.0
    incidence = E(incidence)
    outputs = [
        CompensatedArray.from_cpu(incidence @ value.cpu(), dtype=wp.float64, device=value.high.device)
        for value in [point_map, moment_map]
    ]
    tape = wp._src.context.runtime.tape
    if tape is not None and len(pairs):

        def backward():
            for source, result in zip([point_map, moment_map], outputs):
                add_gradient(source.high, incidence.T @ E(result.high.grad.numpy()))

        tape.record_func(
            backward,
            [*[a.high for a in outputs], *[a.high for a in [point_map, moment_map] if a.high.requires_grad]],
        )
    return tuple(outputs)


def predict_pair_contacts(
    position, moments, duration, history, point_map, moment_map, pairs, parameters, diameter
):
    W, C = pair_maps(point_map, moment_map, pairs)
    relative, velocity = predict_motion(position, moments, duration, W, C)
    return PairContacts(C, relative, velocity, history, parameters, diameter)
