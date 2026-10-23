"""Permanent reference quadrature and cohesive state for growing cut surfaces.

Accepted patches retain their quadrature order, material parameters and damage
history. New points start at zero opening in their owner's current affine map.
"""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import warp as wp

from .cohesive_geometry import InterfaceGeometry
from .cohesive_grid_update import advance_material
from .reference_cut import CutQuadrature
from .reference_grid import build_trace
from .reference_surface import ReferenceSurface
from .reference_visibility import ReferenceVisibility
from ._exact_surface import cross
from .reference_coordinates import rational


@wp.kernel
def _join(a: wp.array(dtype=Any), b: wp.array(dtype=Any), out: wp.array(dtype=Any)):
    i = wp.tid()
    if i < a.shape[0]:
        out[i] = a[i]
    else:
        out[i] = b[i - a.shape[0]]


@wp.kernel
def _gather(source: wp.array(dtype=Any), owners: wp.array(dtype=int), out: wp.array(dtype=Any)):
    i = wp.tid()
    out[i] = source[owners[i]]


def _concatenate(a, b):
    if not len(a):
        return b
    if not len(b):
        return a
    out = wp.empty(len(a) + len(b), dtype=a.dtype, device=a.device, requires_grad=True)
    wp.launch(_join, len(out), inputs=[a, b], outputs=[out], device=a.device)
    return out


def _birth_sides(normals, owners, deformation, rotation):
    if rotation.shape != (1,) or rotation.dtype != wp.quatd or rotation.device != normals.device:
        raise ValueError("Birth orientation requires one matching float64 knife quaternion")
    values = rotation.numpy()
    if not np.isfinite(values).all() or np.all(values == 0):
        raise ValueError("Birth knife orientation must be finite and nonzero")
    q = rational(values[0])
    v, w = q[:3], q[3]
    axis = np.array([1, 0, 0], dtype=object)
    positive_axis = ((w * w - np.dot(v, v)) * axis + 2 * v * v[0] + 2 * w * cross(v, axis)) / np.dot(q, q)
    frames = deformation.numpy()
    directions = {}
    for owner in np.unique(owners):
        F = rational(frames[owner])
        determinant = np.dot(F[0], cross(F[1], F[2]))
        if determinant <= 0:
            raise ValueError("Birth side transport requires positive deformation determinants")
        adjugate = np.array([cross(F[1], F[2]), cross(F[2], F[0]), cross(F[0], F[1])], dtype=object).T
        directions[owner] = adjugate @ positive_axis / determinant
    signs = np.array(
        [np.dot(normal, directions[owner]) for normal, owner in zip(rational(normals.numpy()), owners)]
    )
    if np.any(signs == 0):
        raise ValueError("A swept interface must separate the knife's two lateral sides")
    return np.where(signs > 0, 2, 1).astype(np.int8)


@dataclass(frozen=True)
class CohesiveHistory:
    surface: ReferenceSurface
    quadrature: CutQuadrature
    geometry: InterfaceGeometry
    history: wp.array
    material: wp.array
    cut_indices: np.ndarray
    positive_sides: np.ndarray | None = None

    def __post_init__(self):
        count = len(self.quadrature.area)
        arrays = [
            self.quadrature.points,
            self.quadrature.area,
            self.quadrature.normals,
            self.geometry.gap,
            self.geometry.frame,
            self.history,
            self.material,
        ]
        kinds = [wp.vec3d, wp.float64, wp.vec3d, wp.vec3d, wp.mat33d, wp.float64, wp.vec3d]
        if any(
            a.shape != (count,) or a.dtype != kind or a.device != self.surface.vertices.device
            for a, kind in zip(arrays, kinds)
        ):
            raise ValueError("Cohesive history arrays must match the reference quadrature")
        cuts = np.asarray(self.cut_indices)
        owners = self.quadrature.owners
        if (
            cuts.shape != (count,)
            or not np.issubdtype(cuts.dtype, np.integer)
            or np.any(cuts < 0)
            or owners.shape != (count,)
            or np.any(owners < 0)
            or np.any(owners >= len(self.surface.roundoff))
        ):
            raise ValueError("Each quadrature point requires a cut index and a particle owner")
        if any(not np.isfinite(a.numpy()).all() for a in arrays) or np.any(self.history.numpy() < 0):
            raise ValueError("Cohesive state must be finite with nonnegative damage history")
        object.__setattr__(self, "cut_indices", cuts.astype(np.int64, copy=True))
        positive = (
            np.zeros(count, dtype=np.int8) if self.positive_sides is None else np.asarray(self.positive_sides)
        )
        if (
            positive.shape != (count,)
            or not np.issubdtype(positive.dtype, np.integer)
            or not np.isin(positive, [0, 1, 2]).all()
        ):
            raise ValueError("Birth sides must be ternary digits matching the quadrature")
        object.__setattr__(self, "positive_sides", positive.astype(np.int8, copy=True))

    @classmethod
    def empty(cls, particle_count, *, device):
        def empty(dtype):
            return wp.empty(0, dtype=dtype, device=device, requires_grad=True)

        return cls(
            ReferenceSurface.empty(particle_count, device=device),
            CutQuadrature(empty(wp.vec3d), empty(wp.float64), empty(wp.vec3d), np.empty(0, dtype=np.int64)),
            InterfaceGeometry(empty(wp.vec3d), empty(wp.mat33d)),
            empty(wp.float64),
            empty(wp.vec3d),
            np.empty(0, dtype=np.int64),
        )

    def append(self, candidate, reference, deformation, material, *, cut_index, tool_rotation=None):
        """Append newly swept reference area, including zero-measure boundary terms.

        Deformation and (strength, Gc, unused) parameters are indexed by particle.
        The birth frame is the affine map used to pull back the material sweep.
        Cut indices label creation; changing the current knife does not reset them.
        A birth knife quaternion transports its positive lateral axis by F^-1;
        the sign against each reference normal determines the permanent side order.
        """
        if (
            isinstance(cut_index, (bool, np.bool_))
            or not isinstance(cut_index, (int, np.integer))
            or cut_index < 0
        ):
            raise ValueError("A cut index must be a nonnegative integer")
        count, device = len(self.surface.roundoff), self.surface.vertices.device
        if any(
            a.shape != (count,) or a.dtype != dtype or a.device != device
            for a, dtype in zip([reference, deformation, material], [wp.vec3d, wp.mat33d, wp.vec3d])
        ):
            raise ValueError("Birth reference, deformation and material must match the particle set")
        if candidate.vertices.device != device or candidate.roundoff.shape != self.surface.roundoff.shape:
            raise ValueError("New surfaces must share the existing particle set and device")
        fresh = candidate.subtract(self.surface)
        if not fresh.polygons:
            return self
        quadrature = fresh.quadrature(reference)
        positive = (
            np.zeros(len(quadrature.area), dtype=np.int8)
            if tool_rotation is None
            else _birth_sides(quadrature.normals, quadrature.owners, deformation, tool_rotation)
        )
        owners = wp.array(quadrature.owners, dtype=int, device=device)
        frame = wp.empty(len(owners), dtype=wp.mat33d, device=device, requires_grad=True)
        parameters = wp.empty(len(owners), dtype=wp.vec3d, device=device, requires_grad=True)
        wp.launch(_gather, len(owners), inputs=[deformation, owners], outputs=[frame], device=device)
        wp.launch(_gather, len(owners), inputs=[material, owners], outputs=[parameters], device=device)
        gap = wp.zeros(len(owners), dtype=wp.vec3d, device=device, requires_grad=True)
        history = wp.zeros(len(owners), dtype=wp.float64, device=device, requires_grad=True)
        geometry = InterfaceGeometry(
            _concatenate(self.geometry.gap, gap),
            _concatenate(self.geometry.frame, frame),
            np.concatenate([self.geometry.gap_remainder, np.zeros((len(owners), 3))]),
            np.concatenate([self.geometry.frame_remainder, np.zeros((len(owners), 3, 3))]),
        )
        return CohesiveHistory(
            self.surface.join(fresh),
            CutQuadrature(
                _concatenate(self.quadrature.points, quadrature.points),
                _concatenate(self.quadrature.area, quadrature.area),
                _concatenate(self.quadrature.normals, quadrature.normals),
                np.concatenate([self.quadrature.owners, quadrature.owners]),
            ),
            geometry,
            _concatenate(self.history, history),
            _concatenate(self.material, parameters),
            np.r_[self.cut_indices, np.full(len(owners), cut_index, dtype=np.int64)],
            np.r_[self.positive_sides, positive],
        )

    def updated(self, geometry, history):
        if history.shape != self.history.shape or np.any(history.numpy() < self.history.numpy()):
            raise ValueError("Updated damage history must retain every point and its previous maximum")
        return replace(self, geometry=geometry, history=history)

    def advance(
        self,
        model,
        state,
        *,
        topology,
        sides=None,
        positive_sides=None,
        duration=None,
        impulse=None,
        gravity=(0.0, 0.0, -9.81)
    ):
        """Advance retained points using side codes ordered along each face normal."""
        if not len(self.history):
            result = model.step(state, topology=topology, duration=duration, impulse=impulse, gravity=gravity)
            return result, self
        if sides is None:
            trace = self.reference_trace(model, topology, positive_sides)
        elif positive_sides is not None:
            raise ValueError("Specify explicit side codes or natural face orientations")
        else:
            trace = build_trace(model, topology, self.quadrature.points, self.quadrature.normals, sides)
        step = advance_material(
            model,
            state,
            self.geometry,
            self.history,
            self.quadrature.normals,
            self.material,
            trace.separation_map,
            trace.frame_map,
            self.quadrature.area,
            topology=topology,
            duration=duration,
            impulse=impulse,
            gravity=gravity,
        )
        return step.state, self.updated(step.geometry, step.interface.history)

    def reference_trace(self, model, topology, positive_sides=None, *, equivalent_slots=None):
        """Map both sides using birth orientation and current finite cut geometry.

        Positive sides are ternary digits (1 or 2) on the positive reference
        normal side of each quadrature point. Other cuts use segment visibility.
        """
        positive = self.positive_sides if positive_sides is None else np.asarray(positive_sides)
        if positive.shape != self.cut_indices.shape or not np.isin(positive, [1, 2]).all():
            raise ValueError("Each quadrature point requires its positive-normal cut side")
        digits = np.asarray([3 ** int(cut) for cut in self.cut_indices], dtype=object)
        if any(2 * digit > np.iinfo(np.int64).max for digit in digits):
            raise ValueError("Trace cut codes exceed the integer field range")
        sides = np.asarray(digits[:, None] * np.stack([3 - positive, positive], axis=1), dtype=np.int64)
        face_cuts, start = [], 0
        for polygon in self.surface.polygons:
            count = 3 * (len(polygon) - 2)
            labels = self.cut_indices[start : start + count]
            if len(labels) != count or not np.all(labels == labels[0]):
                raise ValueError("Surface polygons and quadrature cut labels must agree")
            face_cuts.append(labels[0])
            start += count
        if start != len(self.cut_indices):
            raise ValueError("Surface polygons and quadrature counts must agree")
        faces = ReferenceVisibility.from_surface(
            self.surface, model.reference, np.asarray(face_cuts, dtype=np.int64)
        )
        visible = faces.particle_mask(self.quadrature.points.numpy(), model.reference, self.cut_indices)
        return build_trace(
            model,
            topology,
            self.quadrature.points,
            self.quadrature.normals,
            sides,
            visibility=visible,
            equivalent_slots=equivalent_slots,
        )

    def snapshot(self):
        return dict(
            vertices=self.surface.vertices.numpy().copy(),
            exact_vertices=self.surface.coordinates().copy(),
            reference_remainder=self.surface.reference_remainder.copy(),
            polygons=self.surface.polygons,
            surface_owners=self.surface.owners.copy(),
            roundoff=self.surface.roundoff.copy(),
            surface_normals=self.surface.normals.numpy().copy(),
            points=self.quadrature.points.numpy().copy(),
            area=self.quadrature.area.numpy().copy(),
            normals=self.quadrature.normals.numpy().copy(),
            owners=self.quadrature.owners.copy(),
            geometry=self.geometry.snapshot(),
            history=self.history.numpy().copy(),
            material=self.material.numpy().copy(),
            cut_indices=self.cut_indices.copy(),
            positive_sides=self.positive_sides.copy(),
        )

    def differentiable_arrays(self):
        """Continuous checkpoint state, including geometry used by later sweeps."""
        return dict(
            vertices=self.surface.vertices,
            surface_normals=self.surface.normals,
            points=self.quadrature.points,
            area=self.quadrature.area,
            normals=self.quadrature.normals,
            gap=self.geometry.gap,
            frame=self.geometry.frame,
            history=self.history,
            material=self.material,
        )

    @classmethod
    def restore(cls, snapshot, *, device, requires_grad=True):
        def array(name, dtype):
            return wp.array(snapshot[name], dtype=dtype, device=device, requires_grad=requires_grad)

        return cls(
            ReferenceSurface(
                array("vertices", wp.vec3d),
                snapshot["polygons"],
                snapshot["surface_owners"].copy(),
                snapshot["roundoff"].copy(),
                array("surface_normals", wp.vec3d),
                snapshot["exact_vertices"].copy(),
                snapshot["reference_remainder"].copy(),
            ),
            CutQuadrature(
                array("points", wp.vec3d),
                array("area", wp.float64),
                array("normals", wp.vec3d),
                snapshot["owners"].copy(),
            ),
            InterfaceGeometry.restore(snapshot["geometry"], device=device, requires_grad=requires_grad),
            array("history", wp.float64),
            array("material", wp.vec3d),
            snapshot["cut_indices"],
            snapshot["positive_sides"],
        )
