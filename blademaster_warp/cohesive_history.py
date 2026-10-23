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


@dataclass(frozen=True)
class CohesiveHistory:
    surface: ReferenceSurface
    quadrature: CutQuadrature
    geometry: InterfaceGeometry
    history: wp.array
    material: wp.array
    cut_indices: np.ndarray

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

    def append(self, candidate, reference, deformation, material, *, cut_index):
        """Append newly swept reference area, including zero-measure boundary terms.

        Deformation and (strength, Gc, unused) parameters are indexed by particle.
        The birth frame is the affine map used to pull back the material sweep.
        Cut indices label creation; changing the current knife does not reset them.
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
        )

    def updated(self, geometry, history):
        if history.shape != self.history.shape or np.any(history.numpy() < self.history.numpy()):
            raise ValueError("Updated damage history must retain every point and its previous maximum")
        return replace(self, geometry=geometry, history=history)

    def advance(
        self, model, state, *, topology, sides, duration=None, impulse=None, gravity=(0.0, 0.0, -9.81)
    ):
        """Advance retained points using side codes ordered along each face normal."""
        if not len(self.history):
            result = model.step(state, topology=topology, duration=duration, impulse=impulse, gravity=gravity)
            return result, self
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
        )
