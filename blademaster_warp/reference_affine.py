"""Affine velocity moments on fixed reference stencils.

A = dv/dX is in s^-1. With D = sum_i w_ip r_ip r_ip^T = L L^T,
the four velocity vectors per particle are [v, columns(A L)], all in m/s.
Their common mass is the particle mass, so kinetic energy includes the
subparticle affine moment. Reference coordinates and stencils are constant.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from .geometry import grid_cells


@wp.kernel
def _pack(
    velocity: wp.array(dtype=wp.vec3d),
    gradient: wp.array(dtype=wp.mat33d),
    root: wp.array(dtype=wp.mat33d),
    result: wp.array(dtype=wp.vec3d),
):
    p = wp.tid()
    affine = gradient[p] * root[p]
    result[4 * p] = velocity[p]
    for axis in range(3):
        result[4 * p + 1 + axis] = wp.vec3d(affine[0, axis], affine[1, axis], affine[2, axis])


@wp.kernel
def _unpack(
    moments: wp.array(dtype=wp.vec3d),
    inverse_root: wp.array(dtype=wp.mat33d),
    velocity: wp.array(dtype=wp.vec3d),
    gradient: wp.array(dtype=wp.mat33d),
):
    p = wp.tid()
    velocity[p] = moments[4 * p]
    affine = wp.matrix_from_cols(moments[4 * p + 1], moments[4 * p + 2], moments[4 * p + 3])
    gradient[p] = affine * inverse_root[p]


@wp.kernel
def _compose(
    separation: wp.array2d(dtype=wp.float64),
    frame: wp.array2d(dtype=wp.vec3d),
    transfer: wp.array2d(dtype=wp.float64),
    result_separation: wp.array2d(dtype=wp.float64),
    result_frame: wp.array2d(dtype=wp.vec3d),
):
    q, p = wp.tid()
    value = wp.float64(0.0)
    derivative = wp.vec3d(0.0)
    for i in range(transfer.shape[0]):
        value += separation[q, i] * transfer[i, p]
        derivative += frame[q, i] * transfer[i, p]
    result_separation[q, p] = value
    result_frame[q, p] = derivative


@wp.kernel
def _kinematics(
    flow: wp.array2d(dtype=wp.float64),
    gradient: wp.array2d(dtype=wp.vec3d),
    moments: wp.array(dtype=wp.vec3d),
    velocity: wp.array(dtype=wp.vec3d),
    deformation_rate: wp.array(dtype=wp.mat33d),
):
    p = wp.tid()
    value = wp.vec3d(0.0)
    rate = wp.mat33d(0.0)
    for j in range(moments.shape[0]):
        value += flow[p, j] * moments[j]
        rate += wp.outer(moments[j], gradient[p, j])
    velocity[p] = value
    deformation_rate[p] = rate


@wp.kernel
def _force(
    flow: wp.array2d(dtype=wp.float64),
    gradient: wp.array2d(dtype=wp.vec3d),
    force: wp.array(dtype=wp.vec3d),
    moment: wp.array(dtype=wp.mat33d),
    result: wp.array(dtype=wp.vec3d),
):
    j = wp.tid()
    value = wp.vec3d(0.0)
    for p in range(force.shape[0]):
        value += flow[p, j] * force[p] + moment[p] * gradient[p, j]
    result[j] = value


@wp.kernel
def _energy(moments: wp.array(dtype=wp.vec3d), mass: wp.float64, result: wp.array(dtype=wp.float64)):
    p = wp.tid()
    value = wp.float64(0.0)
    for a in range(4):
        value += wp.dot(moments[4 * p + a], moments[4 * p + a])
    result[p] = wp.float64(0.5) * mass * value


@dataclass(frozen=True)
class ReferenceAffineTransfer:
    model: object
    grid_map: wp.array
    flow_map: wp.array
    gradient_map: wp.array
    root: wp.array
    inverse_root: wp.array
    grid_mass: np.ndarray
    equivalent_slots: np.ndarray

    def _check(self, value, dtype, shape):
        if value.dtype != dtype or value.shape != shape or value.device != self.model.device:
            raise ValueError("Reference-affine arrays must have matching float64 shapes and devices")

    def pack(self, velocity, gradient):
        self._check(velocity, wp.vec3d, (self.model.n,))
        self._check(gradient, wp.mat33d, (self.model.n,))
        result = self.model.zeros(4 * self.model.n, wp.vec3)
        wp.launch(_pack, self.model.n, [velocity, gradient, self.root, result], device=self.model.device)
        return result

    def unpack(self, moments):
        self._check(moments, wp.vec3d, (4 * self.model.n,))
        velocity = self.model.zeros(self.model.n, wp.vec3)
        gradient = self.model.zeros(self.model.n, wp.mat33)
        wp.launch(
            _unpack, self.model.n, [moments, self.inverse_root, velocity, gradient], device=self.model.device
        )
        return velocity, gradient

    def compose_trace(self, trace):
        """Retain the cut-point and normal derivatives in the composed maps."""
        count = trace.separation_map.shape[0]
        shape = (count, self.grid_map.shape[0])
        self._check(trace.separation_map, wp.float64, shape)
        self._check(trace.frame_map, wp.vec3d, shape)
        separation = self.model.zeros((count, 4 * self.model.n), float)
        frame = self.model.zeros(separation.shape, wp.vec3)
        if count:
            wp.launch(
                _compose,
                separation.shape,
                [trace.separation_map, trace.frame_map, self.grid_map, separation, frame],
                device=self.model.device,
            )
        return separation, frame

    def kinematics(self, moments):
        self._check(moments, wp.vec3d, (4 * self.model.n,))
        velocity = self.model.zeros(self.model.n, wp.vec3)
        rate = self.model.zeros(self.model.n, wp.mat33)
        wp.launch(
            _kinematics,
            self.model.n,
            [self.flow_map, self.gradient_map, moments, velocity, rate],
            device=self.model.device,
        )
        return velocity, rate

    def pullback_force(self, force, moment):
        """Map particle force (N) and F-conjugate moment (J) by virtual power."""
        self._check(force, wp.vec3d, (self.model.n,))
        self._check(moment, wp.mat33d, (self.model.n,))
        result = self.model.zeros(4 * self.model.n, wp.vec3)
        wp.launch(
            _force,
            len(result),
            [self.flow_map, self.gradient_map, force, moment, result],
            device=self.model.device,
        )
        return result

    def kinetic_energy(self, moments):
        self._check(moments, wp.vec3d, (4 * self.model.n,))
        result = self.model.zeros(self.model.n, float)
        wp.launch(
            _energy, self.model.n, [moments, wp.float64(self.model.mass), result], device=self.model.device
        )
        return result


def build_transfer(model, topology):
    if model.precision != "float64":
        raise ValueError("Reference-affine transfer requires float64 material states")
    slots = topology.slots.numpy()
    weights = model.weights.numpy()
    gradients = model.gradients.numpy()
    cells = grid_cells(model.config.size, model.config.grid_spacing)
    keys = topology.keys.numpy()
    indices = np.array(np.unravel_index(keys % model.node_count, tuple(cells + 1))).T
    origin = -np.array([model.config.size[0] / 2, model.config.size[1] / 2, 0.0])
    coordinates = origin + indices * model.config.grid_spacing
    offset = coordinates[slots] - model.reference[:, None, :]
    covariance = np.einsum("pi,pia,pib->pab", weights, offset, offset)
    root = np.linalg.cholesky(covariance)
    inverse_root = np.linalg.inv(root)
    feature = np.concatenate(
        [np.ones((*weights.shape, 1)), np.einsum("pab,pib->pia", inverse_root, offset)], axis=2
    )
    numerator = np.zeros((topology.count, 4 * model.n))
    values = np.zeros((model.n, topology.count))
    derivative = np.zeros((model.n, topology.count, 3))
    for p in range(model.n):
        for a, slot in enumerate(slots[p]):
            numerator[slot, 4 * p : 4 * p + 4] += model.mass * weights[p, a] * feature[p, a]
            values[p, slot] += weights[p, a]
            derivative[p, slot] += gradients[p, a]
    masses = model.mass * values.sum(axis=0)
    if np.any(masses <= 0) or not np.isfinite(masses).all():
        raise ValueError("Every reference-affine grid slot requires positive finite mass")
    mapping = np.zeros_like(numerator)
    equivalent = np.arange(topology.count)
    offsets, sources = [getattr(topology, name).numpy() for name in ["offsets", "sources"]]
    for node in np.unique(keys % model.node_count):
        group = np.flatnonzero(keys % model.node_count == node)
        total = np.sum(masses[group])
        fully_shared = all(offsets[i + 1] - offsets[i] == len(group) for i in group)
        if fully_shared:
            mapping[group] = np.sum(numerator[group], axis=0) / total
            equivalent[group] = group[0]
        else:
            for i in group:
                compatible = np.sort(sources[offsets[i] : offsets[i + 1]])
                excluded = np.setdiff1d(group, compatible)
                mapping[i] = np.sum(numerator[compatible], axis=0) / total
                mapping[i] += np.sum(masses[excluded]) / total * (numerator[i] / masses[i])
    return ReferenceAffineTransfer(
        model,
        model.array(mapping, float, False),
        model.array(values @ mapping, float, False),
        model.array(np.einsum("isa,sj->ija", derivative, mapping), wp.vec3, False),
        model.array(root, wp.mat33, False),
        model.array(inverse_root, wp.mat33, False),
        masses,
        equivalent,
    )
