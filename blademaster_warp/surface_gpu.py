"""CUDA nodal embedding for the linked-volume surface."""

from functools import cached_property
import numpy as np
import warp as wp
from .material import compatible
from .surface import MaterialEmbedding, OFFSETS


@wp.func
def node_field(
    i: int,
    code: wp.int64,
    extend: int,
    nodes: wp.array(dtype=wp.vec3d),
    reference: wp.array(dtype=wp.vec3d),
    offsets: wp.array(dtype=int),
    indices: wp.array(dtype=int),
    weights: wp.array(dtype=wp.float64),
    extra_offsets: wp.array(dtype=int),
    extra_indices: wp.array(dtype=int),
    extra_weights: wp.array(dtype=wp.float64),
    x: wp.array(dtype=wp.vec3d),
    deformation: wp.array(dtype=wp.mat33d),
    particle_codes: wp.array(dtype=wp.int64),
):
    """Weighted sums of compatible affine-extrapolated positions and F at node i.

    Falls back to the next support ring when extend is set and the first ring has
    no compatible particle. Returns (position sum, F sum, weight sum, extended).
    """
    total = wp.float64(0.0)
    position = wp.vec3d(0.0)
    gradient = wp.mat33d(0.0)
    for a in range(offsets[i], offsets[i + 1]):
        p = indices[a]
        if compatible(code, particle_codes[p]):
            w = weights[a]
            position += w * (x[p] + deformation[p] * (nodes[i] - reference[p]))
            gradient += w * deformation[p]
            total += w
    extended = int(0)
    if total == wp.float64(0.0) and extend != 0:
        for a in range(extra_offsets[i], extra_offsets[i + 1]):
            p = extra_indices[a]
            if compatible(code, particle_codes[p]):
                w = extra_weights[a]
                position += w * (x[p] + deformation[p] * (nodes[i] - reference[p]))
                gradient += w * deformation[p]
                total += w
        if total > wp.float64(0.0):
            extended = 1
    return position, gradient, total, extended


@wp.kernel
def evaluate_nodes(
    nodes: wp.array(dtype=wp.vec3d),
    reference: wp.array(dtype=wp.vec3d),
    offsets: wp.array(dtype=int),
    indices: wp.array(dtype=int),
    weights: wp.array(dtype=wp.float64),
    extra_offsets: wp.array(dtype=int),
    extra_indices: wp.array(dtype=int),
    extra_weights: wp.array(dtype=wp.float64),
    x: wp.array(dtype=wp.vec3d),
    deformation: wp.array(dtype=wp.mat33d),
    particle_codes: wp.array(dtype=wp.int64),
    node_codes: wp.array(dtype=wp.int64),
    uniform_code: wp.int64,
    per_node_codes: int,
    extend: int,
    positions: wp.array(dtype=wp.vec3d),
    gradients: wp.array(dtype=wp.mat33d),
    extensions: wp.array(dtype=int),
    nan: wp.float64,
):
    i = wp.tid()
    code = uniform_code
    if per_node_codes != 0:
        code = node_codes[i]
    position, gradient, total, extended = node_field(
        i,
        code,
        extend,
        nodes,
        reference,
        offsets,
        indices,
        weights,
        extra_offsets,
        extra_indices,
        extra_weights,
        x,
        deformation,
        particle_codes,
    )
    if extended != 0:
        wp.atomic_add(extensions, 0, 1)
    if total > wp.float64(0.0):
        positions[i] = position / total
        gradients[i] = gradient / total
    else:
        positions[i] = wp.vec3d(nan)
        gradients[i] = wp.mat33d(nan)


@wp.kernel
def embed_vertices(
    vertex_codes: wp.array(dtype=wp.int64),
    corner_nodes: wp.array2d(dtype=int),
    corner_weights: wp.array2d(dtype=wp.float64),
    nodes: wp.array(dtype=wp.vec3d),
    reference: wp.array(dtype=wp.vec3d),
    offsets: wp.array(dtype=int),
    indices: wp.array(dtype=int),
    weights: wp.array(dtype=wp.float64),
    extra_offsets: wp.array(dtype=int),
    extra_indices: wp.array(dtype=int),
    extra_weights: wp.array(dtype=wp.float64),
    x: wp.array(dtype=wp.vec3d),
    deformation: wp.array(dtype=wp.mat33d),
    particle_codes: wp.array(dtype=wp.int64),
    vertices: wp.array(dtype=wp.vec3d),
    extensions: wp.array(dtype=int),
    unsupported: wp.array(dtype=int),
):
    """Trilinear interpolation of each vertex's own-code field over its grid cell."""
    v = wp.tid()
    result = wp.vec3d(0.0)
    for k in range(8):
        position, gradient, total, extended = node_field(
            corner_nodes[v, k],
            vertex_codes[v],
            1,
            nodes,
            reference,
            offsets,
            indices,
            weights,
            extra_offsets,
            extra_indices,
            extra_weights,
            x,
            deformation,
            particle_codes,
        )
        if extended != 0:
            wp.atomic_add(extensions, 0, 1)
        if total > wp.float64(0.0):
            result += corner_weights[v, k] * position / total
        elif corner_weights[v, k] > wp.float64(1.0e-9):
            wp.atomic_add(unsupported, 0, 1)
    vertices[v] = result


class CudaMaterialEmbedding(MaterialEmbedding):
    @classmethod
    def from_reference(cls, reference, device="cuda:0"):
        self = cls.__new__(cls)
        self.__dict__.update(reference.__dict__)
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("CUDA surface embedding requires a CUDA device")
        if reference.material_grid is not None:
            self.material_grid = cls.from_reference(reference.material_grid, self.device)

        self.g_nodes = self._upload(self.nodes, wp.vec3d)
        self.g_ref = self._upload(self.ref, wp.vec3d)
        self.g_x = wp.empty(len(self.ref), dtype=wp.vec3d, device=self.device)
        self.g_f = wp.empty(len(self.ref), dtype=wp.mat33d, device=self.device)
        self.g_codes = wp.empty(len(self.ref), dtype=wp.int64, device=self.device)
        self.g_node_codes = wp.empty(len(self.nodes), dtype=wp.int64, device=self.device)
        self.g_positions = wp.empty(len(self.nodes), dtype=wp.vec3d, device=self.device)
        self.g_gradients = wp.empty(len(self.nodes), dtype=wp.mat33d, device=self.device)
        self.g_extensions = wp.zeros(1, dtype=int, device=self.device)
        self.g_unsupported = wp.zeros(1, dtype=int, device=self.device)
        return self

    def _upload(self, data, dtype):
        return wp.array(np.asarray(data), dtype=dtype, device=self.device)

    def _support_arrays(self, support):
        return (
            self._upload(support.offsets.astype(np.int32), int),
            self._upload(support.indices.astype(np.int32), int),
            self._upload(support.weights, wp.float64),
        )

    @cached_property
    def g_support(self):
        return self._support_arrays(self.support)

    @cached_property
    def g_extra(self):
        return self._support_arrays(self.extension_support)

    @staticmethod
    def _codes(values, count):
        a = np.asarray(values)
        if a.shape != (count,) or not np.issubdtype(a.dtype, np.integer) or np.any(a < 0):
            raise ValueError("Cut codes must be nonnegative int64 integers")
        return np.ascontiguousarray(a, dtype=np.int64)

    def _upload_particles(self, x, F, particle_codes):
        x = np.ascontiguousarray(x, dtype=np.float64)
        F = np.ascontiguousarray(F, dtype=np.float64)
        if (
            x.shape != self.ref.shape
            or F.shape != (len(self.ref), 3, 3)
            or not np.isfinite(x).all()
            or not np.isfinite(F).all()
        ):
            raise ValueError("Expected finite particle positions and deformation gradients")
        self.g_x.assign(x)
        self.g_f.assign(F)
        self.g_codes.assign(self._codes(particle_codes, len(self.ref)))

    def _support_inputs(self, extend=True):
        # Without extension the kernels never read the second ring; skip building it.
        extra = self.g_extra if extend else self.g_support
        return [self.g_nodes, self.g_ref, *self.g_support, *extra, self.g_x, self.g_f, self.g_codes]

    def _evaluate(self, x, F, particle_codes, node_codes=None, code=0, extend=False):
        self._upload_particles(x, F, particle_codes)
        if node_codes is not None:
            self.g_node_codes.assign(self._codes(node_codes, len(self.nodes)))
        self.g_extensions.zero_()
        wp.launch(
            evaluate_nodes,
            len(self.nodes),
            [
                *self._support_inputs(extend),
                self.g_node_codes,
                wp.int64(int(code)),
                int(node_codes is not None),
                int(extend),
                self.g_positions,
                self.g_gradients,
                self.g_extensions,
                wp.float64(float("nan")),
            ],
            device=self.device,
        )
        if extend:
            self.extension_nodes += int(self.g_extensions.numpy()[0])

    def field(self, x, F, particle_codes, code):
        self._codes(np.array([code]), 1)
        self._evaluate(x, F, particle_codes, code=code, extend=True)
        return self.g_positions.numpy()

    def node_state(self, x, F, particle_codes, node_codes):
        self._evaluate(x, F, particle_codes, node_codes=node_codes)
        return self.g_positions.numpy(), self.g_gradients.numpy()

    def vertices(self, mesh, x, F, particle_codes):
        """All vertices in one launch; extension_nodes counts extended corner lookups."""
        grid = self.material_grid
        if grid is None:
            return super().vertices(mesh, x, F, particle_codes)
        if not len(mesh.reference):
            return np.empty((0, 3))
        fractional = (mesh.reference - self.grid_origin) / self.h
        base = np.minimum(np.maximum(np.floor(fractional).astype(int), 0), self.grid_cells - 1)
        local = fractional - base
        corners = np.ravel_multi_index(
            (base[:, None, :] + OFFSETS[None, :, :]).transpose(2, 0, 1), tuple(self.grid_cells + 1)
        )
        weights = np.where(OFFSETS[None, :, :] == 1, local[:, None, :], 1 - local[:, None, :]).prod(2)
        grid._upload_particles(x, F, particle_codes)
        device = self.device
        out = wp.empty(len(mesh.reference), dtype=wp.vec3d, device=device)
        grid.g_extensions.zero_()
        grid.g_unsupported.zero_()
        wp.launch(
            embed_vertices,
            len(mesh.reference),
            [
                wp.array(self._codes(mesh.codes, len(mesh.codes)), dtype=wp.int64, device=device),
                wp.array(corners.astype(np.int32), dtype=int, device=device),
                wp.array(weights, dtype=wp.float64, device=device),
                *grid._support_inputs(),
                out,
                grid.g_extensions,
                grid.g_unsupported,
            ],
            device=device,
        )
        counts = np.r_[grid.g_extensions.numpy(), grid.g_unsupported.numpy()]
        grid.extension_nodes += int(counts[0])
        if counts[1]:
            raise RuntimeError(
                "Surface region has no compatible material support; refine reference grid/support"
            )
        return out.numpy()
