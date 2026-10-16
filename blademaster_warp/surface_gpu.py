"""CUDA nodal embedding for the linked-volume surface."""

import numpy as np
import warp as wp
from .material import compatible
from .surface import MaterialEmbedding


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
    particle_codes: wp.array(dtype=int),
    node_codes: wp.array(dtype=int),
    uniform_code: int,
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
    if total == wp.float64(0.0) and extend != 0:
        for a in range(extra_offsets[i], extra_offsets[i + 1]):
            p = extra_indices[a]
            if compatible(code, particle_codes[p]):
                w = extra_weights[a]
                position += w * (x[p] + deformation[p] * (nodes[i] - reference[p]))
                gradient += w * deformation[p]
                total += w
        if total > wp.float64(0.0):
            wp.atomic_add(extensions, 0, 1)
    if total > wp.float64(0.0):
        positions[i] = position / total
        gradients[i] = gradient / total
    else:
        positions[i] = wp.vec3d(nan)
        gradients[i] = wp.mat33d(nan)


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

        def upload(data, dtype):
            return wp.array(np.asarray(data), dtype=dtype, device=self.device)

        def support_arrays(support):
            offsets = np.r_[0, np.cumsum([len(ids) for ids, _ in support])]
            indices = np.concatenate([ids for ids, _ in support]).astype(np.int32)
            weights = np.concatenate([w for _, w in support])
            return upload(offsets, int), upload(indices, int), upload(weights, wp.float64)

        self.g_nodes = upload(self.nodes, wp.vec3d)
        self.g_ref = upload(self.ref, wp.vec3d)
        self.g_support = support_arrays(self.support)
        self.g_extra = support_arrays(self.extension_support)
        self.g_x = wp.empty(len(self.ref), dtype=wp.vec3d, device=self.device)
        self.g_f = wp.empty(len(self.ref), dtype=wp.mat33d, device=self.device)
        self.g_codes = wp.empty(len(self.ref), dtype=int, device=self.device)
        self.g_node_codes = wp.empty(len(self.nodes), dtype=int, device=self.device)
        self.g_positions = wp.empty(len(self.nodes), dtype=wp.vec3d, device=self.device)
        self.g_gradients = wp.empty(len(self.nodes), dtype=wp.mat33d, device=self.device)
        self.g_extensions = wp.zeros(1, dtype=int, device=self.device)
        return self

    @staticmethod
    def _codes(values, count):
        a = np.asarray(values)
        if (
            a.shape != (count,)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0)
            or np.any(a > np.iinfo(np.int32).max)
        ):
            raise ValueError("Cut codes must be nonnegative int32-representable integers")
        return np.ascontiguousarray(a, dtype=np.int32)

    def _evaluate(self, x, F, particle_codes, node_codes=None, code=0, extend=False):
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
        if node_codes is not None:
            self.g_node_codes.assign(self._codes(node_codes, len(self.nodes)))
        self.g_extensions.zero_()
        wp.launch(
            evaluate_nodes,
            len(self.nodes),
            [
                self.g_nodes,
                self.g_ref,
                *self.g_support,
                *self.g_extra,
                self.g_x,
                self.g_f,
                self.g_codes,
                self.g_node_codes,
                int(code),
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
