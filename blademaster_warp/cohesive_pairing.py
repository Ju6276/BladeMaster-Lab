"""Current-space particle pairs and friction history indexed by physical particle IDs."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
import warp as wp

from ._cohesive_expansion import E, assign, norm, positive
from .compensated_state import CompensatedArray, as_compensated, add_gradient
from .topology import compatible


def pairs_at(position, reference, codes, diameter, spacing):
    """Select centre distances <= diameter; compatible near reference neighbours share material."""
    position = as_compensated(position)
    x = position.cpu()
    count = len(x.high)
    reference, codes = np.asarray(reference, dtype=float), np.asarray(codes)
    if position.high.dtype != wp.vec3d or position.high.shape != (count,):
        raise ValueError("Pair selection requires float64 particle positions")
    if reference.shape != (count, 3) or codes.shape != (count,) or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("Reference positions and integer side codes must match particles")
    if (
        not np.isfinite(x.high).all()
        or not np.isfinite(reference).all()
        or np.any(codes < 0)
        or not np.isfinite([diameter, spacing]).all()
        or min(diameter, spacing) <= 0
    ):
        raise ValueError("Pair geometry requires finite positions, nonnegative codes and positive lengths")
    if count < 2:
        return np.empty((0, 2), dtype=np.int64)
    reach = E(diameter) + 2 * np.max(abs(x.low))
    radius = np.nextafter(float(reach.numpy()), np.inf)
    candidates = cKDTree(x.high).query_pairs(radius, p=np.inf, output_type="ndarray")
    selected = []
    for p, q in candidates:
        if compatible(codes[p], codes[q]) and positive(
            2 * E(spacing) - norm(E(reference[p]) - E(reference[q]))
        ):
            continue
        if not positive(norm(x[p] - x[q]) - diameter):
            selected.append((int(p), int(q)))
    return np.asarray(sorted(selected), dtype=np.int64).reshape(-1, 2)


def _pairs(value, count):
    result = np.asarray(value)
    if result.ndim != 2 or result.shape[1] != 2 or not np.issubdtype(result.dtype, np.integer):
        raise ValueError("Contact pairs require an integer (count, 2) array")
    if np.any(result < 0) or np.any(result >= count) or np.any(result[:, 0] >= result[:, 1]):
        raise ValueError("Contact pairs require 0 <= p < q < particle_count")
    order = np.lexsort((result[:, 1], result[:, 0]))
    if not np.array_equal(order, np.arange(len(result))) or (
        len(result) > 1 and np.any(np.all(result[1:] == result[:-1], axis=1))
    ):
        raise ValueError("Contact pairs must be sorted and unique")
    result = result.astype(np.int64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PairHistory:
    particle_count: int
    pairs: np.ndarray
    values: CompensatedArray

    def __post_init__(self):
        if (
            isinstance(self.particle_count, bool)
            or not isinstance(self.particle_count, (int, np.integer))
            or self.particle_count < 0
        ):
            raise ValueError("Particle count must be a nonnegative integer")
        pairs = _pairs(self.pairs, self.particle_count)
        values = as_compensated(self.values)
        if (
            values.high.dtype != wp.vec3d
            or values.high.shape != (len(pairs),)
            or not np.isfinite(values.high.numpy()).all()
        ):
            raise ValueError("Friction history requires one finite float64 vector per pair")
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "values", values)

    @classmethod
    def empty(cls, particle_count, *, device):
        return cls(
            particle_count,
            np.empty((0, 2), dtype=np.int64),
            CompensatedArray(wp.zeros(0, dtype=wp.vec3d, device=device, requires_grad=True)),
        )

    def reindex(self, pairs):
        """Return current rows and retired rows, preserving both sets' state derivatives."""
        pairs = _pairs(pairs, self.particle_count)
        lookup = {tuple(pair): i for i, pair in enumerate(self.pairs)}
        indices = np.asarray([lookup.get(tuple(pair), -1) for pair in pairs], dtype=int)
        retained = set(indices[indices >= 0])
        retired = np.asarray([i for i in range(len(self.pairs)) if i not in retained], dtype=int)
        original = self.values.cpu()
        current = E(np.zeros((len(pairs), 3)))
        existing = indices >= 0
        assign(current, existing, original[indices[existing]])
        arrays = [
            CompensatedArray.from_cpu(v, dtype=wp.vec3d, device=self.values.high.device)
            for v in [current, original[retired]]
        ]
        tape = wp._src.context.runtime.tape
        if tape is not None and len(self.pairs):

            def backward():
                gradient = E(np.zeros_like(original.high))
                if np.any(existing):
                    assign(gradient, indices[existing], E(arrays[0].high.grad.numpy()[existing]))
                if len(retired):
                    assign(gradient, retired, E(arrays[1].high.grad.numpy()))
                add_gradient(self.values.high, gradient)

            tape.record_func(
                backward,
                [a for a in [*[v.high for v in arrays], self.values.high] if a.size and a.requires_grad],
            )
        return (
            PairHistory(self.particle_count, pairs, arrays[0]),
            PairHistory(self.particle_count, self.pairs[retired], arrays[1]),
        )

    def snapshot(self):
        return dict(
            particle_count=self.particle_count, pairs=self.pairs.copy(), values=self.values.snapshot()
        )

    @classmethod
    def restore(cls, snapshot, *, device):
        return cls(
            snapshot["particle_count"],
            snapshot["pairs"],
            CompensatedArray.restore(snapshot["values"], dtype=wp.vec3d, device=device),
        )
