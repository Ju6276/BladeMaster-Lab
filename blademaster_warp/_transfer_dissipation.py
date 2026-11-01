"""APIC kinetic loss as weighted velocity variances before and after field exchange."""

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix

from ._compensated import Expansion
from ._compensated_kernels import sparse_matrix_product


def product(matrix, value):
    if not isinstance(value, Expansion):
        return matrix @ value
    vector = value.ndim == 1
    columns = value.reshape(-1, 1) if vector else value
    high, low = sparse_matrix_product(matrix.indptr, matrix.indices, matrix.data, columns.high, columns.low)
    result = Expansion(high, low)
    return result.reshape(-1) if vector else result


@dataclass(frozen=True)
class TransferDissipation:
    differences: csr_matrix
    weights: np.ndarray
    transpose: csr_matrix = field(init=False)

    def __post_init__(self):
        matrix = self.differences.copy()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        weights = np.asarray(self.weights, dtype=float).copy()
        if weights.shape != (matrix.shape[0],) or np.any(weights < 0.0):
            raise ValueError("Velocity variances require nonnegative matching weights")
        if not np.isfinite(weights).all() or not np.isfinite(matrix.data).all():
            raise ValueError("Velocity variance coefficients must be finite")
        object.__setattr__(self, "differences", matrix)
        object.__setattr__(self, "transpose", matrix.T.tocsr())
        object.__setattr__(self, "weights", weights)

    def action(self, value):
        residual = product(self.differences, value)
        weight = self.weights if residual.ndim == 1 else self.weights[:, None]
        return product(self.transpose, weight * residual)

    def power(self, value):
        residual = product(self.differences, value)
        weight = self.weights if residual.ndim == 1 else self.weights[:, None]
        return (weight * residual * residual).sum()

    def trace(self):
        squares = np.asarray(self.differences.multiply(self.differences).sum(axis=1)).ravel()
        return float(self.weights @ squares)


def build(slots, weights, feature, unmixed, mapping, exchange):
    """Exchange tuples are (destination, source, dimensionless mass * fraction)."""
    values, indices, offsets, factors = [], [], [0], []

    def append(row, weight):
        columns = np.flatnonzero(row)
        values.extend(row[columns])
        indices.extend(columns)
        offsets.append(len(values))
        factors.append(weight)

    for particle in range(len(slots)):
        for corner, slot in enumerate(slots[particle]):
            row = -unmixed[slot].copy()
            row[4 * particle : 4 * particle + 4] += feature[particle, corner]
            append(row, weights[particle, corner])
    for destination, source, weight in exchange:
        append(unmixed[source] - mapping[destination], weight)
    matrix = csr_matrix((values, indices, offsets), shape=(len(factors), unmixed.shape[1]))
    return TransferDissipation(matrix, np.asarray(factors))
