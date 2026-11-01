"""Compensated dense solves and complete orthogonal decompositions."""

import heapq

import numpy as np
from scipy.linalg import qr
from scipy.linalg.lapack import dgeequb

from ._compensated import Expansion
from ._compensated_kernels import array_divide, matrix_residual, lu_factor, lu_substitute


def divide(numerator, denominator):
    numerator, denominator = map(Expansion.from_value, [numerator, denominator])
    high, low = array_divide.ufunc(numerator.high, numerator.low, denominator.high, denominator.low)
    return Expansion(high, low)


def _assign(target, index, value):
    value = Expansion.from_value(value)
    target.high[index], target.low[index] = value.high, value.low


def _norm(value):
    squared = (value * value).sum()
    result = Expansion(np.sqrt(squared.high))
    if result.high == 0.0:
        return result
    return result + divide(squared - result * result, 2.0 * result)


def _columns(values):
    return Expansion(np.stack([a.high for a in values], axis=1), np.stack([a.low for a in values], axis=1))


def residual(matrix, solution, rhs):
    matrix, solution, rhs = map(Expansion.from_value, [matrix, solution, rhs])
    vector = rhs.ndim == 1
    if vector:
        solution, rhs = solution.reshape(-1, 1), rhs.reshape(-1, 1)
    high, low = matrix_residual(matrix.high, matrix.low, solution.high, solution.low, rhs.high, rhs.low)
    result = Expansion(high, low)
    return result[:, 0] if vector else result


def _singleton_order(matrix):
    nonzero = (matrix.high != 0.) | (matrix.low != 0.)
    degrees = nonzero.sum(axis=1)
    active_rows = np.ones(matrix.shape[0], dtype=bool)
    active_columns = np.ones(matrix.shape[1], dtype=bool)
    queue = list(np.flatnonzero(degrees == 1))
    heapq.heapify(queue)
    first_rows, first_columns = [], []
    while queue:
        row = heapq.heappop(queue)
        if not active_rows[row] or degrees[row] != 1:
            continue
        column = int(np.flatnonzero(nonzero[row] & active_columns)[0])
        active_rows[row] = False
        active_columns[column] = False
        first_rows.append(row)
        first_columns.append(column)
        affected = np.flatnonzero(nonzero[:, column] & active_rows)
        degrees[affected] -= 1
        for index in affected[degrees[affected] == 1]:
            heapq.heappush(queue, int(index))
    return (np.r_[first_rows, np.flatnonzero(active_rows)].astype(int),
            np.r_[first_columns, np.flatnonzero(active_columns)].astype(int), len(first_rows))


class Factorization:
    def __init__(self, matrix):
        matrix = Expansion.from_value(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError('A square matrix is required')
        if not np.isfinite(matrix.high).all() or not np.isfinite(matrix.low).all():
            raise ValueError('Finite matrix entries are required')
        self.matrix = Expansion(matrix.high.copy(), matrix.low.copy())
        rows, columns, singletons = _singleton_order(matrix)
        permuted = matrix[rows][:, columns]
        high, low, self.pivots, status = lu_factor(permuted.high, permuted.low, rows, singletons)
        if status >= 0:
            raise np.linalg.LinAlgError('Singular compensated matrix')
        self.lu = Expansion(high, low)
        self.inverse_columns = np.argsort(columns)

    def _substitute(self, rhs):
        high, low = lu_substitute(self.lu.high, self.lu.low,
                               rhs.high[self.pivots], rhs.low[self.pivots])
        return Expansion(high[self.inverse_columns], low[self.inverse_columns])

    def solve(self, rhs):
        rhs = Expansion.from_value(rhs)
        if rhs.ndim not in [1, 2] or rhs.shape[0] != len(self.pivots):
            raise ValueError("Right-hand side must match the matrix")
        if not np.isfinite(rhs.high).all() or not np.isfinite(rhs.low).all():
            raise ValueError("Finite right-hand side entries are required")
        if not rhs.high.size:
            return Expansion(rhs.high.copy(), rhs.low.copy())
        shape = rhs.shape
        rhs = rhs.reshape(rhs.shape[0], -1)
        largest = np.max(abs(rhs.high), axis=0)
        exponents = np.where(largest > 0.0, np.frexp(largest)[1] - 1, 0)
        components = np.concatenate([abs(rhs.high), abs(rhs.low)], axis=0)
        smallest = np.min(np.where(components > 0.0, components, np.inf), axis=0)
        safe_downshift = np.maximum(0, np.frexp(smallest)[1] - np.finfo(float).minexp - 1)
        exponents = np.minimum(exponents, safe_downshift)
        rhs = Expansion(np.ldexp(rhs.high, -exponents), np.ldexp(rhs.low, -exponents))
        initial = self._substitute(rhs)
        result = initial
        for iteration in range(33):
            defect = residual(self.matrix, result, rhs)
            scale = abs(rhs.high) + abs(self.matrix.high) @ abs(result.high)
            error = float(np.max(abs(defect.high) / np.maximum(scale, np.finfo(float).tiny)))
            underflow = (self.matrix.shape[1] + 1) * np.nextafter(0.0, 1.0)
            tolerance = 64 * np.finfo(float).eps ** 2 * scale + underflow
            if np.isfinite(error) and np.all(abs(defect.high) <= tolerance):
                self.backward_error = error
                self.underflow_residual = float(
                    np.max(abs(defect.high)[64 * np.finfo(float).eps ** 2 * scale < underflow], initial=0.0)
                )
                self.refinements = iteration
                return Expansion(np.ldexp(result.high, exponents), np.ldexp(result.low, exponents)).reshape(
                    shape
                )
            if iteration == 32:
                break
            result = result + self._substitute(defect)
        from ._compensated_krylov import refine

        result, self.krylov_iterations = refine(self, rhs, initial)
        defect = residual(self.matrix, result, rhs)
        scale = abs(rhs.high) + abs(self.matrix.high) @ abs(result.high)
        self.backward_error = float(np.max(abs(defect.high) / np.maximum(scale, np.finfo(float).tiny)))
        self.underflow_residual = float(
            np.max(abs(defect.high)[64 * np.finfo(float).eps ** 2 * scale < underflow], initial=0.0)
        )
        self.refinements = iteration
        return Expansion(np.ldexp(result.high, exponents), np.ldexp(result.low, exponents)).reshape(shape)


def solve(matrix, rhs):
    from ._compensated_sparse import try_solve

    matrix, rhs = map(Expansion.from_value, [matrix, rhs])
    refined = try_solve(matrix, rhs)
    if refined is not None:
        return refined
    return Factorization(matrix).solve(rhs)


def _orthonormal_columns(matrix):
    matrix = Expansion.from_value(matrix)
    if not matrix.shape[1]:
        return Expansion(np.zeros(matrix.shape))
    columns = []
    for j in range(matrix.shape[1]):
        vector = matrix[:, j]
        for _ in range(2):
            for previous in columns:
                vector = vector - previous * (previous * vector).sum()
        length = _norm(vector)
        if length.high == 0:
            raise np.linalg.LinAlgError("Dependent retained orthogonal columns")
        columns.append(divide(vector, length))
    return _columns(columns)


def column_basis(matrix):
    """Pivoted, twice-reorthogonalized basis at two-component roundoff."""
    matrix = Expansion.from_value(matrix)
    if matrix.ndim != 2 or not np.isfinite(matrix.high).all() or not np.isfinite(matrix.low).all():
        raise ValueError("A finite matrix is required")
    residual = Expansion(matrix.high.copy(), matrix.low.copy())
    scale = max(float(np.max(abs(matrix.high), initial=0.0)), np.finfo(float).tiny)
    bound = 64 * np.finfo(float).eps ** 2 * max(matrix.shape) * scale
    vectors = []
    for _ in range(min(matrix.shape)):
        lengths = (residual * residual).sum(axis=0)
        j = int(np.argmax(lengths.high))
        vector = residual[:, j]
        for _ in range(2):
            for previous in vectors:
                vector = vector - previous * (previous * vector).sum()
        length = _norm(vector)
        if float(length.high) <= bound:
            break
        unit = divide(vector, length)
        vectors.append(unit)
        for _ in range(2):
            residual = residual - unit[:, None] * (unit[:, None] * residual).sum(axis=0)[None, :]
    if not vectors:
        return Expansion(np.zeros((matrix.shape[0], 0)))
    return _columns(vectors)



def minimum_norm(matrix, rhs):
    matrix = Expansion.from_value(matrix)
    rhs = Expansion.from_value(rhs)
    if matrix.ndim != 2 or rhs.ndim not in (1, 2) or rhs.shape[0] != matrix.shape[0]:
        raise ValueError("Least-squares right-hand sides must match the matrix rows")
    if not all(np.isfinite(a).all() for a in (matrix.high, matrix.low, rhs.high, rhs.low)):
        raise ValueError("Least-squares inputs must be finite")
    Q = column_basis(matrix)
    if not Q.shape[1]:
        return Expansion(np.zeros((matrix.shape[1], *rhs.shape[1:])))
    R = Q.T @ matrix
    Z = _orthonormal_columns(R.T)
    return Z @ solve(R @ Z, Q.T @ Expansion.from_value(rhs))


def row_basis_product(matrix, value):
    """Share redundant row evaluations when the reconstruction is backward stable.

    The retained rows are original rows, preserving cancellation within a row.
    A remainder exceeding the expansion arithmetic bound keeps the full matrix.
    """
    matrix, value = map(Expansion.from_value, [matrix, value])
    if matrix.ndim != 2 or value.ndim not in (1, 2) or matrix.shape[1] != value.shape[0]:
        raise ValueError("A matrix and matching vector or matrix are required")
    if not matrix.shape[0] or not matrix.shape[1]:
        return matrix @ value
    singular = np.linalg.svd(matrix.high, compute_uv=False)
    rank = np.count_nonzero(singular > singular[0] * max(matrix.shape) * np.finfo(float).eps)
    if not rank or rank == matrix.shape[0]:
        return matrix @ value
    rows = qr(matrix.high.T, pivoting=True, mode="economic")[2][:rank]
    basis = matrix[rows]
    coefficients = minimum_norm(basis.T, matrix.T).T
    coefficients.high[rows] = np.eye(rank)
    coefficients.low[rows] = 0.0
    defect = residual(coefficients, basis, matrix)
    scale = abs(matrix.high) + abs(coefficients.high) @ abs(basis.high)
    bound = 64 * np.finfo(float).eps ** 2 * scale
    if np.any(abs(defect.high) > bound):
        return matrix @ value
    return coefficients @ (basis @ value)


def equilibrated_solve(matrix, rhs):
    """Correct a float64 Jacobian; callers check the complete original residual."""
    rows, columns, _, _, _, info = dgeequb(matrix)
    if info:
        raise np.linalg.LinAlgError(f"Matrix equilibration failed at row or column {info}")
    balanced = Expansion(rows[:, None]) * Expansion(matrix) * columns[None, :]
    target = rows * Expansion.from_value(rhs)
    left, singular, right = np.linalg.svd(balanced.high, full_matrices=False)
    rank = np.count_nonzero(singular > singular[0] * max(matrix.shape) * np.finfo(float).eps)
    if rank == matrix.shape[0]:
        result = solve(balanced, target)
    else:
        U, V = Expansion(left[:, :rank]), Expansion(right[:rank].T)
        result = V @ solve(U.T @ balanced @ V, U.T @ target)
    return columns * result
