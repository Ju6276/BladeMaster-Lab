"""Compensated dense solves and complete orthogonal decompositions."""

import numpy as np
from scipy.linalg.lapack import dgeequ

from ._compensated import Expansion, _two_sum, _two_product


def divide(numerator, denominator):
    numerator, denominator = map(Expansion.from_value, [numerator, denominator])
    result = Expansion(numerator.high / denominator.high)
    for _ in range(2):
        residual = numerator - denominator * result
        result = result + residual.high / denominator.high
    return result


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
    parts = np.zeros((4, *rhs.shape))

    def add(value):
        for i in range(4):
            parts[i], value = _two_sum(parts[i], value)

    add(rhs.high)
    add(rhs.low)
    for k in range(matrix.shape[1]):
        for left in [matrix.high[:, k, None], matrix.low[:, k, None]]:
            for right in [solution.high[None, k, :], solution.low[None, k, :]]:
                high, low = _two_product(left, right)
                add(-high)
                add(-low)
    value = Expansion(parts[0]) + Expansion(parts[1]) + Expansion(parts[2]) + Expansion(parts[3])
    return value[:, 0] if vector else value


class Factorization:
    def __init__(self, matrix):
        matrix = Expansion.from_value(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("A square matrix is required")
        if not np.isfinite(matrix.high).all() or not np.isfinite(matrix.low).all():
            raise ValueError("Finite matrix entries are required")
        self.matrix = Expansion(matrix.high.copy(), matrix.low.copy())
        self.lu = Expansion(matrix.high.copy(), matrix.low.copy())
        self.pivots = np.arange(matrix.shape[0])
        n = len(self.pivots)
        for k in range(n):
            pivot = k + np.argmax(abs(self.lu.high[k:, k]))
            if self.lu.high[pivot, k] == 0.0:
                raise np.linalg.LinAlgError("Singular compensated matrix")
            if pivot != k:
                _assign(self.lu, [k, pivot], self.lu[[pivot, k]])
                self.pivots[[k, pivot]] = self.pivots[[pivot, k]]
            if k + 1 < n:
                column = divide(self.lu[k + 1 :, k], self.lu[k, k])
                _assign(
                    self.lu,
                    (slice(k + 1, None), slice(k + 1, None)),
                    self.lu[k + 1 :, k + 1 :] - column[:, None] * self.lu[k, None, k + 1 :],
                )
                _assign(self.lu, (slice(k + 1, None), k), column)

    def _substitute(self, rhs):
        n = len(self.pivots)
        value = Expansion(rhs.high[self.pivots].copy(), rhs.low[self.pivots].copy())
        for k in range(n):
            if k:
                _assign(value, k, value[k] - self.lu[k, :k] @ value[:k])
        for k in range(n - 1, -1, -1):
            residual = value[k] - self.lu[k, k + 1 :] @ value[k + 1 :]
            _assign(value, k, divide(residual, self.lu[k, k]))
        return value

    def solve(self, rhs):
        rhs = Expansion.from_value(rhs)
        if rhs.ndim not in [1, 2] or rhs.shape[0] != len(self.pivots):
            raise ValueError("Right-hand side must match the matrix")
        if not np.isfinite(rhs.high).all() or not np.isfinite(rhs.low).all():
            raise ValueError("Finite right-hand side entries are required")
        if not len(self.pivots):
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
        result = self._substitute(rhs)
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
        raise RuntimeError(f"Compensated dense solve backward error: {error}")


def solve(matrix, rhs):
    return Factorization(matrix).solve(rhs)


def _orthonormal_columns(matrix):
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


def minimum_norm(matrix, rhs):
    matrix = Expansion.from_value(matrix)
    rhs = Expansion.from_value(rhs)
    if matrix.ndim != 2 or rhs.ndim not in (1, 2) or rhs.shape[0] != matrix.shape[0]:
        raise ValueError("Least-squares right-hand sides must match the matrix rows")
    if not all(np.isfinite(a).all() for a in (matrix.high, matrix.low, rhs.high, rhs.low)):
        raise ValueError("Least-squares inputs must be finite")
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
        return Expansion(np.zeros((matrix.shape[1], *Expansion.from_value(rhs).shape[1:])))
    Q = _columns(vectors)
    R = Q.T @ matrix
    Z = _orthonormal_columns(R.T)
    return Z @ solve(R @ Z, Q.T @ Expansion.from_value(rhs))


def equilibrated_solve(matrix, rhs):
    rows, columns, _, _, _, info = dgeequ(matrix)
    if info:
        raise np.linalg.LinAlgError(f"Matrix equilibration failed at row or column {info}")
    balanced = Expansion(rows[:, None]) * Expansion(matrix) * columns[None, :]
    target = rows * Expansion.from_value(rhs)
    try:
        result = solve(balanced, target)
    except np.linalg.LinAlgError:
        result = minimum_norm(balanced, target)
    return columns * result
