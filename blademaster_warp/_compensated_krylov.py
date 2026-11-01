"""Right-preconditioned GMRES corrections at expansion precision."""

import numpy as np

from ._compensated import Expansion as E
from ._compensated_linalg import divide, residual


def _assign(target, index, value):
    value = E.from_value(value)
    target.high[index], target.low[index] = value.high, value.low


def _norm(value):
    largest = float(np.max(abs(value.high), initial=0.0))
    if largest == 0.0:
        return E(0.0)
    exponent = int(np.frexp(largest)[1])
    scaled = E(np.ldexp(value.high, -exponent), np.ldexp(value.low, -exponent))
    squared = (scaled * scaled).sum()
    length = E(np.sqrt(squared.high))
    length = length + divide(squared - length * length, 2.0 * length)
    return E(np.ldexp(length.high, exponent), np.ldexp(length.low, exponent))


def _error(matrix, solution, rhs):
    defect = residual(matrix, solution, rhs)
    scale = abs(rhs.high) + abs(matrix.high) @ abs(solution.high)
    tolerance = 64 * np.finfo(float).eps ** 2 * scale
    tolerance += (matrix.shape[1] + 1) * np.nextafter(0.0, 1.0)
    error = float(np.max(abs(defect.high) / np.maximum(scale, np.finfo(float).tiny)))
    return defect, error, bool(np.isfinite(error) and np.all(abs(defect.high) <= tolerance))


def _column(factor, rhs, solution):
    matrix = factor.matrix
    size = matrix.shape[0]
    iterations = 0
    for restart in range(8):
        defect, best_error, converged = _error(matrix, solution, rhs)
        if converged:
            return solution, iterations
        beta = _norm(defect)
        vectors = [divide(defect, beta)]
        directions, rotations = [], []
        h = E(np.zeros((size + 1, size)))
        g = E(np.zeros(size + 1))
        _assign(g, 0, beta)
        initial = solution
        for j in range(size):
            direction = factor._substitute(vectors[j].reshape(size, 1))[:, 0]
            if not np.isfinite(direction.high).all() or not np.isfinite(direction.low).all():
                raise RuntimeError("GMRES preconditioner left the finite domain")
            directions.append(direction)
            value = matrix @ direction
            for _ in range(2):
                for i, vector in enumerate(vectors):
                    coefficient = (vector * value).sum()
                    _assign(h, (i, j), h[i, j] + coefficient)
                    value = value - coefficient * vector
            length = _norm(value)
            _assign(h, (j + 1, j), length)
            for i, (c, s) in enumerate(rotations):
                first, second = h[i, j], h[i + 1, j]
                _assign(h, (i, j), c * first + s * second)
                _assign(h, (i + 1, j), -s * first + c * second)
            radius = _norm(h[j:j + 2, j])
            if float(radius.high) == 0.0:
                break
            c, s = divide(h[j, j], radius), divide(h[j + 1, j], radius)
            rotations.append((c, s))
            _assign(h, (j, j), radius)
            _assign(h, (j + 1, j), 0.0)
            first, second = g[j], g[j + 1]
            _assign(g, j, c * first + s * second)
            _assign(g, j + 1, -s * first + c * second)
            coefficients = E(np.zeros(j + 1))
            for i in reversed(range(j + 1)):
                tail = (h[i, i + 1:j + 1] * coefficients[i + 1:]).sum()
                _assign(coefficients, i, divide(g[i] - tail, h[i, i]))
            candidate = initial
            for i, direction in enumerate(directions):
                candidate = candidate + coefficients[i] * direction
            _, error, converged = _error(matrix, candidate, rhs)
            iterations += 1
            if converged:
                return candidate, iterations
            if error < best_error:
                best_error, solution = error, candidate
            if float(length.high) == 0.0:
                break
            vectors.append(divide(value, length))
    raise RuntimeError(f"GMRES correction backward error: {best_error}")


def refine(factor, rhs, initial):
    result = E(initial.high.copy(), initial.low.copy())
    iterations = 0
    for i in range(rhs.shape[1]):
        column, count = _column(factor, rhs[:, i], initial[:, i])
        _assign(result, (slice(None), i), column)
        iterations += count
    return result, iterations
