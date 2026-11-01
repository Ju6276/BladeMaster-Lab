"""Sparse factorization with residual refinement in expansion arithmetic."""

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from ._compensated import Expansion


def try_solve(matrix, rhs):
    """Return a refined solution, or None when full-precision factorization is needed."""
    from ._compensated_linalg import residual

    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not matrix.shape[0]
        or rhs.ndim not in (1, 2)
        or rhs.shape[0] != matrix.shape[0]
        or not all(np.isfinite(a).all() for a in (matrix.high, matrix.low, rhs.high, rhs.low))
    ):
        return None
    try:
        factor = splu(csc_matrix(matrix.high))
        solution = Expansion(factor.solve(rhs.high))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            for iteration in range(33):
                defect = residual(matrix, solution, rhs)
                scale = abs(rhs.high) + abs(matrix.high) @ abs(solution.high)
                tolerance = 64 * np.finfo(float).eps**2 * scale
                tolerance += (matrix.shape[1] + 1) * np.nextafter(0.0, 1.0)
                if not np.isfinite(defect.high).all() or not np.isfinite(scale).all():
                    return None
                if np.all(abs(defect.high) <= tolerance):
                    return solution
                if iteration == 32:
                    break
                correction = factor.solve(defect.high)
                if not np.isfinite(correction).all():
                    return None
                solution = solution + correction
    except (RuntimeError, ValueError):
        return None
    return None
