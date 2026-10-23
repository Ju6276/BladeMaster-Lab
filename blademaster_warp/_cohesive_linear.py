"""Eliminate local reaction blocks and retain their exact constraint nullspaces."""

import numpy as np
from scipy.linalg import qr


def _rank(singular, shape):
    if not len(singular):
        return 0
    return np.count_nonzero(singular > singular[0] * max(shape) * np.finfo(float).eps)


def _orthogonal_columns(vectors):
    vectors = np.asarray(vectors, dtype=np.longdouble)
    basis = np.zeros_like(vectors)
    triangular = np.zeros((vectors.shape[1], vectors.shape[1]), dtype=np.longdouble)
    for j in range(vectors.shape[1]):
        vector = vectors[:, j].copy()
        for _ in range(2):
            coefficient = basis[:, :j].T @ vector
            vector -= basis[:, :j] @ coefficient
            triangular[:j, j] += coefficient
        length = np.sqrt(np.sum(vector * vector))
        if length == 0.0:
            raise RuntimeError("Selected constraint directions lost rank")
        triangular[j, j] = length
        basis[:, j] = vector / length
    return basis, triangular


def minimum_norm(matrix, rhs):
    """Retain row-space orthogonality as well as the least-squares residual."""
    matrix = np.asarray(matrix, dtype=np.longdouble)
    rhs = np.asarray(rhs, dtype=np.longdouble)
    singular = np.linalg.svd(matrix.astype(float), compute_uv=False)
    rank = _rank(singular, matrix.shape)
    if rank == 0:
        return np.zeros((matrix.shape[1], *rhs.shape[1:]), dtype=np.longdouble)
    selected = qr(matrix.astype(float).T, mode="economic", pivoting=True)[2][:rank]
    basis, _ = _orthogonal_columns(matrix[selected].T)
    left, triangular = _orthogonal_columns(matrix @ basis)
    solution = left.T @ rhs
    for j in range(rank - 1, -1, -1):
        solution[j] = (solution[j] - triangular[j, j + 1 :] @ solution[j + 1 :]) / triangular[j, j]
    return basis @ solution


def _block_step(matrix, rhs, size):
    count = (len(rhs) - size) // 3
    hessian = matrix[:size, :size].copy()
    reduced_rhs = rhs[:size].copy()
    force_map = matrix[:size, size:].reshape(size, count, 3)
    constraints = matrix[size:, :size].reshape(count, 3, size)
    local_rhs = rhs[size:].reshape(count, 3)
    inverses, nullspaces, rows, targets, reactions = [], [], [], [], []
    for i in range(count):
        block = matrix[size + 3 * i : size + 3 * i + 3, size + 3 * i : size + 3 * i + 3]
        left, singular, right = np.linalg.svd(block)
        rank = _rank(singular, block.shape)
        inverse = (right[:rank].T / singular[:rank]) @ left[:, :rank].T
        nullspace = right[rank:].T
        inverses.append(inverse)
        nullspaces.append(nullspace)
        hessian -= force_map[:, i] @ inverse @ constraints[i]
        reduced_rhs -= force_map[:, i] @ inverse @ local_rhs[i]
        rows.append(left[:, rank:].T @ constraints[i])
        targets.append(left[:, rank:].T @ local_rhs[i])
        reactions.append(force_map[:, i] @ nullspace)
    constraint = np.concatenate(rows)
    target = np.concatenate(targets)
    reaction = np.concatenate(reactions, axis=1)
    if len(target):
        left, singular, right = np.linalg.svd(constraint, full_matrices=constraint.shape[0] < size)
        rank = _rank(singular, constraint.shape)
        particular = right[:rank].T @ ((left[:, :rank].T @ target) / singular[:rank])
        nullspace = right[rank:].T
        reaction_left, singular, _ = np.linalg.svd(reaction, full_matrices=reaction.shape[1] < size)
        rank = _rank(singular, reaction.shape)
        complement = reaction_left[:, rank:]
        reduced = complement.T @ hessian @ nullspace
        projected_rhs = complement.T @ (reduced_rhs - hessian @ particular)
        primal = particular + nullspace @ np.linalg.lstsq(reduced, projected_rhs, rcond=None)[0]
        multiplier = np.linalg.lstsq(reaction, reduced_rhs - hessian @ primal, rcond=None)[0]
    else:
        primal = np.linalg.lstsq(hessian, reduced_rhs, rcond=None)[0]
        multiplier = np.empty(0)
    dual, offset = [], 0
    for i, (inverse, nullspace) in enumerate(zip(inverses, nullspaces)):
        width = nullspace.shape[1]
        dual.append(
            inverse @ (local_rhs[i] - constraints[i] @ primal)
            + nullspace @ multiplier[offset : offset + width]
        )
        offset += width
    return np.r_[primal, np.array(dual).ravel()]


def solve_blocks(matrix, rhs, size):
    """Solve [H K; C diag(D_i)] with three reaction components per point."""
    solution = _block_step(matrix, rhs, size)
    precise_matrix = matrix.astype(np.longdouble)
    precise_rhs = rhs.astype(np.longdouble)
    best, previous = solution.copy(), np.inf
    for iteration in range(9):
        residual = precise_rhs - precise_matrix @ solution.astype(np.longdouble)
        bound = np.abs(precise_rhs) + np.abs(precise_matrix) @ np.abs(solution)
        error = np.max(np.abs(residual) / np.maximum(bound, np.finfo(float).tiny))
        if not np.isfinite(error) or error >= previous:
            return best
        best, previous = solution.copy(), error
        if error <= 64 * np.finfo(float).eps or iteration == 8:
            return solution
        solution += _block_step(matrix, residual.astype(float), size)
    return best
