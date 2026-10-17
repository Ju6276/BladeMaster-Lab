"""Pose-invariant spectral shape reward (TopoCut, Wang et al., CoRL 2025, Eqs. 2-3 and 12-17).

Each fragment is a point cloud. Its descriptor is the smallest eigenpairs of the
combinatorial Laplacian L = D - W of a k-nearest-neighbour graph with Gaussian
weights W_ij = exp(-d_ij^2 / sigma^2). Both shapes of a comparison use the same
sigma (metres), so the descriptor is sensitive to size but not to pose.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

NUM_POINTS = 512
NEIGHBORS = 30
NUM_EIGEN = 30


def sample(points, num_points=NUM_POINTS, jitter=0.0, seed=0):
    """Uniform subsample; a cloud smaller than num_points is resampled with
    replacement plus uniform jitter of half-width `jitter` (e.g. half the particle spacing)."""
    points = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(points) >= num_points:
        return points[rng.choice(len(points), num_points, replace=False)]
    picked = points[rng.choice(len(points), num_points, replace=True)]
    return picked + rng.uniform(-jitter, jitter, picked.shape)


def default_sigma(points, neighbors=NEIGHBORS):
    """Median distance to the k-th neighbour."""
    distance, _ = cKDTree(points).query(points, neighbors + 1)
    return float(np.median(distance[:, -1]))


def descriptor(points, sigma, neighbors=NEIGHBORS, num_eigen=NUM_EIGEN):
    """Smallest num_eigen eigenvalues and eigenvectors of the kNN-graph Laplacian."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    distance, index = cKDTree(points).query(points, neighbors + 1)
    weights = np.zeros((n, n))
    rows = np.repeat(np.arange(n), neighbors)
    weights[rows, index[:, 1:].ravel()] = np.exp(-(distance[:, 1:].ravel() ** 2) / sigma**2)
    weights = np.maximum(weights, weights.T)
    laplacian = np.diag(weights.sum(1)) - weights
    values, vectors = np.linalg.eigh(laplacian)
    return values[:num_eigen], vectors[:, :num_eigen]


def distance(x, y, alpha=1.0, beta=1.0):
    """Eq. 2: alpha ||Lambda_X - Lambda_Y||^2 + beta ||Phi_X^T Phi_X - Phi_Y^T Phi_Y||_F^2.

    eigh returns orthonormal eigenvectors, so the second term is zero; it is kept
    to follow the paper's definition.
    """
    (lx, px), (ly, py) = x, y
    return float(alpha * np.sum((lx - ly) ** 2) + beta * np.sum((px.T @ px - py.T @ py) ** 2))


def reward(d, r_max=1.0, gamma=0.05, tau=10.0, delta=0.1):
    """Eq. 16: linear decay with slope gamma up to tau, slope delta beyond.

    The defaults put resampling noise of one 512-point cloud (d ~ 2) near r_max
    and a 5x5x32 mm stick against a 6x6x32 mm one (d ~ 55) below zero.
    """
    if d <= tau:
        return r_max - gamma * d
    return r_max - gamma * tau - delta * (d - tau)


def match(fragments, goals, sigma=None, jitter=0.0, seed=0, **reward_kwargs):
    """Eq. 17 with a one-to-one assignment of fragments to goal pieces.

    Returns (total reward, distance matrix, list of (fragment, goal) pairs).
    The pairing minimises the summed spectral distance; unmatched fragments or
    goal pieces add nothing.
    """
    goal_points = [sample(g, jitter=jitter, seed=seed) for g in goals]
    if sigma is None:
        sigma = float(np.median([default_sigma(g) for g in goal_points]))
    goal_desc = [descriptor(g, sigma) for g in goal_points]
    frag_desc = [descriptor(sample(f, jitter=jitter, seed=seed), sigma) for f in fragments]
    d = np.array([[distance(f, g) for g in goal_desc] for f in frag_desc])
    rows, cols = linear_sum_assignment(d)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    total = sum(reward(d[i, j], **reward_kwargs) for i, j in pairs)
    return total, d, pairs
