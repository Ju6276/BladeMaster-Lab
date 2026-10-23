"""Reaction-block elimination preserves weak and redundant kinematic constraints."""

import numpy as np
import pytest

from blademaster_warp._cohesive_linear import solve_blocks


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("size", [3, 9])
def test_reaction_block_solution_matches_coupled_system(rank, size):
    rng = np.random.default_rng(51)
    count = 2
    c = rng.normal(size=(3 * count, size))
    if rank == 0:
        c[3:] = c[:3]
    d = np.zeros((3 * count, 3 * count))
    for i in range(count):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        block = rotation[:, :rank] @ np.diag(np.geomspace(0.1, 10.0, rank)) @ rotation[:, :rank].T
        d[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = block
    a = np.block([[np.eye(size), c.T], [c, -d]])
    expected = rng.normal(size=len(a))
    rhs = a @ expected
    actual = solve_blocks(a, rhs, size)
    np.testing.assert_allclose(a @ actual, rhs, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual[:size], expected[:size], rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("weak", [1e-6, 1e-10, 1e-14])
def test_local_weak_reaction_derivative_is_retained(weak):
    matrix = np.block([[np.eye(3), np.eye(3)], [np.eye(3), -np.diag([0.0, weak, 1.0])]])
    expected = np.arange(1.0, 7.0)
    rhs = matrix @ expected
    actual = solve_blocks(matrix, rhs, 3)
    np.testing.assert_allclose(matrix @ actual, rhs, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
