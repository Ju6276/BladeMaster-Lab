"""Redundant row actions, weak independent directions and kinematic pullbacks."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp._compensated import Expansion as E
from blademaster_warp._compensated_linalg import row_basis_product
from blademaster_warp.compensated_state import CompensatedArray
from blademaster_warp.cohesive_geometry import InterfaceGeometry


def redundant_matrix():
    coefficients = E([[1.0, 0.0], [0.0, 1.0], [0.3, -0.7]])
    basis = E([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    return coefficients @ basis, coefficients, basis


@pytest.mark.parametrize("exponent", [-20, -60, -80])
def test_shared_row_action_retains_small_relative_motion(exponent):
    matrix, coefficients, basis = redundant_matrix()
    velocity = E(np.ones(3), np.ldexp([0.0, 1.0, 3.0], exponent))
    expected = coefficients @ (basis @ velocity)
    actual = row_basis_product(matrix, velocity)
    error = abs((actual - expected).high)
    assert np.max(error) <= 64 * np.finfo(float).eps ** 2 * np.max(abs(expected.high))
    assert np.all(actual.high != 0)


def test_independent_direction_in_remainder_is_retained():
    matrix = E(np.ones((2, 2)), [[0.0, 0.0], [0.0, 2.0**-54]])
    velocity = E([1.0, -1.0])
    actual = row_basis_product(matrix, velocity)
    np.testing.assert_array_equal(actual.high, [0.0, -(2.0**-54)])
    np.testing.assert_array_equal(actual.low, 0.0)


def test_full_row_rank_cancellation_and_empty_action():
    actual = row_basis_product(E([[2.0**60, 1.0, -(2.0**60)]]), E(np.ones(3)))
    np.testing.assert_array_equal(actual.high, [1.0])
    assert row_basis_product(E(np.zeros((0, 3))), E(np.ones((3, 2)))).shape == (0, 2)


def test_redundant_geometry_action_and_total_state_pullbacks():
    def array(value, dtype=wp.float64):
        return wp.array(value, dtype=dtype, requires_grad=True)

    matrix, _, _ = redundant_matrix()
    separation = CompensatedArray(array(matrix.high), matrix.low)
    frame_map = array(np.zeros((3, 3, 3)), wp.vec3d)
    velocity = array([[0.5, 0.2, -0.1], [-0.4, 0.7, 0.3], [0.1, -0.5, 0.4]], wp.vec3d)
    duration = array([0.003])
    geometry = InterfaceGeometry(
        array(np.zeros((3, 3)), wp.vec3d), array(np.tile(np.eye(3), (3, 1, 1)), wp.mat33d)
    )
    seed = np.random.default_rng(71).normal(size=(3, 3))
    with wp.Tape() as tape:
        result = geometry.advance(separation, frame_map, velocity, duration)
    tape.backward(grads={result.gap: array(seed, wp.vec3d)})
    np.testing.assert_allclose(
        velocity.grad.numpy(), duration.numpy()[0] * matrix.high.T @ seed, rtol=2e-13, atol=1e-16
    )
    np.testing.assert_allclose(
        separation.high.grad.numpy(), duration.numpy()[0] * seed @ velocity.numpy().T, rtol=2e-13, atol=1e-16
    )
    for source in [separation.high, velocity, duration]:
        base = source.numpy().copy()
        direction = np.random.default_rng(18).normal(size=base.shape) * 0.01
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            values = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                moved = geometry.advance(separation, frame_map, velocity, duration)
                values.append(np.sum(moved.gap.numpy() * seed))
            np.testing.assert_allclose(ad, (values[1] - values[0]) / (2 * epsilon), rtol=2e-6, atol=1e-12)
        source.assign(base)
