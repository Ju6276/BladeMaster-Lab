"""Analytic contractions with cancelling cotangents in interface motion."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_geometry import InterfaceGeometry


@pytest.mark.parametrize("large", [2.0**54, 2.0**56, 2.0**80])
def test_geometry_pullback_preserves_small_remainder_between_opposing_seeds(large):
    def array(value, kind):
        return wp.array(value, dtype=kind, requires_grad=True)

    geometry = InterfaceGeometry(
        array(np.zeros((3, 3)), wp.vec3d), array(np.tile(np.eye(3), (3, 1, 1)), wp.mat33d)
    )
    B = array(np.ones((3, 1)), wp.float64)
    gradient = np.array([0.25, -0.125, 0.5])
    G = array(np.tile(gradient, (3, 1, 1)), wp.vec3d)
    value = np.array([[1.25, -0.5, 0.75]])
    velocity = array(value, wp.vec3d)
    duration = array([0.125], wp.float64)
    with wp.Tape() as tape:
        result = geometry.advance(B, G, velocity, duration)
    coefficients = np.array([large, 1.0, -large])
    vector = np.array([1.0, -2.0, 0.5])
    matrix = np.array([[0.25, -0.125, 0.5], [0.125, 0.25, -0.25], [-0.5, 0.5, 0.125]])
    tape.backward(
        grads={
            result.gap: array(coefficients[:, None] * vector, wp.vec3d),
            result.frame: array(coefficients[:, None, None] * matrix, wp.mat33d),
        }
    )
    expected = vector + matrix @ gradient
    np.testing.assert_array_equal(velocity.grad.numpy(), 0.125 * expected[None])
    np.testing.assert_array_equal(duration.grad.numpy(), [float(value[0] @ expected)])
