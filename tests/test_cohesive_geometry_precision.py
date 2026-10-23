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


@pytest.mark.parametrize("integrated", [False, True])
@pytest.mark.parametrize("remainder_storage", [False, True])
def test_unmapped_opening_residual_uses_total_geometry(integrated, remainder_storage):
    from blademaster_warp.differentiable_cohesive import _evaluate
    from test_differentiable_cohesive import arrays

    opening, compliance, area = 2.0**-40, 2.0**-42, 0.25
    gap = np.array([[opening, 0.0, 0.0]])
    values = [
        np.zeros((1, 3)) if remainder_storage else gap,
        np.eye(3)[None],
        np.array([[1.0, 0.0, 0.0]]),
        np.zeros(1),
        np.array([[2.0, 2.0, 0.0]]),
        np.zeros((1, 1)),
        np.zeros((1, 1, 3)),
        np.array([area]),
        np.ones(1),
        np.ones(1),
    ]
    sources = arrays(values)
    displacement = wp.array(np.zeros((1, 3)), dtype=wp.vec3d, requires_grad=True)
    measure = area if integrated else 1.0
    traction = wp.array([[measure * (2.0 - opening), 0.0, 0.0]], dtype=wp.vec3d, requires_grad=True)
    remainders = (gap if remainder_storage else np.zeros_like(gap), np.zeros((1, 3, 3)), np.zeros((1, 3)))
    with wp.Tape() as tape:
        _, _, residual = _evaluate(
            sources,
            displacement,
            traction,
            compliance,
            integrated_force=integrated,
            remainders=remainders,
        )
    np.testing.assert_allclose(residual.numpy(), 0.0, atol=1e-27)
    tape.backward(grads={residual: wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3d)})
    stiffness = measure
    expected = -compliance * stiffness / (1.0 - compliance * stiffness)
    np.testing.assert_allclose(sources[0].grad.numpy()[0, 0], expected, rtol=2e-13)
