"""Loading tangents remain finite immediately beyond the strength threshold."""

import numpy as np
import pytest
import warp as wp

from test_cohesive_measure import inputs, evaluate
from test_cohesive_proximal import response, evaluate as evaluate_density, array


@pytest.mark.parametrize("integrated", [False, True])
@pytest.mark.parametrize("fraction", [0.17, 1e-12])
@pytest.mark.parametrize("bits", [20, 35, 45])
def test_new_opening_force_and_storage_match_analytic_radial_tangents(integrated, fraction, bits):
    strength, gc, area = 5e4, 70.0, 1.3061e-8
    failure = 2 * gc / strength
    capacity = strength * (area if integrated else 1.0)
    compliance = fraction * failure / capacity
    threshold = compliance * capacity
    coordinate = threshold * (1 + 2.0**-bits)
    parameters = (strength, gc, compliance)
    if integrated:
        source = inputs([[coordinate, 0, 0]], 0, area, parameters)
        forward = evaluate
    else:
        source, _ = response([[coordinate, 0, 0]], 0, parameters)
        forward = evaluate_density
    with wp.Tape() as tape:
        result = forward(source)
    opened = (coordinate - threshold) / (1 - fraction)
    assert opened > 0
    np.testing.assert_allclose(result[0].numpy()[0, 0], opened, rtol=2e-14)
    np.testing.assert_allclose(result[1].numpy()[0, 0], capacity * (1 - opened / failure), rtol=2e-14)
    tape.backward(grads={result[1]: array([[1.0, 0, 0]], True)})
    expected = -capacity / failure / (1 - fraction)
    np.testing.assert_allclose(source[0].grad.numpy()[0], [expected, 0, 0], rtol=2e-13, atol=1e-14)
    tape.zero()
    tape.backward(grads={result[3]: array([1.0])})
    expected = 0.5 * strength * (1 - 2 * opened / failure) / (1 - fraction)
    np.testing.assert_allclose(source[0].grad.numpy()[0], [expected, 0, 0], rtol=2e-13, atol=1e-14)
