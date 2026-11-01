from types import SimpleNamespace

import numpy as np
import pytest

from blademaster_warp._compensated import Expansion as E
from blademaster_warp._compensated_krylov import refine
from blademaster_warp._compensated_linalg import residual


@pytest.mark.parametrize('exponent', [-900, 0, 900])
def test_gmres_multiple_rhs_preserves_low_parts_across_scales(exponent):
    matrix = np.array([
        [1., 1., 1., 1., 0.],
        [-1., 1., 1., 1., 1.],
        [0., -1., 1., 1., 1.],
        [0., 0., -1., 1., 1.],
        [0., 0., 0., -1., 1.],
    ])
    expected = E(
        np.ldexp(np.array([[1., -2., 0.], [2., 1., 0.], [-3., 1., 0.],
                          [4., -1., 0.], [-1., 3., 0.]]), exponent),
        np.ldexp(np.array([[1., 2., 0.], [-1., -3., 0.], [2., -1., 0.],
                          [-3., 4., 0.], [1., -2., 0.]]), exponent - 60),
    )
    A = E(matrix)
    rhs = A @ expected
    before = rhs.high.copy(), rhs.low.copy()
    factor = SimpleNamespace(matrix=A, _substitute=lambda value: value)
    answer, iterations = refine(factor, rhs, E(np.zeros(rhs.shape)))
    assert iterations > 1
    error = abs((answer - expected).high)
    bound = 128 * np.finfo(float).eps**2 * np.linalg.cond(matrix, np.inf)
    assert np.all(error[:, :2] < bound * np.max(abs(expected.high[:, :2]), axis=0))
    np.testing.assert_array_equal(answer.high[:, 2], 0.0)
    np.testing.assert_array_equal(answer.low[:, 2], 0.0)
    defect = residual(A, answer, rhs)
    scale = abs(rhs.high) + abs(matrix) @ abs(answer.high)
    assert np.all(abs(defect.high) <= 64 * np.finfo(float).eps**2 * scale + 6 * np.nextafter(0., 1.))
    np.testing.assert_array_equal(rhs.high, before[0])
    np.testing.assert_array_equal(rhs.low, before[1])
