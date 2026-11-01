"""Transfer energy loss, affine invariance and cancellation-safe sparse actions."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from blademaster_warp._compensated import Expansion
from blademaster_warp._transfer_dissipation import product
from test_reference_affine import setup


@pytest.mark.parametrize("exponent,small", [(80, 2.0**-20), (20, 2.0**-80)])
def test_sparse_action_retains_small_terms_between_large_cancelling_terms(exponent, small):
    matrix = csr_matrix([[2.0**exponent, 1.0, -(2.0**exponent)], [0.0, 3.0, 0.0]])
    value = Expansion(np.ones((3, 2)), np.array([[0, 0], [small, -small], [0, 0]]))
    result = product(matrix, value)
    expected = Expansion([[1.0, 1.0], [3.0, 3.0]], [[small, -small], [3 * small, -3 * small]])
    np.testing.assert_array_equal(result.high, expected.high)
    np.testing.assert_array_equal(result.low, expected.low)


@pytest.mark.parametrize("spacing", [0.006, 0.004])
@pytest.mark.parametrize("shape,seed", [("block", 0), ("block", 1), ("cylinder", 0)])
def test_variance_agrees_with_independent_kinetic_loss_and_retains_affine_fields(spacing, shape, seed):
    model, _, transfer, _ = setup(spacing, shape, seed)
    factor = transfer.dissipation
    rng = np.random.default_rng(2750)
    z = rng.normal(size=(4 * model.n, 3))
    P = transfer.grid_map.numpy()
    grid = P @ z
    independent = np.sum(z * z) - np.sum(transfer.grid_mass[:, None] / model.mass * grid * grid)
    power = factor.power(Expansion(z))
    np.testing.assert_allclose(power.high, independent, rtol=2e-13, atol=2e-14)
    old = z - P.T @ (transfer.grid_mass[:, None] / model.mass * grid)
    np.testing.assert_allclose(factor.action(Expansion(z)).numpy(), old, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(np.sum(z * factor.action(z)), power.high, rtol=2e-13)

    uniform = np.zeros_like(z)
    uniform[::4] = [0.4, -0.1, 0.3]
    for scale in [1e-6, 1e-8]:
        perturbation = Expansion(scale * z)
        small = factor.power(perturbation)
        translated = factor.power(Expansion(uniform) + perturbation)
        assert small.high > 0.0 and translated.high > 0.0
        np.testing.assert_allclose(translated.high, small.high, rtol=2e-5, atol=1e-30)

    A = np.array([[0.3, -50.0, -20.0], [50.0, -0.2, -10.0], [20.0, 10.0, 0.1]])
    affine = np.empty_like(z).reshape(model.n, 4, 3)
    affine[:, 0] = model.reference @ A.T + [0.1, 0.2, 0.3]
    affine[:, 1:] = (A @ transfer.root.numpy()).transpose(0, 2, 1)
    np.testing.assert_allclose(factor.action(Expansion(affine.reshape(-1, 3))).numpy(), 0.0, atol=2e-14)
