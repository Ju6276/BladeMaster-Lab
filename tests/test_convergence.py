import numpy as np
import pytest
from blademaster_warp.convergence import impulse_bins


def test_impulses_are_conserved_across_incommensurate_bins():
    times = [.003, .007, .011, .02]
    forces = np.array([[2, -3, 5], [7, 2, -1], [-8, 3, 2], [4, -6, 9.]])
    bins = impulse_bins(times, forces, [0, .005, .01, .015, .02])
    np.testing.assert_allclose(bins.sum(axis=0), (forces*np.diff([0, *times])[:, None]).sum(axis=0))
    np.testing.assert_allclose(bins[0], forces[0]*.003+forces[1]*.002)
    np.testing.assert_allclose(bins[1], forces[1]*.002+forces[2]*.003)


def test_resampling_constant_force_does_not_create_rate_difference():
    force = np.array([1., -2., 4.])
    edges = np.linspace(0, 1., 101)
    for n in [100, 250, 1000]:
        bins = impulse_bins(np.linspace(1/n, 1, n), np.tile(force, (n, 1)), edges)
        np.testing.assert_allclose(bins, np.tile(force*.01, (100, 1)), atol=1.e-12)


def test_no_extrapolated_force_or_nonfinite_trace():
    with pytest.raises(ValueError):
        impulse_bins([.1], [[1., 2., 3.]], [0, .2])
    with pytest.raises(ValueError):
        impulse_bins([.1], [[1., np.nan, 3.]], [0, .1])
