"""Physical-time partitions for natural cut, withdrawal and push trajectories."""

import numpy as np
import pytest

from scripts.diff_cut_trajectory import PATH_TIMES, motion
from scripts.diff_natural_validate import time_edges
from blademaster_warp.solver import Config


@pytest.mark.parametrize('dt', [.003, .0005])
@pytest.mark.parametrize('phase', [0., .25, .5])
def test_trajectory_partition_covers_each_motion_knot_once(dt, phase):
    end = 1.05
    edges = time_edges(round(end / dt), dt, phase, 0., 'cut-withdraw-push')
    assert edges[0] == 0. and edges[-1] == end
    assert np.all(np.diff(edges) > 1e-12)
    for knot in PATH_TIMES:
        assert np.count_nonzero(edges == knot) == 1
    cfg = Config()
    controls = [.4, .8, 1.2]
    position = motion(cfg, 0., controls)[0].position.copy()
    for start, stop in zip(edges[:-1], edges[1:]):
        tool = motion(cfg, start, controls)[0]
        position += (stop - start) * tool.linear_velocity
        np.testing.assert_allclose(position, motion(cfg, stop, controls)[0].position, atol=1e-13)


@pytest.mark.parametrize('gravity', [0., 9.81])
@pytest.mark.parametrize('time', [.0004, .001, .004])
def test_analytic_rebound_sensitivity(gravity, time):
    from scripts.diff_contact_convergence import rebound

    gap, speed, stiffness, epsilon = .0001, -.12, 2.5e6, 1e-9
    state, gradient = rebound(gap, speed, time, stiffness, gravity)
    lo = rebound(gap - epsilon, speed, time, stiffness, gravity)[0]
    hi = rebound(gap + epsilon, speed, time, stiffness, gravity)[0]
    np.testing.assert_allclose(gradient, (hi - lo) / (2 * epsilon), rtol=1e-7, atol=1e-7)
    if time == .004 and not gravity:
        np.testing.assert_allclose(state[1], -speed, atol=1e-15)
        np.testing.assert_array_equal(gradient, [-1., 0.])
