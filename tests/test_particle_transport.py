"""TLMPM paper Eqs. 11--16, independently of the DexLab comparison oracle."""
import numpy as np
import pytest
import warp as wp
from blademaster_warp.material import g2p


@pytest.mark.parametrize('flip', [0., .95, 1.])
def test_grid_velocity_advects_positions_even_with_particle_residual(flip):
    # Two particles retain opposite FLIP residuals. The updated grid is at
    # rest. Eq. 13 must leave positions unchanged, although Eq. 16 retains
    # particle momentum. Moving with the residual is the old failure mode.
    device = 'cuda:0'
    initial = np.array([[0., 0., 0.], [.003, 0., 0.]], np.float32)
    velocity = np.array([[10., 0., 0.], [-10., 0., 0.]], np.float32)
    x = wp.array(initial, dtype=wp.vec3, device=device)
    v = wp.array(velocity, dtype=wp.vec3, device=device)
    slots = wp.array(np.tile(np.arange(8), (2,1)), dtype=int, device=device)
    weights = wp.full((2,8), 1/8, dtype=float, device=device)
    grid = wp.zeros(8, dtype=wp.vec3, device=device)
    wp.launch(g2p, 2, [x,v,slots,weights,grid,grid,.001,flip], device=device)
    np.testing.assert_array_equal(x.numpy(), initial)
    np.testing.assert_allclose(v.numpy(), flip*velocity, atol=1e-6)


@pytest.mark.parametrize('flip', [0., .95, 1.])
def test_uniform_grid_acceleration_and_particle_momentum_are_distinct(flip):
    device = 'cuda:0'
    x = wp.zeros(1, dtype=wp.vec3, device=device)
    v = wp.array([[7., 2., -1.]], dtype=wp.vec3, device=device)
    slots = wp.array(np.arange(8).reshape(1,8), dtype=int, device=device)
    weights = wp.full((1,8), 1/8, dtype=float, device=device)
    old = wp.array(np.tile([1.,2.,3.], (8,1)), dtype=wp.vec3, device=device)
    new = wp.array(np.tile([2.,4.,6.], (8,1)), dtype=wp.vec3, device=device)
    wp.launch(g2p, 1, [x,v,slots,weights,old,new,.001,flip], device=device)
    np.testing.assert_allclose(x.numpy(), [[.002,.004,.006]], rtol=1e-6)
    np.testing.assert_allclose(v.numpy(), [[2.+6*flip,4.,6.-4*flip]], rtol=1e-6)
