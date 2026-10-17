"""Spectral shape reward (TopoCut Eqs. 2, 16, 17)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from blademaster_warp import spectral as sp


def box(a, b, c, spacing=0.001, seed=0):
    axes = [np.arange(spacing / 2, s, spacing) for s in (a, b, c)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return grid + np.random.default_rng(seed).uniform(-spacing / 4, spacing / 4, grid.shape)


STICK = box(0.005, 0.005, 0.032)


def describe(points, sigma, seed=0):
    return sp.descriptor(sp.sample(points, seed=seed), sigma)


def test_rigid_motion_does_not_change_the_descriptor():
    sigma = sp.default_sigma(sp.sample(STICK))
    moved = Rotation.from_euler("xyz", [30, 50, 70], degrees=True).apply(STICK) + [0.1, -0.2, 0.3]
    assert sp.distance(describe(STICK, sigma), describe(moved, sigma)) < 1.0e-6


def test_thickness_change_is_far_above_resampling_noise():
    sigma = sp.default_sigma(sp.sample(STICK))
    goal = describe(STICK, sigma)
    noise = sp.distance(describe(box(0.005, 0.005, 0.032, seed=2), sigma, seed=1), goal)
    thicker = sp.distance(describe(box(0.006, 0.006, 0.032), sigma), goal)
    assert thicker > 10.0 * noise
    assert sp.reward(noise) > 0.8 > 0.0 > sp.reward(thicker)


def test_reward_is_continuous_and_decreasing():
    tau = 10.0
    assert sp.reward(tau - 1.0e-9) == pytest.approx(sp.reward(tau + 1.0e-9))
    values = [sp.reward(d) for d in np.linspace(0.0, 3 * tau, 31)]
    assert np.all(np.diff(values) < 0.0)


def test_small_clouds_are_upsampled_inside_the_jitter():
    cloud = box(0.003, 0.003, 0.003)
    picked = sp.sample(cloud, jitter=0.0005)
    assert picked.shape == (sp.NUM_POINTS, 3)
    assert picked.min() >= cloud.min() - 0.0005 and picked.max() <= cloud.max() + 0.0005


def test_fragments_are_paired_with_their_goal_pieces():
    cube = box(0.012, 0.012, 0.012)
    slab = box(0.03, 0.03, 0.004)
    moved_slab = Rotation.from_euler("z", 40, degrees=True).apply(box(0.03, 0.03, 0.004, seed=3))
    total, d, pairs = sp.match([moved_slab, cube + 0.05], [cube, slab])
    assert sorted(pairs) == [(0, 1), (1, 0)]
    assert total > 1.5
