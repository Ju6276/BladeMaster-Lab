"""Single-field baseline of paper Fig. 4: grid transfers ignore cut-side codes."""

import numpy as np
from blademaster_warp.solver import Config, Solver


def pushed_halves(single_field, steps=40):
    s = Solver(Config(size=(0.06, 0.03, 0.03), grid_spacing=0.0075, dt=0.0005))
    left = s.reference[:, 0] < 0
    s.codes.assign(np.where(left, 1, 2).astype(np.int64))
    v = np.zeros((s.n, 3))
    v[left, 0] = -0.2
    s.v.assign(v)
    s.set_single_field(single_field)
    for _ in range(steps):
        s.step(target_edge=(0.0, 0.0, 0.2), cutting=False)
    state = s.state()
    return state["velocity"][~left, 0].mean(), state["velocity"][left, 0].mean(), state["sides"]


def test_cut_halves_move_independently():
    right, left, sides = pushed_halves(False)
    assert abs(right) < 0.005
    assert left < -0.1
    assert set(np.unique(sides)) == {1, 2}


def test_single_field_drags_the_other_half():
    right, _, sides = pushed_halves(True)
    assert right < -0.05
    assert set(np.unique(sides)) == {1, 2}


def test_switching_back_restores_separation():
    s = Solver(Config(size=(0.06, 0.03, 0.03), grid_spacing=0.0075, dt=0.0005, cache_topology=True))
    left = s.reference[:, 0] < 0
    s.codes.assign(np.where(left, 1, 2).astype(np.int64))
    s.set_single_field(True)
    s.advance(np.tile([0.0, 0.0, 0.2], (4, 1)), cutting=False)
    assert s.single_field
    s.set_single_field(False)
    v = np.zeros((s.n, 3))
    v[left, 0] = -0.2
    s.v.assign(v)
    s.advance(np.tile([0.0, 0.0, 0.2], (40, 1)), cutting=False)
    assert abs(s.v.numpy()[~left, 0].mean()) < 0.005
