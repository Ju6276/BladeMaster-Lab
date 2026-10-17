"""Several free knives (paper Fig. 6)."""

import numpy as np
import pytest
from blademaster_warp.solver import Config, Solver


def two_knives(**kwargs):
    cfg = Config(
        size=(0.06, 0.03, 0.03),
        grid_spacing=0.0075,
        dt=0.0005,
        free_blade=True,
        tool_count=2,
        blade_half_size=(0.0005, 0.025, 0.02),
        **kwargs,
    )
    return Solver(cfg)


def test_only_free_knives_may_be_several():
    for kwargs in ({}, {"pose_servo": True}, {"external_tool": True}):
        with pytest.raises(ValueError):
            Config(tool_count=2, **kwargs)
    with pytest.raises(ValueError):
        Config(tool_count=0)


def dropped(xs, steps):
    s = two_knives()
    z = 0.03 + 0.02 + 0.002
    s.tool_pos.assign([[x, 0.0, z] for x in xs])
    s.tool_vel.assign([[0.0, 0.0, -0.6]] * 2)
    for _ in range(steps):
        s.step(cutting=True)
    return s, s.state()


def test_knife_index_does_not_matter():
    _, a = dropped([-0.015, 0.015], 100)
    _, b = dropped([0.015, -0.015], 100)
    # Atomic accumulation order alone gives ~1e-7 m between identical runs.
    np.testing.assert_allclose(a["blade_position"], b["blade_position"][::-1], atol=1.0e-7)
    np.testing.assert_allclose(a["position"], b["position"], atol=1.0e-6)


def test_mirrored_knives_move_alike_and_label_their_own_digits():
    # 60 steps keeps the edges above the board, whose sequential vertex solve is
    # not mirror symmetric.
    steps = 60
    s, state = dropped([-0.015, 0.015], steps)
    p, v = state["blade_position"], state["blade_velocity"]
    np.testing.assert_allclose(p[0] * [-1, 1, 1], p[1], atol=1.0e-7)
    np.testing.assert_allclose(v[0] * [-1, 1, 1], v[1], atol=1.0e-5)
    # Both knives entered the block and were slowed by it.
    assert 0.0 < p[0, 2] - 0.02 < 0.028
    assert v[0, 2] > -0.6 - 9.81 * steps * 0.0005
    codes = state["sides"]
    first, second = codes % 3, (codes // 3) % 3
    assert set(np.unique(first)) == {0, 1, 2} and set(np.unique(second)) == {0, 1, 2}
    assert np.all(codes < 9)
    # Each digit splits the block at its own knife.
    ref = s.reference
    assert ref[first == 1, 0].max() < -0.015 < ref[first == 2, 0].min()
    assert ref[second == 1, 0].max() < 0.015 < ref[second == 2, 0].min()


def test_knife_beside_the_block_falls_freely():
    s = two_knives()
    z0 = 0.2
    s.tool_pos.assign([[-0.015, 0.0, 0.03 + 0.02 + 0.002], [0.2, 0.0, z0]])
    n = 100
    for _ in range(n):
        s.step(cutting=True)
    state = s.state()
    dt, g = 0.0005, 9.81
    assert state["blade_velocity"][1, 2] == pytest.approx(-g * n * dt, rel=1.0e-5)
    assert state["blade_position"][1, 2] == pytest.approx(z0 - g * dt * dt * n * (n + 1) / 2, abs=1.0e-6)
    np.testing.assert_allclose(state["blade_position"][1, :2], [0.2, 0.0], atol=1.0e-9)


@pytest.mark.parametrize("first", [0, 1])
def test_each_knife_pays_its_own_cutting_work(first):
    s = two_knives(cutting_energy=1.0)
    x = s.reference.copy()
    x[:, 0] += 10
    x[0] = [0, 0, 0.039]
    s.x.assign(x)
    positions = np.array([[3.0, 0, 0.06], [3.0, 0, 0.06]])
    positions[first, 0] = 0
    s.tool_pos.assign(positions)
    s.tool_vel.assign([[0, 0, -1], [0, 0, -1]])
    s.rigid_contacts(True)
    initial_impulse = s.reaction.numpy()[first].copy()
    assert s.tool_damage[first].numpy()[0] == 1
    assert s.tool_damage[1 - first].numpy()[0] == 0
    positions[:, 0] = positions[::-1, 0]
    s.tool_pos.assign(positions)
    s.reaction.zero_()
    s.torque.zero_()
    s.rigid_contacts(True)
    np.testing.assert_allclose(s.reaction.numpy()[1 - first], initial_impulse, rtol=1e-6)
    assert s.tool_damage[1 - first].numpy()[0] == 1
    s._set_cut(2)
    assert all(not damage.numpy().any() for damage in s.tool_damage)
