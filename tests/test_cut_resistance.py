"""Cutting-resistance variants: paper Eq. 5 against holding intact particles at the edge."""

import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation
from blademaster_warp import contact
from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.paper_impact import SIDE_ROTATION, edge_height, penetration
from blademaster_warp.solver import Config, Solver

SIZE = (0.04, 0.02, 0.024)


@wp.kernel
def edge_resistance_probe(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    tool_velocity: wp.array(dtype=wp.vec3),
    rotation: wp.quat,
    damage: wp.array(dtype=float),
    work: wp.array(dtype=float),
    result: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    result[p] = contact.resist_cut(
        p,
        wp.quat_rotate(rotation, wp.vec3(0.002, 0.001, -0.004)),
        x,
        v,
        wp.vec3(0.0, 0.0, 0.02),
        tool_velocity[p],
        rotation,
        wp.vec3(0.0),
        wp.vec3(0.0005, 0.025, 0.02),
        0.0015,
        0.006,
        6.0,
        damage,
        0.001,
        contact.CUT_ON,
        work,
    )


@pytest.mark.parametrize("angle", [0, 35])
def test_completed_cut_resistance_stays_released_during_hold_and_reversal(angle):
    rotation = Rotation.from_euler("y", angle, degrees=True)
    edge = np.array([0.0, 0.0, 0.02]) + rotation.apply([0.0, 0.0, -0.02])
    positions = edge + rotation.apply([[0, 0, -0.001]] * 5 + [[0, 0, 0.001]])
    velocities = rotation.apply([[0, 0, -speed] for speed in [0.01, 0, -0.01, -0.01, 0.01, 0]])
    device = wp.get_preferred_device()
    x = wp.array(positions, dtype=wp.vec3, device=device)
    v = wp.zeros(6, dtype=wp.vec3, device=device)
    tool_velocity = wp.array(velocities, dtype=wp.vec3, device=device)
    damage = wp.array([1, 1, 1, 0.5, 0.5, 1], dtype=float, device=device)
    work = wp.zeros(1, dtype=float, device=device)
    result = wp.zeros(6, dtype=wp.vec3, device=device)
    wp.launch(
        edge_resistance_probe,
        6,
        [x, v, tool_velocity, wp.quat(*rotation.as_quat()), damage, work, result],
        device=device,
    )
    expected = rotation.apply([[0.002, 0.001, 0]] * 3 + [[0.002, 0.001, -0.004]] * 3)
    np.testing.assert_allclose(result.numpy(), expected, atol=2e-9)
    np.testing.assert_allclose(damage.numpy(), [1, 1, 1, 0.5, 0.54, 1], atol=1e-7)
    assert work.numpy()[0] == pytest.approx(0.004 * 0.01, rel=1e-6)


def test_cut_work_is_validated():
    with pytest.raises(ValueError):
        Config(cut_work="absolute")


def steady_force(cutting_energy, **kwargs):
    """Mean knife force while the edge is pushed from 4 to 12 mm depth at 50 mm/s."""
    cfg = Config(
        size=SIZE,
        grid_spacing=0.006,
        dt=0.0001,
        young_modulus=400000.0,
        poisson_ratio=0.4,
        yield_stress=40000.0,
        cutting_energy=cutting_energy,
        blade_half_size=(0.0005, 0.015, 0.02),
        external_tool=True,
        grid_storage="compact",
        **kwargs,
    )
    s = Solver(cfg)
    port = ExternalToolPort(s)
    hz, speed, interval = cfg.blade_half_size[2], 0.05, 0.002
    edges, forces = [], []
    for i in range(int(0.016 / speed / interval)):
        edge = SIZE[2] + 0.002 - speed * i * interval
        state = ToolState([0.0, 0.0, edge + hz], [0, 0, 0, 1], [0, 0, -speed], [0, 0, 0])
        forces.append(port.advance(state, interval, cutting=True, cut_index=0).force_n[2])
        edges.append(edge)
    depth = SIZE[2] - np.array(edges)
    return np.mean(np.array(forces)[(depth > 0.004) & (depth < 0.012)])


def test_paper_resistance_stops_growing_with_cutting_energy():
    # Particles slide around the edge before their damage work is paid.
    low, high = steady_force(100.0), steady_force(1000.0)
    assert high == pytest.approx(low, rel=0.05)


def test_held_particles_make_the_force_grow_with_cutting_energy():
    low, high = steady_force(100.0, cut_hold=True), steady_force(1000.0, cut_hold=True)
    assert high > 3.0 * low > 0.0


def test_edge_height_follows_the_rotated_edge():
    half = (0.0005, 0.04, 0.03)
    upright = Rotation.identity().as_quat()
    flat = Rotation.from_euler("y", 90, degrees=True).as_quat()
    lowest, tilt = edge_height([[[0.0, 0.0, 0.05], [0.0, 0.0, 0.0005]]], [[upright, flat]], half)
    np.testing.assert_allclose(lowest, [[0.05 - 0.03, 0.0005]], atol=1.0e-12)
    np.testing.assert_allclose(tilt, [[0.0, 90.0]], atol=1.0e-9)


def test_side_launch_axes_and_penetration():
    # Cut direction (tool -z) along +y, edge along x, blade plane horizontal.
    np.testing.assert_allclose(SIDE_ROTATION.apply([0.0, 0.0, -1.0]), [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(SIDE_ROTATION.apply([0.0, 1.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-12)
    half = (0.0005, 0.012, 0.03)
    q = SIDE_ROTATION.as_quat()
    depth, angle = penetration([[[0.0, -0.025, 0.02]]], [[q]], half, np.array([0.0, 1.0, 0.0]), -0.025)
    np.testing.assert_allclose(depth, [[0.03]], atol=1e-12)
    np.testing.assert_allclose(angle, [[0.0]], atol=1e-6)
