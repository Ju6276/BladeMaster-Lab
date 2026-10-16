from dataclasses import replace
import numpy as np
from blademaster_warp.solver import Solver, Config


def lowest_vertex(s):
    from scipy.spatial.transform import Rotation

    h, bevel = np.array(s.config.blade_half_size), s.config.blade_bevel_height
    vertices = np.array(
        [[0, -h[1], -h[2]], [0, h[1], -h[2]], [h[0], -h[1], -h[2] + bevel], [-h[0], h[1], -h[2] + bevel]]
    )
    pos, rot = s.tool_pos.numpy()[0], s.tool_rot.numpy()[0]
    return (Rotation.from_quat(rot).apply(vertices) + pos)[:, 2].min()


def drive(s, edge_target, rotation, steps=3000):
    lowest = []
    for i in range(steps):
        s._drive_tool(edge_target, rotation, None, None, 0)
        if i % 20 == 0:
            lowest.append(lowest_vertex(s))
    assert np.isfinite(s.tool_vel.numpy()).all()
    return np.array(lowest)


def run_tool(free=False):
    cfg = replace(Config(), dt=0.00025, pose_servo=not free, free_blade=free, blade_bevel_height=0.005)
    s = Solver(cfg)
    s.tool_pos.assign([[0.12, 0.0, 0.05]])
    angle = 0.2
    q = np.array([0, np.sin(angle / 2), 0, np.cos(angle / 2)])
    s.tool_rot.assign([q])
    states = drive(s, [0.12, 0.0, -0.01], q)
    assert min(states) > -2e-5
    assert max(states) - min(states) > 0.01
    return s


def test_six_dof_servo_contacts_plane_without_target_clip():
    s = run_tool()
    assert s.tool_pos.numpy()[0, 2] > 0.018
    assert s.tool_table_impulse.numpy()[0, 2] > 0


def test_free_tool_falls_and_contacts_plane():
    run_tool(free=True)


def test_servo_graph_accepts_rotations_and_material_reaction():
    cfg = replace(Config(), pose_servo=True, dt=0.00025)
    eager, graph = Solver(cfg), Solver(cfg)
    for s in (eager, graph):
        s.tool_pos.assign([[0.0, 0.0, 0.045]])
    targets = np.tile([0.0, 0.0, 0.02], (40, 1))
    q = np.tile([0.0, np.sin(0.05), 0.0, np.cos(0.05)], (40, 1))
    eager.advance(targets, orientations=q, graph=False)
    graph.advance(targets, orientations=q, graph=True)
    a, b = eager.state(), graph.state()
    np.testing.assert_allclose(a["blade_orientation"], b["blade_orientation"], atol=3e-6)
    np.testing.assert_allclose(a["position"], b["position"], atol=3e-7)
    assert abs(b["blade_orientation"][0, 1]) > 0.001


def test_guided_knife_stops_on_the_shared_board_contact_without_rotating():
    s = Solver(replace(Config(), dt=0.00025, blade_bevel_height=0.005))
    s.tool_pos.assign([[0.0, 0.0, 0.05]])
    states = drive(s, [0.0, 0.0, -0.01], [0.0, 0.0, 0.0, 1.0])
    assert min(states) > -2e-5 and states[-1] < 1e-4
    np.testing.assert_allclose(s.tool_rot.numpy()[0], [0, 0, 0, 1], atol=1e-7)
    assert s.tool_table_impulse.numpy()[0, 2] > 0


def test_board_contact_ends_at_the_board_edge():
    cfg = replace(Config(), dt=0.00025, free_blade=True, board_half_extent=(0.05, 0.05))
    on, off = Solver(cfg), Solver(cfg)
    on.tool_pos.assign([[0.0, 0.0, 0.05]])
    off.tool_pos.assign([[0.2, 0.0, 0.05]])
    assert drive(on, [0, 0, 0], [0, 0, 0, 1], 1500).min() > -2e-5
    assert drive(off, [0, 0, 0], [0, 0, 0, 1], 1500).min() < -0.05
