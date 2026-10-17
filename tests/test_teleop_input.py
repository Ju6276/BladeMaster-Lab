import numpy as np
import pytest
from blademaster_warp.teleop_input import Command, ContactStrokes, LatestInput, PoseTarget, hid_twist


def test_sequence_timeout_and_events_are_not_replayed():
    data = LatestInput(timeout=0.25)
    c = Command(1, (0.02, 0, 0), (0, 0, 0.1), "begin_cut")
    data.feed(c.packet(), now=0.0)
    cmd, events, stale = data.sample(now=0.1)
    assert cmd == c and events == [(1, "begin_cut")] and not stale
    cmd, events, stale = data.sample(now=0.3)
    assert stale and not any(cmd.velocity) and not events
    with pytest.raises(ValueError):
        data.feed(c.packet(), now=0.4)


def test_world_axes_and_integrated_pose_do_not_depend_on_view():
    p = PoseTarget((1, 2, 3))
    cmd = Command(0, (0.1, 0, 0), (0, 0, np.pi))
    x, q = p.integrate(cmd, [(0, "begin_cut")], 0.01, 100)
    np.testing.assert_allclose(x[-1], [1.1, 2, 3])
    np.testing.assert_allclose(q[-1], [0, 0, 1, 0], atol=1e-12)
    assert p.cut_index == 0 and p.cutting
    p.integrate(Command(1, (0, 0, 0), (0, 0, 0)), [(1, "end_cut")], 0.01, 1)
    assert p.cut_index == 0 and not p.cutting
    p.integrate(cmd, [(2, "begin_cut")], 0.01, 1)
    assert p.cut_index == 1


def test_hid_reuses_existing_fixed_base_axes():
    # Raw forward push is -Y, upward lift -Z on the previously used HID device.
    np.testing.assert_allclose(hid_twist([0, -1, 0, 0, 0, 0])[0], [0.06, 0, 0])
    np.testing.assert_allclose(hid_twist([0, 0, -1, 0, 0, 0])[0], [0, 0, 0.06])
    np.testing.assert_allclose(hid_twist([1, 0, 0, 0, 0, 0])[0], [0, -0.06, 0])
    np.testing.assert_allclose(hid_twist([0, 0, 0, 1, 0, 0])[1], [0, -0.5, 0])


def test_nonfinite_and_wrong_frame_rejected_not_clipped():
    p = Command(0, (1, 2, 3), (0, 0, 0)).packet()
    p["frame"] = "camera"
    with pytest.raises(ValueError):
        Command.parse(p)
    p["frame"] = "world"
    p["linear_velocity_m_s"] = [float("nan"), 0, 0]
    with pytest.raises(ValueError):
        Command.parse(p)


def test_contact_strokes_split_on_clear_gaps_and_advance_before_the_next_touch():
    strokes, dt = ContactStrokes(clear_s=0.05), 0.001
    pattern = [0] * 20 + [1] * 100 + [0] * 30 + [1] * 50 + [0] * 80 + [1] * 40 + [0] * 60
    indices = [strokes.update(bool(c), dt) for c in pattern]
    # A 30 ms gap stays inside the first stroke; the 80 ms gap ends the stroke.
    assert strokes.strokes == 2
    assert indices[199] == 0 and indices[-1] == 2
    first_touch_of_second = 20 + 100 + 30 + 50 + 80
    assert indices[first_touch_of_second - 1] == 1


def test_pose_target_limits_tilt_workspace_and_floor():
    from blademaster_warp.teleop_input import quaternion_rotate

    edge = [(0.0, -0.04, -0.02), (0.0, 0.04, -0.02)]
    target = PoseTarget(
        (0.0, 0.0, 0.03),
        (0.0, 0.0, 0.0, 1.0),
        workspace=((-0.1, -0.1, 0.0), (0.1, 0.1, 0.3)),
        edge=edge,
        floor_z=-0.002,
        max_tilt=np.radians(45.0),
    )
    # Roll 2 rad about x and push down and sideways for 1 s.
    command = Command(0, (0.5, 0.0, -0.5), (2.0, 0.0, 0.0))
    for _ in range(100):
        positions, rotations = target.integrate(command, [], 0.01, 1)
    axis = quaternion_rotate(target.orientation, np.array([0.0, 0.0, 1.0]))
    assert np.degrees(np.arccos(axis[2])) == pytest.approx(45.0, abs=1e-6)
    assert target.position[0] == pytest.approx(0.1)
    lowest = min(quaternion_rotate(target.orientation, np.asarray(e))[2] for e in edge)
    assert target.position[2] + lowest == pytest.approx(-0.002)
    # Yaw is not limited.
    yaw = PoseTarget((0.0, 0.0, 0.1), max_tilt=np.radians(45.0))
    yaw.integrate(Command(0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), [], 1.0, 1)
    assert yaw.orientation[2] == pytest.approx(np.sin(0.5))


def test_yaw_only_keeps_the_blade_upright_and_operator_frame_turns_input():
    from blademaster_warp.franka import yaw_matrix
    from blademaster_warp.teleop_input import quaternion_rotate

    target = PoseTarget((0.0, 0.0, 0.1), input_rotation=yaw_matrix(np.pi), yaw_only=True)
    target.integrate(Command(0, (0.01, 0.0, 0.0), (1.0, 1.0, 0.5)), [], 1.0, 1)
    axis = quaternion_rotate(target.orientation, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(axis, [0.0, 0.0, 1.0], atol=1e-12)
    # Operator facing the arm: their forward is world -x for an arm reaching along +x.
    np.testing.assert_allclose(target.position, [-0.01, 0.0, 0.1], atol=1e-12)
    assert abs(target.orientation[2]) == pytest.approx(np.sin(0.25))


def test_idle_input_keeps_the_limited_pose():
    target = PoseTarget((0.0, 0.0, 0.1), workspace=((-0.1, -0.1, 0.0), (0.1, 0.1, 0.05)))
    first, _ = target.integrate(Command(0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), [], 0.001, 1)
    np.testing.assert_allclose(first[-1], [0.0, 0.0, 0.05])
    again, rotations = target.integrate(Command(1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), [], 0.001, 1)
    np.testing.assert_allclose(again[-1], [0.0, 0.0, 0.05])
    np.testing.assert_allclose(rotations[-1], target.orientation)


def test_session_reset_discards_motion_and_events_but_keeps_sequence():
    data = LatestInput()
    data.feed(Command(10, (0.1, 0, 0), (0, 0, 1), "begin_cut").packet(), now=1.0)
    data.reset_session()
    command, events, stale = data.sample(now=1.0)
    assert stale and command.seq == 10 and not events
    assert not any(command.velocity) and not any(command.omega)
    with pytest.raises(ValueError, match="sequence"):
        data.feed(Command(10, (0, 0, 0), (0, 0, 0)).packet())
    data.feed(Command(11, (0, 0, 0), (0, 0, 0), "begin_cut").packet(), now=2.0)
    assert data.sample(now=2.0)[1:] == ([(11, "begin_cut")], False)


@pytest.mark.parametrize("newer_motion", [False, True])
def test_hold_stops_its_own_command_but_not_a_newer_motion(newer_motion):
    data = LatestInput()
    data.feed(Command(0, (0.1, 0, 0), (0, 0, 1), "hold").packet(), now=0.0)
    if newer_motion:
        data.feed(Command(1, (0.2, 0, 0), (0, 0, 2)).packet(), now=0.01)
    command, events, _ = data.sample(now=0.01)
    target = PoseTarget((0, 0, 0))
    positions, rotations = target.integrate(command, events, 0.01, 1)
    angle = 0.02 if newer_motion else 0.0
    np.testing.assert_allclose(positions[-1], [0.002 if newer_motion else 0.0, 0, 0])
    np.testing.assert_allclose(rotations[-1], [0, 0, np.sin(angle / 2), np.cos(angle / 2)])


def test_hold_remains_stationary_until_a_new_command_arrives():
    data = LatestInput()
    data.feed(Command(0, (0.1, 0, 0), (0, 0, 1), "hold").packet(), now=0.0)
    target = PoseTarget((0, 0, 0))
    for now in (0.0, 0.01):
        command, events, _ = data.sample(now=now)
        positions, rotations = target.integrate(command, events, 0.01, 1)
        np.testing.assert_array_equal(positions[-1], [0, 0, 0])
        np.testing.assert_array_equal(rotations[-1], [0, 0, 0, 1])
    data.feed(Command(1, (0.1, 0, 0), (0, 0, 1)).packet(), now=0.02)
    command, events, _ = data.sample(now=0.02)
    positions, rotations = target.integrate(command, events, 0.01, 1)
    np.testing.assert_allclose(positions[-1], [0.001, 0, 0])
    np.testing.assert_allclose(rotations[-1], [0, 0, np.sin(0.005), np.cos(0.005)])
