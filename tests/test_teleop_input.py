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


@pytest.mark.parametrize("seed", [0, 17, 913])
def test_seeded_six_axis_replay_lifecycle_and_reconnect(seed):
    import json
    from scripts.stress_teleop import replay, seeded_replay

    rows = seeded_replay(seed)
    trace, report = replay(rows)
    assert replay(json.loads(json.dumps(rows))) == (trace, report)
    assert report["sessions"] == 2 and report["cuts_per_session"] == [3, 1]
    assert report["rejected_packets"] == 1 and report["stale_frames"] > 0
    assert all(not r["active"] for r in trace[150:166] + trace[240:249])
    assert trace[249]["cut_index"] == -1 and not trace[249]["cutting"]
    assert trace[250]["cut_index"] == 0 and trace[250]["cutting"]
    assert trace[205]["seq"] == trace[206]["seq"] < trace[207]["seq"]
    assert all(r["stale"] for r in trace[196:205])
    np.testing.assert_array_equal(trace[196]["position"], trace[204]["position"])
    np.testing.assert_array_equal(trace[196]["orientation"], trace[204]["orientation"])
    np.testing.assert_array_equal(trace[150]["position"], trace[165]["position"])
    velocity = np.array([r["packet"]["linear_velocity_m_s"] for r in rows if "packet" in r])
    omega = np.array([r["packet"]["angular_velocity_rad_s"] for r in rows if "packet" in r])
    assert np.all(np.max(abs(velocity), axis=0) > 0)
    assert np.all(np.max(abs(omega), axis=0) > 0)
    assert np.min(velocity[:, 2]) <= -0.15 and np.max(velocity[:, 2]) >= 0.15
    assert trace[23]["position"][2] < 0.01 and trace[64]["position"][2] > 0.04


def test_mixed_axis_rotation_composes_in_world_frame():
    from blademaster_warp.teleop_input import quaternion_rotate

    target = PoseTarget((0, 0, 0), orientation=(0, 0, np.sin(0.3), np.cos(0.3)))
    omega = np.array([0.3, -0.4, 0.2])
    target.integrate(Command(0, (0.1, -0.2, 0.3), tuple(omega)), [], 0.01, 100)
    # Rodrigues' matrix provides an independent oracle for the six-axis integrator.
    angle = np.linalg.norm(omega)
    x, y, z = omega / angle
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    increment = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
    initial = np.array([[np.cos(0.6), -np.sin(0.6), 0], [np.sin(0.6), np.cos(0.6), 0], [0, 0, 1]])
    actual = np.column_stack([quaternion_rotate(target.orientation, axis) for axis in np.eye(3)])
    np.testing.assert_allclose(actual, increment @ initial, atol=1e-12)
    np.testing.assert_allclose(target.position, [0.1, -0.2, 0.3], atol=1e-12)


def test_reconnected_http_stream_uses_watermark_and_rejects_replayed_events():
    import gzip
    import http.client
    import json
    from threading import Thread
    from blademaster_warp.live import LiveState, make_server

    live = LiveState()
    live.teleop = LatestInput()
    live._set(dict(state="ready"))
    server = make_server(live, 0)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()

    def request(method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            data = gzip.decompress(response.read()).decode()
            return response.status, json.loads(data) if response.status == 200 else data
        finally:
            conn.close()

    def send(command):
        packet = (json.dumps(command.packet()) + "\n").encode()
        # JSON split across chunks must still yield exactly one command.
        chunks = [packet[:13], packet[13:]]
        body = b"".join(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n" for chunk in chunks)
        return request("POST", "/input", body + b"0\r\n\r\n", {"Transfer-Encoding": "chunked"})

    try:
        first = Command(42, (0.1, 0, 0), (0, 0.2, 0), "begin_cut")
        assert send(first)[0] == 200
        assert live.teleop.sample()[1] == [(42, "begin_cut")]
        status, snapshot = request("GET", "/state")
        assert status == 200 and snapshot["last_seq"] == 42
        assert send(first)[0] == 400
        assert live.teleop.sample()[1] == []
        assert send(Command(snapshot["last_seq"] + 1, (0, 0, 0), (0, 0, 0), "end_cut"))[0] == 200
        assert live.teleop.sample()[1] == [(43, "end_cut")]
        assert live.teleop.latest.seq == 43
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)


def test_replay_can_retain_many_cut_histories_in_one_session():
    from scripts.stress_teleop import replay, seeded_replay

    trace, report = replay(seeded_replay(17, extra_cuts=6))
    assert report["cuts_per_session"] == [3, 7]
    assert trace[-1]["cut_index"] == 6 and not trace[-1]["cutting"]
    # Each withdrawal returns to the same height, so later strokes still enter.
    for cycle in range(6):
        np.testing.assert_allclose(
            trace[330 + 80 * (cycle + 1) - 1]["position"][2], trace[329]["position"][2], atol=1e-12
        )
        assert trace[330 + 80 * cycle + 29]["position"][2] < 0.005
