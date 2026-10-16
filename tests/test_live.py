"""Online transport exposes current states and never creates replay files."""

import gzip
import json
from threading import Thread
from urllib.request import urlopen, Request
import numpy as np
from blademaster_warp.live import LiveState, make_server


def test_live_state_protocol_and_pause_resume():
    live = LiveState()
    live._set(dict(state="ready", sim_s=0.0))
    server = make_server(live, 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/state") as response:
            packet = json.loads(gzip.decompress(response.read()))
        assert packet["sim_s"] == 0.0
        with urlopen(base + f"/state?after={packet['sequence']}") as response:
            assert response.status == 204
        urlopen(Request(base + "/command/start", method="POST")).close()
        assert live.play.is_set()
        urlopen(Request(base + "/command/pause", method="POST")).close()
        assert not live.play.is_set()
        urlopen(Request(base + "/command/stop", method="POST")).close()
        assert live.stop_requested and live.play.is_set()
        live.finish(dict(status="completed"))
        urlopen(Request(base + "/command/start", method="POST")).close()
        assert live.restart.is_set() and live.autostart
    finally:
        server.shutdown()
        server.server_close()


def test_live_input_stream_feeds_teleop_source():
    import http.client
    from blademaster_warp.teleop_input import Command, LatestInput

    live = LiveState()
    live.teleop = LatestInput()
    live._set(dict(state="ready", sim_s=0.0))
    server = make_server(live, 0)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/state") as response:
            packet = json.loads(gzip.decompress(response.read()))
        assert packet["ready"] and packet["last_seq"] == -1
        chunks = b""
        for seq, event in ((0, "begin_cut"), (1, "")):
            data = (json.dumps(Command(seq, (0, 0, -0.01), (0, 0, 0), event).packet()) + "\n").encode()
            chunks += f"{len(data):x}\r\n".encode() + data + b"\r\n"
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request("POST", "/input", body=chunks + b"0\r\n\r\n", headers={"Transfer-Encoding": "chunked"})
        assert json.loads(gzip.decompress(conn.getresponse().read()))["accepted_packets"] == 2
        command, events, stale = live.teleop.sample()
        assert command.seq == 1 and events == [(0, "begin_cut")] and not stale
    finally:
        server.shutdown()
        server.server_close()


def test_franka_teleop_follows_operator_twist_and_cut_events():
    from scipy.spatial.transform import Rotation
    from blademaster_warp.franka_run import parser, FrankaCuttingRun

    args = parser().parse_args(["--no-material", "--motion", "teleop", "--duration", "4.0"])
    run = FrankaCuttingRun(args)
    start = run.robot.position.copy()
    start_rotation = run.robot.rotation.copy()
    seen = []
    original = run._exchange_wrench

    def exchange():
        wrench = original()
        seen.append((run.robot.data.time, run.cutting, run.cut_index))
        return wrench

    run._exchange_wrench = exchange
    report = run.execute()
    assert report["status"] == "completed" and report["teleop"]["cut_strokes"] == 2
    # Mock input: begin_cut at 0 s, end_cut at 1.7 s, a twist over 2.3-3.2 s, begin_cut at 3.2 s.
    switches = [
        (round(t, 3), on, index) for (t, on, index), (_, before, _) in zip(seen[1:], seen) if on != before
    ]
    assert seen[0][1:] == (True, 0)
    assert switches == [(1.7, False, 0), (3.2, True, 1)]
    target = run.teleop.position
    assert np.linalg.norm(run.robot.position - target) < 0.005
    assert np.linalg.norm(target[:2] - start[:2] - [0.06 * 0.5, 0.006 * 0.5]) < 1.0e-6

    def angle(a, b):
        return np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))

    # The libfranka example's rotational gains (10 Nm/rad, damping 2 sqrt(k)) settle in about 0.6 s.
    commanded = Rotation.from_quat(run.teleop.orientation).as_matrix()
    assert angle(start_rotation, commanded) > 5.0
    assert angle(run.robot.rotation, commanded) < 0.5 * angle(start_rotation, commanded)


def test_recording_is_opt_in(tmp_path):
    from blademaster_warp.franka_run import parser, run

    for record in (False, True):
        out = tmp_path / str(record)
        args = parser().parse_args(
            ["--no-material", "--duration", ".002", "--output", str(out)] + (["--record"] if record else [])
        )
        report = run(args)
        assert report["recording"] == record
        assert (out / "manifest.json").exists() and (out / "report.json").exists()
        assert (out / "trajectory.npz").exists() == record
        if not record:
            assert {p.name for p in out.iterdir()} == {"manifest.json", "report.json"}


def test_surface_can_run_without_storage(tmp_path, monkeypatch):
    from blademaster_warp.solver import Solver
    from blademaster_warp.surface_runtime import SurfaceRecorder
    from blademaster_warp.coupling import ToolState

    s = Solver()
    monkeypatch.chdir(tmp_path)
    surface = SurfaceRecorder(s, None)
    surface.update(
        0.0,
        s.x.numpy(),
        s.F.numpy(),
        s.codes.numpy(),
        ToolState([0, 0, 0.08], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0]),
        cutting=False,
    )
    assert len(surface.current["vertices"]) > 0
    assert not list(tmp_path.iterdir())


def test_live_preview_does_not_insert_unpaired_recorded_frame(tmp_path):
    from blademaster_warp.franka_run import parser, run

    out = tmp_path / "live_record"
    args = parser().parse_args(
        [
            "--material",
            "paper-banana",
            "--duration",
            ".042",
            "--surface",
            "--surface-embedding",
            "cuda",
            "--surface-ccd",
            "cuda",
            "--record",
            "--output",
            str(out),
        ]
    )
    live = LiveState()
    live.autostart = True
    run(args, observer=live)
    trajectory = np.load(out / "trajectory.npz")
    frames = json.loads((out / "surface/frames.json").read_text())
    assert len(trajectory["qpos"]) == len(trajectory["particles"]) == len(frames) == 3
    np.testing.assert_allclose([f["time_s"] for f in frames], trajectory["control_tcp"][:, 0])
