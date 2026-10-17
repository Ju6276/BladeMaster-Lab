"""Local-only regressions; mock device, robot and asset. No solver imports."""

import contextlib
import io
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from blademaster_warp import live as lm, spacemouse_client as client, spacemouse
from blademaster_warp.teleop_input import Command, LatestInput, PoseTarget


def robot_stub():
    return SimpleNamespace(
        data=SimpleNamespace(time=0.0, geom_xpos=np.zeros((0, 3)), geom_xmat=np.zeros((0, 9))),
        material_origin=np.zeros(3),
    )


class CaptureInput(LatestInput):

    def __init__(self):
        super().__init__()
        self.captured = []

    def feed(self, packet, now=None):
        super().feed(packet, now)
        self.captured.append(packet)


class FakeMouse:

    def __init__(self):
        neutral = [0.0] * 6
        active = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.snapshots = [
            (neutral, 0, (), 0.0),
            (active, 1, ("close",), 0.0),
            (active, 2, ("open",), 1.0),
            (neutral, 0, (), 0.0),
            (active, 0, (), 0.0),
        ]
        self.i = 0
        self.state = self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def sample(self):
        (axes, buttons, presses, age) = self.snapshots[min(self.i, len(self.snapshots) - 1)]
        self.i += 1
        return spacemouse.MouseSnapshot(np.array(axes), np.full(2, time.monotonic() - age), buttons, presses)


class TeleopRegressions(unittest.TestCase):

    def server(self, live):
        server = lm.make_server(live, 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def test_restart_drops_previous_session_cut_events(self):
        live = lm.LiveState()
        live.teleop = LatestInput()
        live.finish({"status": "completed"})
        live.teleop.feed(
            Command(10, (0.06, 0, 0), (0, 0, 0), "begin_cut").packet(), now=time.monotonic() - 1.0
        )
        live.command("start")
        with patch.object(lm, "scene_asset", return_value={"geoms": []}):
            live.initialize(robot_stub(), None, None)
        command, events, _ = live.teleop.sample()
        target = PoseTarget()
        target.integrate(command, events, 0.001, 1)
        self.assertFalse(
            target.cutting, f"old command {command.seq} replayed events={events}; cutting={target.cutting}"
        )

    def test_right_button_survives_stale_motion_in_both_modes(self):
        for mode in ("session", "cut"):
            with self.subTest(mode=mode):
                live = lm.LiveState()
                live.teleop = CaptureInput()
                live._set({"state": "ready"})
                server = self.server(live)
                calls = []
                with patch.object(spacemouse, "SpaceMouse", return_value=FakeMouse()), patch.object(
                    client, "session_command", side_effect=lambda h, p, n: calls.append(n)
                ), contextlib.redirect_stdout(io.StringIO()):
                    client.run(port=server.server_port, duration=0.1, buttons=mode)
                packets = live.teleop.captured
                if mode == "session":
                    self.assertIn("stop", calls, f"session calls={calls}")
                else:
                    events = [p["event"] for p in packets]
                    self.assertIn("begin_cut", events)
                    self.assertIn("end_cut", events)
                    ended = next((p for p in packets if p["event"] == "end_cut"))
                    self.assertFalse(any(ended["linear_velocity_m_s"]))
                    self.assertFalse(any(ended["angular_velocity_rad_s"]))
                    target = PoseTarget()
                    for packet in packets:
                        command = Command.parse(packet)
                        target.integrate(
                            command, [(command.seq, command.event)] if command.event else [], 0.001, 1
                        )
                    self.assertFalse(
                        target.cutting, f"events={events}; cutting after recenter={target.cutting}"
                    )

    def test_terminal_pause_does_not_prevent_restart(self):
        live = lm.LiveState()
        live.finish({"status": "completed"})
        live.command("pause")
        live.command("start")
        self.assertTrue(
            live.restart.is_set(), f"state={live.snapshot['state']}; restart={live.restart.is_set()}"
        )

    def test_stop_during_restart_initialization_is_honored(self):
        live = lm.LiveState()
        live.finish({"status": "completed"})
        live.command("start")
        entered = threading.Event()
        release = threading.Event()

        def scene_asset(robot):
            entered.set()
            release.wait(2.0)
            return {"geoms": []}

        with patch.object(lm, "scene_asset", side_effect=scene_asset):
            worker = threading.Thread(target=live.initialize, args=(robot_stub(), None, None), daemon=True)
            worker.start()
            self.assertTrue(entered.wait(2.0))
            live.command("stop")
            release.set()
            worker.join(2.0)
        self.assertTrue(live.stop_requested, f"stop={live.stop_requested}; play={live.play.is_set()}")

    def test_replacement_session_client_can_attach_after_stop(self):
        live = lm.LiveState()
        live.teleop = LatestInput()
        live.teleop.feed(Command(42, (0, 0, 0), (0, 0, 0)).packet())
        live.finish({"status": "stopped"})
        server = self.server(live)
        error = None
        ticks = iter([0.0, 121.0, 121.0, 121.0])
        with patch.object(
            client, "time", SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _: None)
        ), patch.object(spacemouse, "SpaceMouse") as mouse:
            try:
                client.run(port=server.server_port, duration=0.0, buttons="session")
            except TimeoutError as exc:
                error = str(exc)
        self.assertIsNone(error, f"error={error}; mouse opened={mouse.called}")
        self.assertTrue(mouse.called)
        self.assertEqual(live.teleop.latest.seq, 43)


class WarmingTransitions(unittest.TestCase):

    def test_controls_during_initialization(self):
        for restarting in (False, True):
            for commands, playing, stopped in (
                (("start",), True, False),
                (("start", "pause"), False, False),
                (("pause", "start"), True, False),
                (("stop",), True, True),
                (("stop", "pause", "start"), True, True),
            ):
                with self.subTest(restarting=restarting, commands=commands):
                    live = lm.LiveState()
                    if restarting:
                        live.finish({"status": "completed"})
                        live.command("start")
                    (entered, release, published) = (threading.Event(), threading.Event(), threading.Event())
                    errors = []

                    def asset(robot):
                        entered.set()
                        if not release.wait(2):
                            raise TimeoutError("test did not release scene setup")
                        return {"geoms": []}

                    original = live.publish

                    def publish(*args, **kwargs):
                        original(*args, **kwargs)
                        published.set()

                    def initialize():
                        try:
                            live.initialize(robot_stub(), None, None)
                        except Exception as error:
                            errors.append(error)

                    with patch.object(lm, "scene_asset", side_effect=asset), patch.object(
                        live, "publish", side_effect=publish
                    ):
                        worker = threading.Thread(target=initialize, daemon=True)
                        worker.start()
                        try:
                            self.assertTrue(entered.wait(2))
                            for command in commands:
                                live.command(command)
                            self.assertEqual(live.snapshot["state"], "loading")
                            release.set()
                            self.assertTrue(published.wait(2))
                            self.assertEqual(live.play.is_set(), playing)
                            self.assertEqual(live.stop_requested, stopped)
                            self.assertEqual(live.snapshot["state"], "running" if playing else "ready")
                        finally:
                            release.set()
                            live.play.set()
                            worker.join(2)
                        self.assertFalse(worker.is_alive())
                        self.assertEqual(errors, [])

    def test_terminal_pause_preserves_restart_policy(self):
        for status in ("completed", "stopped", "failed"):
            for can_restart in (False, True):
                with self.subTest(status=status, can_restart=can_restart):
                    live = lm.LiveState()
                    live.can_restart = can_restart
                    live.finish({"status": status})
                    live.command("pause")
                    self.assertEqual(live.snapshot["state"], status)
                    if can_restart:
                        live.command("start")
                        self.assertTrue(live.restart.is_set())
                    else:
                        with self.assertRaises(ValueError):
                            live.command("start")
                        self.assertFalse(live.restart.is_set())

    def test_restart_boundary_preserves_new_input(self):
        live = lm.LiveState()
        live.teleop = LatestInput()
        live.finish({"status": "stopped"})
        live.teleop.feed(Command(8, (0.1, 0, 0), (0, 0, 0), "begin_cut").packet())
        live.command("start")
        live.teleop.feed(Command(9, (0, 0, 0), (0, 0, 0), "end_cut").packet())
        live.command("start")
        (command, events, stale) = live.teleop.sample()
        self.assertEqual(command.seq, 9)
        self.assertEqual(events, [(9, "end_cut")])
        self.assertFalse(stale)
