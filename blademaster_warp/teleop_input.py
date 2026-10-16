"""Velocity command packets and pose integration for teleoperation."""

from dataclasses import dataclass
import json
import threading
import time
import numpy as np


def quaternion_product(a, b):
    a_vec, b_vec = np.asarray(a[:3]), np.asarray(b[:3])
    return np.r_[a[3] * b_vec + b[3] * a_vec + np.cross(a_vec, b_vec), a[3] * b[3] - a_vec @ b_vec]


def rotation_increment(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return np.r_[np.asarray(vector) / 2, 1.0]
    return np.r_[np.asarray(vector) * np.sin(angle / 2) / angle, np.cos(angle / 2)]


@dataclass(frozen=True)
class Command:
    seq: int
    velocity: tuple
    omega: tuple
    event: str = ""

    @classmethod
    def parse(cls, obj):
        if obj.get("schema") != "blademaster.teleop.v1" or obj.get("frame") != "world":
            raise ValueError("Expected blademaster.teleop.v1, fixed world frame")
        seq = obj.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            raise ValueError("Invalid seq")
        v = np.asarray(obj.get("linear_velocity_m_s"), dtype=float)
        w = np.asarray(obj.get("angular_velocity_rad_s"), dtype=float)
        if v.shape != (3,) or w.shape != (3,) or not np.isfinite(np.r_[v, w]).all():
            raise ValueError("Invalid twist")
        event = obj.get("event", "")
        if event not in ("", "begin_cut", "end_cut", "hold"):
            raise ValueError("Unknown event")
        return cls(seq, tuple(v), tuple(w), event)

    def packet(self):
        return {
            "schema": "blademaster.teleop.v1",
            "frame": "world",
            "seq": self.seq,
            "linear_velocity_m_s": self.velocity,
            "angular_velocity_rad_s": self.omega,
            "event": self.event,
        }


class LatestInput:
    def __init__(self, timeout=0.25):
        self.timeout = timeout
        self.lock = threading.Lock()
        self.latest = None
        self.received = -np.inf
        self.events = []

    def feed(self, packet, now=None):
        cmd = Command.parse(packet)
        with self.lock:
            if self.latest is not None and cmd.seq <= self.latest.seq:
                raise ValueError("Stale or duplicate sequence")
            self.latest = cmd
            self.received = time.monotonic() if now is None else now
            if cmd.event:
                self.events.append((cmd.seq, cmd.event))

    def sample(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            stale = self.latest is None or now - self.received > self.timeout
            events = self.events
            self.events = []
            cmd = self.latest
            if stale:
                cmd = Command(-1 if cmd is None else cmd.seq, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            return cmd, events, stale


def read_chunked_packets(headers, rfile, feed, running=lambda: True):
    """Feed each NDJSON packet of a chunked HTTP request body; return the count."""
    if headers.get("Transfer-Encoding", "").lower() != "chunked":
        raise ValueError("Streaming input requires HTTP chunked transfer")
    accepted = 0
    while running():
        header = rfile.readline(64)
        if not header:
            break
        size = int(header.strip(), 16)
        if size == 0:
            if rfile.readline(3) != b"\r\n":
                raise ValueError("Unexpected HTTP trailers")
            break
        if not 0 < size <= 8192:
            raise ValueError("Packet too long")
        line = rfile.read(size)
        if len(line) != size or rfile.read(2) != b"\r\n":
            raise ValueError("Incomplete packet")
        feed(json.loads(line))
        accepted += 1
    return accepted


class PoseTarget:
    def __init__(self, position=(-0.03, 0.0, 0.036), orientation=(0.0, 0.0, 0.0, 1.0)):
        self.position = np.asarray(position, dtype=float).copy()
        self.orientation = np.asarray(orientation, dtype=float).copy()
        self.orientation /= np.linalg.norm(self.orientation)
        self.cut_index = -1
        self.cutting = False

    def integrate(self, command, events, dt, substeps):
        for _, event in events:
            if event == "begin_cut":
                self.cut_index += 1
                self.cutting = True
            elif event == "end_cut":
                self.cutting = False
            elif event == "hold":
                command = Command(command.seq, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        indices = np.arange(1, substeps + 1)
        positions = self.position + indices[:, None] * dt * np.asarray(command.velocity)
        rotations = np.array(
            [
                quaternion_product(rotation_increment(np.asarray(command.omega) * dt * i), self.orientation)
                for i in indices
            ]
        )
        rotations /= np.linalg.norm(rotations, axis=1)[:, None]
        self.position = positions[-1].copy()
        self.orientation = rotations[-1].copy()
        return positions, rotations


def hid_twist(raw, speed=0.06, angular_speed=0.5, deadzone=0.08):
    """Exactly the existing WowSkin fixed-base HID axis convention.

    Raw signs [1,-1,-1,1,-1,1], then (forward,-right,up). X forward,
    Y left, Z up; rotations are right-handed world-axis angular velocities.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (6,) or not np.isfinite(raw).all() or not 0 <= deadzone < 1:
        raise ValueError("Invalid HID axes or deadzone")
    axes = np.sign(raw) * np.maximum(abs(raw) - deadzone, 0) / (1 - deadzone)
    axes *= np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0])
    v = np.array([axes[1], -axes[0], axes[2]])
    w = np.array([axes[4], -axes[3], axes[5]])
    return v * speed, w * angular_speed


def mock_command(t, seq, previous_phase=-1):
    """A deterministic input fixture, not a pre-cut object or future solver path.

    Exercises all six axes, two explicitly started strokes, hold and release.
    The engine receives only the current velocity/event packet.
    """
    phases = [
        (0.4, (0, 0, 0), (0, 0, 0), "begin_cut"),
        (1.3, (0, 0, -0.04), (0, 0, 0), ""),
        (1.7, (0, 0, 0), (0, 0, 0), ""),
        (2.3, (0, 0, 0.06), (0, 0, 0), "end_cut"),
        (2.8, (0.06, 0.006, 0), (0.06, -0.04, 0.2), ""),
        (3.2, (0, 0, 0), (-0.075, 0.05, 0), ""),
        (4.1, (0, 0, -0.04), (0, 0, 0), "begin_cut"),
        (4.7, (0, 0, 0), (0, 0, 0), ""),
        (5.3, (0, 0, 0.06), (0, 0, 0), "end_cut"),
        (6.0, (0, -0.004, 0), (0, 0, -0.14), ""),
        (float("inf"), (0, 0, 0), (0, 0, 0), "hold"),
    ]
    phase = next(i for i, row in enumerate(phases) if t < row[0])
    _, v, w, event = phases[phase]
    return Command(seq, v, w, event if phase != previous_phase else ""), phase
