"""Velocity command packets and pose integration for teleoperation."""

from dataclasses import dataclass
import json
import threading
import time
import numpy as np


def quaternion_product(a, b):
    a_vec, b_vec = np.asarray(a[:3]), np.asarray(b[:3])
    return np.concatenate(
        (a[3] * b_vec + b[3] * a_vec + np.cross(a_vec, b_vec), (a[3] * b[3] - a_vec @ b_vec,))
    )


def rotation_increment(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return np.concatenate((np.asarray(vector) / 2, (1.0,)))
    return np.concatenate((np.asarray(vector) * np.sin(angle / 2) / angle, (np.cos(angle / 2),)))


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

    def reset_session(self):
        """Discard previous motion/events while retaining the stream sequence watermark."""
        with self.lock:
            if self.latest is not None:
                self.latest = Command(self.latest.seq, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            self.received = -np.inf
            self.events.clear()

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
    pending = b""
    while running():
        header = rfile.readline(64)
        if not header:
            break
        size = int(header.split(b";", 1)[0].strip(), 16)
        if size == 0:
            if rfile.readline(3) != b"\r\n":
                raise ValueError("Unexpected HTTP trailers")
            if pending.strip():
                feed(json.loads(pending))
                accepted += 1
            break
        if size < 0:
            raise ValueError("Invalid chunk size")
        # HTTP chunks can split a packet or contain several complete NDJSON lines.
        while size:
            length = min(size, 8192)
            data = rfile.read(length)
            if len(data) != length:
                raise ValueError("Incomplete packet")
            size -= length
            lines = (pending + data).split(b"\n")
            pending = lines.pop()
            for line in lines:
                if len(line) > 8192:
                    raise ValueError("Packet too long")
                if line.strip():
                    feed(json.loads(line))
                    accepted += 1
            if len(pending) > 8192:
                raise ValueError("Packet too long")
        if rfile.read(2) != b"\r\n":
            raise ValueError("Incomplete packet")
    return accepted


def quaternion_rotate(q, v):
    """Rotate v by the unit quaternion q (x, y, z, w)."""
    u, w = np.asarray(q[:3]), q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


class PoseTarget:
    """Integrated tool pose target.

    Operator twists are given in the operator frame (x forward, y left, z up)
    and turned into the world by `input_rotation`; `yaw_only` keeps only the
    rotation about the vertical, so the blade stays upright and its edge level.

    Optional limits keep the operator inside what the arm and the scene allow:
    `workspace` (lo, hi) boxes the tool origin, `edge` (local edge end points)
    with `floor_z` keeps the target edge from sinking below the board, so the
    impedance target cannot wind up under it, and `max_tilt` bounds the angle
    between the tool z axis and its initial direction (yaw stays free).
    """

    def __init__(
        self,
        position=(-0.03, 0.0, 0.036),
        orientation=(0.0, 0.0, 0.0, 1.0),
        workspace=None,
        edge=None,
        floor_z=None,
        max_tilt=None,
        input_rotation=None,
        yaw_only=False,
    ):
        self.position = np.asarray(position, dtype=float).copy()
        self.orientation = np.asarray(orientation, dtype=float).copy()
        self.orientation /= np.linalg.norm(self.orientation)
        self.cut_index = -1
        self.cutting = False
        self.workspace = None if workspace is None else np.asarray(workspace, dtype=float)
        self.edge = None if edge is None else np.asarray(edge, dtype=float)
        self.floor_z = floor_z
        self.max_tilt = max_tilt
        self.axis0 = quaternion_rotate(self.orientation, np.array([0.0, 0.0, 1.0]))
        self.input_rotation = np.eye(3) if input_rotation is None else np.asarray(input_rotation, dtype=float)
        self.yaw_only = yaw_only
        # The current pose has passed `limit`, so an idle input may keep it as is.
        self._limited = False

    def limit(self, position, orientation):
        """Pose clamped to the tilt cone, the workspace box and the floor."""
        q = orientation / np.linalg.norm(orientation)
        if self.max_tilt is not None:
            x, y, z, w = (float(v) for v in q)
            axis = np.array([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)])
            tilt = np.arccos(np.clip(axis @ self.axis0, -1.0, 1.0))
            if tilt > self.max_tilt:
                hinge = np.cross(axis, self.axis0)
                norm = np.linalg.norm(hinge)
                # Antiparallel axes: any horizontal hinge brings the tool back.
                hinge = hinge / norm if norm > 1.0e-9 else np.array([1.0, 0.0, 0.0])
                q = quaternion_product(rotation_increment(hinge * (tilt - self.max_tilt)), q)
                q /= np.linalg.norm(q)
        p = np.asarray(position, dtype=float).copy()
        if self.workspace is not None:
            p = np.clip(p, self.workspace[0], self.workspace[1])
        if self.edge is not None and self.floor_z is not None:
            x, y, z, w = (float(v) for v in q)
            row = (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))
            lowest = min(row[0] * e[0] + row[1] * e[1] + row[2] * e[2] for e in self.edge)
            p[2] = max(p[2], self.floor_z - lowest)
        return p, q

    def integrate(self, command, events, dt, substeps):
        if command.event == "hold":
            command = Command(command.seq, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for seq, event in events:
            if event == "begin_cut":
                self.cut_index += 1
                self.cutting = True
            elif event == "end_cut":
                self.cutting = False
            elif event == "hold" and seq == command.seq:
                command = Command(command.seq, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        if self._limited and not any(command.velocity) and not any(command.omega):
            return np.repeat(self.position[None], substeps, 0), np.repeat(self.orientation[None], substeps, 0)
        indices = np.arange(1, substeps + 1)
        velocity = self.input_rotation @ np.asarray(command.velocity)
        omega = self.input_rotation @ np.asarray(command.omega)
        if self.yaw_only:
            omega = np.array([0.0, 0.0, omega[2]])
        positions = self.position + indices[:, None] * dt * velocity
        rotations = np.array(
            [quaternion_product(rotation_increment(omega * dt * i), self.orientation) for i in indices]
        )
        rotations /= np.linalg.norm(rotations, axis=1)[:, None]
        for i in range(len(positions)):
            positions[i], rotations[i] = self.limit(positions[i], rotations[i])
        self.position = positions[-1].copy()
        self.orientation = rotations[-1].copy()
        self._limited = True
        return positions, rotations


class ContactStrokes:
    """Cut strokes from the blade's history in the material instead of operator events.

    `in_material` must mean the blade is embedded (particles beside its body), not
    that it carries contact force: the kerf behind a sharp blade is often open, so
    the force drops to zero while the blade is still inside. A stroke ends once the
    blade has been out for `clear_s`. The index advances at that moment, not at the
    next touch, because the first interval's side labels use the index already
    passed to the solver.
    """

    def __init__(self, clear_s=0.02):
        self.clear_s = clear_s
        self.cut_index = 0
        self.strokes = 0
        self.touched = False
        self.clear_for = 0.0

    def update(self, in_material, dt):
        if in_material:
            self.strokes += not self.touched
            self.touched = True
            self.clear_for = 0.0
        elif self.touched:
            self.clear_for += dt
            if self.clear_for >= self.clear_s:
                self.cut_index += 1
                self.touched = False
        return self.cut_index


def hid_twist(raw, speed=0.06, angular_speed=0.5, deadzone=0.08):
    """SpaceMouse axis convention for a fixed-base tool.

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
