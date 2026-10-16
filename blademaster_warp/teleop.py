"""Standalone knife teleoperation over HTTP (pose servo, no robot)."""

import argparse
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
from pathlib import Path
import threading
import time
import zlib
import numpy as np
from .run import preset, write
from .presets import DEFAULT_BANANA_PRESET
from .solver import Solver
from .teleop_input import LatestInput, PoseTarget, mock_command
from .provenance import git_revision


class Server:
    def __init__(self, source, host="127.0.0.1", port=8765):
        self.source = source
        self.lock = threading.Lock()
        self.status = {"ready": False}
        self.running = True
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            def log_message(self, *args):
                pass

            def send_json(self, status, obj):
                data = json.dumps(obj, allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Encoding", "gzip")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    compressor = zlib.compressobj(wbits=31)
                    try:
                        while owner.running:
                            with owner.lock:
                                status = owner.status
                            event = (
                                b"data: "
                                + json.dumps(status, allow_nan=False, separators=(",", ":")).encode()
                                + b"\n\n"
                            )
                            self.wfile.write(compressor.compress(event) + compressor.flush(zlib.Z_SYNC_FLUSH))
                            self.wfile.flush()
                            time.sleep(1 / 30)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    self.close_connection = True
                    return
                if self.path == "/":
                    data = Path(__file__).with_name("viewer.html").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if self.path != "/state":
                    self.send_json(404, {"error": "Use /state"})
                    return
                with owner.lock:
                    status = dict(owner.status)
                with source.lock:
                    status["last_seq"] = -1 if source.latest is None else source.latest.seq
                self.send_json(200, status)

            def do_POST(self):
                if self.path == "/input":
                    self.close_connection = True
                    accepted = 0
                    try:
                        if self.headers.get("Transfer-Encoding", "").lower() != "chunked":
                            raise ValueError("Streaming input requires HTTP chunked transfer")
                        while owner.running:
                            header = self.rfile.readline(64)
                            if not header:
                                break
                            size = int(header.strip(), 16)
                            if size == 0:
                                if self.rfile.readline(3) != b"\r\n":
                                    raise ValueError("Unexpected HTTP trailers")
                                break
                            if not 0 < size <= 8192:
                                raise ValueError("Packet too long")
                            line = self.rfile.read(size)
                            if len(line) != size or self.rfile.read(2) != b"\r\n":
                                raise ValueError("Incomplete packet")
                            source.feed(json.loads(line))
                            accepted += 1
                        self.send_json(200, {"accepted_packets": accepted})
                    except (ValueError, TypeError, KeyError) as exc:
                        self.send_json(400, {"error": str(exc)})
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                if self.path != "/command":
                    self.send_json(404, {"error": "Use /command"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    if not 0 < length <= 8192:
                        raise ValueError("Invalid packet length")
                    source.feed(json.loads(self.rfile.read(length)))
                    self.send_json(200, {"accepted": True})
                except (ValueError, TypeError, KeyError) as exc:
                    self.send_json(400, {"error": str(exc)})

        self.http = ThreadingHTTPServer((host, port), Handler)
        self.http.daemon_threads = True
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.running = False
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()

    def publish(self, status):
        with self.lock:
            self.status = status


def run(output, name=DEFAULT_BANANA_PRESET, duration=8.0, mock=False, port=8765, pace=False, dt=None):
    cfg = replace(preset(name), pose_servo=True)
    if dt is not None:
        cfg = replace(cfg, dt=dt)
    steps_per_frame = round(0.01 / cfg.dt)
    if not np.isclose(steps_per_frame * cfg.dt, 0.01):
        raise ValueError("dt must divide the 100 Hz control interval")
    frames = round(duration / 0.01)
    if frames < 1 or not np.isclose(frames * 0.01, duration):
        raise ValueError("duration must be a positive multiple of .01 s")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    s = Solver(cfg)
    target = PoseTarget()
    s.tool_pos.assign([target.position + [0.0, 0.0, cfg.blade_half_size[2]]])
    source = LatestInput()
    server = Server(source, port=port)
    # Capture fixed-size input graphs without advancing any initial state.
    # Inputs during capture are ignored; packets start when ready=True.
    positions = np.tile(target.position, (steps_per_frame, 1))
    rotations = np.tile(target.orientation, (steps_per_frame, 1))
    s.prepare(positions, False, 0, rotations)
    s.prepare(positions, True, 0, rotations)
    server.start()
    initial = s.state()
    snapshots = {k: [v] for k, v in initial.items()}
    commands = []
    trace = []
    phase = -1
    metadata = {
        "status": "running",
        "mode": "mock-input-fixture" if mock else "online-HTTP-SpaceMouse",
        "config": asdict(cfg),
        "preset": name,
        "device_name": s.device.name,
        "particles": s.n,
        "sampling_hz": 100.0,
        "input_frame": "fixed world: X forward, Y left, Z up; quaternion xyzw",
        "command_schema": "blademaster.teleop.v1",
        "input_timeout_s": source.timeout,
        "physics": "independent TLMPM with dynamic six-DOF knife servo and unilateral plane contact",
        "control_provenance": "local pose servo: translational DexLab gains; rotational gains and plane contact solver documented in README",
        "display": "HTTP state stream; video rendering and final NPZ compression excluded from compute timing",
        "predefined_cut_surfaces": False,
        "future_trajectory_given_to_solver": False,
        "revision": git_revision(),
        "surface": "recorded physical particles; no online smooth-surface reconstruction",
        "total_duration_s": duration,
        "pace_wall_clock": pace,
    }
    write(out / "manifest.json", metadata)
    server.publish({"ready": True, "time_s": 0.0, "target_edge_m": target.position.tolist()})
    print(f"READY http://127.0.0.1:{server.http.server_port}/state", flush=True)
    compute_s = 0.0
    start = time.perf_counter()
    latencies = []
    reference_u = (s.reference[:, 0] - s.reference[:, 0].min()) / np.ptp(s.reference[:, 0])
    try:
        for frame in range(frames):
            t = frame * 0.01
            tick_start = time.perf_counter()
            if mock:
                packet, phase = mock_command(t, frame, phase)
                source.feed(packet.packet(), now=t)
            command, events, stale = source.sample(now=t if mock else None)
            positions, rotations = target.integrate(command, events, cfg.dt, steps_per_frame)
            s.advance(positions, target.cutting, max(0, target.cut_index), orientations=rotations)
            state = s.state()
            for k, v in state.items():
                snapshots[k].append(v)
            force = s.reaction.numpy()[0] / cfg.dt
            torque = s.torque.numpy()[0] / cfg.dt
            table_force = s.tool_table_impulse.numpy()[0] / cfg.dt
            commands.append(
                {
                    "time_s": t,
                    **command.packet(),
                    "events": events,
                    "stale": stale,
                    "target_edge_m": target.position.tolist(),
                    "target_xyzw": target.orientation.tolist(),
                    "cut_index": target.cut_index,
                    "cutting": target.cutting,
                }
            )
            trace.append(
                {
                    "time_s": state["time"],
                    "blade_reaction_n": force.tolist(),
                    "blade_torque_nm": torque.tolist(),
                    "blade_table_force_n": table_force.tolist(),
                    "minimum_det_f": float(np.linalg.det(state["deformation"]).min()),
                }
            )
            server.publish(
                {
                    "ready": True,
                    "time_s": state["time"],
                    "stale_input": stale,
                    "target_edge_m": target.position.tolist(),
                    "blade_position_m": state["blade_position"][0].tolist(),
                    "blade_xyzw": state["blade_orientation"][0].tolist(),
                    "particles_m": state["position"].tolist(),
                    "blade_reaction_n": force.tolist(),
                    "cut_index": target.cut_index,
                    "cutting": target.cutting,
                    "preset": name,
                    "blade_half_size": cfg.blade_half_size,
                    "blade_bevel_height": cfg.blade_bevel_height,
                    "particle_radius_m": s.radius,
                    "reference_u": reference_u.tolist(),
                }
            )
            elapsed = time.perf_counter() - tick_start
            compute_s += elapsed
            latencies.append(elapsed)
            if pace:
                left = start + (frame + 1) * 0.01 - time.perf_counter()
                if left > 0:
                    time.sleep(left)
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata.update(status="failed", error=repr(exc))
        raise
    finally:
        wall = time.perf_counter() - start
        metadata.update(
            completed_steps=s.step_count,
            wall_s=wall,
            compute_and_record_s=compute_s,
            graph_build_s=s.graph_build_s,
        )
        server.close()
        np.savez_compressed(
            out / "states.npz", **{k: np.asarray(v) for k, v in snapshots.items()}, reference=s.reference
        )
        write(out / "commands.json", commands)
        write(out / "trace.json", trace)
        write(out / "manifest.json", metadata)
    result = {
        "status": "complete",
        "simulation_s": s.step_count * cfg.dt,
        "wall_s": wall,
        "compute_and_record_s": compute_s,
        "compute_realtime_factor": s.step_count * cfg.dt / compute_s,
        "frame_compute_ms_percentiles": np.percentile(latencies, [50, 95, 99, 100]).__mul__(1000).tolist(),
        "peak_sampled_force_n": max(float(np.linalg.norm(r["blade_reaction_n"])) for r in trace),
        "minimum_det_f": min(r["minimum_det_f"] for r in trace),
        "cut_epochs": target.cut_index + 1,
        "stale_frames": sum(r["stale"] for r in commands),
    }
    write(out / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--preset",
        default=DEFAULT_BANANA_PRESET,
        choices=["dexlab-baseline", "paper-material"],
        help="Default: paper-material; legacy soft-material demo requires explicit dexlab-baseline",
    )
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--pace", action="store_true")
    p.add_argument("--dt", type=float)
    a = p.parse_args()
    run(a.output, a.preset, a.duration, a.mock, a.port, a.pace, a.dt)
