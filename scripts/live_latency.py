"""Input-to-frame latency of the live teleoperation path.

Streams the mock operator input at 60 Hz over the same chunked POST /input as the SpaceMouse client,
polls GET /state like the browser and measures the age of the exact command
acknowledged by each frame's input_seq. Run against `franka_run --live --motion teleop`.
The script changes the run state only when asked: --start, --restart, --stop.
"""

import argparse
import gzip
import http.client
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np

from blademaster_warp.spacemouse_client import body
from blademaster_warp.teleop_input import mock_command
from blademaster_warp.provenance import git_revision


def wait_ready(host, port, restart=False, timeout=300.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2.0)
            conn.request("GET", "/state")
            status = json.loads(body(conn.getresponse()))
            if restart and status.get("state") in ("completed", "stopped", "failed"):
                # Restart a finished run; it starts playing as soon as it is ready.
                conn.request("POST", "/command/start")
                conn.getresponse().read()
            conn.close()
            if status.get("ready"):
                return status
        except (OSError, http.client.HTTPException, ValueError):
            pass
        time.sleep(0.2)
    raise TimeoutError("Simulation not ready")


def stream_input(host, port, first_seq, sent, stop, rate):
    stream = socket.create_connection((host, port), timeout=2.0)
    stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    stream.sendall(
        f"POST /input HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/x-ndjson\r\n"
        "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n".encode()
    )
    seq, phase, start = first_seq, -1, time.monotonic()
    while not stop.is_set():
        tick = time.monotonic()
        command, phase = mock_command(tick - start, seq, phase)
        data = (json.dumps(command.packet()) + "\n").encode()
        sent[seq] = time.perf_counter()
        stream.sendall(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        seq += 1
        time.sleep(max(0.0, tick + 1 / rate - time.monotonic()))
    stream.sendall(b"0\r\n\r\n")
    stream.close()


def percentiles(values):
    v = np.asarray(values, float) * 1e3
    return f"median {np.median(v):6.1f}  p95 {np.percentile(v, 95):6.1f}  max {v.max():6.1f} ms"


def acknowledged_latencies(sent, frames):
    """First received frame for each exactly acknowledged input sequence.

    LatestInput coalesces axis updates. A frame using sequence 10 does not
    establish that sequences 8 and 9 were ever integrated by the simulation.
    """
    seen, ages = set(), []
    for frame in frames:
        sequence = frame["seq"]
        if sequence in sent and sequence not in seen:
            ages.append(frame["received"] - sent[sequence])
            seen.add(sequence)
    return ages


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--rate", type=float, default=60.0, help="Operator input rate, Hz")
    p.add_argument("--start", action="store_true", help="Start the run if it is ready or paused")
    p.add_argument("--restart", action="store_true", help="Restart a finished run")
    p.add_argument("--stop", action="store_true", help="Stop the run at the end")
    p.add_argument("--output", type=Path, help="Save timing samples and configuration as JSON")
    a = p.parse_args()
    if not np.isfinite([a.seconds, a.rate]).all() or min(a.seconds, a.rate) <= 0:
        p.error("--seconds and --rate must be finite and positive")

    status = wait_ready(a.host, a.port, a.restart)
    sent, stop = {}, threading.Event()
    sender = threading.Thread(
        target=stream_input, args=(a.host, a.port, status.get("last_seq", -1) + 1, sent, stop, a.rate)
    )
    sender.start()
    time.sleep(0.2)
    conn = http.client.HTTPConnection(a.host, a.port, timeout=5.0)
    if a.start and status["state"] in ("ready", "paused"):
        conn.request("POST", "/command/start")
        conn.getresponse().read()

    frames, sequence, end = [], -1, time.monotonic() + a.seconds
    while time.monotonic() < end:
        t0 = time.perf_counter()
        conn.request("GET", f"/state?after={sequence}")
        response = conn.getresponse()
        body = response.read()
        t1 = time.perf_counter()
        if response.status == 204:
            continue
        raw = gzip.decompress(body)
        packet = json.loads(raw)
        t2 = time.perf_counter()
        sequence = packet["sequence"]
        if packet.get("state") == "running" and packet.get("input_seq") in sent:
            frames.append(
                dict(
                    seq=packet["input_seq"],
                    age=t1 - sent[packet["input_seq"]],
                    request=t1 - t0,
                    decode=t2 - t1,
                    received=t1,
                    gzip=len(body),
                    raw=len(raw),
                    sim=packet["sim_s"],
                    ratio=packet["ratio"],
                )
            )
        if packet.get("state") in ("completed", "stopped", "failed"):
            break
    stop.set()
    sender.join()
    if a.stop:
        conn.request("POST", "/command/stop")
        conn.getresponse().read()
    conn.close()
    if len(frames) < 3:
        raise RuntimeError(f"Only {len(frames)} running frames with a known input_seq")

    interval = np.diff([f["received"] for f in frames])
    first = acknowledged_latencies(sent, frames)
    print(
        f"{len(frames)} frames, sim {frames[0]['sim']:.3f}-{frames[-1]['sim']:.3f} s, ratio {frames[-1]['ratio']:.3f}"
    )
    print("acknowledged input -> frame ", percentiles(first), f"({len(first)} of {len(sent)} sent commands)")
    print("input age at frame receipt  ", percentiles([f["age"] for f in frames]))
    print("frame interval              ", percentiles(interval))
    print("GET /state request          ", percentiles([f["request"] for f in frames]))
    print("gunzip + JSON decode        ", percentiles([f["decode"] for f in frames]))
    print(
        f"payload  gzip {np.median([f['gzip'] for f in frames]) / 1e3:.0f} kB, "
        f"raw {np.median([f['raw'] for f in frames]) / 1e3:.0f} kB (median)"
    )
    if a.output:
        report = dict(
            revision=git_revision(),
            host=a.host,
            port=a.port,
            input_rate_hz=a.rate,
            elapsed_s=a.seconds,
            scope="command send to HTTP frame receipt, before browser rendering",
            sent_commands=len(sent),
            acknowledged_commands=len(first),
            acknowledged_latency_ms=dict(
                zip(["p50", "p95", "max"], (1000 * np.percentile(first, [50, 95, 100])).tolist())
            ),
            received_frames_hz=float(1 / np.mean(interval)),
            frames=frames,
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
