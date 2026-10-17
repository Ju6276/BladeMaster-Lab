"""Stream SpaceMouse input from the Mac to the simulation over HTTP."""

import argparse
import gzip
import http.client
import json
import socket
import time
import numpy as np
from .teleop_input import Command, hid_twist, mock_command


def body(response):
    """Response body; the simulation server gzips every response."""
    data = response.read()
    return gzip.decompress(data) if data and response.getheader("Content-Encoding") == "gzip" else data


def session_command(host, port, name):
    """Press a page button (start/pause/stop) on the simulation server."""
    conn = http.client.HTTPConnection(host, port, timeout=2.0)
    try:
        conn.request("POST", "/command/" + name)
        response = conn.getresponse()
        result = body(response)
        if response.status != 200:
            print(f"{name}: {result.decode()}", flush=True)
    except (OSError, http.client.HTTPException) as error:
        print(f"{name}: {error}", flush=True)
    finally:
        conn.close()


def run(
    host="127.0.0.1",
    port=8765,
    duration=30.0,
    mock=False,
    speed=0.06,
    angular_speed=0.5,
    buttons="session",
):
    conn = http.client.HTTPConnection(host, port, timeout=2.0)
    # Do not start the device/motion clock until shader/kernel warmup is done.
    end = time.monotonic() + 120
    while True:
        try:
            conn.request("GET", "/state")
            response = conn.getresponse()
            status = json.loads(body(response))
            ready = status.get("ready", False)
            if ready:
                break
        except (OSError, http.client.HTTPException):
            conn.close()
        if time.monotonic() > end:
            raise TimeoutError("Simulation not ready")
        time.sleep(0.2)
    mouse = None
    if not mock:
        from .spacemouse import SpaceMouse

        mouse = SpaceMouse()
        mouse.__enter__()
    conn.close()
    stream = socket.create_connection((host, port), timeout=2.0)
    stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    stream.sendall(
        f"POST /input HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n".encode()
    )

    def send(command):
        data = (json.dumps(command.packet()) + "\n").encode()
        stream.sendall(f"{len(data):x}\r\n".encode() + data + b"\r\n")

    seq = status.get("last_seq", -1) + 1
    phase = -1
    armed = False
    start = time.monotonic()
    try:
        while time.monotonic() - start < duration:
            tick = time.monotonic()
            t = tick - start
            if mock:
                command, phase = mock_command(t, seq, phase)
            else:
                snapshot = mouse.state.sample()
                fresh = np.repeat(tick - snapshot.received <= 0.25, 3)
                stale = np.any((abs(snapshot.axes) > 0.08) & ~fresh)
                if snapshot.error or stale:
                    armed = False
                    v = w = np.zeros(3)
                    event = "hold"
                else:
                    if not armed and np.max(abs(snapshot.axes)) <= 0.08 and snapshot.buttons == 0:
                        armed = True
                    v, w = (
                        hid_twist(snapshot.axes * fresh, speed, angular_speed)
                        if armed
                        else (np.zeros(3), np.zeros(3))
                    )
                    event = ""
                    if buttons == "session":
                        # Left starts the run, right ends it, both pause; same as the page buttons.
                        for press in snapshot.presses:
                            session_command(host, port, {"close": "start", "open": "stop", "hold": "pause"}[press])
                    elif snapshot.presses and armed:
                        events = {"close": "begin_cut", "open": "end_cut", "hold": "hold"}
                        event = events.get(snapshot.presses[-1], "")
                command = Command(seq, tuple(v), tuple(w), event)
            send(command)
            seq += 1
            if t // 1 != (time.monotonic() - start) // 1 or seq == 1:
                print(f'{t:.1f}s seq={seq} input={"mock" if mock else "HID"}', flush=True)
            remaining = tick + 1 / 60 - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
    finally:
        if mouse:
            mouse.__exit__()
        try:
            stop = Command(seq, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "hold")
            send(stop)
            stream.sendall(b"0\r\n\r\n")
            response = http.client.HTTPResponse(stream)
            response.begin()
            result = body(response)
            if response.status != 200:
                raise RuntimeError(result.decode())
            print(result.decode(), flush=True)
        except (OSError, http.client.HTTPException):
            pass
        stream.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--speed", type=float, default=0.06)
    p.add_argument("--angular-speed", type=float, default=0.5)
    p.add_argument(
        "--buttons",
        choices=["session", "cut"],
        default="session",
        help="session: left start, right stop, both pause; cut: left begin_cut, right end_cut (manual cut events)",
    )
    a = p.parse_args()
    run(a.host, a.port, a.seconds, a.mock, a.speed, a.angular_speed, a.buttons)
