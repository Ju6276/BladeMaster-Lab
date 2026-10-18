import gzip
import json
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import live_latency
from scripts.live_latency import acknowledged_latencies


def test_coalesced_commands_are_not_counted_as_observed():
    sent = {8: 1.00, 9: 1.01, 10: 1.02, 11: 1.03}
    frames = [
        {"seq": 10, "received": 1.10},
        {"seq": 10, "received": 1.20},
        {"seq": 11, "received": 1.30},
        {"seq": 12, "received": 1.40},
    ]
    assert acknowledged_latencies(sent, frames) == [1.10 - 1.02, 1.30 - 1.03]


def response(packet=None, *, raw=None, status=200):
    data = json.dumps({} if packet is None else packet).encode() if raw is None else raw
    result = Mock(status=status)
    result.read.return_value = gzip.compress(data)
    result.getheader.return_value = "gzip"
    return result


def connection(*responses):
    result = Mock()
    result.getresponse.side_effect = responses
    return result


@pytest.fixture
def offline(monkeypatch):
    ready_response = response(dict(ready=True, state="running", last_seq=10))
    poll_response = response(dict(sequence=1, state="completed"))
    stop_response = response()
    ready = connection(ready_response)
    poll = connection(poll_response)
    control = connection(stop_response)
    connect = Mock(side_effect=[ready, poll, control])
    monkeypatch.setattr(live_latency.http.client, "HTTPConnection", connect)
    input_response = response()
    monkeypatch.setattr(live_latency.http.client, "HTTPResponse", Mock(return_value=input_response))
    stream = Mock()
    first_packet = threading.Event()

    def send(data):
        if not data.startswith(b"POST ") and data != b"0\r\n\r\n":
            first_packet.set()

    stream.sendall.side_effect = send
    open_socket = Mock(return_value=stream)
    monkeypatch.setattr(live_latency.socket, "create_connection", open_socket)
    monkeypatch.setattr(sys, "argv", ["live_latency", "--host", "offline.invalid", "--seconds", "30"])

    threads, stops = [], []
    original_thread = threading.Thread
    original_stream_input = live_latency.stream_input

    class TrackedThread(original_thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.joined = False
            threads.append(self)

        def join(self, *args, **kwargs):
            self.joined = True
            return super().join(*args, **kwargs)

    def track_input(host, port, first_seq, sent, stop, rate):
        stops.append(stop)
        return original_stream_input(host, port, first_seq, sent, stop, rate)

    monkeypatch.setattr(live_latency.threading, "Thread", TrackedThread)
    monkeypatch.setattr(live_latency, "stream_input", track_input)
    yield SimpleNamespace(
        ready=ready,
        ready_response=ready_response,
        poll=poll,
        poll_response=poll_response,
        control=control,
        stop_response=stop_response,
        connect=connect,
        stream=stream,
        open_socket=open_socket,
        input_response=input_response,
        first_packet=first_packet,
        threads=threads,
        stops=stops,
    )
    # Also clean up when an assertion detects the original leaking-thread bug.
    for stop in stops:
        stop.set()
    for thread in threads:
        if thread.ident is not None:
            original_thread.join(thread, timeout=3.0)
            assert not thread.is_alive()


def assert_sender_stopped(offline):
    assert len(offline.threads) == 1
    assert offline.threads[0].joined
    assert not offline.threads[0].is_alive()
    assert offline.stops[0].is_set()
    offline.stream.close.assert_called_once()


@pytest.mark.parametrize("stop_run", [False, True])
@pytest.mark.parametrize("failure", ["request", "headers", "read", "gzip", "json", "interrupt"])
def test_poll_failure_stops_and_joins_sender(offline, failure, stop_run, capsys):
    if stop_run:
        sys.argv.append("--stop")
        offline.control.request.side_effect = lambda *args: assert_sender_stopped(offline)
    error = TimeoutError("poll failed")
    if failure == "request":
        offline.poll.request.side_effect = error
    elif failure == "headers":
        offline.poll.getresponse.side_effect = error
    elif failure in ("read", "interrupt"):
        if failure == "interrupt":
            error = KeyboardInterrupt()
        offline.poll_response.read.side_effect = error
    elif failure == "gzip":
        offline.poll_response.read.return_value = b"not gzip"
        error = gzip.BadGzipFile()
    else:
        offline.poll_response.read.return_value = gzip.compress(b"not JSON")
        error = json.JSONDecodeError("invalid", "", 0)

    with pytest.raises(type(error)) as caught:
        live_latency.main()
    if failure not in ("gzip", "json"):
        assert caught.value is error
    assert offline.first_packet.is_set()
    assert_sender_stopped(offline)
    offline.ready.close.assert_called_once()
    offline.poll.close.assert_called_once()
    if failure not in ("request", "headers"):
        offline.poll_response.close.assert_called_once()
    assert offline.connect.call_count == 2 + stop_run
    if stop_run:
        offline.control.request.assert_called_once_with("POST", "/command/stop")
        offline.control.close.assert_called_once()
        offline.stop_response.close.assert_called_once()
    else:
        offline.control.request.assert_not_called()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("failure", ["connect", "options", "header", "packet"])
def test_sender_failure_propagates_and_attempts_stop(offline, failure, tmp_path, capsys):
    output = tmp_path / "latency.json"
    sys.argv.extend(["--stop", "--output", str(output)])
    offline.connect.side_effect = [offline.ready, offline.control]
    error = OSError(f"sender {failure} failed")
    if failure == "connect":
        offline.open_socket.side_effect = error
    elif failure == "options":
        offline.stream.setsockopt.side_effect = error
    elif failure == "header":
        offline.stream.sendall.side_effect = error
    else:
        offline.stream.sendall.side_effect = [None, error]

    with pytest.raises(OSError) as caught:
        live_latency.main()
    assert caught.value is error
    assert offline.threads[0].joined and not offline.threads[0].is_alive()
    assert offline.stops[0].is_set()
    if failure != "connect":
        offline.stream.close.assert_called_once()
    offline.poll.request.assert_not_called()
    offline.ready.close.assert_called_once()
    offline.control.request.assert_called_once_with("POST", "/command/stop")
    offline.control.close.assert_called_once()
    assert not output.exists()
    assert capsys.readouterr().out == ""


def running_responses():
    return [
        response(dict(sequence=i, state="running", input_seq=11, sim_s=i / 10, ratio=1.0))
        for i in range(1, 4)
    ] + [response(dict(sequence=4, state="completed"))]


@pytest.mark.parametrize("failure", ["terminator", "headers", "read", "rejected"])
def test_late_sender_failure_prevents_success_report(offline, failure, tmp_path, capsys):
    output = tmp_path / "latency.json"
    sys.argv.extend(["--stop", "--output", str(output)])
    offline.poll.getresponse.side_effect = running_responses()
    error = BrokenPipeError("input stream failed during shutdown")
    if failure == "terminator":
        send = offline.stream.sendall.side_effect

        def fail_at_end(data):
            if data == b"0\r\n\r\n":
                raise error
            send(data)

        offline.stream.sendall.side_effect = fail_at_end
    elif failure == "headers":
        offline.input_response.begin.side_effect = error
    elif failure == "read":
        offline.input_response.read.side_effect = error
    else:
        offline.input_response.status = 400
        error = RuntimeError("POST /input failed: HTTP 400")

    with pytest.raises(type(error), match=str(error)):
        live_latency.main()
    assert_sender_stopped(offline)
    offline.poll.close.assert_called_once()
    if failure != "terminator":
        offline.input_response.close.assert_called_once()
    offline.control.request.assert_called_once_with("POST", "/command/stop")
    offline.control.close.assert_called_once()
    assert not output.exists()
    assert capsys.readouterr().out == ""


def test_sender_failure_during_polling_ends_measurement(offline):
    sys.argv.extend(["--stop", "--rate", "1000"])
    poll_started = threading.Event()
    error = ConnectionResetError("sender disconnected while polling")
    send = offline.stream.sendall.side_effect

    def fail_after_poll(data):
        send(data)
        if poll_started.is_set():
            raise error

    def poll(method, path):
        assert offline.first_packet.wait(1.0)
        poll_started.set()
        assert offline.stops[0].wait(1.0)

    offline.stream.sendall.side_effect = fail_after_poll
    offline.poll.request.side_effect = poll
    offline.poll_response.status = 204
    with pytest.raises(ConnectionResetError) as caught:
        live_latency.main()
    assert caught.value is error
    offline.poll.request.assert_called_once()
    assert_sender_stopped(offline)
    offline.poll.close.assert_called_once()
    offline.control.close.assert_called_once()


@pytest.mark.parametrize("rate", [0.001, 1e-300])
def test_low_rate_sender_stops_promptly(offline, rate):
    stop = threading.Event()
    errors = []

    def send():
        try:
            live_latency.stream_input("offline.invalid", 8765, 42, {}, stop, rate)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=send, daemon=True)
    thread.start()
    try:
        assert offline.first_packet.wait(1.0)
    finally:
        start = time.monotonic()
        stop.set()
        thread.join(timeout=1.0)
    assert not thread.is_alive(), "Stopping must interrupt the input-rate wait"
    assert time.monotonic() - start < 1.0
    assert not errors
    offline.stream.sendall.assert_called_with(b"0\r\n\r\n")
    offline.stream.close.assert_called_once()
    offline.input_response.close.assert_called_once()


@pytest.mark.parametrize("failure", ["request", "headers", "read", "gzip", "json", "restart"])
def test_wait_ready_closes_failed_attempt_before_retry(offline, monkeypatch, failure):
    first_response = response(dict(ready=False, state="stopped"))
    first = connection(first_response)
    error = TimeoutError("not ready yet")
    if failure == "request":
        first.request.side_effect = error
    elif failure == "headers":
        first.getresponse.side_effect = error
    elif failure == "read":
        first_response.read.side_effect = error
    elif failure == "gzip":
        first_response.read.return_value = b"not gzip"
    elif failure == "json":
        first_response.read.return_value = gzip.compress(b"not JSON")
    else:
        restart_response = response()
        restart_response.read.side_effect = error
        first.getresponse.side_effect = [first_response, restart_response]
    offline.connect.side_effect = [first, offline.ready]

    def retry_delay(seconds):
        first.close.assert_called_once()
        if failure not in ("request", "headers"):
            first_response.close.assert_called_once()

    monkeypatch.setattr(live_latency.time, "sleep", retry_delay)
    assert live_latency.wait_ready("offline.invalid", 8765, restart=True)["ready"]
    offline.ready.close.assert_called_once()
    offline.ready_response.close.assert_called_once()
    if failure == "restart":
        restart_response.close.assert_called_once()


@pytest.mark.parametrize("operation", ["ready", "restart", "start", "poll"])
def test_rejected_http_status_fails_and_still_stops(offline, operation):
    sys.argv.append("--stop")
    rejected = response(status=503)
    if operation in ("ready", "restart"):
        offline.connect.side_effect = [offline.ready, offline.control]
        offline.ready.getresponse.side_effect = [rejected]
        endpoint = "GET /state"
        if operation == "restart":
            sys.argv.append("--restart")
            offline.ready.getresponse.side_effect = [response(dict(ready=False, state="stopped")), rejected]
            endpoint = "POST /command/start"
    else:
        offline.poll.getresponse.side_effect = [rejected]
        endpoint = "GET /state"
        if operation == "start":
            sys.argv.append("--start")
            offline.ready_response.read.return_value = gzip.compress(
                b'{"ready": true, "state": "ready", "last_seq": 10}'
            )
            endpoint = "POST /command/start"

    with pytest.raises(RuntimeError, match=f"{endpoint} failed: HTTP 503"):
        live_latency.main()
    rejected.close.assert_called_once()
    offline.ready.close.assert_called_once()
    if operation in ("start", "poll"):
        assert_sender_stopped(offline)
        offline.poll.close.assert_called_once()
    else:
        assert not offline.threads
    offline.control.request.assert_called_once_with("POST", "/command/stop")
    offline.control.close.assert_called_once()


@pytest.mark.parametrize("failure", ["request", "headers", "read", "rejected"])
def test_stop_failure_preserves_original_poll_error(offline, failure):
    sys.argv.append("--stop")
    original = TimeoutError("original poll error")
    offline.poll.request.side_effect = original
    stop_error = OSError("stop failed")
    if failure == "request":
        offline.control.request.side_effect = stop_error
    elif failure == "headers":
        offline.control.getresponse.side_effect = stop_error
    elif failure == "read":
        offline.stop_response.read.side_effect = stop_error
    else:
        offline.stop_response.status = 403

    with pytest.raises(TimeoutError) as caught:
        live_latency.main()
    assert caught.value is original
    assert any("stop failed" in note or "HTTP 403" in note for note in original.__notes__)
    assert_sender_stopped(offline)
    offline.poll.close.assert_called_once()
    offline.control.close.assert_called_once()
    if failure in ("read", "rejected"):
        offline.stop_response.close.assert_called_once()


def test_poll_error_preserved_when_sender_also_fails_during_shutdown(offline):
    sys.argv.append("--stop")
    original = TimeoutError("original poll error")
    offline.poll.request.side_effect = original
    offline.input_response.read.side_effect = ConnectionResetError("sender shutdown failed")
    with pytest.raises(TimeoutError) as caught:
        live_latency.main()
    assert caught.value is original
    assert any("sender shutdown failed" in note for note in original.__notes__)
    assert_sender_stopped(offline)
    offline.control.request.assert_called_once_with("POST", "/command/stop")


def test_stop_rejection_prevents_success_report(offline, tmp_path, capsys):
    output = tmp_path / "latency.json"
    sys.argv.extend(["--stop", "--output", str(output)])
    offline.poll.getresponse.side_effect = running_responses()
    offline.stop_response.status = 400
    with pytest.raises(RuntimeError, match="POST /command/stop failed: HTTP 400"):
        live_latency.main()
    assert_sender_stopped(offline)
    offline.control.close.assert_called_once()
    offline.stop_response.close.assert_called_once()
    assert not output.exists()
    assert capsys.readouterr().out == ""


def test_success_closes_transports_and_writes_report(offline, monkeypatch, tmp_path, capsys):
    output = tmp_path / "latency.json"
    sys.argv.extend(["--stop", "--output", str(output)])
    offline.poll.getresponse.side_effect = [response(status=204), *running_responses()]
    monkeypatch.setattr(live_latency, "git_revision", lambda: "offline-test")
    live_latency.main()
    assert_sender_stopped(offline)
    offline.ready.close.assert_called_once()
    offline.poll.close.assert_called_once()
    offline.control.close.assert_called_once()
    offline.input_response.close.assert_called_once()
    report = json.loads(output.read_text())
    assert len(report["frames"]) == 3
    assert report["acknowledged_commands"] == 1
    assert report["sent_commands"] >= 1
    assert report["revision"] == "offline-test"
    assert "3 frames" in capsys.readouterr().out


@pytest.mark.parametrize(
    "failure", ["readiness", "thread_start", "connection", "start_request", "start_read"]
)
def test_setup_failure_attempts_stop(offline, monkeypatch, failure):
    sys.argv.extend(["--start", "--stop"])
    original = RuntimeError(f"{failure} failed")
    if failure == "readiness":
        monkeypatch.setattr(live_latency, "wait_ready", Mock(side_effect=original))
        offline.connect.side_effect = [offline.control]
    elif failure == "thread_start":
        monkeypatch.setattr(live_latency.threading.Thread, "start", Mock(side_effect=original))
        offline.connect.side_effect = [offline.ready, offline.control]
    elif failure == "connection":
        offline.connect.side_effect = [offline.ready, original, offline.control]
    else:
        offline.ready_response.read.return_value = gzip.compress(
            b'{"ready": true, "state": "paused", "last_seq": 10}'
        )
        if failure == "start_request":
            offline.poll.request.side_effect = original
        else:
            offline.poll_response.read.side_effect = original
    with pytest.raises(RuntimeError) as caught:
        live_latency.main()
    assert caught.value is original
    if failure in ("connection", "start_request", "start_read"):
        assert_sender_stopped(offline)
    if failure.startswith("start_"):
        offline.poll.close.assert_called_once()
    if failure == "start_read":
        offline.poll_response.close.assert_called_once()
    offline.control.request.assert_called_once_with("POST", "/command/stop")
    offline.control.close.assert_called_once()


@pytest.mark.parametrize("status", [200, 400])
def test_input_stream_and_response_over_socket_pair(monkeypatch, status):
    client, peer = socket.socketpair()
    stop, received = threading.Event(), threading.Event()
    sent = {}
    client.settimeout(2.0)
    peer.settimeout(2.0)
    transport = Mock(wraps=client)
    # AF_UNIX socket pairs support the byte stream but not TCP_NODELAY.
    transport.setsockopt = Mock()
    monkeypatch.setattr(live_latency.socket, "create_connection", lambda *args, **kwargs: transport)

    def receive():
        with peer.makefile("rb") as stream:
            assert stream.readline() == b"POST /input HTTP/1.1\r\n"
            headers = []
            while (line := stream.readline()) != b"\r\n":
                assert line
                headers.append(line)
            assert b"Transfer-Encoding: chunked\r\n" in headers
            packets = []
            while size := int(stream.readline(), 16):
                packets.append(json.loads(stream.read(size)))
                assert stream.read(2) == b"\r\n"
                received.set()
            assert stream.readline() == b"\r\n"
            peer.sendall(
                f"HTTP/1.1 {status} Result\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}".encode()
            )
            return packets

    with closing(client), closing(peer), ThreadPoolExecutor(max_workers=2) as executor:
        reader = executor.submit(receive)
        sender = executor.submit(live_latency.stream_input, "offline.invalid", 8765, 42, sent, stop, 10.0)
        try:
            assert received.wait(1.0)
        finally:
            stop.set()
        if status == 200:
            sender.result(timeout=2.0)
        else:
            with pytest.raises(RuntimeError, match="POST /input failed: HTTP 400"):
                sender.result(timeout=2.0)
        assert client.fileno() == -1
        packets = reader.result(timeout=2.0)
        assert packets and packets[0]["seq"] == 42
        assert list(sent) == [packet["seq"] for packet in packets]
