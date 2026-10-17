import gzip
import http.client
import io
import json
import socket
from threading import Thread
import time
import zlib

import pytest

from blademaster_warp.live import LiveState, make_server
from blademaster_warp.teleop import Server
from blademaster_warp.teleop_input import LatestInput, Command, read_chunked_packets


@pytest.fixture(params=["standalone", "live"])
def input_server(request):
    source = LatestInput()
    if request.param == "standalone":
        server = Server(source, port=0)
        server.start()
        try:
            yield source, server.http.server_port
        finally:
            server.close()
    else:
        live = LiveState()
        live.teleop = source
        server = make_server(live, 0)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield source, server.server_port
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


@pytest.mark.parametrize("chunk_size", [None, 1, 23])
def test_ndjson_packets_are_independent_of_http_chunk_boundaries(input_server, chunk_size):
    source, port = input_server
    events = ["begin_cut", "", "end_cut", "hold"]
    packets = b"".join(
        (json.dumps(Command(seq, (0.01, 0, 0), (0, 0, 0), event).packet()) + "\n").encode()
        for seq, event in enumerate(events)
    )
    size = len(packets) if chunk_size is None else chunk_size
    chunks = (packets[i : i + size] for i in range(0, len(packets), size))
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
    try:
        connection.request("POST", "/input", body=chunks, encode_chunked=True)
        response = connection.getresponse()
        body = response.read()
        if response.getheader("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        assert response.status == 200, body
        assert json.loads(body)["accepted_packets"] == 4
        command, received, _ = source.sample()
        assert command.seq == 3
        assert received == [(0, "begin_cut"), (2, "end_cut"), (3, "hold")]
    finally:
        connection.close()


def test_large_http_chunk_can_contain_many_small_packets():
    source = LatestInput()
    packets = b"".join(
        (json.dumps(Command(seq, (0, 0, 0), (0, 0, 0)).packet()) + "\n").encode() for seq in range(64)
    )
    assert len(packets) > 8192
    body = f"{len(packets):x};extension=value\r\n".encode() + packets + b"\r\n0\r\n\r\n"
    accepted = read_chunked_packets({"Transfer-Encoding": "chunked"}, io.BytesIO(body), source.feed)
    assert accepted == 64 and source.latest.seq == 63


@pytest.mark.parametrize("chunk_size", [2048, 16384])
def test_packet_length_limit_applies_across_http_chunks(chunk_size):
    packet = b" " * 8193 + b"{}\n"
    chunks = [packet[i : i + chunk_size] for i in range(0, len(packet), chunk_size)]
    body = b"".join(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n" for chunk in chunks)
    received = []
    with pytest.raises(ValueError, match="Packet too long"):
        read_chunked_packets(
            {"Transfer-Encoding": "chunked"}, io.BytesIO(body + b"0\r\n\r\n"), received.append
        )
    assert received == []


def test_http_packets_state_and_disconnect_timeout():
    source = LatestInput(timeout=0.02)
    server = Server(source, port=0)
    server.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.http.server_port, timeout=2.0)
    try:
        packet = Command(0, (0.01, 0, 0), (0, 0, 0.2), "begin_cut").packet()
        connection.request("POST", "/command", json.dumps(packet))
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        server.publish({"ready": True})
        connection.request("GET", "/state")
        response = connection.getresponse()
        assert json.loads(response.read()) == {"ready": True, "last_seq": 0}
        connection.request("POST", "/command", json.dumps(packet))
        response = connection.getresponse()
        assert response.status == 400
        response.read()
        cmd, events, stale = source.sample()
        assert events == [(0, "begin_cut")]
        connection.close()
        time.sleep(0.03)
        cmd, events, stale = source.sample()
        assert stale and not any(cmd.velocity) and events == []
    finally:
        connection.close()
        server.close()


def test_streaming_input_preserves_events_without_roundtrip_per_packet():
    source = LatestInput()
    server = Server(source, port=0)
    server.start()
    s = socket.create_connection(("127.0.0.1", server.http.server_port), timeout=2.0)
    try:
        s.sendall(
            b"POST /input HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        )
        for seq, event in enumerate(["begin_cut", "", "end_cut", "hold"]):
            data = (json.dumps(Command(seq, (0, 0, 0), (0, 0, 0), event).packet()) + "\n").encode()
            s.sendall(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        s.sendall(b"0\r\n\r\n")
        response = http.client.HTTPResponse(s)
        response.begin()
        assert response.status == 200 and json.loads(response.read())["accepted_packets"] == 4
        command, events, stale = source.sample()
        assert command.seq == 3 and events == [(0, "begin_cut"), (2, "end_cut"), (3, "hold")]
    finally:
        s.close()
        server.close()


def test_compressed_state_stream_preserves_actual_values():
    server = Server(LatestInput(), port=0)
    state = {"ready": True, "particles_m": [[0.0123456789, -0.00321, 0.001234]], "time_s": 0.01}
    server.publish(state)
    server.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.http.server_port, timeout=2.0)
    try:
        connection.request("GET", "/events")
        response = connection.getresponse()
        assert response.status == 200 and response.getheader("Content-Encoding") == "gzip"
        decoder = zlib.decompressobj(wbits=31)
        data = b""
        while b"\n\n" not in data:
            data += decoder.decompress(response.read1(4096))
        assert json.loads(data.split(b"\n\n")[0].removeprefix(b"data: ")) == state
    finally:
        connection.close()
        server.close()
