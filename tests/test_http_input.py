import http.client
import json
import time
import socket
import zlib
from blademaster_warp.teleop import Server
from blademaster_warp.teleop_input import LatestInput,Command


def test_http_packets_state_and_disconnect_timeout():
    source=LatestInput(timeout=.02)
    server=Server(source,port=0);server.start()
    connection=http.client.HTTPConnection('127.0.0.1',server.http.server_port,timeout=2.)
    try:
        packet=Command(0,(.01,0,0),(0,0,.2),'begin_cut').packet()
        connection.request('POST','/command',json.dumps(packet))
        response=connection.getresponse();assert response.status==200;response.read()
        server.publish({'ready':True})
        connection.request('GET','/state');response=connection.getresponse()
        assert json.loads(response.read())=={'ready':True,'last_seq':0}
        connection.request('POST','/command',json.dumps(packet))
        response=connection.getresponse();assert response.status==400;response.read()
        cmd,events,stale=source.sample()
        assert events==[(0,'begin_cut')]
        connection.close()
        time.sleep(.03)
        cmd,events,stale=source.sample()
        assert stale and not any(cmd.velocity) and events==[]
    finally:
        connection.close();server.close()


def test_streaming_input_preserves_events_without_roundtrip_per_packet():
    source=LatestInput();server=Server(source,port=0);server.start()
    s=socket.create_connection(('127.0.0.1',server.http.server_port),timeout=2.)
    try:
        s.sendall(b'POST /input HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n')
        for seq,event in enumerate(['begin_cut','','end_cut','hold']):
            data=(json.dumps(Command(seq,(0,0,0),(0,0,0),event).packet())+'\n').encode()
            s.sendall(f'{len(data):x}\r\n'.encode()+data+b'\r\n')
        s.sendall(b'0\r\n\r\n')
        response=http.client.HTTPResponse(s);response.begin()
        assert response.status==200 and json.loads(response.read())['accepted_packets']==4
        command,events,stale=source.sample()
        assert command.seq==3 and events==[(0,'begin_cut'),(2,'end_cut'),(3,'hold')]
    finally:s.close();server.close()


def test_compressed_state_stream_preserves_actual_values():
    server=Server(LatestInput(),port=0)
    state={'ready':True,'particles_m':[[.0123456789,-.00321,.001234]],'time_s':.01}
    server.publish(state);server.start()
    connection=http.client.HTTPConnection('127.0.0.1',server.http.server_port,timeout=2.)
    try:
        connection.request('GET','/events');response=connection.getresponse()
        assert response.status==200 and response.getheader('Content-Encoding')=='gzip'
        decoder=zlib.decompressobj(wbits=31);data=b''
        while b'\n\n' not in data:data+=decoder.decompress(response.read1(4096))
        assert json.loads(data.split(b'\n\n')[0].removeprefix(b'data: '))==state
    finally:connection.close();server.close()
