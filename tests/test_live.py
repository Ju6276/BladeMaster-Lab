"""Online transport exposes current states and never creates replay files."""
import gzip
import json
from threading import Thread
from urllib.request import urlopen, Request
import numpy as np
from blademaster_warp.live import LiveState, make_server


def test_live_state_protocol_and_pause_resume():
    live=LiveState()
    live._set(dict(state='ready',sim_s=0.))
    server=make_server(live,0)
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base+'/state') as response:
            packet=json.loads(gzip.decompress(response.read()))
        assert packet['sim_s']==0.
        with urlopen(base+f"/state?after={packet['sequence']}") as response:
            assert response.status==204
        urlopen(Request(base+'/command/start',method='POST')).close()
        assert live.play.is_set()
        urlopen(Request(base+'/command/pause',method='POST')).close()
        assert not live.play.is_set()
        urlopen(Request(base+'/command/stop',method='POST')).close()
        assert live.stop_requested and live.play.is_set()
        live.finish(dict(status='completed'))
        urlopen(Request(base+'/command/start',method='POST')).close()
        assert live.restart.is_set() and live.autostart
    finally:
        server.shutdown();server.server_close()


def test_recording_is_opt_in(tmp_path):
    from blademaster_warp.franka_run import parser,run
    for record in (False,True):
        out=tmp_path/str(record)
        args=parser().parse_args(['--no-material','--duration','.002','--output',str(out)]+(['--record'] if record else []))
        report=run(args)
        assert report['recording']==record
        assert (out/'manifest.json').exists() and (out/'report.json').exists()
        assert (out/'trajectory.npz').exists()==record
        if not record:
            assert {p.name for p in out.iterdir()}=={'manifest.json','report.json'}


def test_surface_can_run_without_storage(tmp_path,monkeypatch):
    from blademaster_warp.solver import Solver
    from blademaster_warp.surface_runtime import SurfaceRecorder
    from blademaster_warp.coupling import ToolState
    s=Solver()
    monkeypatch.chdir(tmp_path)
    surface=SurfaceRecorder(s,None)
    surface.update(0.,s.x.numpy(),s.F.numpy(),s.codes.numpy(),
        ToolState([0,0,.08],[0,0,0,1],[0,0,0],[0,0,0]),cutting=False)
    assert len(surface.current['vertices'])>0
    assert not list(tmp_path.iterdir())


def test_live_preview_does_not_insert_unpaired_recorded_frame(tmp_path):
    from blademaster_warp.franka_run import parser,run
    out=tmp_path/'live_record'
    args=parser().parse_args(['--material','paper-banana','--duration','.042',
        '--surface','--surface-embedding','cuda','--surface-ccd','cuda',
        '--record','--output',str(out)])
    live=LiveState();live.autostart=True
    run(args,observer=live)
    trajectory=np.load(out/'trajectory.npz')
    frames=json.loads((out/'surface/frames.json').read_text())
    assert len(trajectory['qpos'])==len(trajectory['particles'])==len(frames)==3
    np.testing.assert_allclose([f['time_s'] for f in frames],trajectory['control_tcp'][:,0])
