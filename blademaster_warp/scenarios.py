"""Cut-history validation scenarios inspired by BladeMaster Figs. 4/7/8.

Prescribed rigid tool isolates material mechanics. These paths are explicit
experiment inputs, not preset cut surfaces and not claimed author trajectories.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .coupling import ToolState,ExternalToolPort
from .geometry import DEMO_KNIFE_HALF_SIZE_M, DEMO_KNIFE_BEVEL_HEIGHT_M
from .provenance import git_revision


def keyframes(name):
    # time, position, rotation-euler-z-y-x, cut event (-1 means passive contact)
    if name=='parallel':
        return [(0,[-.012,0,.06],[0,0,0],0),(.3,[-.012,0,.06],[0,0,0],0),
            (2.3,[-.012,0,.0202],[0,0,0],0),(2.5,[-.012,0,.0202],[0,0,0],-1),
            (3.5,[-.012,0,.07],[0,0,0],-1),(4.2,[.012,0,.07],[0,0,0],1),
            (6.5,[.012,0,.0202],[0,0,0],1),(6.7,[.012,0,.0202],[0,0,0],-1),
            (8,[.012,0,.07],[0,0,0],-1)]
    if name=='cross-shallow':
        return [(0,[0,0,.06],[0,0,0],0),(2,[0,0,.035],[0,0,0],0),
            (2.2,[0,0,.035],[0,0,0],-1),(3.2,[0,0,.07],[0,0,0],-1),
            (4.2,[0,0,.07],[90,0,0],1),(6.2,[0,0,.035],[90,0,0],1),
            (6.4,[0,0,.035],[90,0,0],-1),(7.5,[0,0,.07],[90,0,0],-1)]
    if name=='slice-lift':
        base=keyframes('parallel')
        return base+[(9,[.065,0,.07],[0,0,0],-1),
            (10,[.065,0,.05],[0,90,0],-1),(11,[.065,0,.003],[0,90,0],-1),
            (13,[0,0,.003],[0,90,0],-1),(15,[0,0,.035],[0,90,0],-1),
            (16,[0,0,.035],[0,90,0],-1)]
    raise ValueError(name)


def command_at(t,keys):
    times=np.array([k[0] for k in keys]);j=min(np.searchsorted(times,t,side='right')-1,len(keys)-2);j=max(j,0)
    a,b=keys[j:j+2];duration=b[0]-a[0];u=np.clip((t-a[0])/duration,0,1)
    p=(1-u)*np.array(a[1])+u*np.array(b[1]);v=(np.array(b[1])-a[1])/duration
    rotations=Rotation.from_euler('ZYX',[a[2],b[2]],degrees=True)
    r=Slerp([0,1],rotations)(u)
    w=rotations[0].apply((rotations[0].inv()*rotations[1]).as_rotvec())/duration
    active=a[3]>=0
    index=max(k[3] for k in keys[:j+1])
    return ToolState(p,r.as_quat(),v,w),active,index


def run(args):
    from .solver import Solver,Config
    from .topology import components
    from .diagnostics import Metrics
    from .surface_runtime import SurfaceRecorder
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=Config(size=(.06,.03,.03),grid_spacing=.006,dt=args.dt,young_modulus=6000,
        poisson_ratio=.35,external_tool=True,blade_half_size=DEMO_KNIFE_HALF_SIZE_M,
        blade_bevel_height=DEMO_KNIFE_BEVEL_HEIGHT_M)
    keys=keyframes(args.scenario);interval=.002;duration=keys[-1][0]
    warm=Solver(cfg);ExternalToolPort(warm).advance(command_at(0,keys)[0],interval);del warm
    s=Solver(cfg);port=ExternalToolPort(s);meter=Metrics(s)
    surface=SurfaceRecorder(s,out/'surface',args.surface_spacing,getattr(args,'surface_embedding','cpu'),getattr(args,'surface_ccd','cpu')) if args.surface else None
    meta={'scenario':args.scenario,'config':asdict(cfg),'keyframes':keys,'driver':'prescribed tool (infinite effective mass)',
        'parameter_provenance':'Fig.7 elastic E/nu/rho; local geometry, path, Gc=.5 and contact numerics',
        'revision':git_revision(),
        'no_presplit':True,'surface_interval_s':.04 if surface else None,'status':'running'}
    (out/'manifest.json').write_text(json.dumps(meta,indent=2))
    saved={k:[] for k in s.state()};forces=[];energy=[];events=[];minimum=1.;status='completed';error=None
    begin=time.perf_counter()
    try:
        for i in range(round(duration/interval)):
            t=i*interval;tool,active,index=command_at(t,keys)
            reaction=port.advance(tool,interval,cutting=active,cut_index=index)
            forces.append([t,*reaction.force_n,*reaction.torque_nm])
            if i%20==0 or i==round(duration/interval)-1:
                state=s.state()
                for k,v in state.items():saved[k].append(v)
                energy.append([t,*meter.read()])
                minimum=min(minimum,energy[-1][-1])
                if surface:surface.update(t,state['position'],state['deformation'],state['sides'],tool,active,index)
            if (i+1)%250==0:
                print(f'{args.scenario} t={t+interval:.2f}s wall={time.perf_counter()-begin:.1f}s',flush=True)
            if i%500==0:
                _,counts=components(s.node_numpy,s.weight_numpy,s.codes.numpy())
                events.append(dict(time_s=t,components=counts.tolist()))
    except Exception as exc:status='failed';error=repr(exc)
    wall=time.perf_counter()-begin
    np.savez_compressed(out/'states.npz',**{k:np.asarray(v) for k,v in saved.items()},reference=s.reference,
        forces=np.array(forces),energy=np.array(energy))
    _,counts=components(s.node_numpy,s.weight_numpy,s.codes.numpy())
    meta.update(status=status,error=error,wall_s=wall,completed_sim_s=s.step_count*cfg.dt,
        realtime_ratio=s.step_count*cfg.dt/wall,components=counts.tolist(),events=events,min_det_F=float(minimum),
        surface_wall_s=surface.wall_s if surface else 0.)
    (out/'manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2),flush=True)
    if error:raise RuntimeError(error)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--scenario',choices=['parallel','cross-shallow','slice-lift'],default='parallel')
    p.add_argument('--dt',type=float,default=.00025);p.add_argument('--surface',action='store_true')
    p.add_argument('--surface-embedding',choices=['cpu','cuda'],default='cpu',
                   help='Nodal field evaluation only; topology remains the CPU reference')
    p.add_argument('--surface-ccd',choices=['cpu','cuda'],default='cpu',
                   help='Finite moving-segment intersection backend; Hermite/QEF/triangles remain CPU')
    p.add_argument('--surface-spacing',type=float,default=None,help='Surface grid in m; default half the MPM spacing, no effect on physics')
    run(p.parse_args())
