"""Paper-material diagnostics with explicit provenance, not author experiments.

One prescribed vertical cut removes the DexLab PD actuator as a confounder.
The banana geometry and trajectory are independently specified, not measured
author assets. No result from this runner alone establishes reproduction.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import time
import numpy as np
import warp as wp
from .solver import Solver
from .run import preset, write
from .coupling import ExternalToolPort, ToolState, Reaction
from .topology import components


def vertical_state(t, half_height):
    # Explicit test fixture: 0.036 -> 0 m in 1.8 s, followed by withdrawal.
    if t < 1.8: edge=.036-.02*t; speed=-.02; active=True
    elif t < 2.: edge=0.; speed=0.; active=True
    else: edge=.02*(t-2.); speed=.02; active=False
    return ToolState([0.,0.,edge+half_height],[0.,0.,0.,1.],
                     [0.,0.,speed],[0.,0.,0.]),active


def single_cut(output,dt=.000025,interval=.002,duration=3.8,graph=True,cache_topology=False,
               driver='prescribed',spacing=.006):
    if driver not in ('prescribed','guided'): raise ValueError(driver)
    cfg=replace(preset('paper-material'),dt=dt,grid_spacing=spacing,external_tool=driver=='prescribed',
                cache_topology=cache_topology,accumulate_tool_impulses=True)
    if interval <= 0 or not np.isfinite([interval,duration]).all() or duration <= 0:
        raise ValueError('Invalid coupling time')
    if not np.isclose(interval/dt,round(interval/dt),atol=1.e-9,rtol=0.) or interval < dt:
        raise ValueError('Coupling interval must be an integer number of substeps')
    if not np.isclose(duration/interval,round(duration/interval),atol=1.e-9,rtol=0.):
        raise ValueError('Duration must be an integer number of coupling intervals')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    state,_=vertical_state(0.,cfg.blade_half_size[2])
    def get_port(solver): return ExternalToolPort(solver) if cfg.external_tool else None
    def advance(solver,port,command,active):
        if port is not None: return port.advance(command,interval,cutting=active,graph=graph)
        edge=np.asarray(command.position)-[0.,0.,cfg.blade_half_size[2]]
        solver.interval_impulse.zero_();solver.interval_angular_impulse.zero_()
        solver.advance(np.tile(edge,(round(interval/dt),1)),cutting=active,graph=graph)
        # Accumulation uses one fixed origin even while the tool moves.
        world=Reaction(interval,solver.interval_impulse.numpy()[0].astype(float),
                       solver.interval_angular_impulse.numpy()[0].astype(float),np.zeros(3))
        return world.about(solver.bx.numpy()[0])
    warm=Solver(cfg);advance(warm,get_port(warm),state,True);del warm
    s=Solver(cfg);port=get_port(s)
    s.bx.assign([state.position])
    meta={'status':'running','implementation':'independent paper-equation Warp implementation',
          'experiment':'single_vertical_cut_driver_diagnostic','preset':f'paper-material-{driver}',
          'config':asdict(cfg),'device':s.device.name,'warp_version':wp.__version__,
          'coupling_interval_s':interval,'tool_effective_inverse_mass':0. if cfg.external_tool else 1/cfg.blade_mass,
          'driver':driver,'driver_note':'prescribed pose/twist' if cfg.external_tool else
              'existing DexLab guided PD; translational effective mass and local actuator settings',
          'paper_parameters':['young_modulus','poisson_ratio','density','yield_stress',
                              'blade_mass','friction_static','friction_dynamic','cutting_energy','sharpness'],
          'independent_choices':['synthetic DexLab banana geometry','knife geometry','particle sampling',
                                 'material timestep','coupling interval','20 mm/s diagnostic motion',
                                 'PIC/FLIP fraction','contact scales','Hencky plasticity return map'],
          'surface':'not recorded by this diagnostic; linked-volume reconstruction is a separate pipeline',
          'acceptance':'a diagnostic, not reproduction of the author force curve',
          'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in Path(__file__).parent.glob('*.py')}}
    write(out/'manifest.json',meta)
    saved={k:[v] for k,v in s.state().items()}; traces=[]; minimum=1.
    start=time.perf_counter()
    try:
        for i in range(round(duration/interval)):
            command,active=vertical_state(i*interval,cfg.blade_half_size[2])
            r=advance(s,port,command,active)
            traces.append({'time_s':(i+1)*interval,'blade_reaction_n':r.force_n.tolist(),
                           'blade_reaction_nm':r.torque_nm.tolist(),
                           'origin_world_m':r.origin_world_m.tolist(),
                           'target_origin_world_m':list(command.position),
                           'actual_origin_world_m':s.bx.numpy()[0].tolist(),
                           'cutting_enabled':active})
            if (i+1)%max(1,round(.01/interval))==0 or i==round(duration/interval)-1:
                state=s.state(); minimum=min(minimum,float(np.linalg.det(state['deformation']).min()))
                if not all(np.isfinite(v).all() for v in state.values()):
                    raise FloatingPointError('Nonfinite material or tool state')
                for k,v in state.items():saved[k].append(v)
            if (i+1)%max(1,round(.5/interval))==0:
                print(f't={s.step_count*dt:.3f}s wall={time.perf_counter()-start:.2f}s',flush=True)
        wp.synchronize_device(s.device)
        meta.update(status='complete',wall_s_excluding_jit=time.perf_counter()-start)
    except Exception as exc:
        meta.update(status='failed',error=repr(exc),wall_s_excluding_jit=time.perf_counter()-start)
        raise
    finally:
        meta.update(completed_steps=s.step_count,minimum_det_f_sampled=minimum)
        np.savez_compressed(out/'states.npz',**{k:np.asarray(v) for k,v in saved.items()},reference=s.reference)
        write(out/'trace.json',traces);write(out/'manifest.json',meta)
    _,counts=components(s.node_numpy,s.weight_numpy,saved['sides'][-1])
    events=[]
    for when in [0.,1.8,2.,duration]:
        idx=np.argmin(abs(np.asarray(saved['time'])-when))
        _,sizes=components(s.node_numpy,s.weight_numpy,saved['sides'][idx])
        events.append({'time_s':float(saved['time'][idx]),'component_sizes':sizes.tolist()})
    result={'status':'complete','component_sizes':counts.tolist(),'events':events,
            'particle_count':s.n,'sampled_material_mass_kg':s.n*s.mass,
            'final_center_of_mass_m':saved['position'][-1].mean(axis=0).tolist(),
            'final_position_bounds_m':[saved['position'][-1].min(axis=0).tolist(),
                                       saved['position'][-1].max(axis=0).tolist()],
            'final_mean_det_f':float(np.linalg.det(saved['deformation'][-1]).mean()),
            'wall_s':meta['wall_s_excluding_jit'],'realtime_factor':duration/meta['wall_s_excluding_jit'],
            'maximum_interval_force_n':float(np.linalg.norm([v['blade_reaction_n'] for v in traces],axis=1).max()),
            'minimum_det_f_sampled':minimum,'graph_build_s':s.graph_build_s,
            'note':'Controlled driver diagnostic; not a robot, author trajectory, or force-curve validation.'}
    write(out/'result.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--dt',type=float,default=.000025)
    p.add_argument('--interval',type=float,default=.002);p.add_argument('--duration',type=float,default=3.8)
    p.add_argument('--spacing',type=float,default=.006,help='Reference-grid spacing in metres; geometry and material parameters stay fixed')
    p.add_argument('--no-graph',action='store_true')
    p.add_argument('--cache-topology',action='store_true',help='Experimental CUDA conditional topology-map cache')
    p.add_argument('--driver',choices=['prescribed','guided'],default='prescribed')
    a=p.parse_args();single_cut(a.output,a.dt,a.interval,a.duration,not a.no_graph,a.cache_topology,a.driver,a.spacing)
