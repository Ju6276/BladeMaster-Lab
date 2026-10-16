"""Reproducible articulated coupling diagnostic; not a task success benchmark."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .franka import Franka, HALF, BEVEL, SOURCE
from .geometry import BANANA_DEMO_SIZE_M


def target_at(t, initial):
    # Explicit diagnostic reference only. Geometry is not pre-split.
    z = np.interp(t, [0, .3, 2.3, 2.6, 4.6], [0, 0, -.04, -.04, .01])
    return initial + np.array([0., 0., z])


def material_config(args):
    from .solver import Config
    from .presets import banana_config
    # Shared banana material and assets. Only the rigid-body integration owner
    # changes; material/contact kernels and their numerical settings do not.
    cfg=(banana_config() if args.material=='paper-banana' else
         Config(size=(.06,.03,.03),grid_spacing=.0075,dt=.00025,
                density=1100.,young_modulus=6000.,poisson_ratio=.4,
                blade_mass=.1,blade_half_size=HALF,blade_bevel_height=BEVEL))
    overrides={name:getattr(args,name) for name in ('dt','cutting_energy') if getattr(args,name) is not None}
    if args.spacing is not None:
        overrides['grid_spacing']=args.spacing
    return replace(cfg,external_tool=True,grid_storage=getattr(args,'grid_storage',cfg.grid_storage),
                   cache_topology=(getattr(args,'grid_storage',cfg.grid_storage)=='compact'),**overrides)


def run(args, observer=None):
    import warp as wp
    from .solver import Solver
    from .coupling import ExternalToolPort
    from .topology import components
    from .cut_cycle import CutCycle, CycleSettings, evaluate_cut
    record = bool(getattr(args,'record',False))
    if (record or args.video) and not args.output:
        raise ValueError('--record/--video requires --output')
    if args.video and not record:
        raise ValueError('--video requires --record; live display does not use a video file')
    out = Path(args.output) if args.output else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
    if not np.isfinite([args.duration,args.interval]).all() or min(args.duration,args.interval) <= 0:
        raise ValueError('Duration and timesteps must be finite and positive')
    robot = Franka(args.interval, args.stiffness, controller=args.controller, attachment=args.attachment, grip_force_per_finger=args.grip_force_per_finger)
    cycle = None
    if args.motion == 'feedback':
        cycle = CutCycle(robot.position, robot.material_origin[2], HALF[2],
            CycleSettings(descent_speed_m_s=args.descent_speed,
                          withdrawal_speed_m_s=args.withdrawal_speed,
                          position_tolerance_m=args.goal_tolerance))
    banana = args.material == 'paper-banana'
    cfg = material_config(args)
    provenance = {
        'robot': json.loads((SOURCE.parent.parent/'SOURCE.json').read_text()),
        'controller': args.controller,
        'controller_source': json.loads((SOURCE.parents[2]/('libfranka/SOURCE.json' if args.controller == 'libfranka' else 'serl_controller/manifest.json')).read_text()),
        'controller_parameters': ({'stiffness':robot.k.tolist(),'damping':robot.damping.tolist()} if args.controller == 'libfranka' else
            {'pos_gains':[100.,100.,100.],'ori_gains':[2000.,2000.,200.],'damping_ratio':1.,
             'nullspace_stiffness':.5,'max_pos_acceleration':None,'max_ori_acceleration':None}),
        'controller_modification': 'qfrc_bias includes gravity in MuJoCo; preserve upstream actuator force limits',
        'mount': robot.attachment,
        'knife_geometry': {'source':'shared DexLab synthetic demo wedge; not author CAD',
                           'half_size_m':list(HALF),'bevel_height_m':BEVEL,
                           'fixture':'local MuJoCo attachment, not part of the shared blade'},
        'grip_force_per_finger_n': robot.grip_force_per_finger if args.attachment=='grasp' else None,
        'geometry': ({'source':'DexLab b955d85 synthetic banana geometry; original 120 mm length, shared with standalone demo',
                      'domain_size_m':list(BANANA_DEMO_SIZE_M), 'author_asset':False,
                      'surface':'online linked-volume reconstruction; no preassigned display seams'} if banana else
                     {'source':'synthetic rectangular block', 'author_asset':False}),
        'material': ('paper banana E=400kPa, nu=.45, yield=30kPa, rho=1100, Gc=700; independent trajectory and contact numerics'
            if banana else 'elastic block: paper Fig.3 E=6kPa, nu=.4, rho=1100; geometry/numerics/contact/cutting energy are declared independent test settings'),
        'cutting_reference': ('measured knife pose gates descent/withdrawal; independent robot adapter, not author controller'
                             if cycle else 'straight timed descent/withdrawal diagnostic; no pre-split particle topology'),
        'cycle_settings': asdict(cycle.settings) if cycle else None,
        'cycle_observations': 'actual tool pose only; topology is evaluated separately and never drives commands' if cycle else None,
        'material_origin_world_m': robot.material_origin.tolist(),
        'config': asdict(cfg), 'args': vars(args),
        'versions': {'mujoco': __import__('mujoco').__version__, 'warp': wp.__version__},
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        'surface': 'online linked-volume reference at 25 Hz' if args.surface else 'disabled',
        'surface_embedding_backend':getattr(args,'surface_embedding','cpu') if args.surface else None,
        'surface_ccd_backend':getattr(args,'surface_ccd','cpu') if args.surface else None,
        'limitations': ['single tool; cut event index explicitly set to 0',
            'default replay shows physical particles; optional surface has separate data and renderer',
            'synthetic handle and blade CAD; fixed mount or free frictional grasp as declared in mount',
            'MPM ground is an infinite plane; finite visible/MuJoCo board',
            'not a physical force calibration or full BladeMaster reproduction']}
    if out is not None:
        (out/'manifest.json').write_text(json.dumps(provenance, indent=2))
        if record:
            (out/'scene.xml').write_text(robot.xml)
    s = port = None
    warm = time.perf_counter()
    if not args.no_material:
        # Warm separate disposable state. Reset all material history afterward.
        disposable = Solver(cfg)
        ExternalToolPort(disposable).advance(robot.state(), args.interval)
        del disposable
        s = Solver(cfg)
        port = ExternalToolPort(s)
        # Capture schedules without advancing physics. First graph construction
        # remains in the measured loop (JIT already warmed).
    warm_s = time.perf_counter()-warm
    from .diagnostics import Metrics
    meter=Metrics(s) if s else None
    surface=None
    if args.surface and s:
        from .surface_runtime import SurfaceRecorder
        surface=SurfaceRecorder(s,out/'surface' if record else None,args.surface_spacing,getattr(args,'surface_embedding','cpu'),getattr(args,'surface_ccd','cpu'))
    logs, frames, particles, codes, energies, deformations, controls = [], [], [], [], [], [], []
    velocities, plastic_deformations = [], []
    task_observations, cycle_trace, edge_heights = [], [], []
    phase_time = dict(robot_state=0., material=0., robot_step=0., observation=0.)
    status, exception = 'completed', None
    if observer:
        observer.initialize(robot,s,surface)
        if surface:
            surface.wall_s = 0.
            surface.timing_s = dict.fromkeys(surface.timing_s,0.)
    paused_s = 0.
    begin = time.perf_counter()
    count = int(round(args.duration/args.interval))
    stride = max(1, round(.04/args.interval))
    try:
        for i in range(count):
            if observer:
                paused_s += observer.before_step()
                if observer.stop_requested:
                    status = 'stopped'
                    break
            t = i*args.interval
            tool_target = cycle.update(t, robot.position, robot.edge_points) if cycle else None
            if cycle and cycle.done:
                break
            # Free-grasp mode uses the measured tool-to-TCP offset. Fixed
            # mounting has zero offset. No body pose is assigned by control.
            target = (tool_target+robot.control_position-robot.position) if cycle else target_at(t, robot.initial_position)
            tick = time.perf_counter()
            state = robot.state() if port else None
            phase_time['robot_state'] += time.perf_counter()-tick
            tick = time.perf_counter()
            wrench = np.zeros(6)
            if port:
                cutting = not args.no_cutting and (cycle is None or cycle.phase == 'descend')
                reaction = port.advance(state, args.interval, cutting=cutting)
                wrench = np.r_[reaction.force_n, reaction.torque_nm]
            phase_time['material'] += time.perf_counter()-tick
            tick = time.perf_counter()
            requested, applied = robot.step(target, wrench)
            phase_time['robot_step'] += time.perf_counter()-tick
            tick = time.perf_counter()
            logs.append(np.r_[robot.data.time, target, robot.position, wrench,
                robot.data.qpos, robot.data.qvel, requested, robot.data.ctrl[:7], applied])
            heights = robot.edge_points[:,2]-robot.material_origin[2]
            edge_heights.append(heights)
            if cycle:
                cycle_trace.append([robot.data.time, cycle.phases.index(cycle.phase),
                    *tool_target, *heights, robot.board_normal_force()])
            if i % stride == 0 or i == count-1:
                frames.append(robot.data.qpos.copy())
                controls.append(np.r_[robot.data.time,robot.control_position,robot.position-robot.control_position])
                if s:
                    particles.append(s.x.numpy().copy())
                    codes.append(s.codes.numpy().copy())
                    deformations.append(s.F.numpy().copy())
                    if record:
                        velocities.append(s.v.numpy().copy())
                        plastic_deformations.append(s.Fp.numpy().copy())
                    if record:
                        energies.append(np.r_[robot.data.time,meter.read(),s.work.numpy()[0]])
                    if cycle:
                        _, sizes = components(s.node_numpy,s.weight_numpy,codes[-1])
                        task_observations.append(dict(time_s=float(robot.data.time), phase=cycle.phase,
                            component_sizes=sizes.tolist(),
                            knife_clear_of_material=bool(min(heights)>particles[-1][:,2].max()+s.radius)))
                    if surface:
                        surface.update(robot.data.time,particles[-1],deformations[-1],codes[-1],robot.state(),cutting)
                if observer:
                    observer.publish(robot,s,surface,particles[-1] if s else None,codes[-1] if s else None,
                        time.perf_counter()-begin-paused_s,cycle.phase if cycle else 'timed',wrench)
                if not record:
                    # Only current visualization state is retained. No particle,
                    # deformation, or image history accumulates in live mode.
                    frames.clear(); controls.clear(); particles.clear(); codes.clear(); deformations.clear()
            phase_time['observation'] += time.perf_counter()-tick
            if (i+1)%max(1,round(.5/args.interval))==0:
                print(f't={robot.data.time:.3f}s wall={time.perf_counter()-begin:.2f}s'+
                      (f' phase={cycle.phase} edge_mm={min(heights)*1000:.3f}' if cycle else ''),flush=True)
    except Exception as error:
        status, exception = 'failed', repr(error)
    wall_s = time.perf_counter()-begin-paused_s
    log = np.array(logs)
    if record:
        np.savez_compressed(out/'trajectory.npz', log=log, qpos=np.array(frames),
            particles=np.array(particles), codes=np.array(codes), origin=robot.material_origin,
            deformation=np.array(deformations),plastic_deformation=np.array(plastic_deformations),
            velocity=np.array(velocities),energy=np.array(energies),control_tcp=np.array(controls),
            cycle_trace=np.array(cycle_trace),edge_heights=np.array(edge_heights),
            radius=s.radius if s else .001, interval=args.interval, frame_stride=stride)
        if cycle:
            (out/'cycle.json').write_text(json.dumps(dict(events=cycle.events,observations=task_observations,
                trace_columns=['time_s','phase_index','tool_target_x','tool_target_y','tool_target_z',
                               'edge_end_1_height_m','edge_end_2_height_m','board_normal_force_n'],
                phases=cycle.phases),indent=2))
        (out/'energy_columns.json').write_text(json.dumps(['time_s']+Metrics.columns+['directional_contact_work_J']))
    nq,nv=robot.model.nq,robot.model.nv
    qend=13+nq;vend=qend+nv;requested_start=vend;commanded_start=vend+7
    columns=dict(time=0,target=[1,4],tool_position=[4,7],wrench=[7,13],
        qpos=[13,qend],qvel=[qend,vend],requested_arm_torque=[vend,vend+7],
        commanded_arm_torque=[vend+7,vend+14],material_generalized_force=[vend+14,vend+14+nv])
    if record:
        (out/'columns.json').write_text(json.dumps(columns,indent=2))
    report = dict(status=status, exception=exception, sim_s=robot.data.time,
        wall_s=wall_s, realtime_ratio=robot.data.time/wall_s, warmup_s=warm_s,
        recording=record, display='live WebGL' if observer else 'none', pause_wall_s=paused_s,
        live_setup_s=getattr(observer,'setup_s',0.),
        timing_scope='physics, observation and state publication; excludes JIT, viewer initialization, user pause and client rendering',
        time_s=phase_time, material_present=not args.no_material,
        warnings=robot.data.warning.number.tolist(),surface_wall_s=surface.wall_s if surface else 0.,
        surface_phase_s=surface.timing_s if surface else {})
    if cycle:
        report.update(motion_status='completed' if cycle.done else 'timeout' if status=='completed' else 'failed',
                      final_phase=cycle.phase,
                      task_evaluation=evaluate_cut(task_observations, cycle.done and status=='completed'))
    if len(log):
        report.update(peak_material_force_n=float(np.linalg.norm(log[:,7:10],axis=1).max()),
            peak_material_torque_nm=float(np.linalg.norm(log[:,10:13],axis=1).max()),
            max_tracking_error_m=float(np.linalg.norm(log[:,1:4]-log[:,4:7],axis=1).max()),
            saturated_steps=int(np.any(np.abs(log[:,requested_start:requested_start+7]-log[:,commanded_start:commanded_start+7])>1.e-10,axis=1).sum()),
            min_edge_height_m=float(np.min(edge_heights)),
            arm_joint_motion_rad=(np.ptp(log[:,13:20],axis=0)).tolist())
    if s:
        labels, counts = components(s.node_numpy, s.weight_numpy, s.codes.numpy())
        report.update(components=counts.tolist(), min_det_F=float(np.linalg.det(s.F.numpy()).min()),
            gpu_errors=s.errors.numpy().tolist(), particle_count=s.n)
        if record:
            np.save(out/'component_labels.npy',labels)
        if s.compact:
            report['active_grid_nodes'] = int(s.active_count.numpy()[0])
            report['lookup_capacity'] = s.capacity
    if out is not None:
        (out/'report.json').write_text(json.dumps(report,indent=2))
    if observer:
        observer.finish(report)
    print(json.dumps(report,indent=2),flush=True)
    if args.video and len(frames):
        if surface:
            from .render_franka import render as render_surface_robot
            render_surface_robot(out)
        else:
            render(out)
    if status == 'failed':
        raise RuntimeError(exception)
    return report


def render(out):
    import mujoco as mj
    import imageio.v2 as imageio
    import xml.etree.ElementTree as ET
    from PIL import Image, ImageDraw
    out = Path(out)
    z = np.load(out/'trajectory.npz')
    report = json.loads((out/'report.json').read_text())
    outcome = report.get('task_evaluation', {}).get('success')
    result_label = 'diagnostic only' if outcome is None else 'cycle complete' if outcome else 'cycle incomplete'
    audit_path = out/'material_geometry_validation.json'
    folded = audit_path.exists() and json.loads(audit_path.read_text())['status']=='sampled_material_map_folded'
    quality = 'MATERIAL GEOMETRY FOLDED (final audit)' if folded else 'Physics not validated'
    root = ET.parse(out/'scene.xml').getroot()
    root.find('compiler').set('meshdir',str(SOURCE.parent/'assets'))
    m = mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    d = mj.MjData(m)
    renderer = mj.Renderer(m, height=540, width=720, max_geom=20000)
    camera = mj.MjvCamera()
    camera.lookat[:] = [.27, 0, .4]
    camera.distance, camera.azimuth, camera.elevation = 1.6, 135, -22
    close = mj.MjvCamera()
    close.lookat[:] = z['origin']+[0,0,.035]
    close.distance, close.azimuth, close.elevation = .27, 135, -20
    option = mj.MjvOption()
    option.geomgroup[3] = 0
    palette = np.array([[.96,.72,.22,1],[.35,.7,.9,1],[.7,.45,.8,1]])
    fps = 1/(float(z['interval'])*int(z['frame_stride']))
    with imageio.get_writer(out/'replay.mp4', fps=fps, macro_block_size=2) as writer:
        for i,q in enumerate(z['qpos']):
            d.qpos[:] = q
            mj.mj_forward(m,d)
            panels=[]
            for cam in (camera, close):
                renderer.update_scene(d,camera=cam,scene_option=option)
                if len(z['particles']):
                    for p,code in zip(z['particles'][i]+z['origin'],z['codes'][i]):
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        mj.mjv_initGeom(geom,mj.mjtGeom.mjGEOM_SPHERE,
                            np.full(3,float(z['radius'])*.85),p,np.eye(3).ravel(),palette[int(code)%3])
                        renderer.scene.ngeom += 1
                panels.append(renderer.render().copy())
            frame=Image.fromarray(np.concatenate(panels,axis=1))
            draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,1440,42),fill=(18,24,30))
            time_s=float(z['control_tcp'][i,0]) if 'control_tcp' in z else i/fps
            phase=''
            if 'cycle_trace' in z and z['cycle_trace'].size:
                from .cut_cycle import CutCycle
                idx=np.argmin(abs(z['cycle_trace'][:,0]-time_s))
                phase=' | '+CutCycle.phases[int(z['cycle_trace'][idx,1])]
            draw.text((14,8),f'Franka torque dynamics + GPU TLMPM | t={time_s:.2f}s{phase} | final: {result_label}',fill='white')
            draw.text((14,23),f'{quality} | Actual particles; colors = side codes, not piece count.',fill=(255,150,130) if folded else 'white')
            writer.append_data(np.asarray(frame))
            if i in (0,len(z['qpos'])//2,len(z['qpos'])-1):
                frame.save(out/f'frame_{i:03d}.png')
    renderer.close()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',help='Optional new directory for configuration and summary')
    p.add_argument('--record',action='store_true',help='Also persist raw trajectory and surface snapshots for offline analysis')
    p.add_argument('--live',action='store_true',help='Display current physics in the browser; no offscreen renderer or replay')
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--grid-storage',choices=['compact','hash'],default='compact',help='Dense active grid with unchanged-topology cache, or original hash baseline')
    p.add_argument('--interval',type=float,default=.001)
    p.add_argument('--material',choices=['elastic-block','paper-banana'],default='elastic-block',
                   help='paper-banana selects published material values, with the original 120 mm synthetic demo geometry; not author assets/controller')
    p.add_argument('--dt',type=float,default=None,help='Defaults: elastic block 250 us; paper banana 25 us')
    p.add_argument('--spacing',type=float,default=None)
    p.add_argument('--duration',type=float,default=4.6)
    p.add_argument('--motion',choices=['timed','feedback'],default='timed',help='Feedback gates phases using actual knife pose; independent robot adapter')
    p.add_argument('--descent-speed',type=float,default=.02,help='Feedback task reference speed, m/s')
    p.add_argument('--withdrawal-speed',type=float,default=.025,help='Feedback task reference speed, m/s')
    p.add_argument('--goal-tolerance',type=float,default=.0005,help='Feedback endpoint tolerance, m; applies to both blade-edge ends')
    p.add_argument('--stiffness',type=float,default=150.)
    p.add_argument('--controller',choices=['libfranka','serl'],default='libfranka')
    p.add_argument('--attachment',choices=['fixed','grasp'],default='fixed')
    p.add_argument('--grip-force-per-finger',type=float,default=10.,help='Grasp test actuator command in N per finger, not a measured contact force')
    p.add_argument('--cutting-energy',type=float,default=None)
    p.add_argument('--no-material',action='store_true')
    p.add_argument('--no-cutting',action='store_true')
    p.add_argument('--video',action='store_true')
    p.add_argument('--surface',action='store_true',help='Online reconstruction, included in elapsed time')
    p.add_argument('--surface-embedding',choices=['cpu','cuda'],default='cpu',
                   help='Compatible nodal field backend only; topology/QEF/triangles remain CPU')
    p.add_argument('--surface-ccd',choices=['cpu','cuda'],default='cpu',
                   help='Finite moving-segment intersection backend; Hermite/QEF/triangles remain CPU')
    p.add_argument('--surface-spacing',type=float,default=None,
                   help='Surface-only reference spacing in m; default half the MPM spacing. Does not change physics')
    return p


def main():
    args=parser().parse_args()
    if args.live:
        if not args.surface:
            raise ValueError('--live currently requires --surface')
        from .live import serve
        serve(args,run)
    else:
        run(args)


if __name__ == '__main__':
    main()
