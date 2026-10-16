"""Single-variable replay of recorded knife states, NOT a closed-loop robot run.

Compare particle advection with TLMPM Eq. 13 (de Vaucorbeil et al., 2020).
All material/contact/cutting settings and recorded tool inputs are preserved.
The legacy kernel exists only here to reproduce the previously observed error.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import mujoco as mj
import warp as wp
from blademaster_warp import material
from blademaster_warp.solver import Config, Solver
from blademaster_warp.franka import Franka
from blademaster_warp.coupling import ExternalToolPort
from blademaster_warp.check_material_geometry import check


@wp.kernel
def legacy_g2p(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
               slots: wp.array2d(dtype=int), weights: wp.array2d(dtype=float),
               old_v: wp.array(dtype=wp.vec3), new_v: wp.array(dtype=wp.vec3),
               dt: float, flip: float):
    p = wp.tid()
    pic = wp.vec3(0.0)
    dv = wp.vec3(0.0)
    for a in range(8):
        s = slots[p, a]
        w = weights[p, a]
        pic += w * new_v[s]
        dv += w * (new_v[s] - old_v[s])
    v[p] = (1.0 - flip) * pic + flip * (v[p] + dv)
    x[p] += dt * v[p]


@wp.kernel
def paper_g2p(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
              slots: wp.array2d(dtype=int), weights: wp.array2d(dtype=float),
              old_v: wp.array(dtype=wp.vec3), new_v: wp.array(dtype=wp.vec3),
              dt: float, flip: float):
    p = wp.tid()
    pic = wp.vec3(0.0)
    dv = wp.vec3(0.0)
    for a in range(8):
        s = slots[p, a]
        w = weights[p, a]
        pic += w * new_v[s]
        dv += w * (new_v[s] - old_v[s])
    v[p] = (1.0 - flip) * pic + flip * (v[p] + dv)
    x[p] += dt * pic


def run(args):
    source, out = args.source, args.output
    meta = json.loads((source/'manifest.json').read_text())
    data = np.load(source/'trajectory.npz')
    cfg = Config(**meta['config'])
    if meta['mount'] != 'fixed':
        raise ValueError('Recorded replay currently supports the fixed mount only')
    material.g2p = legacy_g2p if args.transport == 'legacy' else paper_g2p
    out.mkdir(parents=True, exist_ok=False)
    (out/'manifest.json').write_text(json.dumps(dict(
        source=str(source.resolve()), source_trajectory_sha256=hashlib.sha256((source/'trajectory.npz').read_bytes()).hexdigest(),
        experiment='recorded tool-state replay, not closed-loop robot dynamics',
        transport=args.transport, config=asdict(cfg),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(material.__file__).parent.glob('*.py')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    interval = float(data['interval'])
    robot = Franka(interval, controller=meta['controller'], attachment='fixed')
    solver = Solver(cfg)
    port = ExternalToolPort(solver)
    rows, positions, codes, Fs, Fps, velocities, times = [], [], [], [], [], [], []
    start = time.perf_counter()
    status, error = 'completed', None
    input_hash = hashlib.sha256()
    try:
        for i, row in enumerate(data['log']):
            # Logs contain the POST-step state. MPM consumed the previous state.
            if i:
                robot.data.qpos[:] = data['log'][i-1, 13:22]
                robot.data.qvel[:] = data['log'][i-1, 22:31]
                mj.mj_forward(robot.model, robot.data)
            state = robot.state()
            input_hash.update(np.r_[state.position,state.orientation_xyzw,state.linear_velocity,
                                   state.angular_velocity,state.mobility.ravel()].tobytes())
            cutting = int(data['cycle_trace'][i,1]) == 1
            reaction = port.advance(state, interval, cutting=cutting)
            rows.append(np.r_[row[0],reaction.force_n,reaction.torque_nm])
            if i%40==0 or i==len(data['log'])-1:
                positions.append(solver.x.numpy().copy()); codes.append(solver.codes.numpy().copy())
                Fs.append(solver.F.numpy().copy()); Fps.append(solver.Fp.numpy().copy())
                velocities.append(solver.v.numpy().copy()); times.append(row[0])
            if (i+1)%1000==0:
                print(f'{args.transport}: t={row[0]:.3f}s wall={time.perf_counter()-start:.1f}s', flush=True)
    except Exception as exc:
        status,error = 'failed',repr(exc)
    np.savez_compressed(out/'states.npz',reference=solver.reference,time=times,
        position=positions,sides=codes,deformation=Fs,plastic_deformation=Fps,velocity=velocities,reaction=rows)
    result = dict(status=status,error=error,wall_s=time.perf_counter()-start,
                  input_sha256=input_hash.hexdigest(),intervals=len(rows),
                  peak_force_n=float(np.linalg.norm(np.asarray(rows)[:,1:4],axis=1).max()))
    if times:
        result['geometry']=check(out)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--transport',choices=['legacy','paper'],required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
