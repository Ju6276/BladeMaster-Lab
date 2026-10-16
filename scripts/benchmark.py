"""Warm CPU/GPU TLMPM timing, with identical work and one final synchronization.

FP64 Numba reference vs FP32 Warp kernels; excludes contact, rendering, disk I/O,
and compilation. This is an implementation comparison, not hardware-only speedup.
"""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import warp as wp
from dexlab.cutting import material_step
from blademaster_warp.solver import Solver
from blademaster_warp.run import preset


def run(steps=2000):
    cfg=preset('dexlab-baseline');s=Solver(cfg)
    # Four assigned history values for topology work, not a cutting demo or four-fragment claim.
    codes=(np.searchsorted([-.03,0.,.03],s.reference[:,0])+1).astype(np.int32)
    s.codes.assign(codes)
    vel=(.01*s.reference/.1).astype(np.float32);s.v.assign(vel)
    impulse=np.zeros_like(vel);impulse[:,2]=-9.81*s.mass*cfg.dt;s.impulses.assign(impulse)
    x=s.x.numpy().astype(float);v=vel.astype(float);f=s.F.numpy().astype(float)
    nodes=s.nodes.numpy().astype(np.int64);w=s.weights.numpy().astype(float);g=s.gradients.numpy().astype(float)
    args=(codes.astype(np.int64),nodes,w,g,impulse.astype(float),s.mass,s.volume,s.mu,s.lam,cfg.dt,cfg.flip_fraction,s.node_count,0.)
    material_step(x.copy(),v.copy(),f.copy(),*args)
    warm=Solver(cfg);warm.material_step();warm.state();del warm
    start=time.perf_counter()
    for _ in range(steps):material_step(x,v,f,*args)
    cpu=time.perf_counter()-start
    wp.synchronize_device(s.device);start=time.perf_counter()
    for _ in range(steps):s.material_step()
    wp.synchronize_device(s.device);gpu=time.perf_counter()-start
    state=s.state()
    return {'particles':s.n,'steps':steps,'scope':__doc__,'cpu_s':cpu,'gpu_s':gpu,
        'ratio_cpu_over_gpu':cpu/gpu,'cpu_precision':'float64','gpu_precision':'float32',
        'final_position_rmse_m':float(np.sqrt(np.mean((state['position']-x)**2))),
        'config':asdict(cfg)}


if __name__=='__main__':
    r=run();Path('runs').mkdir(exist_ok=True);Path('runs/material_benchmark.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
