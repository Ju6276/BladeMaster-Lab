"""Reconstruct a Franka surface from saved observations without rerunning physics.

Uses only the same current/past particle states, F, codes and actual blade poses
available to the online recorder. The original states and surface stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from .geometry import points
from .surface_runtime import SurfaceRecorder
from .provenance import git_revision


def rebuild(directory, output, spacing=None):
    directory,output=Path(directory),Path(output)
    meta=json.loads((directory/'manifest.json').read_text())
    state_path=directory/'trajectory.npz'
    before=hashlib.sha256(state_path.read_bytes()).hexdigest()
    frames=json.loads((directory/'surface/frames.json').read_text())
    ref,origin,_=points(meta['config']['size'],meta['config']['grid_spacing'],meta['config']['shape'])
    source=SimpleNamespace(config=SimpleNamespace(**meta['config']),reference=ref,origin=origin)
    recorder=SurfaceRecorder(source,output,spacing)
    begin=time.perf_counter()
    with np.load(state_path) as z:
        if len(frames)!=len(z['particles']):
            raise ValueError('Surface observations and saved physics must have identical sampling')
        for i,info in enumerate(frames):
            t=info['time_s']
            if abs(t-z['energy'][i,0])>1.e-5:
                raise ValueError('Surface/physics timestamps do not match')
            with np.load(directory/'surface'/f'{i:05d}.npz') as old:
                tool=SimpleNamespace(position=old['tool_position'].copy(),orientation_xyzw=old['tool_orientation'].copy())
            cutting=not meta['args'].get('no_cutting',False)
            if z['cycle_trace'].size:
                trace=z['cycle_trace']
                cutting=cutting and int(trace[np.argmin(abs(trace[:,0]-t)),1])==1
            recorder.update(t,z['particles'][i],z['deformation'][i],z['codes'][i],tool,cutting,0)
            if i%25==0:
                print(f'reconstructed {i+1}/{len(frames)} frames, wall={time.perf_counter()-begin:.1f}s',flush=True)
    after=hashlib.sha256(state_path.read_bytes()).hexdigest()
    if before!=after:
        raise RuntimeError('Physical state changed during surface reconstruction')
    result=dict(source=str(directory.resolve()),kind='offline replay of online surface observations',
                trajectory_sha256=before,physics_unchanged=True,frames=len(frames),wall_s=time.perf_counter()-begin,
                revision=git_revision())
    (output/'reconstruction.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--output',type=Path,required=True,help='New surface directory; must not already exist')
    p.add_argument('--spacing',type=float,default=None,help='Surface spacing in m; default half the material grid spacing')
    args=p.parse_args()
    print(json.dumps(rebuild(args.directory,args.output,args.spacing),indent=2))
