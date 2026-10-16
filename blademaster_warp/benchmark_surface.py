"""Compare CPU/CUDA surface stages on the same recorded Franka cut.

Reconstructs sequentially from recorded current/past states. No future trajectory
is passed to the surface code. Measures reconstruction plus recording, excludes
physics/rendering and one-time setup/JIT; never labels this full-pipeline speed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time
from types import SimpleNamespace
import numpy as np
from .geometry import points
from .surface_runtime import SurfaceRecorder


def mesh_violations(triangles):
    """A closed oriented surface has each edge twice, in opposite directions."""
    edges=np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]])
    directions=np.where(edges[:,0]<edges[:,1],1,-1)
    _,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    orientations=np.bincount(inverse,weights=directions)
    return dict(non_two_incidence_edges=int(np.count_nonzero(counts!=2)),
                inconsistent_edge_orientations=int(np.count_nonzero(orientations)))


def benchmark(directory, output, cuda_ccd=False):
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(output)
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in Path(__file__).parent.glob('surface*.py')}
    manifest = json.loads((directory/'manifest.json').read_text())
    cfg = SimpleNamespace(**manifest['config'])
    cycle = json.loads((directory/'cycle.json').read_text())
    descend = cycle['phases'].index('descend')
    with np.load(directory/'trajectory.npz') as z:
        x, F, codes = z['particles'], z['deformation'], z['codes']
        times, phase_trace = z['control_tcp'][:, 0], z['cycle_trace']
    frames = json.loads((directory/'surface/frames.json').read_text())
    if len(frames) != len(x):
        raise ValueError('Surface and material recordings must have the same frame count')
    poses = []
    for i, frame in enumerate(frames):
        if abs(frame['time_s']-times[i]) > 1.e-9:
            raise ValueError('Surface and material timestamps do not match')
        with np.load(directory/'surface'/f'{i:05d}.npz') as z:
            poses.append(SimpleNamespace(position=z['tool_position'], orientation_xyzw=z['tool_orientation']))
    active = [int(phase_trace[np.argmin(abs(phase_trace[:, 0]-t)), 1]) == descend for t in times]
    ref, origin, _ = points(cfg.size, cfg.grid_spacing, cfg.shape)
    solver = SimpleNamespace(config=cfg, reference=ref, origin=origin, device='cuda:0')
    result = dict(input_directory=str(directory.resolve()), frame_count=len(x),
                  simulation_duration_s=float(times[-1]-times[0]), platform=platform.platform(),
                  input_sha256=hashlib.sha256((directory/'trajectory.npz').read_bytes()).hexdigest(),
                  scope='surface reconstruction and compressed recording only; excludes setup/JIT, physics, rendering and UI',
                  precision='FP64 embedding on CPU and GPU; recorded material states unchanged',
                  equivalence_tolerance_m=1.e-9, backends={})
    result['cuda_ccd']=bool(cuda_ccd)
    spacing = json.loads((directory/'surface/method.json').read_text())['surface_spacing_m']
    with tempfile.TemporaryDirectory(prefix='blademaster-surface-') as temporary:
        root = Path(temporary)
        for backend in ['cpu', 'cuda']:
            ccd_backend='cuda' if backend=='cuda' and cuda_ccd else 'cpu'
            recorder = SurfaceRecorder(solver, root/backend, spacing, backend, ccd_backend)
            # Warm only the field kernels; do not precompute/skip topology updates.
            recorder.embedding.node_state(x[0], F[0], codes[0], recorder.surface.codes)
            recorder.embedding.material_grid.field(x[0], F[0], codes[0], 0)
            # Compile CCD on disposable coordinates, without touching topology.
            if ccd_backend=='cuda':
                sample=np.array([[0.,0.,0.]])
                edge=np.array([[0.,-.01,0.],[0.,.01,0.]])
                recorder.ccd(sample, sample+[.01,0,0], sample, sample+[.01,0,0], edge, edge)
            frame_wall = []
            for i, t in enumerate(times):
                start = time.perf_counter()
                recorder.update(t, x[i], F[i], codes[i], poses[i], active[i])
                frame_wall.append(time.perf_counter()-start)
            result['backends'][backend] = dict(
                wall_s=float(sum(frame_wall)), phase_s=recorder.timing_s,
                frame_latency_ms_quantiles=dict(zip(['p50', 'p95', 'p99', 'max'],
                                                    (1000*np.percentile(frame_wall, [50, 95, 99, 100])).tolist())),
                frame_wall_s=frame_wall, broken_links=int(recorder.surface.broken.sum()),
                final_triangles=recorder.history[-1]['triangles'])
            print(backend, result['backends'][backend]['wall_s'], recorder.timing_s, flush=True)
        maximum = 0.
        mismatch_frames = []
        reference_maximum = 0.
        reference_mismatch_frames = []
        reference_code_mismatch_frames=[]
        reference_shape_mismatch_frames=[]
        mesh_errors=[]
        original_orientation_errors=[]
        for i in range(len(x)):
            with np.load(root/'cpu'/f'{i:05d}.npz') as a, np.load(root/'cuda'/f'{i:05d}.npz') as b:
                if not np.isfinite(a['vertices']).all() or not np.isfinite(b['vertices']).all():
                    raise FloatingPointError(f'Nonfinite surface at frame {i}')
                if not all(np.array_equal(a[k], b[k]) for k in ['triangles', 'codes']):
                    mismatch_frames.append(i)
                else:
                    maximum = max(maximum, float(np.max(abs(a['vertices']-b['vertices']))))
                for name, mesh in [('cpu',a),('cuda',b)]:
                    violations=mesh_violations(mesh['triangles'])
                    if any(violations.values()):
                        mesh_errors.append(dict(frame=i,backend=name,**violations))
                with np.load(directory/'surface'/f'{i:05d}.npz') as original:
                    if not np.array_equal(a['triangles'],original['triangles']):
                        reference_mismatch_frames.append(i)
                    if not np.array_equal(a['codes'],original['codes']):
                        reference_code_mismatch_frames.append(i)
                    if a['vertices'].shape==original['vertices'].shape:
                        reference_maximum = max(reference_maximum,
                            float(np.max(abs(a['vertices']-original['vertices']))))
                    else:
                        reference_shape_mismatch_frames.append(i)
                    original_orientation_errors.append(mesh_violations(original['triangles']))
        result['comparison'] = dict(topology_mismatch_frames=mismatch_frames,
            maximum_coordinate_difference_m=maximum,
            original_recording_topology_mismatch_frames=reference_mismatch_frames,
            original_recording_code_mismatch_frames=reference_code_mismatch_frames,
            original_recording_vertex_shape_mismatch_frames=reference_shape_mismatch_frames,
            original_recording_maximum_coordinate_difference_m=reference_maximum,
            original_recording_edge_violations=original_orientation_errors,
            current_mesh_violations=mesh_errors,
            original_recording_identical=not reference_mismatch_frames and not reference_code_mismatch_frames
                and not reference_shape_mismatch_frames and reference_maximum<=result['equivalence_tolerance_m'],
            comparison_scope='equivalent compares current CPU and CUDA algorithms strictly; original mesh audit is separate because primal-edge winding replaces the unstable QEF-vertex orientation heuristic',
            cpu_to_cuda_speedup=result['backends']['cpu']['wall_s']/result['backends']['cuda']['wall_s'],
            equivalent=not mismatch_frames and not mesh_errors and maximum <= result['equivalence_tolerance_m'])
    result['source_sha256'] = source_hashes
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
           for p in Path(__file__).parent.glob('surface*.py')}
    if after!=source_hashes:
        raise RuntimeError('Surface source changed during benchmark; rerun before reporting timing')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print({k:v for k,v in result['comparison'].items()
           if k not in ['original_recording_edge_violations','original_recording_topology_mismatch_frames']}, flush=True)
    if not result['comparison']['equivalent']:
        raise RuntimeError('CPU/CUDA surface mismatch; see report, do not enable as an equivalent backend')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--cuda-ccd',action='store_true',help='Also move finite moving-segment detection to CUDA')
    args = parser.parse_args()
    benchmark(args.directory, args.output,args.cuda_ccd)
