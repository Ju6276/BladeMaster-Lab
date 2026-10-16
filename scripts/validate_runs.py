"""Independent saved-state checks. Requires the explicitly pinned DexLab reference.

Connectivity is the reference-grid transfer graph, not the number of colors.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from dexlab.cutting_score import grid_components
from blademaster_warp.geometry import stencil, points


def inspect(path):
    path=Path(path);meta=json.loads((path/'manifest.json').read_text());s=np.load(path/'states.npz')
    cfg=meta['config'];_,origin,cells=points(cfg['size'],cfg['grid_spacing'],cfg['shape'])
    nodes,weights,_=stencil(s['reference'],origin,cfg['grid_spacing'],cells)
    events=[]
    scale=meta.get('trajectory_time_scale',1.)
    times=[0,2*scale,4*scale,5.3*scale,float(s['time'][-1])]
    for t in sorted(set(min(t,float(s['time'][-1])) for t in times)):
        i=int(np.argmin(abs(s['time']-t)))
        labels,sizes=grid_components(nodes,weights,s['sides'][i])
        events.append({'time_s':float(s['time'][i]),'components':len(sizes),'particle_counts':sizes.tolist()})
    labels,sizes=grid_components(nodes,weights,s['sides'][-1])
    before=int(np.argmin(abs(s['time']-5.3*scale)))
    centers=[]
    for i in range(len(sizes)):
        mask=labels==i
        centers.append({'particles':int(sizes[i]),'reference_x_mean_m':float(s['reference'][mask,0].mean()),
            'push_displacement_m':(s['position'][-1,mask].mean(0)-s['position'][before,mask].mean(0)).tolist()})
    result={'status':meta['status'],'topology_events':events,'final_components':centers,
        'min_det_f_sampled':float(np.linalg.det(s['deformation']).min()),
        'max_speed_m_s_sampled':float(np.linalg.norm(s['velocity'],axis=2).max()),
        'minimum_blade_edge_z_m':float(np.min(s['blade_position'][:,0,2]-cfg['blade_half_size'][2]*
            (1-2*np.sum(s['blade_orientation'][:,0,:2]**2,axis=1))))}
    return result


def compare(gpu,cpu):
    g=np.load(Path(gpu)/'states.npz');c=np.load(Path(cpu)/'states.npz')
    np.testing.assert_allclose(g['time'],c['time'],atol=1e-9)
    np.testing.assert_allclose(g['reference'],c['reference'],atol=1e-12)
    diff=g['position']-c['position'];mismatch=np.argwhere(g['sides']!=c['sides'])
    return {'position_rmse_m':float(np.sqrt(np.mean(diff**2))),
            'maximum_coordinate_error_m':float(abs(diff).max()),
            'maximum_particle_distance_error_m':float(np.linalg.norm(diff,axis=2).max()),
            'label_mismatching_frame_particle_count':len(mismatch),
            'label_mismatch_locations':mismatch.tolist(),
            'final_labels_equal':bool(np.array_equal(g['sides'][-1],c['sides'][-1])),
            'final_center_of_mass_error_m':(g['position'][-1].mean(0)-c['position'][-1].mean(0)).tolist(),
            'timing_note':'GPU run excludes JIT; saved CPU run includes JIT and per-step FP64 diagnostics. Not a controlled kernel-only speed ratio.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--cpu');p.add_argument('--output',required=True)
    a=p.parse_args();report={'gpu':inspect(a.gpu)}
    if a.cpu:report['cpu_gpu_comparison']=compare(a.gpu,a.cpu)
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
