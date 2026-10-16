"""Inspect saved optimization runs without inferring fracture from image colors."""
import argparse
import json
from pathlib import Path
import numpy as np
from dexlab.cutting_score import grid_components
from blademaster_warp.geometry import points,stencil


def inspect(path):
    meta=json.loads((path/'manifest.json').read_text())
    s=np.load(path/'states.npz');cfg=meta['config'];times=s['time']
    _,origin,cells=points(cfg['size'],cfg['grid_spacing'],cfg['shape'])
    nodes,weights,_=stencil(s['reference'],origin,cfg['grid_spacing'],cells)
    scale=meta.get('trajectory_time_scale',1.)
    final_labels,final_sizes=grid_components(nodes,weights,s['sides'][-1])
    result={'result':json.loads((path/'result.json').read_text()),
            'particles':len(s['reference']),'dt':cfg['dt'],'grid_spacing':cfg['grid_spacing'],
            'final_components':final_sizes.tolist(),'cut_checks':[]}
    if 'mode' not in meta:
        edge=s['blade_position'][:,0,2]-cfg['blade_half_size'][2]
        for cut in range(3):
            begin,end=(cut*2+.4)*scale,(cut*2+1.3)*scale
            if times[-1]<end-1e-6:break
            i=int(np.argmin(abs(times-end)));mask=(times>=begin-1e-8)&(times<=end+1e-8)
            labels,sizes=grid_components(nodes,weights,s['sides'][i])
            result['cut_checks'].append({'cut':cut+1,'time_s':float(times[i]),
                'components':sizes.tolist(),'edge_at_withdrawal_mm':float(edge[i]*1000),
                'lowest_edge_during_cut_mm':float(edge[mask].min()*1000)})
    else:
        commands=json.loads((path/'commands.json').read_text())
        active=[i for i,c in enumerate(commands) if c['seq']>=0 and not c['stale']]
        result['received_unique_sequences']=len(set(c['seq'] for c in commands if c['seq']>=0))
        result['actual_blade_rotation_range_deg']=float(np.degrees(2*np.arccos(np.clip(abs(s['blade_orientation'][:,0,3]),0,1))).max())
        if active:
            result['fresh_input_interval_s']=[commands[active[0]]['time_s'],commands[active[-1]]['time_s']]
    return result


def compare(a,b):
    x=np.load(a/'states.npz');y=np.load(b/'states.npz')
    n=min(len(x['time']),len(y['time']))
    np.testing.assert_allclose(x['reference'],y['reference'],atol=1e-12)
    np.testing.assert_allclose(x['time'][:n],y['time'][:n],atol=1e-9)
    delta=x['position'][:n]-y['position'][:n]
    mismatch=x['sides'][:n]!=y['sides'][:n]
    per_frame=np.max(np.linalg.norm(delta,axis=2),axis=1)
    return {'compared_until_s':float(x['time'][n-1]),
            'position_rmse_m':float(np.sqrt(np.mean(delta**2))),
            'max_particle_error_m':float(per_frame.max()),
            'mismatched_frame_particles':int(mismatch.sum()),
            'last_compared_labels_equal':bool(np.array_equal(x['sides'][n-1],y['sides'][n-1])),
            'first_frame_over_1um_s':next((float(x['time'][i]) for i,e in enumerate(per_frame) if e>1e-6),None)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs');a=p.parse_args()
    root=Path(a.runs);out=root/'optimization';results={}
    for path in sorted(out.iterdir()):
        if path.is_dir() and (path/'result.json').exists():
            results[path.name]=inspect(path)
    comparisons={}
    pairs=[(root/'banana_gpu_v1',out/'banana_graph_v2'),
           (root/'paper_material_v1',out/'paper_dt25'),
           (out/'paper_eager_first_cut',out/'paper_graph_first_cut'),
           (out/'paper_graph_first_cut',out/'paper_graph_first_cut_repeat')]
    for a,b in pairs:
        if (a/'states.npz').exists() and (b/'states.npz').exists():
            comparisons[f'{a.name}__{b.name}']=compare(a,b)
    report={'runs':results,'comparisons':comparisons}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
