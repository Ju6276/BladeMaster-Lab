"""Read-only comparison; topology, force, tracking and sampled energy are separate."""
import argparse,json
from pathlib import Path
import numpy as np


def analyze(root):
    names=['paper_banana_25us','corrected_banana_25us','corrected_banana_12us',
           'corrected_banana_half_interval','corrected_banana_refined','friction_grasp']
    rows=[]
    for name in names:
        p=root/name
        if not (p/'report.json').exists():continue
        r=json.loads((p/'report.json').read_text());z=np.load(p/'trajectory.npz')
        x=z['particles'];interval=float(z['interval']);stride=int(z['frame_stride'])
        frame_times=np.arange(len(x))*interval*stride+interval
        frame_times[-1]=r['sim_s']
        speeds=np.linalg.norm(np.diff(x,axis=0)/np.diff(frame_times)[:,None,None],axis=-1)
        row=dict(name=name,status=r['status'],particles=r['particle_count'],components=r['components'],
            peak_force_N=r['peak_material_force_n'],minimum_edge_height_m=r['min_edge_height_m'],
            wall_s=r['wall_s'],maximum_frame_derived_speed_m_s=float(speeds.max()),
            maximum_height_m=float(x[:,:,2].max()),
            final_COM_displacement_m=(x[-1].mean(0)-x[0].mean(0)).tolist())
        if 'energy' in z:
            e=z['energy'];row.update(sampled_max_speed_m_s=float(e[:,4].max()),
                peak_sampled_kinetic_J=float(e[:,1].max()),peak_sampled_elastic_J=float(e[:,3].max()))
        rows.append(row)
    return {'rows':rows,'limitations':[
        'The old 464-particle banana was cropped by its sampling domain; not a complete banana geometry.',
        'Sampled velocity is not a maximum over all material substeps.',
        'Elastic energy is a fixed-corotated diagnostic; plastic dissipation and full work balance are not yet logged.',
        'Corrected banana tests have not passed force/spatial/temporal convergence.'],
        'conclusion':'No claim that the observed kick is validated physical behavior or fully fixed.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(analyze(a.root),indent=2))
