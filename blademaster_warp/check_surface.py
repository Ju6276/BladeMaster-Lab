"""Read-only surface checks: closure alone does not establish cut separation."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def check(directory,surface_directory=None):
    directory=Path(directory)
    surface_directory=Path(surface_directory) if surface_directory else directory/'surface'
    rows=[]
    for path in sorted(surface_directory.glob('*.npz')):
        with np.load(path) as mesh:
            v,t=mesh['vertices'],mesh['triangles']
            edges=np.concatenate([t[:,[0,1]],t[:,[1,2]],t[:,[2,0]]])
            _,incidence=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
            graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(v),len(v)))
            count,_=connected_components(graph,directed=False)
            rows.append(dict(frame=path.name,mesh_components=count,
                boundary_edges=int((incidence==1).sum()),nonmanifold_edges=int((incidence>2).sum()),
                finite_vertices=bool(np.isfinite(v).all())))
    if not rows:raise ValueError('No surface snapshots')
    report_path=directory/'report.json'
    report=json.loads((report_path if report_path.exists() else directory/'manifest.json').read_text())
    mechanical=len(report['components'])
    result=dict(frames=len(rows),all_finite=all(r['finite_vertices'] for r in rows),
        max_boundary_edges=max(r['boundary_edges'] for r in rows),
        max_nonmanifold_edges=max(r['nonmanifold_edges'] for r in rows),
        final_mesh_components=rows[-1]['mesh_components'],final_mechanical_components=mechanical,
        final_component_counts_match=rows[-1]['mesh_components']==mechanical,
        limitations='Checks topology counts and edge incidence, not geometric accuracy, normals, self-intersection or force convergence.',
        observations=rows)
    destination=(directory/'surface_validation.json' if surface_directory==directory/'surface' else surface_directory/'validation.json')
    destination.write_text(json.dumps(result,indent=2))
    return {k:v for k,v in result.items() if k!='observations'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--surface-directory',type=Path)
    args=p.parse_args()
    print(json.dumps(check(args.directory,args.surface_directory),indent=2))
