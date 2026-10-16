"""Read-only material-position audit; independent of stored F and display mesh.

Fixed tetrahedra connect nearest reference-lattice samples. They are diagnostic
probes, NOT FEM elements used by the solver. Probes spanning different cut codes
are excluded. Negative signed-volume ratios expose a folded sampled material
map; positive ratios alone do not validate a continuum solution or its forces.
No simulation state, control command, or existing run report is modified.
"""
import argparse
from itertools import permutations
import json
from pathlib import Path
import numpy as np
from .geometry import points


def lattice_tetrahedra(reference, spacing):
    reference = np.asarray(reference, dtype=float)
    origin = reference.min(axis=0)
    lattice = np.rint((reference-origin)/spacing).astype(int)
    if not np.allclose(origin+lattice*spacing, reference, rtol=0, atol=spacing*1.e-4):
        raise ValueError('Expected the original regular particle lattice')
    lookup = {tuple(v): i for i, v in enumerate(lattice)}
    tetrahedra = []
    for cell in lattice:
        for order in permutations(range(3)):
            pos = cell.copy()
            ids = [lookup[tuple(pos)]]
            for axis in order:
                pos[axis] += 1
                ids.append(lookup.get(tuple(pos), -1))
            if min(ids) >= 0:
                tetrahedra.append(ids)
    return np.asarray(tetrahedra, dtype=int).reshape(-1, 4)


def frame_geometry(reference, positions, codes, tetrahedra, deformation):
    histories = codes[tetrahedra]
    same = np.all(histories == histories[:, :1], axis=1)
    tet = tetrahedra[same]
    if not len(tet):
        return dict(probes=0, inverted_probes=0, assigned_inverted_probes=0,
                    minimum_signed_volume_ratio=None, minimum_det_F=float(np.linalg.det(deformation).min()))
    rest = reference[tet]
    current = positions[tet]
    ratio = np.linalg.det(current[:, 1:]-current[:, :1]) / np.linalg.det(rest[:, 1:]-rest[:, :1])
    return dict(probes=len(tet), inverted_probes=int(np.sum(ratio < 0)),
                assigned_inverted_probes=int(np.sum((ratio < 0) & (histories[same, 0] != 0))),
                minimum_signed_volume_ratio=float(ratio.min()),
                signed_volume_ratio_quantiles=np.percentile(ratio, [0, 5, 50, 95, 100]).tolist(),
                minimum_det_F=float(np.linalg.det(deformation).min()))


def check(directory):
    directory = Path(directory)
    cfg = json.loads((directory/'manifest.json').read_text())['config']
    path = directory/'trajectory.npz'
    if path.exists():
        z = np.load(path)
        x, codes, times = z['particles'], z['codes'], z['energy'][:, 0]
        reference, _, _ = points(cfg['size'], cfg['grid_spacing'], cfg['shape'])
        reference_source = 'analytic sampling from saved config; particle order checked against first observation'
        if x.shape[1:] != reference.shape or not np.allclose(x[0], reference, rtol=0, atol=cfg['grid_spacing']/100):
            raise ValueError('First observation does not match reconstructed particle inventory')
    else:
        z = np.load(directory/'states.npz')
        x, codes, times = z['position'], z['sides'], z['time']
        reference = z['reference'] if 'reference' in z else x[0]
        reference_source = 'saved reference' if 'reference' in z else 'first observation (not exact reference)'
    tet = lattice_tetrahedra(reference, cfg['grid_spacing']/2)
    if not len(tet):
        raise ValueError('No complete reference cells; no geometric validation possible')
    rows = [dict(time_s=float(t), **frame_geometry(reference, p, c, tet, f))
            for t, p, c, f in zip(times, x, codes, z['deformation'], strict=True)]
    affected = [r for r in rows if r['inverted_probes']]
    result = dict(status='sampled_material_map_folded' if affected else 'no_fold_detected',
                  reference_source=reference_source, probe_definition=__doc__,
                  first_fold_time_s=affected[0]['time_s'] if affected else None,
                  max_inverted_probes=max(r['inverted_probes'] for r in rows),
                  final=rows[-1], observations=rows,
                  limitations='Finite point-sample diagnostic, not a full self-intersection or force-convergence test. Identical codes are a conservative subset of compatible material; cut fronts require separate review.')
    (directory/'material_geometry_validation.json').write_text(json.dumps(result, indent=2))
    return {k: v for k, v in result.items() if k != 'observations'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps(check(parser.parse_args().directory), indent=2))
