"""Reproducible numerical-sensitivity audit, without changing material parameters.

The prescribed path isolates the material solver from robot tracking errors.
This is not a comparison against measured forces, nor an author benchmark.
Raw interval forces are retained. Common-time bins integrate impulses exactly;
they do not filter forces inside the simulation or replace raw peak reporting.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


CASES = {
    'baseline': (.006, .000025, .001),
    'repeat': (.006, .000025, .001),
    'dt_50us': (.006, .00005, .001),
    'dt_12_5us': (.006, .0000125, .001),
    'exchange_500us': (.006, .000025, .0005),
    'exchange_250us': (.006, .000025, .00025),
    'grid_3mm': (.003, .000025, .001),
    'grid_2mm': (.002, .000025, .001),
}


def impulse_bins(end_times, forces, edges):
    """Integrate piecewise-constant interval-average forces over requested bins."""
    t = np.asarray(end_times, dtype=float)
    f = np.asarray(forces, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if (t.ndim != 1 or not len(t) or f.shape != (len(t), 3)
            or not np.isfinite(t).all() or not np.isfinite(f).all()
            or t[0] <= 0 or np.any(np.diff(t) <= 0)):
        raise ValueError('Expected finite forces and increasing interval end times')
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] < 0 or edges[-1] > t[-1] + 1.e-10
            or np.any(np.diff(edges) <= 0)):
        raise ValueError('Requested bins must lie within the complete trace')
    dt = np.diff(np.r_[0., t])
    cumulative = np.vstack([np.zeros(3), np.cumsum(f * dt[:, None], axis=0)])
    at_edges = np.column_stack([np.interp(edges, np.r_[0., t], cumulative[:, j])
                                for j in range(3)])
    return np.diff(at_edges, axis=0)


def summarize_case(path, bin_width=.01):
    path = Path(path)
    result = json.loads((path/'result.json').read_text())
    manifest = json.loads((path/'manifest.json').read_text())
    trace = json.loads((path/'trace.json').read_text())
    t = np.array([r['time_s'] for r in trace])
    f = np.array([r['blade_reaction_n'] for r in trace])
    count = round(t[-1]/bin_width)
    if not np.isclose(count*bin_width, t[-1], atol=1.e-9, rtol=0):
        raise ValueError('Duration must contain an integer number of comparison bins')
    edges = np.arange(count+1)*bin_width
    impulses = impulse_bins(t, f, edges)
    dt = np.diff(np.r_[0., t])
    common_force = impulses/bin_width
    sizes = np.asarray(result['component_sizes'])
    geometry_path = path/'material_geometry_validation.json'
    return {
        'config': manifest['config'], 'coupling_interval_s': manifest['coupling_interval_s'],
        'revision': manifest.get('revision'),
        'particle_count': result['particle_count'],
        'sampled_material_mass_kg': result['sampled_material_mass_kg'],
        'component_sizes': sizes.tolist(), 'component_mass_fractions': (sizes/sizes.sum()).tolist(),
        'maximum_interval_force_n': float(np.linalg.norm(f, axis=1).max()),
        'integral_force_vector_ns': (f*dt[:, None]).sum(axis=0).tolist(),
        'integral_force_norm_ns': float(np.dot(np.linalg.norm(f, axis=1), dt)),
        'common_bin_width_s': bin_width, 'common_bin_force_n': common_force.tolist(),
        'maximum_common_bin_force_n': float(np.linalg.norm(common_force, axis=1).max()),
        'minimum_det_f_sampled': result['minimum_det_f_sampled'],
        'final_mean_det_f': result['final_mean_det_f'],
        'final_center_of_mass_m': result['final_center_of_mass_m'],
        'final_position_bounds_m': result['final_position_bounds_m'],
        'wall_s': result['wall_s'], 'realtime_factor': result['realtime_factor'],
        'geometry_check': json.loads(geometry_path.read_text()) if geometry_path.exists() else None,
    }


def compare(first, second):
    a = np.asarray(first['common_bin_force_n'])
    b = np.asarray(second['common_bin_force_n'])
    rms = float(np.sqrt(np.mean(np.sum((a-b)**2, axis=1))))
    scale = float(np.sqrt(np.mean(np.sum(b*b, axis=1))))
    return {
        'force_vector_rmse_n_at_common_bins': rms,
        'relative_force_rmse_to_second': rms/scale if scale else None,
        'signed_impulse_difference_ns': (np.asarray(first['integral_force_vector_ns'])
                                        - second['integral_force_vector_ns']).tolist(),
        'center_of_mass_distance_m': float(np.linalg.norm(np.asarray(first['final_center_of_mass_m'])
                                                         - second['final_center_of_mass_m'])),
        'mass_difference_kg': first['sampled_material_mass_kg']-second['sampled_material_mass_kg'],
        'component_count_equal': len(first['component_sizes']) == len(second['component_sizes']),
    }


def report(root):
    root = Path(root)
    cases, failures = {}, {}
    for name in CASES:
        folder = root/name
        if (folder/'result.json').exists():
            cases[name] = summarize_case(folder)
        elif folder.exists():
            failures[name] = 'Incomplete: inspect case manifest and run.log; excluded from numeric comparison'
    pairs = [('baseline', 'repeat'), ('dt_50us', 'baseline'), ('baseline', 'dt_12_5us'),
             ('baseline', 'exchange_500us'), ('exchange_500us', 'exchange_250us'),
             ('baseline', 'grid_3mm'), ('grid_3mm', 'grid_2mm')]
    comparisons = {f'{a}__{b}': compare(cases[a], cases[b]) for a, b in pairs if a in cases and b in cases}
    payload = {
        'purpose': 'Numerical sensitivity of the same synthetic banana and prescribed single cut',
        'not_proven': ['agreement with measured force', 'robot closed-loop convergence',
                       'multi-cut surface accuracy', 'multi-tool operation', 'full-pipeline real time'],
        'interpretation': 'No pass threshold is invented. Raw peaks, common-bin forces, impulses, geometry and failures are reported together. Contact stiffness and particle contact radii depend on dt/h by the implemented paper equations; refinement is a sensitivity test, not evidence of physical calibration.',
        'cases': cases, 'failures': failures, 'comparisons': comparisons,
    }
    (root/'report.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    return payload


def run(root, names, duration=3.8):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (root/name).exists():
            raise FileExistsError(f'{root/name}: refusing to overwrite or reuse a possibly stale run')
        h, dt, interval = CASES[name]
        command = [sys.executable, '-m', 'blademaster_warp.paper_experiments',
                   '--output', str(root/name), '--spacing', str(h), '--dt', str(dt),
                   '--interval', str(interval), '--duration', str(duration)]
        print(f'Running {name}: h={h} dt={dt} exchange={interval}', flush=True)
        with (root/f'{name}.log').open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        print(f'{name}: exit={completed.returncode}', flush=True)
        report(root)
    return report(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    parser.add_argument('--duration', type=float, default=3.8)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    report(args.output) if args.summarize_only else run(args.output, args.cases, args.duration)
