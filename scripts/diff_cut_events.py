"""Scan natural cutting events under knife-path and fracture-energy perturbations."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from blademaster_warp.coupling import ExternalToolPort
from blademaster_warp.presets import banana_dt, tangent_history_dt
from blademaster_warp.provenance import git_revision
from blademaster_warp.solver import Solver
from blademaster_warp.topology import components
from scripts.diff_cut_trajectory import motion
from scripts.diff_cut_validate import configuration


def run_case(cfg, controls, duration):
    solver = Solver(cfg, precision='float64')
    port = ExternalToolPort(solver)
    label_step = np.full(solver.n, -1, dtype=np.int64)
    release_step = np.full(solver.n, -1, dtype=np.int64)
    steps = round(duration / cfg.dt)
    peak_force = 0.
    impulse = np.zeros(3)
    tool_work = 0.
    started = perf_counter()
    for step in range(steps):
        elapsed = step * cfg.dt
        tool = motion(cfg, elapsed, controls)[0]
        reaction = port.advance(tool, cfg.dt, cutting=elapsed < .24, graph=False)
        codes = solver.codes.numpy()
        damage = solver.damage.numpy()
        label_step[(label_step < 0) & (codes != 0)] = step + 1
        release_step[(release_step < 0) & (damage >= 1)] = step + 1
        peak_force = max(peak_force, float(np.linalg.norm(reaction.force_n)))
        impulse += reaction.force_n * cfg.dt
        tool_work -= float(reaction.force_n @ tool.linear_velocity) * cfg.dt
    errors = solver.errors.numpy().tolist()
    if any(errors):
        raise RuntimeError(f'Solver errors: {errors}')
    x, v = solver.x.numpy(), solver.v.numpy()
    if not np.isfinite(x).all() or not np.isfinite(v).all():
        raise FloatingPointError('Nonfinite particle state')
    _, sizes = components(solver.nodes.numpy(), solver.weights.numpy(), codes)
    arrays = dict(label_step=label_step, release_step=release_step, codes=codes,
                  damage=damage, x=x, v=v, Fp=solver.Fp.numpy())
    metrics = dict(particles=solver.n, steps=steps, duration_s=steps * cfg.dt,
                   labelled_particles=int(np.count_nonzero(label_step >= 0)),
                   released_particles=int(np.count_nonzero(release_step >= 0)),
                   final_piece_sizes=sizes.tolist(), peak_force_n=peak_force,
                   reaction_impulse_ns=impulse.tolist(), tool_work_j=tool_work,
                   cutting_work_j=float(solver.work.numpy()[0]),
                   max_damage=float(np.max(damage)), wall_s=perf_counter() - started)
    for key, values in [('label', label_step), ('release', release_step)]:
        active = values[values >= 0]
        metrics[f'{key}_time_range_s'] = ([float(np.min(active) * cfg.dt),
                                         float(np.max(active) * cfg.dt)] if len(active) else [])
    return metrics, arrays


def compare(reference, perturbed, dt):
    result = dict(changed_side_particles=int(np.count_nonzero(reference['codes'] != perturbed['codes'])),
                  final_x_max_distance_m=float(np.max(np.linalg.norm(perturbed['x'] - reference['x'], axis=1))),
                  final_v_max_distance_m_s=float(np.max(np.linalg.norm(perturbed['v'] - reference['v'], axis=1))))
    for key in ('label', 'release'):
        a, b = reference[f'{key}_step'], perturbed[f'{key}_step']
        common = (a >= 0) & (b >= 0)
        delta = (b[common] - a[common]) * dt
        result[key] = dict(new_events=int(np.count_nonzero((a < 0) & (b >= 0))),
                           lost_events=int(np.count_nonzero((a >= 0) & (b < 0))),
                           shifted_events=int(np.count_nonzero(delta)),
                           max_abs_time_change_s=float(np.max(np.abs(delta))) if len(delta) else 0.)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spacing', type=float, default=.006)
    parser.add_argument('--duration', type=float, default=.24)
    parser.add_argument('--x-offsets-mm', type=float, nargs='*', default=[-.05, .05, -2., 2.])
    parser.add_argument('--depth-scales', type=float, nargs='*', default=[.99, 1.01])
    parser.add_argument('--energy-scales', type=float, nargs='*', default=[.01, .1, .99, 1.01, 10.])
    args = parser.parse_args()
    values = [args.spacing, args.duration, *args.depth_scales, *args.energy_scales]
    if any(not np.isfinite(value) or value <= 0 for value in values):
        parser.error('Spacing, duration and scales must be finite and positive')
    if not np.isfinite(args.x_offsets_mm).all():
        parser.error('Knife offsets must be finite')
    cfg = replace(configuration(args.spacing, 'banana'), blade_grid_constraint=True)
    cfg = replace(cfg, dt=tangent_history_dt(cfg, banana_dt(cfg, args.spacing)))
    cases = [('baseline', [0., 1., 1.], 1.)]
    cases += [(f'x_{offset:g}mm', [offset, 1., 1.], 1.) for offset in args.x_offsets_mm if offset != 0]
    cases += [(f'depth_{scale:g}', [0., scale, 1.], 1.) for scale in args.depth_scales if scale != 1]
    cases += [(f'energy_{scale:g}', [0., 1., 1.], scale) for scale in args.energy_scales if scale != 1]
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(provenance=git_revision(), command=sys.argv, config=asdict(cfg),
                  precision='float64', cases=[])
    reference = None
    for name, controls, scale in cases:
        actual = replace(cfg, cutting_energy=cfg.cutting_energy * scale)
        metrics, arrays = run_case(actual, controls, args.duration)
        row = dict(name=name, controls=controls, cutting_energy_j_m2=actual.cutting_energy, **metrics)
        if reference is None:
            reference = arrays
        else:
            row['vs_baseline'] = compare(reference, arrays, cfg.dt)
        np.savez_compressed(args.output / f'{name}.npz', **arrays)
        report['cases'].append(row)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
