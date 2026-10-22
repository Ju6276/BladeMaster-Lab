"""Accepted-segment energy balances for rest, indentation and natural cutting."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from blademaster_warp.differentiable_geometry import GeometryEventContact
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.differentiable_natural import CutInstruction
from blademaster_warp.energy_ledger import EnergyLedger
from blademaster_warp.presets import banana_dt
from blademaster_warp.provenance import git_revision
from blademaster_warp.topology import components
from scripts.diff_cut_validate import configuration
from scripts.diff_natural_validate import time_edges


def evaluate(cfg, scene, duration, speed, press_depth):
    m = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    c = GeometryEventContact(m)
    ledger = EnergyLedger(c)
    state, history = m.state(), c.history()
    radius = cfg.grid_spacing / 4
    offset = max(cfg.blade_contact_radius or radius, radius - cfg.blade_half_size[0])
    column = (np.abs(m.reference[:, 0] - .0004) <= offset + cfg.blade_half_size[0])
    column &= np.abs(m.reference[:, 1]) <= cfg.blade_half_size[1]
    if not np.any(column):
        raise ValueError('The knife path has no particles in its contact envelope')
    edge_start = float(m.reference[column, 2].max() + offset + .00005)
    depth = press_depth if scene == 'press' else max(edge_start - .0005, 0.)
    stop = depth / speed
    steps = round(duration / cfg.dt)
    if steps < 1 or not np.isclose(steps * cfg.dt, duration, rtol=0., atol=1e-12):
        raise ValueError('Duration must be an integer multiple of the time step')
    edges = time_edges(steps, cfg.dt, 0., 0., 'straight', knots=[stop])
    forces, elapsed = [], perf_counter()
    for index, dt in enumerate(np.diff(edges)):
        time = edges[index]
        position = [.0004, 0., edge_start + cfg.blade_half_size[2] - min(time * speed, depth)]
        velocity = [0., 0., -speed if time < stop else 0.]
        if scene == 'rest':
            position, velocity = [1., 1., 1.], [0., 0., 0.]
        tool = c.tool(position, velocity)
        state, history, force = c.step(state, history, tool,
                                      CutInstruction(cutting=scene == 'cut', duration=float(dt)))
        forces.append(force.numpy()[0])
        if (index + 1) % max(1, len(edges) // 10) == 0:
            print(json.dumps(dict(scene=scene, time_s=float(edges[index + 1]), elapsed_s=perf_counter()-elapsed)), flush=True)
    forces = np.asarray(forces)
    if not np.isfinite(forces).all():
        raise FloatingPointError('Nonfinite knife force')
    report = ledger.report()
    report.update(config=asdict(cfg), scene=scene, knife_speed_m_s=speed, press_depth_m=press_depth,
                  edge_start_m=edge_start, elapsed_s=perf_counter()-elapsed,
                  damage_sum=float(history.damage.numpy().sum()),
                  released_particles=int(np.count_nonzero(history.damage.numpy() >= 1.)),
                  labelled_particles=int(np.count_nonzero(history.codes.numpy())),
                  knife_peak_force_n=float(np.linalg.norm(forces, axis=1).max()),
                  knife_impulse_ns=np.sum(forces * np.diff(edges)[:, None], axis=0).tolist(),
                  plastic_max_abs=float(np.max(np.abs(state.Fp.numpy() - np.eye(3)))))
    _, sizes = components(m.nodes, m.weights.numpy(), history.codes.numpy())
    report['piece_sizes'] = sizes.tolist()
    threshold = cfg.cutting_energy * m.volume**(2/3) * (1-cfg.sharpness) / cfg.sharpness
    report['damage_work_residual_j'] = report['totals']['damage_work_j'] - threshold * report['damage_sum']
    report['minimum_segment_terms_j'] = {name: min(row[name] for row in ledger.rows) for name in
        ('plastic_j', 'friction_viscous_j', 'friction_sliding_j', 'friction_integration_j', 'history_transport_j', 'pic_j')}
    totals, initial = report['totals'], report['initial']
    scale = max(abs(totals['knife_work_j']) + abs(totals['gravity_work_j']),
                sum(abs(initial[name]) for name in ('kinetic_j', 'elastic_j', 'contact_normal_j', 'contact_tangent_j')),
                np.finfo(float).tiny)
    report['energy_scale_j'] = scale
    report['relative_balance_residual'] = totals['global_balance_residual_j'] / scale
    names = list(ledger.rows[0])
    arrays = dict(time_s=edges[1:], force=np.asarray(forces), energy_columns=np.asarray(names),
                  segment_energy=np.asarray([[row[name] for name in names] for row in ledger.rows]),
                  codes=history.codes.numpy(), damage=history.damage.numpy(),
                  **{name: getattr(state, name).numpy() for name in ('x', 'v', 'F', 'Fp', 'stress')})
    return report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenes', choices=['rest', 'press', 'cut'], nargs='+', default=['rest', 'press', 'cut'])
    parser.add_argument('--shape', choices=['banana', 'block', 'cylinder'], default='banana')
    parser.add_argument('--material', choices=['banana', 'soft', 'plastic'], default='banana')
    parser.add_argument('--spacing', type=float, default=.006)
    parser.add_argument('--dt', type=float, default=1/64000)
    parser.add_argument('--dt-divisors', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--flip-reference-dt', type=float, default=1/64000)
    parser.add_argument('--duration', type=float, default=.05)
    parser.add_argument('--cutting-energies', type=float, nargs='+', default=[7., 70., 700., 1400.])
    parser.add_argument('--cut-speed', type=float, default=.12)
    parser.add_argument('--press-speed', type=float, default=.02)
    parser.add_argument('--press-depth', type=float, default=.006)
    args = parser.parse_args()
    numbers = [args.spacing, args.dt, *args.dt_divisors, args.flip_reference_dt, args.duration,
               *args.cutting_energies, args.cut_speed, args.press_speed, args.press_depth]
    if not np.isfinite(numbers).all() or min(numbers) <= 0:
        parser.error('Lengths, time steps, durations, speeds and fracture energies must be positive and finite')
    base = replace(configuration(args.spacing, args.material), shape=args.shape,
                   dt=args.dt, flip_reference_dt=args.flip_reference_dt, blade_grid_constraint=True)
    if args.shape != 'banana':
        base = replace(base, size=(.024, .018, .018))
    if args.dt > banana_dt(base, args.spacing):
        parser.error('Time step exceeds the material CFL bound')
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(provenance=git_revision(), command=sys.argv, cases=[], adjacent=[])
    for scene in args.scenes:
        for gc in (args.cutting_energies if scene == 'cut' else [base.cutting_energy]):
            previous = None
            for divisor in sorted(set(args.dt_divisors)):
                cfg = replace(base, dt=base.dt/divisor, cutting_energy=gc,
                              board_normal_scale=base.board_normal_scale/divisor**2)
                speed = args.press_speed if scene == 'press' else args.cut_speed
                row, arrays = evaluate(cfg, scene, args.duration, speed, args.press_depth)
                name = f'{scene}_{args.shape}_{args.material}_Gc{gc:g}_dt{divisor}'
                row['name'] = name
                report['cases'].append(row)
                if previous is not None:
                    prior, old = previous
                    report['adjacent'].append(dict(scene=scene, cutting_energy=gc,
                        coarse_dt_s=prior['config']['dt'], fine_dt_s=cfg.dt,
                        position_max_distance_m=float(np.linalg.norm(arrays['x'] - old['x'], axis=1).max()),
                        coarse_residual_j=prior['totals']['global_balance_residual_j'],
                        fine_residual_j=row['totals']['global_balance_residual_j'],
                        coarse_integration_residual_j=prior['totals']['integration_residual_j'],
                        fine_integration_residual_j=row['totals']['integration_residual_j'],
                        changed_side_particles=int(np.count_nonzero(arrays['codes'] != old['codes'])),
                        changed_released_particles=int(np.count_nonzero((arrays['damage'] >= 1) != (old['damage'] >= 1)))))
                previous = row, arrays
                np.savez_compressed(args.output / f'{name}.npz', **arrays)
                (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                if scene != 'rest' and row['knife_peak_force_n'] == 0:
                    raise RuntimeError('The trajectory did not reach knife contact; increase its duration')
                if scene == 'cut' and row['labelled_particles'] == 0:
                    raise RuntimeError('The trajectory did not reach a cutting event; increase its duration')
                print(json.dumps(dict(case=name, residual_j=row['totals']['global_balance_residual_j'],
                                      relative_residual=row['relative_balance_residual'])), flush=True)


if __name__ == '__main__':
    main()
