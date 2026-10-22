"""Refine natural cutting windows from identical states at fixed stiffness and PIC rate."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

from blademaster_warp.presets import banana_dt, tangent_history_dt
from blademaster_warp.provenance import git_revision
from scripts.diff_cut_validate import configuration
from scripts.diff_natural_validate import capture_initial, evaluate, intact_initial


def compare_levels(coarse, fine):
    a, x = coarse
    b, y = fine
    g0, g1 = np.asarray(a['gradient_controls']), np.asarray(b['gradient_controls'])
    common = np.isfinite(x['event_times']) & np.isfinite(y['event_times'])
    result = dict(
        coarse_dt_s=a['config']['dt'], fine_dt_s=b['config']['dt'],
        state_max_abs={name: float(np.max(np.abs(x[name] - y[name])))
                       for name in ('x', 'v', 'F', 'Fp', 'stress', 'damage')},
        changed_side_particles=int(np.count_nonzero(x['codes'] != y['codes'])),
        changed_released_particles=int(np.count_nonzero((x['damage'] >= 1) != (y['damage'] >= 1))),
        changed_pair_keys=len(np.setxor1d(x['pair_keys'], y['pair_keys'])),
        changed_event_particles=int(np.count_nonzero(np.isfinite(x['event_times']) != np.isfinite(y['event_times']))),
        event_time_max_abs_s=float(np.max(np.abs(x['event_times'][common] - y['event_times'][common])))
            if np.any(common) else None,
        loss_abs=abs(a['loss'] - b['loss']),
        gradient_controls_abs=np.abs(g1 - g0).tolist(),
        gradient_controls_relative=(np.abs(g1 - g0) / np.maximum(np.maximum(np.abs(g0), np.abs(g1)), 1e-12)).tolist(),
        knife_impulse_max_abs_ns=float(np.max(np.abs(np.asarray(a['knife_impulse_ns']) - b['knife_impulse_ns']))),
        damage_work_abs_j=abs(a['damage_work_j'] - b['damage_work_j']),
    )
    result['gradient_materials_relative'] = {}
    for name in a['gradient_materials']:
        p, q = np.asarray(a['gradient_materials'][name]), np.asarray(b['gradient_materials'][name])
        result['gradient_materials_relative'][name] = (
            np.abs(p - q) / np.maximum(np.maximum(np.abs(p), np.abs(q)), 1e-12)).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spacings', type=float, nargs='+', default=[.006, .004])
    parser.add_argument('--materials', choices=['banana', 'soft', 'plastic'], nargs='+', default=['banana'])
    parser.add_argument('--shapes', choices=['banana', 'block', 'cylinder'], nargs='+', default=['banana'])
    parser.add_argument('--duration', type=float, default=.001)
    parser.add_argument('--from-initial', action='store_true')
    parser.add_argument('--trajectory', choices=['straight', 'cut-withdraw-push'], default='straight')
    parser.add_argument('--cutting-energy', type=float)
    parser.add_argument('--dt-divisors', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--epsilons', type=float, nargs='*', default=[5e-7, 5e-8],
                        help='Finite-difference perturbations; pass no values to compare refinement levels only')
    parser.add_argument('--segment-steps', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--adjoint', choices=['full', 'checkpoint'], default='full')
    parser.add_argument('--parameters', nargs='+', default=['knife_x_mm', 'knife_depth_scale'],
                        choices=['knife_x_mm', 'knife_depth_scale', 'knife_lateral_scale'])
    parser.add_argument('--flip-reference-dt', type=float)
    parser.add_argument('--geometry', choices=['boundary', 'event'], default='event')
    parser.add_argument('--release', choices=['boundary', 'integrated'], default='integrated')
    parser.add_argument('--friction-update', choices=['lagged', 'projected'], default='projected')
    parser.add_argument('--integration', choices=['symplectic', 'midpoint'], default='midpoint')
    parser.add_argument('--phases', type=float, nargs='+', default=[0.])
    args = parser.parse_args()
    if args.geometry == 'event' and args.integration == 'midpoint' and args.friction_update != 'projected':
        parser.error('Midpoint integration requires --friction-update projected')
    numbers = [*args.spacings, args.duration, *args.dt_divisors, *args.epsilons, *args.segment_steps]
    if args.flip_reference_dt is not None:
        numbers.append(args.flip_reference_dt)
    if not np.isfinite(numbers).all() or min(numbers) <= 0 or max(args.epsilons, default=0.) >= 1:
        parser.error('Sizes, times, steps and perturbations must be positive and finite; perturbations must be <1')
    if not np.isfinite(args.phases).all() or min(args.phases) < 0 or max(args.phases) >= 1:
        parser.error('Time-grid phases must be in [0, 1)')
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(provenance=git_revision(), command=sys.argv, cases=[])
    for shape in args.shapes:
        for spacing in args.spacings:
            for material in args.materials:
                base = replace(configuration(spacing, material), shape=shape, blade_grid_constraint=True)
                if shape != 'banana':
                    base = replace(base, size=(.024, .018, .018))
                dt = tangent_history_dt(base, banana_dt(base, spacing))
                base = replace(base, dt=dt, flip_reference_dt=args.flip_reference_dt or dt)
                if args.cutting_energy is not None:
                    base = replace(base, cutting_energy=args.cutting_energy)
                steps = round(args.duration / dt)
                if steps < 32 or abs(steps * dt - args.duration) > 1e-12:
                    parser.error('Duration must contain at least 32 base steps and divide exactly by the base step')
                window = intact_initial(base) if args.from_initial else capture_initial(base, steps)
                for phase in sorted(set(args.phases)):
                    case = dict(shape=shape, material=material, spacing_m=spacing, time_grid_phase=phase,
                                pic_rate_s_inv=(1 - base.flip_fraction) / base.flip_reference_dt,
                                start_time_s=window.start_time, duration_s=args.duration, levels=[], adjacent=[])
                    report['cases'].append(case)
                    previous = None
                    for divisor in sorted(set(args.dt_divisors)):
                        cfg = replace(base, dt=base.dt / divisor,
                                      board_normal_scale=base.board_normal_scale / divisor**2)
                        row, arrays = evaluate(cfg, steps * divisor, args.release, args.epsilons,
                                               args.segment_steps, args.geometry, args.friction_update,
                                               initial_window=window, return_arrays=True, adjoint=args.adjoint,
                                               controls_to_check=args.parameters, integration=args.integration, phase=phase,
                                               trajectory=args.trajectory, progress=True)
                        name = f'{shape}_{material}_h{spacing:g}_dt{divisor}_phase{phase:g}'
                        row.update(name=name, divisor=divisor)
                        np.savez_compressed(args.output / f'{name}.npz', **arrays)
                        case['levels'].append(row)
                        if previous is not None:
                            case['adjacent'].append(compare_levels(previous, (row, arrays)))
                        previous = row, arrays
                        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                        print(json.dumps(dict(case=name, loss=row['loss'], gradient=row['gradient_controls'],
                                              adjacent=case['adjacent'][-1] if case['adjacent'] else None)), flush=True)


if __name__ == '__main__':
    main()
