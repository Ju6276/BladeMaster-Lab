"""Check natural damage-release gradients on banana windows and refined time steps."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp

from blademaster_warp.coupling import ExternalToolPort
from blademaster_warp.differentiable_cutting import EventCutContact, pair_snapshot
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.presets import banana_dt, tangent_history_dt
from blademaster_warp.provenance import git_revision
from blademaster_warp.solver import Solver
from scripts.diff_cut_trajectory import motion, objective, tool_at
from scripts.diff_cut_validate import configuration


def snapshot(solver):
    values = {name: getattr(solver, name).numpy() for name in ('x', 'v', 'F', 'Fp', 'stress')}
    history = dict(rigid=solver.histories.numpy(), damage=solver.damage.numpy(),
                   side=solver.blade_constraint.side.numpy())
    return values, history, solver.codes.numpy()


def before_first_release(cfg):
    solver = Solver(cfg, precision='float64')
    port = ExternalToolPort(solver)
    for step in range(round(.12 / cfg.dt)):
        if step % 32 == 0:
            saved = snapshot(solver)
            start = step
            if pair_snapshot(solver)[0].size:
                raise ValueError('Release window requires an empty initial pair set')
        port.advance(motion(cfg, step * cfg.dt, [0, 1, 1])[0], cfg.dt, cutting=True, graph=False)
        if np.any(solver.damage.numpy() >= 1):
            return start, saved
    raise RuntimeError('No damage release in the first 0.12 seconds')


def boundary_reference(cfg, start_time, steps, values, history, codes):
    solver = Solver(cfg, precision='float64')
    for name, value in values.items():
        getattr(solver, name).assign(value)
    solver.histories.assign(history['rigid'])
    solver.damage.assign(history['damage'])
    solver.codes.assign(codes)
    solver.blade_constraint.side.assign(history['side'])
    port = ExternalToolPort(solver)
    forces = []
    for step in range(steps):
        tool = motion(cfg, start_time + step * cfg.dt, [0, 1, 1])[0]
        forces.append(port.advance(tool, cfg.dt, cutting=True, graph=False).force_n.copy())
        if np.any(solver.codes.numpy() != codes) or pair_snapshot(solver)[0].size:
            raise ValueError('Window crosses a side-code or pair-set change; shorten its duration')
    if solver.errors.numpy().any():
        raise RuntimeError(f'Solver errors: {solver.errors.numpy()}')
    return snapshot(solver)[0], np.asarray(forces), float(solver.work.numpy()[0])


def evaluate(cfg, start_time, duration, saved, epsilons):
    values, initial_history, codes = saved
    steps = round(duration / cfg.dt)
    start = round(start_time / cfg.dt)
    reference, reference_force, reference_work = boundary_reference(
        cfg, start_time, steps, values, initial_history, codes)
    m = MaterialTrajectory(cfg, ordered_transfers=True, precision='float64')
    c = EventCutContact(m)
    initial = m.state(**values)
    history = c.history(**initial_history)
    branch = c.freeze(codes, [], [], np.zeros(m.n, dtype=int))
    q = m.array([0., 1., 1.], float)
    accumulate = objective(m, steps)

    def knife(step):
        return tool_at(m, q, start + step)

    def rollout():
        state, h = initial, history
        loss = m.zeros(1, float)
        event_times = np.full(m.n, -1.)
        event_steps = np.full(m.n, -1, dtype=int)
        work = 0.
        forces = []
        modes = []
        for step in range(steps):
            previous = h.damage.numpy()
            state, h, force = c.step(state, h, knife(step), branch)
            accumulate(step, state, h, force, loss)
            trial = h.trial_damage.numpy()
            crossing = (previous < 1) & (trial >= 1) & (event_steps < 0)
            fraction = (1 - previous[crossing]) / (trial[crossing] - previous[crossing])
            event_times[crossing] = start_time + (step + fraction) * cfg.dt
            event_steps[crossing] = step
            work += float(np.sum(h.damage_work.numpy()))
            forces.append(force.numpy()[0])
            modes.append(h.rigid_modes.numpy())
        return loss, state, h, event_times, event_steps, work, np.asarray(forces), np.asarray(modes)

    with wp.Tape() as tape:
        loss, state, h, event_times, event_steps, work, forces, modes = rollout()
    tape.backward(loss)
    gradient_q = q.grad.numpy().copy()
    gradient_energy = float(c.fracture_energy.grad.numpy()[0])
    gradient_x = initial.x.grad.numpy().copy()
    terminal = {name: getattr(state, name).numpy() for name in values}
    threshold = cfg.cutting_energy * m.volume**(2 / 3) * (1 - cfg.sharpness) / cfg.sharpness
    retained_work = float(threshold * np.sum(h.damage.numpy() - history.damage.numpy()))
    tape.zero()
    timings = {}
    checkpoint = c.checkpoint_vjp(initial, history, [branch] * steps, 16, knife, accumulate,
                                  parameters=(q,), timings=timings)
    gradient_error = dict(controls=float(np.max(np.abs(checkpoint.parameter_gradients[q] - gradient_q))),
                          fracture_energy=abs(float(checkpoint.parameter_gradients[c.fracture_energy][0]) - gradient_energy),
                          initial_x=float(np.max(np.abs(checkpoint.initial_gradient.x.numpy() - gradient_x))))
    np.testing.assert_allclose(checkpoint.parameter_gradients[q], gradient_q, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(checkpoint.parameter_gradients[c.fracture_energy][0], gradient_energy, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(checkpoint.initial_gradient.x.numpy(), gradient_x, rtol=1e-8, atol=1e-7)
    np.testing.assert_allclose(work, retained_work, rtol=1e-11, atol=1e-15)
    result = dict(dt_s=cfg.dt, steps=steps, start_time_s=start_time, loss=float(loss.numpy()[0]),
                   released_particles=int(np.count_nonzero(event_steps >= 0)),
                   gradient_controls=gradient_q.tolist(), gradient_fracture_energy=gradient_energy,
                   checkpoint_gradient_max_abs=gradient_error, recompute_max_abs=timings['recompute_max_abs'],
                   damage_work_j=work, boundary_damage_work_j=reference_work,
                   damage_work_balance_error_j=abs(work - retained_work),
                   vs_boundary_max_abs={name: float(np.max(np.abs(terminal[name] - reference[name]))) for name in values},
                   force_max_difference_n=float(np.max(np.abs(forces - reference_force))), finite_difference=[])
    direction = np.array([.4, .7, 0.])
    for parameter in ('fracture_energy', 'controls'):
        ad = gradient_energy * cfg.cutting_energy if parameter == 'fracture_energy' else float(gradient_q @ direction)
        if not np.isfinite(ad) or abs(ad) < 1e-9:
            raise ValueError(f'Inactive {parameter} direction: {ad}')
        for epsilon in epsilons:
            losses, changes, mode_changes = [], [], []
            for sign in (-1, 1):
                q.assign([0., 1., 1.])
                c.fracture_energy.assign([cfg.cutting_energy])
                if parameter == 'fracture_energy':
                    c.fracture_energy.assign([cfg.cutting_energy * (1 + sign * epsilon)])
                else:
                    q.assign(np.array([0., 1., 1.]) + sign * epsilon * direction)
                shifted_loss, _, _, _, shifted_steps, _, _, shifted_modes = rollout()
                losses.append(float(shifted_loss.numpy()[0]))
                changes.append(int(np.count_nonzero(event_steps != shifted_steps)))
                mode_changes.append(np.count_nonzero(modes != shifted_modes, axis=(0, 2)).tolist())
            fd = (losses[1] - losses[0]) / (2 * epsilon)
            result['finite_difference'].append(dict(parameter=parameter, epsilon=epsilon, ad=ad, fd=fd,
                                                     relative_error=abs(ad - fd) / max(abs(ad), abs(fd), 1e-12),
                                                     changed_event_steps=changes,
                                                     changed_rigid_modes=mode_changes))
    arrays = dict(event_times=event_times, event_steps=event_steps, force=forces,
                  rigid_modes=modes, damage=h.damage.numpy(), **terminal)
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spacings', type=float, nargs='+', default=[.006, .004])
    parser.add_argument('--energies', type=float, nargs='+', default=[7., 70.])
    parser.add_argument('--dt-divisors', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--duration', type=float, default=.002)
    parser.add_argument('--epsilons', type=float, nargs='+', default=[.01, .001, .0001, .00001, .000001])
    args = parser.parse_args()
    numbers = [*args.spacings, *args.energies, *args.dt_divisors, args.duration, *args.epsilons]
    if any(not np.isfinite(value) or value <= 0 for value in numbers):
        parser.error('Grid sizes, energies, divisors, duration and perturbations must be finite and positive')
    if any(value >= 1 for value in args.epsilons):
        parser.error('Relative energy perturbations must be less than one')
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(provenance=git_revision(), command=sys.argv, cases=[])
    for spacing in args.spacings:
        for energy in args.energies:
            base = replace(configuration(spacing, 'banana'), blade_grid_constraint=True, cutting_energy=energy)
            base = replace(base, dt=tangent_history_dt(base, banana_dt(base, spacing)))
            prefix, saved = before_first_release(base)
            start_time = prefix * base.dt
            levels = []
            for divisor in sorted(set(args.dt_divisors)):
                cfg = replace(base, dt=base.dt / divisor)
                if abs(round(args.duration / cfg.dt) * cfg.dt - args.duration) > 1e-12:
                    parser.error('Duration must be divisible by every selected time step')
                row, arrays = evaluate(cfg, start_time, args.duration, saved, args.epsilons)
                name = f'h{spacing:g}_gc{energy:g}_dt{divisor}'
                row.update(name=name, config=asdict(cfg))
                np.savez_compressed(args.output / f'{name}.npz', **arrays)
                report['cases'].append(row)
                levels.append((row, arrays))
                (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps(row), flush=True)
            finest_row, finest = levels[-1]
            for row, arrays in levels:
                common = (arrays['event_times'] >= 0) & (finest['event_times'] >= 0)
                row['vs_finest'] = dict(x_max_distance_m=float(np.max(np.linalg.norm(arrays['x'] - finest['x'], axis=1))),
                                        v_max_distance_m_s=float(np.max(np.linalg.norm(arrays['v'] - finest['v'], axis=1))),
                                        event_time_max_error_s=float(np.max(np.abs(arrays['event_times'][common] - finest['event_times'][common]))) if np.any(common) else 0.,
                                        different_event_particles=int(np.count_nonzero((arrays['event_times'] >= 0) != (finest['event_times'] >= 0))),
                                        gradient_controls_max_abs=float(np.max(np.abs(np.asarray(row['gradient_controls']) - finest_row['gradient_controls']))),
                                        gradient_fracture_energy_abs=abs(row['gradient_fracture_energy'] - finest_row['gradient_fracture_energy']))
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
