"""Compare natural cutting rollouts and differentiate across a side-assignment window."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp

from blademaster_warp.coupling import ExternalToolPort
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.differentiable_geometry import GeometryEventContact
from blademaster_warp.differentiable_natural import CutInstruction, NaturalCutContact
from blademaster_warp.presets import banana_dt, tangent_history_dt
from blademaster_warp.provenance import git_revision
from blademaster_warp.solver import Solver
from scripts.diff_cut_validate import (
    capture_window, configuration, differentiable_tool, numpy_tool, reaction_loss, terminal_loss,
)


def first_side_prefix(cfg):
    solver = Solver(cfg, precision='float64')
    port = ExternalToolPort(solver)
    origin = np.array([.0004, 0., .027 + cfg.blade_half_size[2]])
    for step in range(round(.2 / cfg.dt)):
        port.advance(numpy_tool(origin, step * cfg.dt, [0., 1., 1.]), cfg.dt, graph=False)
        if np.any(solver.codes.numpy()):
            return (step // 32) * 32
    raise RuntimeError('No geometric side assignment in the first 0.2 seconds')


@dataclass
class WindowState:
    origin: np.ndarray
    values: dict
    history: dict
    codes: np.ndarray
    pair_keys: np.ndarray
    start_time: float
    reference_dt: float
    rows: list


def capture_initial(cfg, steps):
    prefix = first_side_prefix(cfg)
    origin, values, saved, rows, _ = capture_window(cfg, prefix, steps, [0., 1., 1.], precision='float64')
    return WindowState(origin, values, saved, rows[0]['codes'], rows[0]['previous'],
                       prefix * cfg.dt, cfg.dt, rows)


def evaluate(cfg, steps, release, epsilons, segment_steps, geometry='boundary', friction_update='projected',
             *, initial_window=None, return_arrays=False, adjoint='full',
             controls_to_check=('knife_x_mm', 'knife_depth_scale'), integration='midpoint', phase=0.):
    directions = dict(knife_x_mm=np.array([1., 0., 0.]),
                      knife_depth_scale=np.array([0., 1., 0.]),
                      knife_lateral_scale=np.array([0., 0., 1.]))
    if adjoint not in ('full', 'checkpoint') or (adjoint == 'checkpoint' and not segment_steps):
        raise ValueError('Use full Tape or checkpoint adjoints with at least one segment length')
    if not controls_to_check or any(name not in directions for name in controls_to_check):
        raise ValueError('Select knife_x_mm, knife_depth_scale or knife_lateral_scale')
    window = capture_initial(cfg, steps) if initial_window is None else initial_window
    if not np.isfinite(phase) or not 0 <= phase < 1:
        raise ValueError('Time-grid phase must be in [0, 1)')
    total_duration = steps * cfg.dt
    edges = np.arange(steps + 1, dtype=float) * cfg.dt
    if phase:
        edges = np.concatenate(([0.], (np.arange(steps) + phase) * cfg.dt, [total_duration]))
    durations = np.diff(edges) if phase else np.full(steps, cfg.dt)
    instructions = [CutInstruction(duration=float(dt)) for dt in durations]
    steps = len(durations)
    origin, values, saved, rows = window.origin, window.values, window.history, window.rows
    controls = np.array([0., 1., 1.])
    m = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    contact_type = GeometryEventContact if geometry == 'event' else NaturalCutContact
    kwargs = dict(integration=integration) if geometry == 'event' else {}
    c = contact_type(m, release=release, friction_update=friction_update, **kwargs)
    initial = m.state(**values)
    history = c.history(**saved, codes=window.codes, pair_keys=window.pair_keys)
    q = m.array(controls, float)

    def knife(step):
        return differentiable_tool(m, q, origin, window.start_time + edges[step])

    def objective(step, state, h, force, loss):
        m.launch(reaction_loss, 1, inputs=[force, float(durations[step])], outputs=[loss], device=m.device)
        if step == steps - 1:
            m.launch(terminal_loss, m.n, inputs=[state.v, state.Fp], outputs=[loss], device=m.device)

    def rollout(compare=False):
        state, h = initial, history
        loss = m.zeros(1, float)
        codes, pairs, released, rigid_modes = [], [], [], []
        geometry_events = []
        forces, damage_work = [], 0.
        error = {name: 0. for name in (*values, 'force', 'rigid_history', 'pair_history', 'damage')}
        discrete = dict(side_particles=0, pair_keys=0)
        for step in range(steps):
            state, h, force = c.step(state, h, knife(step), instructions[step])
            objective(step, state, h, force, loss)
            codes.append(h.codes.numpy())
            pairs.append(h.pair_keys.copy())
            released.append(h.damage.numpy() >= 1)
            rigid_modes.append(h.rigid_modes.numpy())
            geometry_events.append({p: float(t.numpy()[0]) for p, t in h.geometry_events})
            forces.append(force.numpy()[0])
            damage_work += float(np.sum(h.damage_work.numpy()))
            if compare:
                row = rows[step]
                for name in values:
                    error[name] = max(error[name], float(np.max(np.abs(
                        getattr(state, name).numpy() - row['state'][name]))))
                for name, a, b in (
                    ('force', force.numpy()[0], row['force']),
                    ('rigid_history', h.rigid.numpy(), row['rigid']),
                    ('damage', h.damage.numpy(), row['damage']),
                ):
                    error[name] = max(error[name], float(np.max(np.abs(a - b))))
                discrete['side_particles'] += int(np.count_nonzero(codes[-1] != row['next_codes']))
                discrete['pair_keys'] += len(np.setxor1d(pairs[-1], row['keys']))
                if np.array_equal(pairs[-1], row['keys']) and len(pairs[-1]):
                    error['pair_history'] = max(error['pair_history'], float(np.max(
                        np.abs(h.pairs.numpy() - row['pairs']))))
        return loss, state, h, dict(codes=np.asarray(codes), pairs=pairs, released=np.asarray(released),
                                   rigid_modes=np.asarray(rigid_modes), geometry=geometry_events,
                                   force=np.asarray(forces), damage_work=damage_work), error, discrete

    compare = phase == 0 and cfg.dt == window.reference_dt and steps == len(rows)
    with wp.Tape() if adjoint == 'full' else nullcontext() as tape:
        loss, terminal, terminal_history, events, forward_error, discrete_error = rollout(compare=compare)
    parameters = dict(young_modulus=m.young, yield_stress=m.yield_stress,
                      friction=c.friction, fracture_energy=c.fracture_energy)
    gradient, gradient_x, parameter_gradients = None, None, None
    if tape is not None:
        tape.backward(loss)
        gradient = q.grad.numpy().copy()
        gradient_x = initial.x.grad.numpy().copy()
        parameter_gradients = {name: array.grad.numpy().copy() for name, array in parameters.items()}
        tape.zero()
    del tape
    checkpoint = []
    for length in segment_steps:
        timings = {}
        result = c.checkpoint_vjp(initial, history, instructions, length,
                                  knife, objective, parameters=(q,), timings=timings)
        np.testing.assert_allclose(result.loss.numpy(), loss.numpy(), rtol=1e-11, atol=1e-12)
        for name in values:
            np.testing.assert_allclose(getattr(result.terminal, name).numpy(),
                                       getattr(terminal, name).numpy(), rtol=1e-10, atol=1e-10)
        has_gradient_reference = gradient is not None
        if not has_gradient_reference:
            gradient = result.parameter_gradients[q].copy()
            gradient_x = result.initial_gradient.x.numpy().copy()
            parameter_gradients = {name: result.parameter_gradients[array].copy()
                                   for name, array in parameters.items()}
        error_q, error_x, gradient_reference = None, None, None
        material_error = {name: None for name in parameters}
        if has_gradient_reference:
            gradient_reference = 'full_tape' if adjoint == 'full' else f'checkpoint:{segment_steps[0]}'
            error_q = float(np.max(np.abs(result.parameter_gradients[q] - gradient)))
            error_x = float(np.max(np.abs(result.initial_gradient.x.numpy() - gradient_x)))
            np.testing.assert_allclose(result.parameter_gradients[q], gradient, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(result.initial_gradient.x.numpy(), gradient_x, rtol=1e-8, atol=1e-8)
            for name, array in parameters.items():
                actual, expected = result.parameter_gradients[array], parameter_gradients[name]
                np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
                material_error[name] = float(np.max(np.abs(actual - expected)))
        checkpoint.append(dict(segment_steps=length, controls_max_abs=error_q, initial_x_max_abs=error_x,
                               gradient_reference=gradient_reference,
                               material_max_abs=material_error,
                               recompute_max_abs=timings['recompute_max_abs']))
    finite_difference = []
    baseline_loss = float(loss.numpy()[0])
    for parameter in controls_to_check:
        direction = directions[parameter]
        ad = float(gradient @ direction)
        for epsilon in epsilons:
            losses, changes = [], []
            for sign in (-1, 1):
                q.assign(controls + sign * epsilon * direction)
                shifted_loss, _, _, shifted, _, _ = rollout()
                losses.append(float(shifted_loss.numpy()[0]))
                mode_changes = np.argwhere(shifted['rigid_modes'] != events['rigid_modes'])
                changes.append(dict(
                    side_particle_steps=int(np.count_nonzero(shifted['codes'] != events['codes'])),
                    final_side_particles=int(np.count_nonzero(shifted['codes'][-1] != events['codes'][-1])),
                    pair_keys=sum(len(np.setxor1d(a, b)) for a, b in zip(shifted['pairs'], events['pairs'])),
                    release_particle_steps=int(np.count_nonzero(shifted['released'] != events['released'])),
                    rigid_modes=int(np.count_nonzero(shifted['rigid_modes'] != events['rigid_modes'])),
                    rigid_mode_channels=dict(zip(('board', 'blade', 'feature', 'hold', 'work'),
                        np.count_nonzero(shifted['rigid_modes'] != events['rigid_modes'], axis=(0, 2)).tolist())),
                    first_rigid_mode_change=None if not len(mode_changes) else dict(
                        step=int(mode_changes[0, 0]), channel=int(mode_changes[0, 1]),
                        particle=int(mode_changes[0, 2]),
                        baseline=int(events['rigid_modes'][tuple(mode_changes[0])]),
                        perturbed=int(shifted['rigid_modes'][tuple(mode_changes[0])])),
                    geometry_event_ids=sum(len(a.keys() ^ b.keys())
                                           for a, b in zip(shifted['geometry'], events['geometry'])),
                    geometry_order_steps=sum(list(a) != list(b)
                                             for a, b in zip(shifted['geometry'], events['geometry'])),
                    geometry_event_time_max_abs_s=max((abs(a[p] - b[p])
                        for a, b in zip(shifted['geometry'], events['geometry']) for p in a.keys() & b.keys()),
                        default=0.),
                ))
            fd = (losses[1] - losses[0]) / (2 * epsilon)
            finite_difference.append(dict(parameter=parameter, epsilon=epsilon, ad=ad, fd=fd,
                                           left_fd=(baseline_loss - losses[0]) / epsilon,
                                           right_fd=(losses[1] - baseline_loss) / epsilon,
                                           relative_error=abs(ad - fd) / max(abs(ad), abs(fd), 1e-12),
                                           event_changes=changes))
    if not np.isfinite([float(loss.numpy()[0]), *gradient]).all():
        raise FloatingPointError('Nonfinite objective or gradient')
    new_labels = np.count_nonzero(events['codes'][-1] != history.codes.numpy())
    if not new_labels:
        raise RuntimeError('Selected window contains no new side assignments')
    threshold = cfg.cutting_energy * m.volume**(2 / 3) * (1 - cfg.sharpness) / cfg.sharpness
    work_from_damage = threshold * np.sum(terminal_history.damage.numpy() - history.damage.numpy())
    result = dict(config=asdict(cfg), release=release, geometry=geometry, friction_update=friction_update,
                integration=integration if geometry == 'event' else 'symplectic',
                adjoint=adjoint,
                particles=m.n, prefix_steps=round(window.start_time / cfg.dt),
                start_time_s=window.start_time, steps=steps, duration_s=total_duration, time_grid_phase=phase,
                newly_labelled_particles=int(new_labels), peak_pairs=max(map(len, events['pairs'])),
                loss=float(loss.numpy()[0]), gradient_controls=gradient.tolist(),
                gradient_materials={name: value.tolist() for name, value in parameter_gradients.items()},
                damage_work_j=events['damage_work'], damage_work_balance_error_j=abs(events['damage_work']-work_from_damage),
                knife_impulse_ns=np.sum(durations[:, None] * events['force'], axis=0).tolist(),
                vs_original_max_abs=forward_error if compare else None,
                vs_original_discrete=discrete_error if compare else None,
                checkpoint=checkpoint, finite_difference=finite_difference)
    if not return_arrays:
        return result
    event_times = np.full(m.n, np.nan)
    for step, entries in enumerate(events['geometry']):
        for particle, time in entries.items():
            event_times[particle] = window.start_time + edges[step] + time
    arrays = {name: getattr(terminal, name).numpy() for name in values}
    arrays.update(codes=terminal_history.codes.numpy(), damage=terminal_history.damage.numpy(),
                  pair_keys=terminal_history.pair_keys.copy(), force=events['force'], event_times=event_times)
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spacings', type=float, nargs='+', default=[.006, .004])
    parser.add_argument('--materials', choices=['banana', 'soft', 'plastic'], nargs='+', default=['banana'])
    parser.add_argument('--shapes', choices=['banana', 'block', 'cylinder'], nargs='+', default=['banana'])
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--segment-steps', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--adjoint', choices=['full', 'checkpoint'], default='full')
    parser.add_argument('--parameters', nargs='+', default=['knife_x_mm', 'knife_depth_scale'],
                        choices=['knife_x_mm', 'knife_depth_scale', 'knife_lateral_scale'])
    parser.add_argument('--release', choices=['boundary', 'integrated'], default='integrated')
    parser.add_argument('--geometry', choices=['boundary', 'event'], default='boundary')
    parser.add_argument('--friction-update', choices=['lagged', 'projected'], default='projected')
    parser.add_argument('--integration', choices=['symplectic', 'midpoint'], default='midpoint')
    parser.add_argument('--epsilons', type=float, nargs='+', default=[.05, .005, .0005, .00005])
    args = parser.parse_args()
    if args.geometry == 'event' and args.integration == 'midpoint' and args.friction_update != 'projected':
        parser.error('Midpoint integration requires --friction-update projected')
    if (not np.isfinite([*args.spacings, *args.epsilons]).all()
            or min(*args.spacings, *args.epsilons, args.steps, *args.segment_steps) <= 0
            or max(args.epsilons) >= 1):
        parser.error('Sizes, steps and perturbations must be positive and finite; perturbations must be <1')
    if args.steps < 32:
        parser.error('At least 32 steps are needed to include the first side assignment')
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(provenance=git_revision(), command=sys.argv, cases=[])
    for shape in args.shapes:
        for spacing in args.spacings:
            for material in args.materials:
                cfg = replace(configuration(spacing, material), shape=shape, blade_grid_constraint=True)
                if shape != 'banana':
                    cfg = replace(cfg, size=(.024, .018, .018))
                dt = tangent_history_dt(cfg, banana_dt(cfg, spacing))
                cfg = replace(cfg, dt=dt, flip_reference_dt=dt)
                row = evaluate(cfg, args.steps, args.release, args.epsilons, args.segment_steps,
                               args.geometry, args.friction_update, adjoint=args.adjoint,
                               controls_to_check=args.parameters, integration=args.integration)
                row.update(shape=shape, material=material)
                report['cases'].append(row)
                (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
