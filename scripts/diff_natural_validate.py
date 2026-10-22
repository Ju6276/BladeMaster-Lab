"""Compare natural cutting rollouts and differentiate across a side-assignment window."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import warp as wp

from blademaster_warp.coupling import ExternalToolPort
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.differentiable_geometry import GeometryEventContact
from blademaster_warp.differentiable_natural import CutInstruction, NaturalCutContact
from blademaster_warp.energy import energy_snapshot
from blademaster_warp.presets import banana_dt, tangent_history_dt
from blademaster_warp.provenance import git_revision
from blademaster_warp.solver import Solver
from scripts.diff_cut_validate import (
    capture_window, configuration, differentiable_tool, numpy_tool, reaction_loss, terminal_loss,
)
from scripts.diff_cut_trajectory import PATH_TIMES, tool_at_time


STATE_PARAMETERS = ('initial_z_mm', 'young_scale', 'yield_scale', 'friction_scale', 'fracture_scale')


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


def intact_initial(cfg):
    m = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    state = m.state()
    values = {name: getattr(state, name).numpy() for name in ('x', 'v', 'F', 'Fp')}
    values['stress'] = np.zeros((m.n, 3, 3))
    saved = dict(rigid=np.zeros((2, m.n, 3)), pairs=np.zeros((0, 3)),
                 damage=np.zeros(m.n), side=np.zeros(m.n))
    return WindowState(np.array([.0004, 0., .027 + cfg.blade_half_size[2]]), values, saved,
                       np.zeros(m.n, dtype=np.int64), np.empty(0, dtype=np.int64), 0., cfg.dt, [])


def time_edges(steps, dt, phase, start, trajectory, knots=None):
    total = steps * dt
    edges = np.arange(steps + 1, dtype=float) * dt
    if phase:
        edges = np.concatenate(([0.], (np.arange(steps) + phase) * dt, [total]))
    if knots is not None or trajectory == 'cut-withdraw-push':
        tolerance = 8 * np.finfo(float).eps * max(total, dt)
        for knot in (PATH_TIMES[1:] if knots is None else knots):
            relative = knot - start
            if 0 < relative < total:
                nearest = int(np.argmin(np.abs(edges - relative)))
                if abs(edges[nearest] - relative) <= tolerance:
                    edges[nearest] = relative
                else:
                    edges = np.sort(np.append(edges, relative))
    return edges


def difference_result(parameter, epsilon, ad, baseline_loss, losses):
    if len(losses) != 2 or not np.isfinite([epsilon, ad, baseline_loss, *losses]).all() or epsilon <= 0:
        raise FloatingPointError(f'Invalid finite-difference inputs for {parameter}, epsilon={epsilon}')
    fd = (losses[1] - losses[0]) / (2 * epsilon)
    left = (baseline_loss - losses[0]) / epsilon
    right = (losses[1] - baseline_loss) / epsilon
    error = abs(ad - fd) / max(abs(ad), abs(fd), 1e-12)
    if not np.isfinite([fd, left, right, error]).all():
        raise FloatingPointError(f'Nonfinite finite difference for {parameter}, epsilon={epsilon}')
    return dict(parameter=parameter, epsilon=epsilon, ad=ad, fd=fd,
                left_fd=left, right_fd=right, relative_error=error)


def event_time_array(events, instructions, edges, particles, start=0., *, multicut=False):
    shape = (max(item.cut_index for item in instructions) + 1, particles)
    times = np.full(shape, np.nan)
    for step, entries in enumerate(events):
        cut = instructions[step].cut_index
        for particle, time in entries.items():
            if np.isfinite(times[cut, particle]):
                raise RuntimeError(f'Duplicate side-assignment event for cut {cut}, particle {particle}')
            times[cut, particle] = start + edges[step] + time
    return times if multicut else times[0]


def evaluate(cfg, steps, release, epsilons, segment_steps, geometry='boundary', friction_update='projected',
             *, initial_window=None, return_arrays=False, adjoint='full',
             controls_to_check=('knife_x_mm', 'knife_depth_scale'), integration='midpoint', phase=0.,
             trajectory='straight', progress=False, state_parameters=(), checkpoint_device=None, path=None):
    directions = dict(knife_x_mm=np.array([1., 0., 0.]),
                      knife_depth_scale=np.array([0., 1., 0.]),
                      knife_lateral_scale=np.array([0., 0., 1.]))
    if path is not None:
        directions = dict(zip(path.control_names, np.eye(len(path.control_names))))
    if adjoint not in ('full', 'checkpoint') or (adjoint == 'checkpoint' and not segment_steps):
        raise ValueError('Use full Tape or checkpoint adjoints with at least one segment length')
    if not controls_to_check or any(name not in directions for name in controls_to_check):
        raise ValueError(f'Select controls from {tuple(directions)}')
    if trajectory not in ('straight', 'cut-withdraw-push'):
        raise ValueError('Trajectory must be straight or cut-withdraw-push')
    if any(name not in STATE_PARAMETERS for name in state_parameters):
        raise ValueError(f'State parameters must be selected from {STATE_PARAMETERS}')
    window = initial_window
    if window is None:
        window = intact_initial(cfg) if path is not None else capture_initial(cfg, steps)
    if not np.isfinite(phase) or not 0 <= phase < 1:
        raise ValueError('Time-grid phase must be in [0, 1)')
    total_duration = steps * cfg.dt
    edges = time_edges(steps, cfg.dt, phase, window.start_time, trajectory,
                       None if path is None else path.times)
    durations = np.diff(edges) if phase or trajectory != 'straight' or path is not None else np.full(steps, cfg.dt)
    instructions = [CutInstruction(duration=float(dt),
                    cutting=trajectory == 'straight' or window.start_time + edges[i] < PATH_TIMES[1])
                    for i, dt in enumerate(durations)]
    if path is not None:
        instructions = [path.instruction(window.start_time + edges[i], float(dt))
                        for i, dt in enumerate(durations)]
    steps = len(durations)
    origin, values, saved, rows = window.origin, window.values, window.history, window.rows
    controls = np.array([0., 1., 1.]) if path is None else np.zeros(len(path.control_names))
    m = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    contact_type = GeometryEventContact if geometry == 'event' else NaturalCutContact
    kwargs = dict(integration=integration) if geometry == 'event' else {}
    c = contact_type(m, release=release, friction_update=friction_update, **kwargs)
    initial = m.state(**values)
    initial_energy = energy_snapshot(m, initial)
    history = c.history(**saved, codes=window.codes, pair_keys=window.pair_keys)
    q = m.array(controls, float)
    path_tool = None if path is None else path.bind(m, q)

    def knife(step):
        if path_tool is not None:
            return path_tool(window.start_time + edges[step])
        if trajectory == 'cut-withdraw-push':
            return tool_at_time(m, q, window.start_time + edges[step])
        return differentiable_tool(m, q, origin, window.start_time + edges[step])

    def objective(step, state, h, force, loss):
        if path is None or window.start_time + edges[step] >= path.objective_start:
            m.launch(reaction_loss, 1, inputs=[force, float(durations[step])], outputs=[loss], device=m.device)
        if step == steps - 1:
            m.launch(terminal_loss, m.n, inputs=[state.v, state.Fp], outputs=[loss], device=m.device)

    def rollout(compare=False, log=False):
        state, h = initial, history
        loss = m.zeros(1, float)
        codes, pairs, released, rigid_modes = [], [], [], []
        geometry_events = []
        forces, damage_work, damage_increment = [], 0., 0.
        milestones = []
        previous_damage = h.damage.numpy()
        old_cut_codes = h.codes.numpy() % (3 ** h.cut_index)
        pause_damage_error = 0.
        error = {name: 0. for name in (*values, 'force', 'rigid_history', 'pair_history', 'damage')}
        discrete = dict(side_particles=0, pair_keys=0)
        for step in range(steps):
            previous_index = h.cut_index
            if path is not None and instructions[step].cut_index != previous_index:
                old_cut_codes = h.codes.numpy() % (3 ** instructions[step].cut_index)
            state, h, force = c.step(state, h, knife(step), instructions[step])
            objective(step, state, h, force, loss)
            codes.append(h.codes.numpy())
            pairs.append(h.pair_keys.copy())
            current_damage = h.damage.numpy()
            released.append(current_damage >= 1)
            rigid_modes.append(h.rigid_modes.numpy())
            geometry_events.append({p: float(t.numpy()[0]) for p, t in h.geometry_events})
            forces.append(force.numpy()[0])
            damage_work += float(np.sum(h.damage_work.numpy()))
            damage_increment += float(np.sum(current_damage) - (
                np.sum(previous_damage) if h.cut_index == previous_index else 0.))
            if path is not None:
                if not np.array_equal(codes[-1] % (3 ** h.cut_index), old_cut_codes):
                    raise RuntimeError('A later cut changed an earlier cut digit')
                if not instructions[step].cutting and h.cut_index == previous_index:
                    pause_damage_error = max(pause_damage_error, float(np.max(np.abs(
                        current_damage - previous_damage))))
            previous_damage = current_damage
            if path is not None and (step == steps - 1 or path.segment(window.start_time + edges[step + 1])
                                     != path.segment(window.start_time + edges[step])):
                milestones.append(dict(time_s=float(window.start_time + edges[step + 1]),
                    cut_index=h.cut_index, cutting=instructions[step].cutting,
                    labelled_per_cut=[int(np.count_nonzero(codes[-1] // 3**i % 3))
                                      for i in range(h.cut_index + 1)],
                    damage_sum=float(np.sum(current_damage)),
                    released_particles=int(np.count_nonzero(current_damage >= 1)),
                    plastic_max_abs=float(np.max(np.abs(state.Fp.numpy() - np.eye(3))))))
                if progress and log:
                    print(json.dumps(dict(stage='milestone', **milestones[-1])), flush=True)
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
                                   force=np.asarray(forces), damage_work=damage_work,
                                   damage_increment=damage_increment, milestones=milestones,
                                   pause_damage_max_abs=pause_damage_error), error, discrete

    compare = path is None and trajectory == 'straight' and phase == 0 and cfg.dt == window.reference_dt and steps == len(rows)
    begun = perf_counter()
    with wp.Tape() if adjoint == 'full' else nullcontext() as tape:
        loss, terminal, terminal_history, events, forward_error, discrete_error = rollout(compare=compare, log=True)
    forward_s = perf_counter() - begun
    final_energy = energy_snapshot(m, terminal)
    if progress:
        print(json.dumps(dict(stage='forward', duration_s=total_duration, elapsed_s=forward_s,
                               loss=float(loss.numpy()[0]))), flush=True)
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
                                  knife, objective, parameters=(q,), timings=timings,
                                  checkpoint_device=checkpoint_device)
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
        checkpoint.append(dict(segment_steps=length, device=timings['checkpoint_device'],
                               controls_max_abs=error_q, initial_x_max_abs=error_x,
                               gradient_reference=gradient_reference,
                               material_max_abs=material_error,
                               recompute_max_abs=timings['recompute_max_abs'],
                               timings={name: timings[name] for name in ('forward_s', 'recompute_s', 'backward_s')}))
        if progress:
            print(json.dumps(dict(stage='checkpoint', segment_steps=length,
                                   timings=checkpoint[-1]['timings'])), flush=True)
    finite_difference = []
    baseline_loss = float(loss.numpy()[0])
    checks = [(name, q, controls, directions[name], float(gradient @ directions[name]))
              for name in controls_to_check]
    material_sources = dict(young_scale='young_modulus', yield_scale='yield_stress',
                            friction_scale='friction', fracture_scale='fracture_energy')
    for name in state_parameters:
        if name == 'initial_z_mm':
            direction = np.zeros_like(values['x'])
            direction[:, 2] = .001
            checks.append((name, initial.x, values['x'], direction, float(np.sum(gradient_x * direction))))
        else:
            key = material_sources[name]
            source = parameters[key]
            base = source.numpy().copy()
            checks.append((name, source, base, base, float(np.sum(parameter_gradients[key] * base))))
    for parameter, source, base, direction, ad in checks:
        for epsilon in epsilons:
            losses, changes = [], []
            for sign in (-1, 1):
                source.assign(base + sign * epsilon * direction)
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
            difference = difference_result(parameter, epsilon, ad, baseline_loss, losses)
            difference['event_changes'] = changes
            finite_difference.append(difference)
            if progress:
                print(json.dumps(dict(stage='finite_difference', parameter=parameter, epsilon=epsilon,
                                       relative_error=finite_difference[-1]['relative_error'])), flush=True)
        source.assign(base)
    if (not np.isfinite([float(loss.numpy()[0]), *gradient]).all()
            or not np.isfinite(gradient_x).all()
            or any(not np.isfinite(g).all() for g in parameter_gradients.values())):
        raise FloatingPointError('Nonfinite objective or gradient')
    new_labels = np.count_nonzero(events['codes'][-1] != history.codes.numpy())
    if not new_labels:
        raise RuntimeError('Selected window contains no new side assignments')
    threshold = cfg.cutting_energy * m.volume**(2 / 3) * (1 - cfg.sharpness) / cfg.sharpness
    work_from_damage = threshold * events['damage_increment']
    result = dict(config=asdict(cfg), release=release, geometry=geometry, friction_update=friction_update,
                integration=integration if geometry == 'event' else 'symplectic',
                adjoint=adjoint, trajectory=trajectory if path is None else path.name, forward_s=forward_s,
                particles=m.n, prefix_steps=round(window.start_time / cfg.dt),
                start_time_s=window.start_time, steps=steps, duration_s=total_duration, time_grid_phase=phase,
                newly_labelled_particles=int(new_labels), peak_pairs=max(map(len, events['pairs'])),
                released_particles=int(np.count_nonzero(terminal_history.damage.numpy() >= 1)),
                peak_force_n=float(np.max(np.linalg.norm(events['force'], axis=1))),
                initial_x_gradient_l2=float(np.linalg.norm(gradient_x)),
                loss=float(loss.numpy()[0]), gradient_controls=gradient.tolist(),
                gradient_materials={name: value.tolist() for name, value in parameter_gradients.items()},
                damage_work_j=events['damage_work'], damage_work_balance_error_j=abs(events['damage_work']-work_from_damage),
                material_energy_initial=initial_energy, material_energy_final=final_energy,
                knife_impulse_ns=np.sum(durations[:, None] * events['force'], axis=0).tolist(),
                vs_original_max_abs=forward_error if compare else None,
                vs_original_discrete=discrete_error if compare else None,
                checkpoint=checkpoint, finite_difference=finite_difference,
                process_peak_live_cuda_pool_bytes=wp.get_mempool_used_mem_high(m.device))
    if path is not None:
        result.update(path=path.record(), milestones=events['milestones'],
                      pause_damage_max_abs=events['pause_damage_max_abs'])
    if not return_arrays:
        return result
    event_times = event_time_array(events['geometry'], instructions, edges, m.n, window.start_time,
                                   multicut=path is not None)
    arrays = {name: getattr(terminal, name).numpy() for name in values}
    arrays.update(codes=terminal_history.codes.numpy(), damage=terminal_history.damage.numpy(),
                  pair_keys=terminal_history.pair_keys.copy(), force=events['force'])
    arrays['event_times' if path is None else 'event_times_by_cut'] = event_times
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spacings', type=float, nargs='+', default=[.006, .004])
    parser.add_argument('--materials', choices=['banana', 'soft', 'plastic'], nargs='+', default=['banana'])
    parser.add_argument('--shapes', choices=['banana', 'block', 'cylinder'], nargs='+', default=['banana'])
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--from-initial', action='store_true')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--trajectory', choices=['straight', 'cut-withdraw-push'], default='straight')
    parser.add_argument('--cutting-energy', type=float)
    parser.add_argument('--segment-steps', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--checkpoint-device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--adjoint', choices=['full', 'checkpoint'], default='full')
    parser.add_argument('--parameters', nargs='+', default=['knife_x_mm', 'knife_depth_scale'],
                        choices=['knife_x_mm', 'knife_depth_scale', 'knife_lateral_scale'])
    parser.add_argument('--state-parameters', nargs='+', choices=STATE_PARAMETERS, default=[])
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
                if args.cutting_energy is not None:
                    cfg = replace(cfg, cutting_energy=args.cutting_energy)
                steps = args.steps if args.duration is None else round(args.duration / dt)
                if args.duration is not None and (steps < 1 or abs(steps * dt - args.duration) > 1e-12):
                    parser.error('Duration must be positive and divide exactly by the time step')
                window = intact_initial(cfg) if args.from_initial else None
                row = evaluate(cfg, steps, args.release, args.epsilons, args.segment_steps,
                               args.geometry, args.friction_update, adjoint=args.adjoint,
                               controls_to_check=args.parameters, integration=args.integration,
                               initial_window=window, trajectory=args.trajectory, progress=True,
                               state_parameters=args.state_parameters,
                               checkpoint_device='cpu' if args.checkpoint_device == 'cpu' else None)
                row.update(shape=shape, material=material)
                report['cases'].append(row)
                (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
