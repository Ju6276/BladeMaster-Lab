"""Compare natural cutting rollouts and differentiate across a side-assignment window."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp

from blademaster_warp.coupling import ExternalToolPort
from blademaster_warp.differentiable_material import MaterialTrajectory
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


def evaluate(cfg, steps, release, epsilons, segment_steps):
    prefix = first_side_prefix(cfg)
    controls = np.array([0., 1., 1.])
    origin, values, saved, rows, _ = capture_window(cfg, prefix, steps, controls, precision='float64')
    m = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    c = NaturalCutContact(m, release=release)
    initial = m.state(**values)
    history = c.history(**saved, codes=rows[0]['codes'], pair_keys=rows[0]['previous'])
    q = m.array(controls, float)

    def knife(step):
        return differentiable_tool(m, q, origin, (prefix + step) * cfg.dt)

    def objective(step, state, h, force, loss):
        m.launch(reaction_loss, 1, inputs=[force, cfg.dt], outputs=[loss], device=m.device)
        if step == steps - 1:
            m.launch(terminal_loss, m.n, inputs=[state.v, state.Fp], outputs=[loss], device=m.device)

    def rollout(compare=False):
        state, h = initial, history
        loss = m.zeros(1, float)
        codes, pairs, released, rigid_modes = [], [], [], []
        error = {name: 0. for name in (*values, 'force', 'rigid_history', 'pair_history', 'damage')}
        discrete = dict(side_particles=0, pair_keys=0)
        for step in range(steps):
            state, h, force = c.step(state, h, knife(step))
            objective(step, state, h, force, loss)
            codes.append(h.codes.numpy())
            pairs.append(h.pair_keys.copy())
            released.append(h.damage.numpy() >= 1)
            rigid_modes.append(h.rigid_modes.numpy())
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
                                   rigid_modes=np.asarray(rigid_modes)), error, discrete

    with wp.Tape() as tape:
        loss, terminal, terminal_history, events, forward_error, discrete_error = rollout(compare=True)
    tape.backward(loss)
    gradient = q.grad.numpy().copy()
    gradient_x = initial.x.grad.numpy().copy()
    tape.zero()
    checkpoint = []
    for length in segment_steps:
        timings = {}
        result = c.checkpoint_vjp(initial, history, [CutInstruction()] * steps, length,
                                  knife, objective, parameters=(q,), timings=timings)
        error_q = float(np.max(np.abs(result.parameter_gradients[q] - gradient)))
        error_x = float(np.max(np.abs(result.initial_gradient.x.numpy() - gradient_x)))
        np.testing.assert_allclose(result.parameter_gradients[q], gradient, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(result.initial_gradient.x.numpy(), gradient_x, rtol=1e-8, atol=1e-8)
        checkpoint.append(dict(segment_steps=length, controls_max_abs=error_q, initial_x_max_abs=error_x,
                               recompute_max_abs=timings['recompute_max_abs']))
    finite_difference = []
    for parameter, direction in (('knife_x_mm', np.array([1., 0., 0.])),
                                 ('knife_depth_scale', np.array([0., 1., 0.]))):
        ad = float(gradient @ direction)
        for epsilon in epsilons:
            losses, changes = [], []
            for sign in (-1, 1):
                q.assign(controls + sign * epsilon * direction)
                shifted_loss, _, _, shifted, _, _ = rollout()
                losses.append(float(shifted_loss.numpy()[0]))
                changes.append(dict(
                    side_particle_steps=int(np.count_nonzero(shifted['codes'] != events['codes'])),
                    final_side_particles=int(np.count_nonzero(shifted['codes'][-1] != events['codes'][-1])),
                    pair_keys=sum(len(np.setxor1d(a, b)) for a, b in zip(shifted['pairs'], events['pairs'])),
                    release_particle_steps=int(np.count_nonzero(shifted['released'] != events['released'])),
                    rigid_modes=int(np.count_nonzero(shifted['rigid_modes'] != events['rigid_modes'])),
                ))
            fd = (losses[1] - losses[0]) / (2 * epsilon)
            finite_difference.append(dict(parameter=parameter, epsilon=epsilon, ad=ad, fd=fd,
                                           relative_error=abs(ad - fd) / max(abs(ad), abs(fd), 1e-12),
                                           event_changes=changes))
    if not np.isfinite([float(loss.numpy()[0]), *gradient]).all():
        raise FloatingPointError('Nonfinite objective or gradient')
    new_labels = np.count_nonzero(events['codes'][-1] != history.codes.numpy())
    if not new_labels:
        raise RuntimeError('Selected window contains no new side assignments')
    return dict(config=asdict(cfg), release=release, particles=m.n, prefix_steps=prefix,
                start_time_s=prefix * cfg.dt, steps=steps, duration_s=steps * cfg.dt,
                newly_labelled_particles=int(new_labels), peak_pairs=max(map(len, events['pairs'])),
                loss=float(loss.numpy()[0]), gradient_controls=gradient.tolist(),
                vs_original_max_abs=forward_error, vs_original_discrete=discrete_error,
                checkpoint=checkpoint, finite_difference=finite_difference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spacings', type=float, nargs='+', default=[.006, .004])
    parser.add_argument('--materials', choices=['banana', 'soft', 'plastic'], nargs='+', default=['banana'])
    parser.add_argument('--shapes', choices=['banana', 'block', 'cylinder'], nargs='+', default=['banana'])
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--segment-steps', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--release', choices=['boundary', 'integrated'], default='integrated')
    parser.add_argument('--epsilons', type=float, nargs='+', default=[.05, .005, .0005, .00005])
    args = parser.parse_args()
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
                row = evaluate(cfg, args.steps, args.release, args.epsilons, args.segment_steps)
                row.update(shape=shape, material=material)
                report['cases'].append(row)
                (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
