"""Natural multi-cut trajectories, history inheritance and first-cut derivatives."""

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp

from blademaster_warp.differentiable_contact import Tool
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.differentiable_natural import CutInstruction
from blademaster_warp.presets import banana_dt, tangent_history_dt
from blademaster_warp.provenance import git_revision
from blademaster_warp.topology import components
from scripts.diff_cut_validate import configuration
from scripts.diff_natural_validate import evaluate, intact_initial


@wp.kernel
def interpolate_controls(
    q: wp.array(dtype=float), points: wp.array(dtype=wp.vec3),
    basis: wp.array2d(dtype=wp.vec3), index: int, fraction: float, duration: float,
    position: wp.array(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3),
):
    a = wp.vec3(points[index])
    b = wp.vec3(points[index + 1])
    for control in range(q.shape[0]):
        a += q[control] * basis[index, control]
        b += q[control] * basis[index + 1, control]
    position[0] = (1.0 - fraction) * a + fraction * b
    velocity[0] = (b - a) / duration


@dataclass
class CutPath:
    name: str
    times: np.ndarray
    positions: np.ndarray
    yaw: np.ndarray
    cutting: np.ndarray
    cut_indices: np.ndarray
    basis: np.ndarray
    objective_start: float
    control_names: tuple = ('first_knife_x_mm', 'first_knife_depth_mm')

    def __post_init__(self):
        n = len(self.times)
        if (n < 2 or self.times[0] != 0 or np.any(np.diff(self.times) <= 0)
                or self.positions.shape != (n, 3) or self.yaw.shape != (n,)
                or self.cutting.shape != (n - 1,) or self.cut_indices.shape != (n - 1,)
                or self.basis.shape != (n, len(self.control_names), 3)
                or self.cut_indices.dtype.kind not in 'iu' or self.cutting.dtype.kind != 'b'
                or np.any(self.cut_indices < 0) or np.any(np.diff(self.cut_indices) < 0)
                or not 0 <= self.objective_start < self.times[-1]
                or not all(np.isfinite(a).all() for a in (self.times, self.positions, self.yaw, self.basis))):
            raise ValueError('Invalid piecewise linear cut path')

    def segment(self, time):
        return min(int(np.searchsorted(self.times[1:], time, side='right')), len(self.times) - 2)

    def instruction(self, time, duration):
        index = self.segment(time)
        return CutInstruction(bool(self.cutting[index]), int(self.cut_indices[index]), duration)

    def bind(self, model, q):
        points = model.array(self.positions, wp.vec3, False)
        basis = model.array(self.basis, wp.vec3, False)

        def tool_at(time):
            index = self.segment(time)
            duration = self.times[index + 1] - self.times[index]
            fraction = (time - self.times[index]) / duration
            position, velocity = model.zeros(1, wp.vec3), model.zeros(1, wp.vec3)
            model.launch(interpolate_controls, 1,
                         inputs=[q, points, basis, index, float(fraction), float(duration)],
                         outputs=[position, velocity], device=model.device)
            angle = (1 - fraction) * self.yaw[index] + fraction * self.yaw[index + 1]
            angular_speed = (self.yaw[index + 1] - self.yaw[index]) / duration
            return Tool(position, velocity,
                        model.array([[0., 0., np.sin(angle / 2), np.cos(angle / 2)]], wp.quat, False),
                        model.array([[0., 0., angular_speed]], wp.vec3, False))

        return tool_at

    def record(self):
        return dict(name=self.name, times_s=self.times.tolist(), positions_m=self.positions.tolist(),
                    yaw_rad=self.yaw.tolist(), cutting=self.cutting.tolist(),
                    cut_indices=self.cut_indices.tolist(), control_names=list(self.control_names),
                    position_basis_m=self.basis.tolist(), objective_start_s=self.objective_start)


def multicut_path(cfg, name):
    if name not in ('resume', 'two-cuts', 'crossing', 'three-cuts-push'):
        raise ValueError('Unknown multi-cut trajectory')
    top, bottom = cfg.size[2] + .002, .0005
    first_x = .0004 if name in ('resume', 'crossing') else -.02
    times, points, yaw, basis = [0.], [[first_x, 0., top]], [0.], []
    cutting, indices = [], []

    def weights(first, depth):
        value = np.zeros((2, 3))
        value[0, 0] = .001 * first
        value[1, 2] = -.001 * depth
        return value

    basis.append(weights(1, 0))

    def append(duration, point, active, cut, *, angle=0., first=0., depth=0.):
        times.append(round(times[-1] + duration, 12))
        points.append(point)
        yaw.append(angle)
        cutting.append(active)
        indices.append(cut)
        basis.append(weights(first, depth))

    if name == 'resume':
        middle = (top + bottom) / 2
        append(.12, [first_x, 0., middle], True, 0, first=1, depth=.5)
        append(.02, [first_x, 0., middle], False, 0, first=1, depth=.5)
        append(.12, [first_x, 0., top], False, 0, first=1)
        append(.12, [first_x, 0., middle], True, 0, first=1, depth=.5)
        objective_start = times[-1]
        append(.12, [first_x, 0., bottom], True, 0, first=1, depth=1)
        append(.24, [first_x, 0., top], False, 0, first=1)
    else:
        append(.24, [first_x, 0., bottom], True, 0, first=1, depth=1)
        append(.24, [first_x, 0., top], False, 0, first=1)
        later = [(.0004, np.pi / 3)] if name == 'crossing' else [(.02, 0.)]
        if name == 'three-cuts-push':
            later = [(.0004, 0.), (.02, 0.)]
        objective_start = times[-1] + .08
        for cut, (x, angle) in enumerate(later, start=1):
            append(.08, [x, 0., top], False, cut - 1, angle=angle)
            append(.24, [x, 0., bottom], True, cut, angle=angle)
            append(.24, [x, 0., top], False, cut, angle=angle)
        if name == 'three-cuts-push':
            cut = len(later)
            append(.08, [-.03, 0., top], False, cut)
            append(.20, [-.03, 0., .008], False, cut)
            append(.10, [-.015, 0., .008], False, cut)
            append(.20, [-.015, 0., top], False, cut)
    positions = np.asarray(points)
    positions[:, 2] += cfg.blade_half_size[2]
    return CutPath(name, np.asarray(times), positions, np.asarray(yaw), np.asarray(cutting),
                   np.asarray(indices), np.asarray(basis), objective_start)


def validate_differences(rows, controls, tolerance=.02):
    for control in controls:
        selected = [row for row in rows if row['parameter'] == control]
        values = [[row[key] for key in ('epsilon', 'ad', 'fd', 'left_fd', 'right_fd', 'relative_error')]
                  for row in selected]
        if not selected or not np.isfinite(values).all():
            raise RuntimeError(f'Missing or nonfinite finite differences for {control}')
        epsilons = {row['epsilon'] for row in selected}
        if len(epsilons) < 2 or min(epsilons) <= 0:
            raise RuntimeError(f'Two distinct positive perturbations are required for {control}')
        if any(row['relative_error'] < 0 or row['relative_error'] > tolerance for row in selected):
            raise RuntimeError(f'Natural finite-difference error exceeds {tolerance:g} for {control}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['resume', 'two-cuts', 'crossing', 'three-cuts-push'],
                        choices=['resume', 'two-cuts', 'crossing', 'three-cuts-push'])
    parser.add_argument('--spacings', nargs='+', type=float, default=[.006])
    parser.add_argument('--materials', nargs='+', default=['banana'], choices=['banana', 'soft', 'plastic'])
    parser.add_argument('--cutting-energy', type=float)
    parser.add_argument('--dt', type=float, help='Material step in seconds; defaults to the stability estimate')
    parser.add_argument('--flip-reference-dt', type=float,
                        help='PIC reference time in seconds; keep fixed during time-step comparisons')
    parser.add_argument('--segment-steps', nargs='+', type=int, default=[32, 127])
    parser.add_argument('--checkpoint-device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--epsilons', nargs='+', type=float, default=[1e-6, 1e-7])
    args = parser.parse_args()
    numbers = [*args.spacings, *args.segment_steps, *args.epsilons]
    numbers.extend(value for value in (args.cutting_energy, args.dt, args.flip_reference_dt)
                   if value is not None)
    if not np.isfinite(numbers).all() or min(numbers) <= 0:
        parser.error('Spacings, times, segment lengths, perturbations and fracture energy must be positive and finite')
    if len(set(args.epsilons)) < 2:
        parser.error('Provide at least two distinct finite-difference perturbations')
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(provenance=git_revision(), command=sys.argv, cases=[])
    for spacing in args.spacings:
        for material in args.materials:
            cfg = replace(configuration(spacing, material), blade_grid_constraint=True)
            dt = args.dt if args.dt is not None else tangent_history_dt(cfg, banana_dt(cfg, spacing))
            reference_dt = dt if args.flip_reference_dt is None else args.flip_reference_dt
            cfg = replace(cfg, dt=dt, flip_reference_dt=reference_dt)
            if args.cutting_energy is not None:
                cfg = replace(cfg, cutting_energy=args.cutting_energy)
            for name in args.cases:
                path = multicut_path(cfg, name)
                steps = round(path.times[-1] / dt)
                if steps < 1 or abs(steps * dt - path.times[-1]) > 1e-12:
                    parser.error('The trajectory duration must be an integer number of material steps')
                row, arrays = evaluate(cfg, steps, 'integrated', args.epsilons, args.segment_steps,
                    'event', 'midpoint', initial_window=intact_initial(cfg), return_arrays=True,
                    adjoint='checkpoint', controls_to_check=path.control_names,
                    checkpoint_device=args.checkpoint_device, path=path, progress=True)
                model = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
                _, sizes = components(model.nodes, model.weights.numpy(), arrays['codes'])
                row.update(material=material, final_piece_sizes=sizes.tolist())
                stem = f'{name}_{material}_h{spacing:g}'
                np.savez_compressed(args.output / f'{stem}.npz', **arrays)
                report['cases'].append(row)
                (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                validate_differences(row['finite_difference'], path.control_names)
                if not np.isfinite(row['pause_damage_max_abs']) or row['pause_damage_max_abs'] != 0:
                    raise RuntimeError('Damage changed while cutting was paused')
                print(json.dumps(dict(case=stem, loss=row['loss'], pieces=row['final_piece_sizes'],
                    max_fd_error=max(entry['relative_error'] for entry in row['finite_difference']))), flush=True)


if __name__ == '__main__':
    main()
