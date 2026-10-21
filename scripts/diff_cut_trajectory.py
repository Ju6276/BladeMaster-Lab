"""Record and differentiate a complete prescribed banana cut, withdrawal and push."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import warp as wp

from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.differentiable_contact import Tool
from blademaster_warp.differentiable_cutting import FixedCutContact, pair_snapshot
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.presets import banana_dt, tangent_history_dt
from blademaster_warp.provenance import git_revision
from blademaster_warp.solver import Config, Solver
from blademaster_warp.topology import components
from scripts.diff_cut_validate import configuration, reaction_loss, terminal_loss


@wp.kernel
def path_motion(q: wp.array(dtype=float), base: wp.vec3, speed: wp.vec3, origin: wp.vec3,
                position: wp.array(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3)):
    position[0] = wp.vec3(origin[0] + 0.001 * q[0] + q[2] * (base[0] - origin[0]),
                          base[1], origin[2] + q[1] * (base[2] - origin[2]))
    velocity[0] = wp.vec3(q[2] * speed[0], speed[1], q[1] * speed[2])


def motion(cfg, elapsed, controls):
    times = np.array([0., .24, .48, .55, .75, .85, 1.05])
    positions = np.array([[.0004, 0, .027], [.0004, 0, .0005], [.0004, 0, .032],
                          [-.008, 0, .032], [-.008, 0, .008], [.008, 0, .008], [.008, 0, .032]])
    positions[:, 2] += cfg.blade_half_size[2]
    segment = min(np.searchsorted(times[1:], elapsed, side='right'), len(times) - 2)
    speed = (positions[segment + 1] - positions[segment]) / (times[segment + 1] - times[segment])
    base = positions[segment] + (elapsed - times[segment]) * speed
    origin = positions[0]
    q = np.asarray(controls)
    position = origin + (base - origin) * [q[2], 1, q[1]] + [0.001 * q[0], 0, 0]
    velocity = speed * [q[2], 1, q[1]]
    return ToolState(position, [0, 0, 0, 1], velocity, [0, 0, 0]), base, speed, origin


def tool_at(model, q, step):
    _, base, speed, origin = motion(model.config, step * model.config.dt, [0, 1, 1])
    position, velocity = model.zeros(1, wp.vec3), model.zeros(1, wp.vec3)
    vector = wp.vec3d if model.precision == "float64" else wp.vec3
    model.launch(path_motion, 1, inputs=[q, vector(*base), vector(*speed), vector(*origin)],
              outputs=[position, velocity], device=model.device)
    return Tool(position, velocity, model.array([[0, 0, 0, 1]], wp.quat, False), model.zeros(1, wp.vec3, False))


def objective(model, steps):
    def accumulate(step, state, history, force, loss):
        model.launch(reaction_loss, 1, inputs=[force, 1 / steps], outputs=[loss], device=model.device)
        if step == steps - 1:
            model.launch(terminal_loss, model.n, inputs=[state.v, state.Fp], outputs=[loss], device=model.device)
    return accumulate


def freeze_archive(contact, archive):
    frozen = []
    previous = np.empty(0, np.int64)
    for i, code_index in enumerate(archive['code_index']):
        lo, hi = archive['pair_offsets'][i:i + 2]
        keys = archive['pair_keys'][lo:hi]
        branch = contact.freeze(archive['codes'][code_index], keys, previous, archive['released'][i],
                                cutting=bool(archive['cutting'][i]), rigid_modes=archive['rigid_modes'][i],
                                pair_modes=archive['pair_modes'][lo:hi], yielded=archive['yielded'][i],
                                grid_batches=int(archive['grid_batches'][i]))
        frozen.append(branch)
        previous = keys
    return frozen


def record(cfg, controls, duration, output, frame_stride, ordered_transfers=False, precision="float32",
           reference_precision="float32"):
    original = Solver(cfg, precision=reference_precision)
    port = ExternalToolPort(original)
    model = MaterialTrajectory(cfg, ordered_transfers=ordered_transfers, precision=precision)
    contact = FixedCutContact(model)
    values = {name: getattr(original, name).numpy() for name in ('x', 'v', 'F', 'Fp', 'stress')}
    initial = model.state(**values)
    initial_history = contact.history()
    state, history = initial, initial_history
    steps = round(duration / cfg.dt)
    q = model.array(controls, float)
    rows = {key: [] for key in ('code_index', 'released', 'rigid_modes', 'yielded', 'pair_keys',
                                'pair_modes', 'grid_batches', 'cutting')}
    code_table, code_lookup, offsets = [], {}, [0]
    frames = {key: [] for key in ('time', 'x', 'v', 'F', 'codes', 'tool', 'force', 'pieces',
                                  'reference_x', 'reference_v', 'reference_F', 'reference_force', 'pair_count')}
    maximum = dict.fromkeys((*values, 'force'), 0.)
    original_s = replay_s = 0.
    previous = np.empty(0, np.int64)
    frozen = []
    begun = perf_counter()
    peak_pairs, peak_residual = 0, 0.
    force_samples = []

    def frame(elapsed, tool, codes, force, reference_force, pair_count):
        labels, sizes = components(model.nodes, model.weights.numpy(), codes)
        snapshot = dict(time=elapsed, x=state.x.numpy(), v=state.v.numpy(), F=state.F.numpy(), codes=codes,
                        tool=np.r_[tool.position, tool.orientation_xyzw], force=force, pieces=labels,
                        reference_x=original.x.numpy(), reference_v=original.v.numpy(),
                        reference_F=original.F.numpy(), reference_force=reference_force, pair_count=pair_count)
        for key, value in snapshot.items():
            frames[key].append(value)
        return sizes

    tool = motion(cfg, 0., controls)[0]
    sizes = frame(0., tool, original.codes.numpy(), np.zeros(3), np.zeros(3), 0)
    for i in range(steps):
        elapsed = i * cfg.dt
        tool = motion(cfg, elapsed, controls)[0]
        cutting = elapsed < .24
        codes = original.codes.numpy()
        key = codes.tobytes()
        if key not in code_lookup:
            code_lookup[key] = len(code_table)
            code_table.append(codes)
        released = original.damage.numpy() >= 1.
        start = perf_counter()
        reaction = port.advance(tool, cfg.dt, cutting=cutting, graph=False)
        original_s += perf_counter() - start
        keys, _ = pair_snapshot(original)
        b = original.blade_constraint
        batches = int(b.batches.numpy()[0]) if b.needs_solve.numpy()[0] else 0
        branch = contact.freeze(codes, keys, previous, released, cutting=cutting, grid_batches=batches)
        start = perf_counter()
        state, history, force = contact.step(state, history, tool_at(model, q, i), branch)
        force_np = force.numpy()[0]
        replay_s += perf_counter() - start
        if not np.isfinite(force_np).all():
            raise FloatingPointError(f'Nonfinite force at step {i}')
        branch.rigid_modes = history.rigid_modes
        branch.pair_modes = history.pair_modes
        branch.yielded = state.yielded
        frozen.append(branch)
        rows['code_index'].append(code_lookup[key])
        rows['released'].append(released)
        rows['rigid_modes'].append(history.rigid_modes.numpy().astype(np.int8))
        rows['yielded'].append(state.yielded.numpy().astype(np.int8))
        rows['pair_keys'].append(keys)
        rows['pair_modes'].append(history.pair_modes.numpy().astype(np.int8))
        rows['grid_batches'].append(batches)
        rows['cutting'].append(cutting)
        offsets.append(offsets[-1] + len(keys))
        previous = keys
        force_samples.append(force_np)
        maximum['force'] = max(maximum['force'], float(np.max(np.abs(force_np - reaction.force_n))))
        peak_pairs = max(peak_pairs, len(keys))
        peak_residual = max(peak_residual, history.grid_residual)
        if (i + 1) % frame_stride == 0 or i + 1 == steps:
            for name in values:
                actual = getattr(state, name).numpy()
                if not np.isfinite(actual).all():
                    raise FloatingPointError(f'Nonfinite {name} at step {i}')
                maximum[name] = max(maximum[name], float(np.max(np.abs(actual - getattr(original, name).numpy()))))
            sizes = frame((i + 1) * cfg.dt, tool, original.codes.numpy(), force_np, reaction.force_n, len(keys))
            if (i + 1) % (frame_stride * 10) == 0:
                print(json.dumps(dict(step=i + 1, steps=steps, time_s=(i + 1) * cfg.dt,
                                      pieces=sizes.tolist(), forward_max_abs=maximum,
                                      elapsed_s=perf_counter() - begun)), flush=True)
    errors = original.errors.numpy().tolist()
    archive = {key: np.asarray(value) for key, value in rows.items() if key not in ('pair_keys', 'pair_modes')}
    archive.update(codes=np.asarray(code_table), pair_offsets=np.asarray(offsets),
                   pair_keys=np.concatenate(rows['pair_keys']), pair_modes=np.concatenate(rows['pair_modes']))
    archive.update({f'initial_{name}': value for name, value in values.items()})
    np.savez_compressed(output / 'branches.npz', **archive)
    np.savez_compressed(output / 'trajectory.npz', **{key: np.asarray(value) for key, value in frames.items()})
    last = state
    loss = float(np.mean(np.asarray(force_samples).astype(float) @ [0.3, .2, -1])
                 + np.mean(20 * np.sum(last.v.numpy().astype(float)**2, axis=1)
                           + .1 * np.sum(last.Fp.numpy().astype(float)**2, axis=(1, 2))))
    result = dict(steps=steps, duration_s=steps * cfg.dt, particles=model.n, loss=loss,
                  original_s=original_s, differentiable_forward_s=replay_s,
                  recording_wall_s=perf_counter() - begun, forward_max_abs=maximum,
                  original_errors=errors, final_piece_sizes=sizes.tolist(), peak_pairs=peak_pairs,
                  peak_grid_residual=peak_residual, grid_solve_steps=int(np.count_nonzero(rows['grid_batches'])))
    return model, contact, initial, initial_history, frozen, result


def replay(model, contact, initial, history, frozen, controls):
    q = model.array(controls, float)
    state, h = initial, history
    total = 0.
    for step, branch in enumerate(frozen):
        state, h, force = contact.step(state, h, tool_at(model, q, step), branch)
        total += float(force.numpy()[0].astype(float) @ [.3, .2, -1]) / len(frozen)
    total += float(np.mean(20 * np.sum(state.v.numpy().astype(float)**2, axis=1)
                          + .1 * np.sum(state.Fp.numpy().astype(float)**2, axis=(1, 2))))
    return total


def render(directory, comparison=False, playback_speed=.1):
    import subprocess
    from PIL import Image, ImageDraw, ImageFont
    from blademaster_warp.replay_style import ReplayScene
    from blademaster_warp.surface_runtime import SurfaceRecorder
    from blademaster_warp.topology import piece_rgb

    report = json.loads((directory / 'report.json').read_text())
    cfg = Config(**report['config'])
    frames = np.load(directory / 'trajectory.npz')
    geometry = Solver(cfg)
    if comparison and 'reference_x' not in frames:
        raise ValueError('Comparison rendering requires recorded original-solver states')
    surfaces = [SurfaceRecorder(geometry, None, embedding_backend='cuda', ccd_backend='cuda')
                for _ in range(2 if comparison else 1)]
    board = dict(size_mm=[190, 130, 8], center_mm=[0, 0, -4])
    scenes = [ReplayScene(asdict(cfg), board, 640, 600, style='paper') for _ in range(2)]
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = corners * np.asarray(cfg.blade_half_size)
    rotations = frames['tool'][:, 3:]
    cross = 2 * np.cross(rotations[:, None, :3], corners[None, :, :])
    rotated = (corners[None, :, :] + rotations[:, None, 3:] * cross
               + np.cross(rotations[:, None, :3], cross))
    tool_points = rotated + frames['tool'][:, None, :3]
    point_sets = [frames['x'].reshape(-1, 3), tool_points.reshape(-1, 3)]
    if comparison:
        point_sets.append(frames['reference_x'].reshape(-1, 3))
    points = 1000 * np.concatenate(point_sets)
    for scene in scenes:
        camera = scene.renderer.GetActiveCamera()
        forward = np.asarray(camera.GetDirectionOfProjection())
        right = np.cross(forward, camera.GetViewUp())
        right /= np.linalg.norm(right)
        basis = np.array([right, np.cross(right, forward), forward])
        projected = points @ basis.T
        low, high = projected.min(axis=0), projected.max(axis=0)
        center = (low + high) / 2 @ basis
        distance = np.linalg.norm(np.asarray(camera.GetPosition()) - camera.GetFocalPoint())
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center - distance * forward))
        half_width, half_height = (high - low)[:2] / 2
        camera.SetParallelScale(max(half_height, half_width / (640 / 600)) + 1000 * cfg.grid_spacing)
    font = ImageFont.truetype('DejaVuSans.ttf', 22)
    small = ImageFont.truetype('DejaVuSans.ttf', 18)
    stem = 'solver_comparison' if comparison else 'differentiable_cutting'
    suffix = '' if playback_speed == .1 else f'_{playback_speed:g}x'
    filename = directory / f'{stem}{suffix}.mp4'
    output_fps = 20
    duration = float(frames['time'][-1])
    output_times = np.minimum(np.arange(int(np.ceil(duration * output_fps / playback_speed)) + 1)
                              * playback_speed / output_fps, duration)
    indices = np.searchsorted(frames['time'], output_times)
    previous = np.maximum(indices - 1, 0)
    indices = np.where(abs(frames['time'][previous] - output_times)
                       < abs(frames['time'][indices] - output_times), previous, indices)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1280x720', '-r', str(output_fps), '-i', '-', '-an', '-c:v', 'libx264',
               '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(filename)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        # Surface topology consumes every recorded observation, including frames skipped at faster playback.
        output_cursor = 0
        for index, elapsed in enumerate(frames['time']):
            tool = ToolState(frames['tool'][index, :3], frames['tool'][index, 3:], [0, 0, 0], [0, 0, 0])
            for surface_index, surface in enumerate(surfaces):
                prefix = 'reference_' if comparison and surface_index == 0 else ''
                surface.update(float(elapsed), frames[f'{prefix}x'][index], frames[f'{prefix}F'][index],
                               frames['codes'][index], tool, cutting=elapsed <= .24)
            if output_cursor == len(indices) or indices[output_cursor] != index:
                continue
            ids = frames['pieces'][index]
            colors = piece_rgb(ids, [.985, .95, .78])
            image = Image.new('RGB', (1280, 720), (245, 248, 248))
            for column, scene in enumerate(scenes):
                prefix = 'reference_' if comparison and column == 0 else ''
                surface = surfaces[column if comparison else 0]
                scene.update(surface.current, frames[f'{prefix}x'][index], frames['codes'][index], colors)
                image.paste(Image.fromarray(scene.frame(particles=bool(column) and not comparison)), (column * 640, 60))
            draw = ImageDraw.Draw(image)
            phase = 'Cut' if elapsed <= .24 else 'Withdraw' if elapsed <= .48 else 'Reposition / push' if elapsed <= .85 else 'Withdraw'
            title = 'Original vs differentiable TLMPM' if comparison else 'Differentiable TLMPM | Banana cutting'
            draw.text((20, 10), title, font=font, fill=(30, 45, 50))
            draw.text((940, 12), f'{phase}  |  {elapsed:.3f} s', font=small, fill=(30, 45, 50))
            if comparison:
                draw.text((20, 37), f"Original solver | {report['reference_precision']}", font=small, fill=(60, 70, 75))
                draw.text((660, 37), f"Differentiable solver | {report['precision']}", font=small, fill=(60, 70, 75))
                for column, key in enumerate(('reference_force', 'force')):
                    draw.text((20 + 640 * column, 667), f"Knife reaction: {np.linalg.norm(frames[key][index]):.3f} N",
                              font=small, fill=(30, 45, 50))
                error = np.max(np.abs(frames['x'][index] - frames['reference_x'][index]))
                detail = f'Same material, path, step and camera | {playback_speed:g}x speed | position difference: {error:.2e} m'
                draw.text((20, 694), detail, font=small, fill=(30, 45, 50))
            else:
                draw.text((20, 37), 'Surface', font=small, fill=(60, 70, 75))
                draw.text((660, 37), 'Simulated particles | color by separated piece', font=small, fill=(60, 70, 75))
                draw.text((20, 681), f"{int(ids.max()) + 1} pieces | {len(ids)} particles | {playback_speed:g}x speed",
                          font=small, fill=(30, 45, 50))
                draw.text((660, 681), f"Knife reaction: {np.linalg.norm(frames['force'][index]):.3f} N | fixed cutting branches",
                          font=small, fill=(30, 45, 50))
            while output_cursor < len(indices) and indices[output_cursor] == index:
                proc.stdin.write(np.asarray(image).tobytes())
                output_cursor += 1
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError('ffmpeg failed')
    finally:
        for surface in surfaces:
            surface.close()
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
    key = ('comparison_video' if comparison else 'video') + suffix
    source = ('Solver and MaterialTrajectory/FixedCutContact particle states' if comparison
              else 'MaterialTrajectory and FixedCutContact particle states')
    report[key] = dict(file=filename.name, frames=len(indices), fps=output_fps,
                       playback_speed=playback_speed, source=source)
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(filename, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spacing', type=float, default=.006)
    parser.add_argument('--material', choices=['banana', 'soft', 'plastic'], default='banana')
    parser.add_argument('--duration', type=float, default=1.05)
    parser.add_argument('--dt', type=float, help='Material step in seconds; default satisfies grid and sticking-history bounds')
    parser.add_argument('--controls', type=float, nargs=3, default=[0., 1., 1.])
    parser.add_argument('--frame-stride', type=int, default=160)
    parser.add_argument('--segment-steps', type=int, default=32)
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--reuse', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--comparison-video', action='store_true', help='Render original and differentiable states side by side')
    parser.add_argument('--playback-speed', type=float, default=.1, help='Video simulation seconds per playback second')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--ordered-transfers', action='store_true')
    parser.add_argument('--precision', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--reference-precision', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--epsilons', type=float, nargs='+', default=[])
    args = parser.parse_args()
    if not np.isfinite(args.playback_speed) or args.playback_speed <= 0:
        parser.error('Playback speed must be finite and positive')
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'report.json'
    if args.render_only:
        render(args.output, args.comparison_video, args.playback_speed)
        return
    if args.reuse:
        report = json.loads(report_path.read_text())
        if report['provenance'] != git_revision():
            raise ValueError('Recorded trajectory revision differs from the current worktree')
        cfg = Config(**report['config'])
        controls = np.asarray(report['controls'])
        archive = dict(np.load(args.output / 'branches.npz'))
        model = MaterialTrajectory(cfg, ordered_transfers=report.get("ordered_transfers", False), precision=report.get("precision", "float32"))
        contact = FixedCutContact(model)
        initial = model.state(**{key[8:]: value for key, value in archive.items() if key.startswith('initial_')})
        history = contact.history()
        frozen = freeze_archive(contact, archive)
    else:
        cfg = replace(configuration(args.spacing, args.material), blade_grid_constraint=True)
        dt = tangent_history_dt(cfg, banana_dt(cfg, args.spacing)) if args.dt is None else args.dt
        cfg = replace(cfg, dt=dt)
        if cfg.dt <= 0 or args.duration <= 0 or args.frame_stride < 1:
            parser.error("Step, duration and frame stride must be positive")
        controls = np.asarray(args.controls)
        model, contact, initial, history, frozen, forward = record(
            cfg, controls, args.duration, args.output, args.frame_stride, args.ordered_transfers, args.precision,
            args.reference_precision)
        report = dict(provenance=git_revision(), command=sys.argv, config=asdict(cfg),
                      controls=controls.tolist(), ordered_transfers=args.ordered_transfers, precision=args.precision,
                      reference_precision=args.reference_precision, forward=forward)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    if args.backward:
        q = model.array(controls, float)
        timings = {}
        result = contact.checkpoint_vjp(initial, history, frozen, args.segment_steps,
                                        lambda step: tool_at(model, q, step), objective(model, len(frozen)),
                                        parameters=(q,), timings=timings)
        gradient = result.parameter_gradients[q]
        parameters = dict(controls=gradient.tolist(), young_modulus=result.parameter_gradients[model.young].tolist(),
                          yield_stress=result.parameter_gradients[model.yield_stress].tolist(),
                          friction=result.parameter_gradients[contact.friction].tolist())
        report['backward'] = dict(segment_steps=args.segment_steps, loss=float(result.loss.numpy()[0]),
                                  gradients=parameters, timings=timings,
                                  finite=all(np.isfinite(value).all() for value in parameters.values()))
        report['backward_command'] = sys.argv
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        direction = np.array([.7, .3, .4])
        ad = float(gradient @ direction)
        checks = []
        for epsilon in args.epsilons:
            lo = replay(model, contact, initial, history, frozen, controls - epsilon * direction)
            hi = replay(model, contact, initial, history, frozen, controls + epsilon * direction)
            fd = (hi - lo) / (2 * epsilon)
            checks.append(dict(epsilon=epsilon, ad=ad, fd=fd, abs_error=abs(ad-fd),
                               relative_error=abs(ad-fd) / max(abs(ad), abs(fd), 1e-12)))
            report['finite_difference'] = checks
            report_path.write_text(json.dumps(report, indent=2) + '\n')
    report['peak_live_cuda_pool_bytes'] = wp.get_mempool_used_mem_high(model.device)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    if args.video or args.comparison_video:
        render(args.output, args.comparison_video, args.playback_speed)
    print(json.dumps(dict(provenance=report['provenance'], forward=report['forward'],
                          backward={key: value for key, value in report.get('backward', {}).items() if key != 'timings'},
                          finite_difference=report.get('finite_difference', []))), flush=True)


if __name__ == '__main__':
    main()
