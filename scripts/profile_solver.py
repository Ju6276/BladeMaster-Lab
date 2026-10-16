"""Warm end-to-end and CUDA activity timing; no physics parameters changed."""
import argparse
from collections import defaultdict
import cProfile
import json
from pathlib import Path
import pstats
import time
import warp as wp
from blademaster_warp.run import preset, banana_target
from blademaster_warp.solver import Solver


def profile(output, name='dexlab-baseline', steps=1000):
    cfg = preset(name)
    s = Solver(cfg)
    s.bx.assign([[-.03, 0., .036 + cfg.blade_half_size[2]]])
    for i in range(round(1.0 / cfg.dt)):
        target, cut, active = banana_target(i * cfg.dt)
        s.step(target, active, cut)
    s.state()
    captures = []
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    for _ in range(steps):
        target, cut, active = banana_target(s.step_count * cfg.dt)
        s.step(target, active, cut)
    wp.synchronize()
    profiler.disable()
    wall = time.perf_counter() - start
    # CUDA events add overhead: keep this separate from uninstrumented timing.
    with wp.ScopedTimer('gpu', print=False, cuda_filter=wp.TIMING_ALL) as timer:
        for _ in range(100):
            target, cut, active = banana_target(s.step_count * cfg.dt)
            s.step(target, active, cut)
    captures = timer.timing_results
    aggregate = defaultdict(lambda: {'count': 0, 'ms': 0.})
    for r in captures:
        aggregate[r.name]['count'] += 1
        aggregate[r.name]['ms'] += r.elapsed
    start = time.perf_counter()
    for _ in range(100):
        s.state()
        s.reaction.numpy()
        s.work.numpy()
    state_wall = time.perf_counter() - start
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {'preset': name, 'device': s.device.name, 'steps': steps,
              'wall_with_cprofile_s': wall, 'state_capture_mean_ms': state_wall * 10,
              'gpu_activities_100_steps': dict(sorted(aggregate.items(), key=lambda kv: -kv[1]['ms']))}
    output.write_text(json.dumps(result, indent=2) + '\n')
    profiler.dump_stats(str(output.with_suffix('.pstats')))
    pstats.Stats(profiler).sort_stats('cumtime').print_stats(12)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--preset', default='dexlab-baseline')
    p.add_argument('--steps', type=int, default=1000)
    a = p.parse_args()
    profile(a.output, a.preset, a.steps)
