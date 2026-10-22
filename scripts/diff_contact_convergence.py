"""Compare natural penalty-contact trajectories with an analytic elastic rebound."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import warp as wp

from blademaster_warp.differentiable_geometry import GeometryEventContact
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.differentiable_natural import CutInstruction
from blademaster_warp.provenance import git_revision
from blademaster_warp.solver import Config
from scripts.diff_natural_validate import time_edges


def rebound(gap, speed, time, stiffness, gravity):
    omega = np.sqrt(stiffness)
    incoming = -np.sqrt(speed * speed + 2 * gravity * gap)
    incoming_derivative = gravity / incoming
    entry = (speed - incoming) / gravity if gravity else -gap / speed
    entry_derivative = -1 / incoming
    if time <= entry:
        return np.array([gap + speed * time - .5 * gravity * time**2, speed - gravity * time]), np.array([1., 0.])
    departure = 2 * (np.pi + np.arctan(incoming * omega / gravity)) / omega if gravity else np.pi / omega
    departure_derivative = 2 / (incoming * (1 + (incoming * omega / gravity)**2)) if gravity else 0.
    elapsed = time - entry
    if elapsed < departure:
        angle = omega * elapsed
        angle_derivative = -omega * entry_derivative
        state = np.array([gravity / stiffness * (np.cos(angle) - 1) + incoming / omega * np.sin(angle),
                          -gravity / omega * np.sin(angle) + incoming * np.cos(angle)])
        gradient = np.array([
            (-gravity / stiffness * np.sin(angle) + incoming / omega * np.cos(angle)) * angle_derivative
            + incoming_derivative / omega * np.sin(angle),
            (-gravity / omega * np.cos(angle) - incoming * np.sin(angle)) * angle_derivative
            + incoming_derivative * np.cos(angle),
        ])
        return state, gradient
    elapsed -= departure
    elapsed_derivative = -entry_derivative - departure_derivative
    state = np.array([-incoming * elapsed - .5 * gravity * elapsed**2, -incoming - gravity * elapsed])
    gradient = np.array([-incoming_derivative * elapsed + state[1] * elapsed_derivative,
                         -incoming_derivative - gravity * elapsed_derivative])
    return state, gradient


def evaluate(surface, spacing, dt, duration, phase, epsilons):
    cfg = Config(size=(2 * spacing, 2 * spacing, 2 * spacing), grid_spacing=spacing, dt=dt,
                 young_modulus=20000, yield_stress=0, external_tool=True, blade_grid_constraint=False,
                 contact_reference_dt=.0002, flip_reference_dt=.0002,
                 friction_static=0., friction_dynamic=0., tangent_damping=0.)
    m = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    c = GeometryEventContact(m)
    gap, speed = .0001, -.12
    x, v = m.reference.copy(), np.zeros((m.n, 3))
    direction = np.zeros_like(x)
    seed = np.zeros_like(x)
    codes = np.zeros(m.n, dtype=np.int64)
    gravity = 9.81 if surface == 'board' else 0.
    axis = 2 if surface == 'board' else 0
    contact_multiplicity = 1
    if surface == 'pair':
        right = x[:, 0] > 0
        contact_multiplicity = len(np.unique(x[~right, 0]))
        sign = np.where(right, 1., -1.)
        x[:, 0] = sign * (spacing / 2 + gap) / 2
        x[:, 2] += .1
        v[:, 0] = sign * speed / 2
        direction[:, 0] = sign / 2
        seed[:, 0] = sign * 2 / m.n
        codes = np.where(right, 2, 1)
        offset = spacing / 2
        tool = c.tool([1, 1, 1])
    else:
        offset = spacing / 4
        if surface == 'blade':
            offset += cfg.blade_half_size[0]
            x[:, 2] += .1
        x[:, axis] = offset + gap
        v[:, axis] = speed
        direction[:, axis] = 1.
        seed[:, axis] = 1 / m.n
        tool = c.tool([0, 0, .1 + spacing / 2] if surface == 'blade' else [1, 1, 1])
    initial, history = m.state(x=x, v=v), c.history(codes=codes)
    edges = time_edges(round(duration / dt), dt, phase, 0., 'straight')
    instructions = [CutInstruction(cutting=False, duration=float(t)) for t in np.diff(edges)]

    def rollout():
        state, h = initial, history
        for instruction in instructions:
            state, h, _ = c.step(state, h, tool, instruction)
        return state

    def values(state):
        return np.array([np.sum(state.x.numpy() * seed) - offset, np.sum(state.v.numpy() * seed)])

    begun = perf_counter()
    with wp.Tape() as tape:
        terminal = rollout()
    state = values(terminal)
    ad = []
    for name in ('x', 'v'):
        tape.backward(grads={getattr(terminal, name): m.array(seed, wp.vec3, False)})
        ad.append(float(np.sum(initial.x.grad.numpy() * direction)))
        tape.zero()
    del tape
    finite_difference = []
    for epsilon in epsilons:
        shifted = []
        for sign in (-1, 1):
            initial.x.assign(x + sign * epsilon * direction)
            shifted.append(values(rollout()))
        fd = (shifted[1] - shifted[0]) / (2 * epsilon)
        finite_difference.append(dict(epsilon_m=epsilon, central=fd.tolist(),
                                       left=((state - shifted[0]) / epsilon).tolist(),
                                       right=((shifted[1] - state) / epsilon).tolist(),
                                       ad_absolute_error=np.abs(fd - ad).tolist()))
    stiffness = contact_multiplicity * cfg.contact_scales()[0] / dt**2
    exact, gradient = rebound(gap, speed, duration, stiffness, gravity)
    return dict(surface=surface, config=asdict(cfg), phase=phase, steps=len(instructions),
                contact_multiplicity=contact_multiplicity, normal_stiffness_s_inv2=stiffness,
                state=state.tolist(), exact_state=exact.tolist(), state_absolute_error=np.abs(state-exact).tolist(),
                gradient=ad, exact_gradient=gradient.tolist(), gradient_absolute_error=np.abs(ad-gradient).tolist(),
                finite_difference=finite_difference, elapsed_s=perf_counter()-begun,
                process_peak_live_cuda_pool_bytes=wp.get_mempool_used_mem_high(m.device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spacings', type=float, nargs='+', default=[.006, .004])
    parser.add_argument('--surfaces', choices=['board', 'blade', 'pair'], nargs='+', default=['board', 'blade', 'pair'])
    parser.add_argument('--dt-divisors', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--phases', type=float, nargs='+', default=[0., .25, .5])
    parser.add_argument('--epsilons', type=float, nargs='+', default=[1e-8, 1e-10])
    args = parser.parse_args()
    if min(*args.spacings, *args.dt_divisors, *args.epsilons) <= 0 or not np.isfinite(
            [*args.spacings, *args.dt_divisors, *args.epsilons, *args.phases]).all():
        parser.error('Spacing, divisors and perturbations must be positive and finite')
    if min(args.phases) < 0 or max(args.phases) >= 1:
        parser.error('Phases must be in [0, 1)')
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(provenance=git_revision(), command=sys.argv, cases=[])
    for surface in args.surfaces:
        for spacing in args.spacings:
            for phase in args.phases:
                for divisor in args.dt_divisors:
                    row = evaluate(surface, spacing, .00005 / divisor, .004, phase, args.epsilons)
                    report['cases'].append(row)
                    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                    print(json.dumps(dict(surface=surface, spacing=spacing, phase=phase, divisor=divisor,
                                          gradient_error=row['gradient_absolute_error'], elapsed_s=row['elapsed_s'])), flush=True)


if __name__ == '__main__':
    main()
