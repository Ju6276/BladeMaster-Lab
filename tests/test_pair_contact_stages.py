"""Two separated material fields with particle-pair friction in both stages."""

import numpy as np
import warp as wp

from blademaster_warp.cohesive_history import CohesiveHistory
from blademaster_warp.cohesive_pair_contact import predict_pair_contacts
from blademaster_warp.cohesive_stages import AffineState, advance_affine
from blademaster_warp.compensated_state import CompensatedArray
from test_cohesive_stages import setup


def prepared_pairs(dt=2e-6):
    m, _, _, _, old = setup()
    transfer = old["transfer"]
    retained = CohesiveHistory.empty(m.n, device=m.device)
    sign = np.where(m.reference[:, 0] < 0, 1.0, -1.0)
    x = m.reference + sign[:, None] * np.array([1e-4, 0, 0])
    velocity = sign[:, None] * np.array([0.1, 0.2, 0.0])
    state = AffineState.from_material(m.state(x=x, v=velocity), m.zeros(m.n, wp.mat33))
    left = np.flatnonzero(m.reference[:, 0] == np.max(m.reference[sign > 0, 0]))
    right = np.flatnonzero(m.reference[:, 0] == np.min(m.reference[sign < 0, 0]))
    match = {tuple(m.reference[q, 1:]): int(q) for q in right}
    pairs = np.asarray(sorted((int(p), match[tuple(m.reference[p, 1:])]) for p in left), dtype=np.int64)
    W = m.array(np.eye(m.n), float, False)
    C = transfer.flow_map
    parameters = m.array([[1000, 500, 0.2, 0.3, 0.2]], float)
    diameter = m.array([m.config.grid_spacing / 2], float)
    contact = CompensatedArray(m.array(np.tile([0, 1e-7, -2e-7], (len(pairs), 1)), wp.vec3))
    data = dict(
        transfer=transfer,
        reference_normal=retained.quadrature.normals,
        material=retained.material,
        area=retained.quadrature.area,
        separation_map=m.zeros((0, 4 * m.n), float),
        frame_map=m.zeros((0, 4 * m.n), wp.vec3),
        duration=m.array([dt], float),
        damping_rate=m.array([2000.0], float),
    )

    def provider(x, z, tau, elapsed, h):
        return predict_pair_contacts(x, z, tau, h, W, C, pairs, parameters, diameter)

    def advance(s=state, c=contact):
        return advance_affine(
            state=s,
            geometry=retained.geometry,
            history=retained.history,
            **data,
            gravity=(0, 0, 0),
            contacts=provider,
            contact_history=c,
        )

    return m, state, contact, data, pairs, parameters, diameter, advance


def test_pair_material_stage_momentum_and_gradients():
    m, state, contact, data, pairs, parameters, diameter, advance = prepared_pairs()

    def outputs(step):
        return [
            step.state.x,
            step.state.v,
            step.state.affine,
            step.state.F,
            step.state.Fp,
            step.state.stress,
            step.contact_history.high,
            step.contact_stages[-1][1].high,
        ]

    with wp.Tape() as tape:
        step = advance()
    np.testing.assert_allclose(m.mass * np.sum(step.state.v.numpy() - state.v.numpy(), axis=0), 0, atol=1e-18)
    rng = np.random.default_rng(2761)
    result = outputs(step)
    seeds = [
        rng.normal(size=a.numpy().shape) / scale
        for a, scale in zip(result, [0.01, 1, 100, 1, 1, 2e4, 0.001, 1])
    ]
    tape.backward(grads={a: m.array(s, a.dtype, False) for a, s in zip(result, seeds)})
    for source, scale in [(data["duration"], 2e-7), (state.v, 0.1), (diameter, 3e-4)]:
        baseline = source.numpy().copy()
        direction = rng.normal(size=baseline.shape) * scale
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            values = []
            for sign in [-1, 1]:
                source.assign(baseline + sign * epsilon * direction)
                values.append([a.numpy() for a in outputs(advance())])
            fd = sum(np.sum((b - a) * s) for a, b, s in zip(*values, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=2e-8)
        source.assign(baseline)
