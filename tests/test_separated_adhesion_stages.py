"""Permanent adhesion and independent particle contact in complete material stages."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_pair_contact import predict_pair_contacts
from blademaster_warp.cohesive_stages import AffineState, advance_affine
from blademaster_warp.compensated_state import CompensatedArray
from test_cohesive_stages import setup, targets
from test_contact_stage_checkpoint import check_segmented_contact_gradient


def prepared_separated(plastic=False, fraction=0.4, dt=2e-6):
    m, _, geometry, history, data = setup("opened", dt)
    side = np.where(m.reference[:, 0] < 0, -1.0, 1.0)
    gap = np.array([-2e-4, 1e-4, 0.0])
    position = m.reference + 0.5 * side[:, None] * gap
    A = np.array([[0, -20, 10], [20, 0, -15], [-10, 15, 0]], dtype=float)
    if plastic:
        m.yield_stress.assign([20.0])
        A += np.diag([2000.0, -400.0, -200.0])
    velocity = side[:, None] * np.array([-0.1, 0.2, 0.0]) + m.reference @ A.T
    state = AffineState.from_material(
        m.state(x=position, v=velocity), m.array(np.tile(A, (m.n, 1, 1)), wp.mat33)
    )
    geometry.gap.assign(np.tile(gap, (len(history), 1)))
    material = data["material"].numpy()
    history.assign(fraction * 2 * material[:, 1] / material[:, 0])
    left = np.flatnonzero(m.reference[:, 0] == np.max(m.reference[side < 0, 0]))
    right = np.flatnonzero(m.reference[:, 0] == np.min(m.reference[side > 0, 0]))
    match = {tuple(m.reference[q, 1:]): int(q) for q in right}
    pairs = np.asarray(sorted((int(p), match[tuple(m.reference[p, 1:])]) for p in left), dtype=np.int64)
    W = m.array(np.eye(m.n), float, False)
    C = data["transfer"].flow_map
    parameters = m.array([[1000, 500, 0.2, 0.3, 0.2]], float)
    diameter = m.array([m.config.grid_spacing / 2], float)
    contact = CompensatedArray(m.array(np.tile([0, 1e-7, -2e-7], (len(pairs), 1)), wp.vec3))
    data.pop("force")
    data["damping_rate"] = m.array([2000.0], float)

    def provider(x, z, tau, elapsed, h):
        return predict_pair_contacts(x, z, tau, h, W, C, pairs, parameters, diameter)

    def advance(s=state, g=geometry, h=history, c=contact):
        return advance_affine(
            state=s,
            geometry=g,
            history=h,
            **data,
            gravity=(0, 0, 0),
            contacts=provider,
            contact_history=c,
            constrain_normal=False,
        )

    sources = [
        data["duration"],
        m.young,
        m.yield_stress,
        parameters,
        diameter,
        data["damping_rate"],
        data["material"],
        data["area"],
    ]
    prepared = (m, state, geometry, history, contact, data, sources, advance)
    return prepared, pairs, parameters, diameter


def outputs(step):
    return [
        *targets(step),
        step.contact_history.high,
        step.contact_stages[-1][1].high,
        step.interface.force,
        step.interface.storage,
        step.interface.fracture,
        step.interface.surface_moment,
        step.damping_energy,
    ]


@pytest.mark.parametrize("plastic,fraction", [(False, 0.4), (True, 0.4), (False, 1.2)])
def test_separated_material_stages_keep_damage_momentum_and_gradients(plastic, fraction):
    prepared, pairs, parameters, diameter = prepared_separated(plastic, fraction)
    m, state, geometry, history, contact, data, _, advance = prepared
    with wp.Tape() as tape:
        result = advance()
    assert np.all(result.history.numpy() >= history.numpy())
    assert np.max(np.linalg.norm(result.contact_stages[-1][1].high.numpy(), axis=1)) > 0
    np.testing.assert_allclose(
        m.mass * np.sum(result.state.v.numpy() - state.v.numpy(), axis=0), 0, atol=1e-18
    )
    if plastic:
        assert np.count_nonzero(result.state.yielded.numpy()) > 0
    if fraction > 1:
        np.testing.assert_allclose(result.interface.force.numpy(), 0, atol=1e-24)
        np.testing.assert_array_equal(result.history.numpy(), history.numpy())
    else:
        assert np.max(np.linalg.norm(result.interface.force.numpy(), axis=1)) > 0
    rng = np.random.default_rng(4091)
    values = outputs(result)
    scales = [0.01, 1, 100, 1, 1, 2e4, 0.001, 1, 0.001, 0.001, 1, 1, 70, 70, 0.001, 1e-6]
    seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(values, scales)]
    tape.backward(grads={a: m.array(s, a.dtype, False) for a, s in zip(values, seeds)})
    groups = [
        [(data["duration"], 2e-7)],
        [(state.v, 0.1), (state.affine, 10.0), (m.young, 2000.0)],
        [
            (parameters, [100, 50, 0.01, 0.01, 0.01]),
            (diameter, 3e-4),
            (data["material"], [5000.0, 7.0, 0.0]),
            (history, 1e-4),
            (data["area"], 1e-6),
        ],
    ]
    if plastic:
        groups[1].append((m.yield_stress, 2.0))
    for group in groups:
        baseline = [a.numpy().copy() for a, _ in group]
        directions = [rng.normal(size=a.shape) * scale for a, (_, scale) in zip(baseline, group)]
        ad = sum(np.sum(source.grad.numpy() * d) for (source, _), d in zip(group, directions))
        for epsilon in [1e-4, 1e-5]:
            trials = []
            for sign in [-1, 1]:
                for (source, _), original, direction in zip(group, baseline, directions):
                    source.assign(original + sign * epsilon * direction)
                trials.append([a.numpy() for a in outputs(advance())])
            fd = sum(np.sum((b - a) * seed) for a, b, seed in zip(*trials, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=2e-8)
        for (source, _), original in zip(group, baseline):
            source.assign(original)


@pytest.mark.parametrize("plastic", [False, True])
def test_separated_material_checkpoint_matches_complete_backward(plastic):
    prepared, *_ = prepared_separated(plastic)
    check_segmented_contact_gradient(prepared, plastic)
