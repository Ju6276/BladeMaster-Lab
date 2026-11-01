"""Spatial selection agrees with exhaustive contacts and preserves indexed histories."""

from itertools import combinations

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_contact import predict_contacts
from blademaster_warp.cohesive_contact_batch import CombinedContacts, join_states, split_state
from blademaster_warp.cohesive_history import CohesiveHistory
from blademaster_warp.cohesive_pair_contact import predict_pair_contacts
from blademaster_warp.cohesive_pairing import PairHistory, pairs_at
from blademaster_warp.cohesive_spatial_contact import advance_spatial_pairs, pair_storage
from blademaster_warp.cohesive_stages import AffineState, advance_affine
from blademaster_warp.compensated_state import CompensatedArray
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.reference_affine import build_transfer
from blademaster_warp.solver import Config


def array(value, kind=wp.float64):
    return wp.array(np.asarray(value), dtype=kind, device="cuda:0", requires_grad=True)


@pytest.mark.parametrize("shared", [False, True])
def test_pair_storage_and_all_input_derivatives(shared):
    x = array([[0, 0, 0], [0.0028, 0.0001, 0], [0.0065, 0, 0]], wp.vec3d)
    pairs = np.array([[0, 1], [1, 2]])
    values = array([[0, 1e-4, -2e-4], [0, 3e-4, -1e-4]], wp.vec3d)
    history = PairHistory(3, pairs, CompensatedArray(values))
    parameters = array([[1000, 500, 0.2, 0.3, 0.2]] * (1 if shared else 2))
    diameter = array([0.003])
    with wp.Tape() as tape:
        result = pair_storage(x, history, parameters, diameter)
    gaps = np.linalg.norm(x.numpy()[pairs[:, 0]] - x.numpy()[pairs[:, 1]], axis=1) - 0.003
    expected = np.column_stack([500 * np.minimum(gaps, 0) ** 2, 250 * np.sum(values.numpy() ** 2, axis=1)])
    np.testing.assert_allclose(result.numpy(), expected, rtol=2e-13, atol=1e-20)
    rng = np.random.default_rng(4094)
    seed = rng.normal(size=(2, 2))
    tape.backward(grads={result: array(seed)})
    for source, scale in [(x, 1e-4), (values, 1e-5), (parameters, 100.0), (diameter, 1e-4)]:
        baseline = source.numpy().copy()
        direction = rng.normal(size=baseline.shape) * scale
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            values_fd = []
            for sign in [-1, 1]:
                source.assign(baseline + sign * epsilon * direction)
                values_fd.append(pair_storage(x, history, parameters, diameter).numpy())
            fd = np.sum(seed * (values_fd[1] - values_fd[0])) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-13)
        source.assign(baseline)


def setup(mode="entry", mixed=False):
    config = Config(
        size=(0.012, 0.012, 0.012),
        grid_spacing=0.006,
        dt=2e-6,
        young_modulus=20000.0,
        yield_stress=0.0,
        flip_fraction=1.0,
        external_tool=True,
        blade_grid_constraint=False,
    )
    model = MaterialTrajectory(config, precision="float64", ordered_transfers=True)
    codes = np.where(model.reference[:, 0] < 0, 1, 2)
    side = np.where(codes == 1, -1.0, 1.0)
    gap = dict(entry=1e-7, exit=-1e-7, active=-2e-4)[mode]
    velocity = side[:, None] * [0.2 if mode == "exit" else -0.2, 0.03, 0.0]
    position = model.reference + side[:, None] * [0.5 * gap, 0.0, 0.0]
    state = AffineState.from_material(model.state(x=position, v=velocity), model.zeros(model.n, wp.mat33))
    transfer = build_transfer(model, model.topology(codes))
    retained = CohesiveHistory.empty(model.n, device=model.device)
    diameter = model.array([0.003], float)
    parameters = model.array([[1000, 500, 0.2, 0.3, 0.2]], float)
    initial_pairs = pairs_at(state.x, model.reference, codes, diameter.numpy()[0], config.grid_spacing)
    pair_history = PairHistory(
        model.n,
        initial_pairs,
        CompensatedArray(model.array(np.tile([0, 1e-7, -2e-7], (len(initial_pairs), 1)), wp.vec3)),
    )
    W = model.array(np.eye(model.n), float, False)
    C = transfer.flow_map
    extra = CompensatedArray(model.array(np.tile([1e-7, -2e-7, 0], (model.n, 1)), wp.vec3)) if mixed else None
    normal = model.array(np.tile([0, 0, 1.0], (model.n, 1)), wp.vec3, False)
    origin = model.array(model.reference + [0, 0, 2e-4], wp.vec3)
    surface_velocity = model.zeros(1, wp.vec3, False)
    plane_parameters = model.array(np.tile([500, 200, 0.1, 0.3, 0.2], (model.n, 1)), float)

    def other(x, z, tau, elapsed, h):
        return predict_contacts(x, z, tau, h, W, C, origin, surface_velocity, normal, plane_parameters)

    data = dict(
        transfer=transfer,
        state=state,
        geometry=retained.geometry,
        history=retained.history,
        reference_normal=retained.quadrature.normals,
        material=retained.material,
        separation_map=model.zeros((0, 4 * model.n), float),
        frame_map=model.zeros((0, 4 * model.n), wp.vec3),
        area=retained.quadrature.area,
        duration=model.array([config.dt], float),
        damping_rate=model.array([2000.0], float),
        gravity=(0, 0, 0),
        constrain_normal=False,
    )

    def advance():
        return advance_spatial_pairs(
            **data,
            pair_history=pair_history,
            codes=codes,
            parameters=parameters,
            diameter=diameter,
            other_contacts=other if mixed else None,
            other_history=extra,
        )

    all_pairs = np.asarray([(p, q) for p, q in combinations(range(model.n), 2) if codes[p] != codes[q]])

    def exhaustive():
        initial, _ = pair_history.reindex(all_pairs)
        contact_history = initial.values if extra is None else join_states([initial.values, extra])

        def provider(x, z, tau, elapsed, h):
            if mixed:
                pair_h, plane_h = split_state(h, [len(all_pairs), model.n])
            else:
                pair_h = h
            batch = predict_pair_contacts(x, z, tau, pair_h, W, C, all_pairs, parameters, diameter)
            return CombinedContacts([batch, other(x, z, tau, elapsed, plane_h)]) if mixed else batch

        return advance_affine(**data, contacts=provider, contact_history=contact_history)

    return model, state, pair_history, parameters, diameter, extra, data, advance, exhaustive


@pytest.mark.parametrize(
    "mode,mixed", [("active", False), ("entry", False), ("exit", False), ("entry", True)]
)
def test_spatial_stages_match_exhaustive_contacts_and_preserve_history(mode, mixed):
    model, state, history, _, _, _, data, advance, exhaustive = setup(mode, mixed)
    result = advance()
    expected = exhaustive()
    for name in ["x", "v", "affine", "F", "Fp", "stress"]:
        np.testing.assert_allclose(
            getattr(result.material.state, name).numpy(),
            getattr(expected.state, name).numpy(),
            rtol=2e-10,
            atol=2e-11,
        )
    assert len(result.pairs) == 16
    if mode == "entry":
        assert len(history.pairs) == 0 and result.rebuilds >= 1 and len(result.history.pairs) == 16
    if mode == "exit":
        assert len(result.history.pairs) == 0 and len(result.retired.pairs) == 16
        assert np.sum(result.initial_storage.numpy()) > 0
        np.testing.assert_array_equal(result.final_storage.numpy(), 0.0)
        np.testing.assert_array_equal(result.retired.values.high.numpy(), 0.0)
    else:
        assert len(result.history.pairs) == 16
    if not mixed:
        np.testing.assert_allclose(
            model.mass * np.sum(result.material.state.v.numpy() - state.v.numpy(), axis=0), 0, atol=1e-18
        )


@pytest.mark.parametrize("mixed", [False, True])
def test_new_spatial_contact_natural_gradients_include_rebuilt_history(mixed):
    model, state, history, parameters, diameter, extra, data, advance, _ = setup("entry", mixed)

    def outputs(result):
        step = result.material
        return [
            step.state.x,
            step.state.v,
            step.state.affine,
            step.state.F,
            step.state.Fp,
            result.history.values.high,
            result.initial_storage,
            result.final_storage,
            *([] if result.other_history is None else [result.other_history.high]),
        ]

    with wp.Tape() as tape:
        result = advance()
    assert result.rebuilds >= 1
    rng = np.random.default_rng(4095)
    values = outputs(result)
    scales = [0.01, 1, 100, 1, 1, 0.001, 1e-7, 1e-7, 0.001]
    seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(values, scales)]
    tape.backward(grads={a: model.array(seed, a.dtype, False) for a, seed in zip(values, seeds)})
    sources = [
        (data["duration"], 2e-7),
        (state.v, 0.02),
        (diameter, 1e-5),
        (parameters, [100, 50, 0.01, 0.01, 0.01]),
    ]
    if mixed:
        sources.append((extra.high, 1e-7))
    for source, scale in sources:
        baseline = source.numpy().copy()
        direction = rng.normal(size=baseline.shape) * scale
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            trials = []
            for sign in [-1, 1]:
                source.assign(baseline + sign * epsilon * direction)
                trials.append([a.numpy() for a in outputs(advance())])
            fd = sum(np.sum((b - a) * seed) for a, b, seed in zip(*trials, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=2e-8)
        source.assign(baseline)
