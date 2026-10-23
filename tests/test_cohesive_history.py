"""Growing quadrature retains fracture memory, material state and cotangents."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_geometry import InterfaceGeometry
from blademaster_warp.cohesive_history import CohesiveHistory
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.reference_surface import ReferenceSurface, swept_edge
from blademaster_warp.solver import Config
from test_reference_surface import arrays, block


def same_snapshot(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], dict):
            same_snapshot(a[key], b[key])
        elif isinstance(a[key], tuple):
            assert a[key] == b[key]
        else:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.parametrize("spacing", [0.006, 0.004])
def test_sweep_growth_reentry_and_new_cut_keep_old_quadrature_and_history(spacing):
    reference, values = block(spacing)
    ref = wp.array(reference, dtype=wp.vec3d)
    parameters = wp.array(np.tile([5e4, 70.0, 0.0], (len(ref), 1)), dtype=wp.vec3d, requires_grad=True)
    state = CohesiveHistory.empty(len(ref), device=ref.device)
    restored = state
    heights = [0.015, 0.009, 0.003, 0.009, 0.001]
    for start, end, depth in zip(heights, heights[1:], [0.003, 0.009, 0.009, 0.011]):
        _, values = block(spacing, start=start, end=end)
        sources = arrays(values)
        candidate = swept_edge(*sources)
        old = state.snapshot()
        n = len(state.history)
        state = state.append(candidate, ref, sources[3], parameters, cut_index=0)
        restored = restored.append(candidate, ref, sources[3], parameters, cut_index=0)
        same_snapshot(state.snapshot(), restored.snapshot())
        for key in ["points", "area", "normals", "owners", "material", "history", "cut_indices"]:
            np.testing.assert_array_equal(state.snapshot()[key][:n], old[key])
        for key in old["geometry"]:
            np.testing.assert_array_equal(state.geometry.snapshot()[key][:n], old["geometry"][key])
        np.testing.assert_array_equal(state.history.numpy()[n:], 0.0)
        np.testing.assert_array_equal(state.geometry.gap.numpy()[n:], 0.0)
        np.testing.assert_allclose(state.quadrature.area.numpy().sum(), 0.0074 * depth, rtol=4e-14)
        maximum = wp.array(np.maximum(state.history.numpy(), 2e-4), dtype=wp.float64, requires_grad=True)
        geometry = InterfaceGeometry(
            state.geometry.gap,
            state.geometry.frame,
            np.full((len(maximum), 3), 1e-23),
            np.full((len(maximum), 3, 3), 1e-18),
        )
        state = state.updated(geometry, maximum)
        restored = CohesiveHistory.restore(state.snapshot(), device=ref.device)
    old = state.snapshot()
    _, values = block(spacing)
    values[6][0] = values[8][0] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    sources = arrays(values)
    candidate = swept_edge(*sources)
    extended = state.append(candidate, ref, sources[3], parameters, cut_index=1)
    n = len(state.history)
    np.testing.assert_array_equal(extended.history.numpy()[:n], old["history"])
    np.testing.assert_array_equal(extended.cut_indices[:n], 0)
    np.testing.assert_array_equal(extended.cut_indices[n:], 1)
    np.testing.assert_array_equal(extended.history.numpy()[n:], 0.0)
    same_snapshot(state.snapshot(), old)
    with pytest.raises(ValueError, match="previous maximum"):
        state.updated(state.geometry, wp.zeros_like(state.history))


def test_old_fracture_energy_and_front_gradient_survive_surface_append():
    reference, values = block(end=0.0063)
    old_sources = arrays(values)
    _, values = block(start=0.01, end=0.002)
    new_sources = arrays(values)
    ref = wp.array(reference, dtype=wp.vec3d)
    parameters = wp.array(np.tile([5e4, 70.0, 0.0], (len(ref), 1)), dtype=wp.vec3d, requires_grad=True)

    def build():
        initial = CohesiveHistory.empty(len(ref), device=ref.device)
        old = initial.append(swept_edge(*old_sources), ref, old_sources[3], parameters, cut_index=0)
        maximum = wp.full(len(old.history), 2e-4, dtype=wp.float64, requires_grad=True)
        old = old.updated(old.geometry, maximum)
        return old.append(swept_edge(*new_sources), ref, new_sources[3], parameters, cut_index=1), maximum

    with wp.Tape() as tape:
        result, maximum = build()
    weights = result.quadrature.area.numpy()
    history = result.history.numpy()
    strengths = result.material.numpy()[:, 0]
    material_seed = np.zeros((len(weights), 3))
    material_seed[:, 0] = 0.5 * weights * history
    tape.backward(
        grads={
            result.quadrature.area: wp.array(0.5 * history * strengths, dtype=wp.float64),
            result.history: wp.array(0.5 * weights * strengths, dtype=wp.float64),
            result.material: wp.array(material_seed, dtype=wp.vec3d),
        }
    )
    np.testing.assert_allclose(maximum.grad.numpy(), 2.5e4 * weights[: len(maximum)], rtol=2e-14)
    expected = -0.0074 * 0.5 * 5e4 * 2e-4
    np.testing.assert_allclose(old_sources[7].grad.numpy()[0, 2], expected, rtol=3e-13)
    np.testing.assert_allclose(new_sources[7].grad.numpy(), 0.0, atol=1e-20)
    np.testing.assert_allclose(parameters.grad.numpy()[:, 0].sum(), 0.5 * weights @ history, rtol=2e-14)
    base = old_sources[7].numpy().copy()
    for epsilon in [1e-7, 1e-8]:
        losses = []
        for sign in [-1, 1]:
            value = base.copy()
            value[0, 2] += sign * epsilon
            old_sources[7].assign(value)
            trial, _ = build()
            losses.append(
                0.5 * trial.quadrature.area.numpy() @ (trial.history.numpy() * trial.material.numpy()[:, 0])
            )
        np.testing.assert_allclose((losses[1] - losses[0]) / (2 * epsilon), expected, rtol=2e-7)
    old_sources[7].assign(base)


def material_case():
    cfg = Config(
        size=(0.012,) * 3,
        grid_spacing=0.006,
        dt=2e-6,
        young_modulus=20000.0,
        yield_stress=0.0,
        flip_fraction=1.0,
        external_tool=True,
        blade_grid_constraint=False,
    )
    model = MaterialTrajectory(cfg, precision="float64", ordered_transfers=True)
    codes = np.where(model.reference[:, 0] < 0, 1, 2)
    topology = model.topology(codes)
    reference = model.array(model.reference, wp.vec3, False)
    material = model.array(np.tile([5e4, 7.0, 0.0], (model.n, 1)), wp.vec3)
    force = np.zeros((model.n, 3))
    force[:, 0] = np.where(codes == 1, -1, 1) * model.mass * 2e6 * cfg.dt
    impulse = model.array(force, wp.vec3)

    def triangle(height):
        coordinates = np.array(
            [[0.0004, 0.0002, height], [0.0004, 0.0008, height], [0.0004, 0.0002, height + 0.0006]]
        )
        owner = int(np.argmin(np.linalg.norm(model.reference - coordinates.mean(axis=0), axis=1)))
        return ReferenceSurface(
            model.array(coordinates - model.reference[owner], wp.vec3),
            ((0, 1, 2),),
            np.array([owner]),
            np.zeros(model.n),
        )

    surfaces = [triangle(0.0007), triangle(0.002)]
    state = model.state(stress=np.zeros((model.n, 3, 3)))
    history = CohesiveHistory.empty(model.n, device=model.device).append(
        surfaces[0], reference, state.F, material, cut_index=0
    )
    return model, topology, reference, material, impulse, surfaces, state, history


def test_damage_survives_material_loading_new_patch_and_restart():
    m, grid, ref, material, impulse, surfaces, state, history = material_case()
    for _ in range(16):
        state, history = history.advance(
            m,
            state,
            topology=grid,
            sides=np.tile([1, 2], (len(history.history), 1)),
            impulse=impulse,
            gravity=(0, 0, 0),
        )
    assert np.all(history.history.numpy() >= 2 * 7.0 / 5e4)
    old = history.snapshot()
    n = len(history.history)
    history = history.append(surfaces[1], ref, state.F, material, cut_index=1)
    np.testing.assert_array_equal(history.history.numpy()[:n], old["history"])
    np.testing.assert_array_equal(history.history.numpy()[n:], 0.0)
    restored = CohesiveHistory.restore(history.snapshot(), device=m.device)
    sides = np.tile([1, 2], (len(history.history), 1))
    impulse.assign(-impulse.numpy())
    direct_state, direct_history = history.advance(m, state, topology=grid, sides=sides, impulse=impulse)
    replay_state, replay_history = restored.advance(m, state, topology=grid, sides=sides, impulse=impulse)
    same_snapshot(direct_history.snapshot(), replay_history.snapshot())
    for name in ["x", "v", "F", "Fp", "stress"]:
        np.testing.assert_array_equal(
            getattr(direct_state, name).numpy(), getattr(replay_state, name).numpy()
        )
    assert np.all(direct_history.history.numpy()[:n] >= old["history"])


def test_append_checkpoint_keeps_geometry_material_and_damage_pullbacks():
    m, grid, ref, material, impulse, surfaces, state, initial = material_case()
    initial = initial.updated(initial.geometry, m.array([2e-4] * len(initial.history), float))
    rng = np.random.default_rng(31)

    def advance(state, history, step):
        if step == 1:
            history = history.append(surfaces[1], ref, state.F, material, cut_index=1)
        return history.advance(
            m,
            state,
            topology=grid,
            sides=np.tile([1, 2], (len(history.history), 1)),
            impulse=impulse,
            gravity=(0, 0, 0),
        )

    def state_arrays(state, history):
        return [state.x, state.v, state.F, state.Fp, state.stress, *history.differentiable_arrays().values()]

    def snapshot(state, history):
        return {
            name: getattr(state, name).numpy().copy() for name in ["x", "v", "F", "Fp", "stress"]
        }, history.snapshot()

    checkpoints = [snapshot(state, initial)]
    original = state
    with wp.Tape() as tape:
        history = initial
        for step in range(3):
            state, history = advance(state, history, step)
            checkpoints.append(snapshot(state, history))
    targets = state_arrays(state, history)
    seeds = [rng.normal(size=a.numpy().shape) for a in targets]
    tape.backward(grads={a: wp.array(s, dtype=a.dtype) for a, s in zip(targets, seeds)})
    expected = [a.grad.numpy().copy() for a in state_arrays(original, initial)]
    parameters = [material, impulse, m.young, surfaces[1].vertices, surfaces[1].normals]
    expected_parameters = [a.grad.numpy().copy() for a in parameters]
    tape.zero()
    parameter_gradients = [np.zeros_like(a) for a in expected_parameters]
    for step in reversed(range(3)):
        start_state = m.state(**checkpoints[step][0])
        start_history = CohesiveHistory.restore(checkpoints[step][1], device=m.device)
        with wp.Tape() as local:
            state, history = advance(start_state, start_history, step)
        saved = snapshot(state, history)
        same_snapshot(saved[0], checkpoints[step + 1][0])
        same_snapshot(saved[1], checkpoints[step + 1][1])
        local.backward(
            grads={a: wp.array(s, dtype=a.dtype) for a, s in zip(state_arrays(state, history), seeds)}
        )
        seeds = [a.grad.numpy().copy() for a in state_arrays(start_state, start_history)]
        for accumulated, source in zip(parameter_gradients, parameters):
            accumulated += source.grad.numpy()
        local.zero()
    for actual, target in zip(seeds + parameter_gradients, expected + expected_parameters):
        np.testing.assert_allclose(actual, target, rtol=2e-9, atol=2e-9)


def test_empty_history_matches_material_step():
    m, grid, _, _, impulse, _, state, _ = material_case()
    empty = CohesiveHistory.empty(m.n, device=m.device)
    direct = m.step(state, topology=grid, impulse=impulse)
    result, history = empty.advance(m, state, topology=grid, sides=[], impulse=impulse)
    assert history is empty
    same_snapshot(history.snapshot(), CohesiveHistory.restore(history.snapshot(), device=m.device).snapshot())
    for name in ["x", "v", "F", "Fp", "stress"]:
        np.testing.assert_array_equal(getattr(direct, name).numpy(), getattr(result, name).numpy())


def test_material_trajectory_derivatives_through_surface_append():
    m, grid, ref, material, impulse, surfaces, initial_state, initial = material_case()
    initial = initial.updated(initial.geometry, m.array([2e-4] * len(initial.history), float))
    duration = m.array([m.config.dt], float)

    def evaluate():
        state, history = initial_state, initial
        for step in range(3):
            if step == 1:
                history = history.append(surfaces[1], ref, state.F, material, cut_index=1)
            state, history = history.advance(
                m,
                state,
                topology=grid,
                sides=np.tile([1, 2], (len(history.history), 1)),
                impulse=impulse,
                duration=duration,
                gravity=(0, 0, 0),
            )
        return [
            state.x,
            state.v,
            state.F,
            state.stress,
            history.geometry.gap,
            history.geometry.frame,
            history.history,
            history.quadrature.area,
            history.material,
        ]

    with wp.Tape() as tape:
        targets = evaluate()
    rng = np.random.default_rng(72)
    seeds = [
        rng.normal(size=a.numpy().shape) / scale
        for a, scale in zip(targets, [0.01, 10, 1, 1e4, 0.001, 1, 0.001, 1e-6, 5e4])
    ]
    tape.backward(grads={a: wp.array(s, dtype=a.dtype) for a, s in zip(targets, seeds)})
    sources = [
        initial.history,
        initial.geometry.gap,
        initial_state.F,
        material,
        impulse,
        duration,
        surfaces[1].vertices,
    ]
    directions = [
        rng.normal(size=a.numpy().shape) * scale
        for a, scale in zip(sources, [1e-4, 1e-4, 0.01, np.array([5e4, 7, 0]), m.mass, m.config.dt, 0.0])
    ]
    directions[-1][:] = [0.0004, 0.0002, 0.0003]
    for source, direction in zip(sources, directions):
        base = source.numpy().copy()
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-5, 1e-6]:
            trials = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                trials.append([a.numpy() for a in evaluate()])
            fd = sum(np.sum((b - a) * s) for a, b, s in zip(*trials, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-7)
        source.assign(base)
