"""Affine-state motion, energy, restart and multi-step stage derivatives."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_geometry import InterfaceGeometry
from blademaster_warp.cohesive_history import CohesiveHistory
from blademaster_warp.cohesive_stages import AffineState, advance_affine
from blademaster_warp.reference_affine import build_transfer
from test_cohesive_grid_update import patch


def setup(mode="rigid", dt=2e-6):
    m, codes, geometry, history, data = patch(gc=7.0 if mode == "loaded" else 70.0)
    transfer = build_transfer(m, data["topology"])
    B, G = transfer.compose_trace(
        SimpleNamespace(separation_map=data["separation_map"], frame_map=data["frame_map"])
    )
    data = dict(
        transfer=transfer,
        reference_normal=data["reference_normal"],
        material=data["material"],
        separation_map=B,
        frame_map=G,
        area=data["area"],
        duration=m.array([dt], float),
    )
    A = np.array([[0.0, -50.0, -20.0], [50.0, 0.0, -10.0], [20.0, 10.0, 0.0]])
    base = m.state(v=m.reference @ A.T + [0.1, 0.2, 0.3])
    state = AffineState.from_material(base, m.array(np.tile(A, (m.n, 1, 1)), wp.mat33))
    if mode == "opened":
        geometry.gap.assign(np.tile([2e-4, 1e-4, 0.0], (len(history), 1)))
        history.assign(np.full(len(history), 1e-3))
    physical_force = np.zeros((m.n, 3))
    if mode == "loaded":
        physical_force[:, 0] = m.mass * 2e5 * np.where(codes == 1, -1.0, 1.0)
    data["force"] = m.array(physical_force, wp.vec3)
    return m, state, geometry, history, data


def advance(state, geometry, history, data):
    load = data["force"]
    return advance_affine(
        state=state,
        geometry=geometry,
        history=history,
        **{key: value for key, value in data.items() if key != "force"},
        force=lambda state, velocity, elapsed: load,
    )


def rollout(state, geometry, history, data, steps=8):
    for _ in range(steps):
        result = advance(state, geometry, history, data)
        state, geometry, history = result.state, result.geometry, result.history
    return result


def targets(result):
    return [
        result.state.x,
        result.state.v,
        result.state.affine,
        result.state.F,
        result.state.Fp,
        result.state.stress,
        result.geometry.gap,
        result.geometry.frame,
        result.history,
    ]


def test_empty_interface_free_fall_and_midpoint_time_derivative():
    m, _, _, _, _ = setup()
    retained = CohesiveHistory.empty(m.n, device=m.device)
    transfer = build_transfer(m, m.uncut)
    v = np.tile([0.1, 0.2, 0.3], (m.n, 1))
    state = AffineState.from_material(m.state(v=v), m.zeros(m.n, wp.mat33))
    duration = m.array([2e-4], float)
    kwargs = dict(
        transfer=transfer,
        state=state,
        geometry=retained.geometry,
        history=retained.history,
        reference_normal=retained.quadrature.normals,
        material=retained.material,
        separation_map=m.zeros((0, 4 * m.n), float),
        frame_map=m.zeros((0, 4 * m.n), wp.vec3),
        area=retained.quadrature.area,
        duration=duration,
    )
    with wp.Tape() as tape:
        result = advance_affine(**kwargs)
    t = duration.numpy()[0]
    g = np.array([0.0, 0.0, -9.81])
    np.testing.assert_allclose(result.state.x.numpy(), m.reference + t * v + 0.5 * t * t * g, atol=1e-17)
    np.testing.assert_allclose(result.state.v.numpy(), v + t * g, atol=1e-15)
    np.testing.assert_allclose(
        result.path_midpoint.numpy(), m.reference + 0.5 * t * v + 0.125 * t * t * g, atol=1e-17
    )
    np.testing.assert_allclose(result.state.affine.numpy(), 0.0, atol=1e-12)
    tape.backward(grads={result.path_midpoint: m.array(np.ones((m.n, 3)), wp.vec3, False)})
    np.testing.assert_allclose(duration.grad.numpy()[0], np.sum(0.5 * v + 0.25 * t * g), rtol=1e-13)
    direct, updated = retained.advance(m, state, topology=m.uncut, duration=duration)
    assert isinstance(direct, AffineState) and len(updated.history) == 0
    np.testing.assert_array_equal(direct.v.numpy(), result.state.v.numpy())


def test_energy_snapshot_counts_affine_moments():
    from blademaster_warp.energy import energy_snapshot

    m, state, _, _, data = setup()
    transfer = data["transfer"]
    energy = energy_snapshot(m, state, gravity=(0, 0, 0), affine_transfer=transfer)
    moments = transfer.pack(state.v, state.affine)
    np.testing.assert_allclose(
        energy["kinetic_j"] + energy["affine_kinetic_j"],
        np.sum(transfer.kinetic_energy(moments).numpy()),
        rtol=2e-14,
    )
    assert energy["affine_kinetic_j"] > 0
    with pytest.raises(ValueError, match="matching reference transfer"):
        energy_snapshot(m, state)


@pytest.mark.parametrize("mode", ["rigid", "opened", "loaded"])
def test_affine_state_and_permanent_history_restart(mode):
    m, initial, geometry, history, data = setup(mode)
    first = rollout(initial, geometry, history, data, steps=4)
    saved_state = first.state.snapshot()
    saved_geometry = first.geometry.snapshot()
    saved_history = first.history.numpy().copy()
    direct = rollout(first.state, first.geometry, first.history, data, steps=4)
    restored = AffineState.restore(saved_state, device=m.device)
    geometry2 = InterfaceGeometry.restore(saved_geometry, device=m.device)
    history2 = m.array(saved_history, float)
    resumed = rollout(restored, geometry2, history2, data, steps=4)
    for actual, expected in zip(targets(resumed), targets(direct)):
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    assert direct.state.v.shape == (m.n,) and direct.state.affine.shape == (m.n,)
    assert np.all(direct.history.numpy() >= saved_history)
    if mode == "rigid":
        np.testing.assert_array_equal(direct.geometry.gap.numpy(), 0.0)
        np.testing.assert_array_equal(direct.history.numpy(), 0.0)
    elif mode == "loaded":
        assert np.max(direct.history.numpy()) > 0.0


@pytest.mark.parametrize("mode", ["opened", "loaded"])
def test_material_and_interface_stage_directional_derivatives(mode):
    m, initial, geometry, history, data = setup(mode)
    sources = dict(
        velocity=initial.v,
        affine=initial.affine,
        deformation=initial.F,
        young=m.young,
        duration=data["duration"],
        material=data["material"],
        area=data["area"],
        frame=geometry.frame,
        normal=data["reference_normal"],
        force=data["force"],
    )
    if mode == "opened":
        sources.update(gap=geometry.gap, history=history)
    scales = dict(
        velocity=0.1,
        affine=10.0,
        deformation=0.01,
        young=2000.0,
        duration=2e-7,
        material=np.array([5000.0, 0.7, 0.0]),
        area=1e-6,
        frame=0.01,
        normal=0.1,
        force=m.mass * 2e4,
        gap=1e-5,
        history=1e-4,
    )
    rng = np.random.default_rng(940)
    directions = {
        name: rng.normal(size=value.numpy().shape) * scales[name] for name, value in sources.items()
    }
    with wp.Tape() as tape:
        result = rollout(initial, geometry, history, data)
    arrays = targets(result)
    seeds = [
        rng.normal(size=a.numpy().shape) / scale
        for a, scale in zip(arrays, [0.01, 1.0, 100.0, 1.0, 1.0, 2e4, 0.001, 1.0, 0.001])
    ]
    tape.backward(grads={a: m.array(seed, a.dtype, False) for a, seed in zip(arrays, seeds)})
    ads = {name: np.sum(source.grad.numpy() * directions[name]) for name, source in sources.items()}
    for name, source in sources.items():
        original = source.numpy().copy()
        for eps in [1e-4, 1e-5]:
            values = []
            for sign in [-1, 1]:
                source.assign(original + sign * eps * directions[name])
                values.append([a.numpy() for a in targets(rollout(initial, geometry, history, data))])
            fd = sum(np.sum((b - a) * seed) for a, b, seed in zip(*values, seeds)) / (2 * eps)
            np.testing.assert_allclose(ads[name], fd, rtol=2e-5, atol=2e-8, err_msg=f"{mode}/{name}/{eps}")
        source.assign(original)


def test_piecewise_backward_keeps_affine_stress_and_interface_cotangents():
    m, initial, geometry, history, data = setup("loaded")
    rng = np.random.default_rng(29)
    with wp.Tape() as tape:
        direct = rollout(initial, geometry, history, data)
    seeds = [
        rng.normal(size=a.numpy().shape) / scale
        for a, scale in zip(targets(direct), [0.01, 1.0, 100.0, 1.0, 1.0, 2e4, 0.001, 1.0, 0.001])
    ]
    tape.backward(grads={a: m.array(s, a.dtype, False) for a, s in zip(targets(direct), seeds)})
    expected = [initial.v.grad.numpy().copy(), initial.affine.grad.numpy().copy()]
    tape.zero()
    with wp.Tape() as prefix:
        split = rollout(initial, geometry, history, data, steps=4)
    state2 = AffineState.restore(split.state.snapshot(), device=m.device)
    geometry2 = InterfaceGeometry.restore(split.geometry.snapshot(), device=m.device)
    history2 = m.array(split.history.numpy(), float)
    with wp.Tape() as suffix:
        resumed = rollout(state2, geometry2, history2, data, steps=4)
    suffix.backward(grads={a: m.array(s, a.dtype, False) for a, s in zip(targets(resumed), seeds)})
    restored = [
        state2.x,
        state2.v,
        state2.affine,
        state2.F,
        state2.Fp,
        state2.stress,
        geometry2.gap,
        geometry2.frame,
        history2,
    ]
    prefix.backward(
        grads={a: m.array(b.grad.numpy(), a.dtype, False) for a, b in zip(targets(split), restored)}
    )
    for actual, reference in zip([initial.v, initial.affine], expected):
        np.testing.assert_allclose(actual.grad.numpy(), reference, rtol=1e-10, atol=1e-10)


def test_force_callback_receives_transported_velocity_and_stage_time():
    m, initial, geometry, history, data = setup("opened")
    rng = np.random.default_rng(15)
    initial.v.assign(rng.normal(size=(m.n, 3)))
    observed = []

    def force(state, velocity, elapsed):
        moments = data["transfer"].pack(state.v, state.affine)
        expected, _ = data["transfer"].kinematics(moments)
        np.testing.assert_allclose(velocity.numpy(), expected.numpy(), rtol=2e-14, atol=1e-14)
        observed.append(float(elapsed.numpy()[0]))
        return m.zeros(m.n, wp.vec3)

    from blademaster_warp.cohesive_stages import GAMMA

    advance_affine(
        state=initial,
        geometry=geometry,
        history=history,
        **{key: value for key, value in data.items() if key != "force"},
        force=force,
    )
    np.testing.assert_allclose(observed, [0.0, GAMMA * data["duration"].numpy()[0]], atol=0.0)


def test_retained_surface_advance_uses_birth_sides_and_affine_state():
    from test_cohesive_orientation import fixture

    m, append = fixture()
    retained = append(CohesiveHistory.empty(m.n, device=m.device), 0.009, 0.0085)
    grid = m.topology(np.where(m.reference[:, 0] < 0.0007, 1, 2))
    A = np.array([[0.0, -50.0, -20.0], [50.0, 0.0, -10.0], [20.0, 10.0, 0.0]])
    state = AffineState.from_material(
        m.state(v=m.reference @ A.T), m.array(np.tile(A, (m.n, 1, 1)), wp.mat33)
    )
    retained = retained.updated(retained.geometry, m.array(np.full(len(retained.history), 0.001), float))
    initial_damage = retained.history.numpy().copy()
    results, derivatives = [], []
    for explicit in [False, True]:
        saved = CohesiveHistory.restore(retained.snapshot(), device=m.device)
        current = AffineState.restore(state.snapshot(), device=m.device)
        start = current
        with wp.Tape() as tape:
            for _ in range(2):
                current, saved = saved.advance(
                    m,
                    current,
                    topology=grid,
                    gravity=(0, 0, 0),
                    sides=np.tile([2, 1], (len(saved.history), 1)) if explicit else None,
                )
        np.testing.assert_array_equal(saved.history.numpy(), initial_damage)
        results.append(
            [current.x.numpy(), current.v.numpy(), current.affine.numpy(), saved.geometry.gap.numpy()]
        )
        tape.backward(grads={current.affine: m.array(np.ones((m.n, 3, 3)), wp.mat33, False)})
        derivatives.append(start.affine.grad.numpy().copy())
    for a, b in zip(*results):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(*derivatives, rtol=1e-12, atol=1e-12)


def test_active_plastic_return_and_parameter_derivatives():
    from blademaster_warp.energy import material_energy

    m, initial, geometry, history, data = setup()
    m.yield_stress.assign([20.0])
    A = np.array([[100.0, -50.0, -20.0], [50.0, -20.0, -10.0], [20.0, 10.0, 0.0]])
    initial.affine.assign(np.tile(A, (m.n, 1, 1)))
    initial.v.assign(m.reference @ A.T)

    def run():
        return rollout(initial, geometry, history, data)

    with wp.Tape() as tape:
        result = run()
    assert np.count_nonzero(np.linalg.norm(result.state.Fp.numpy() - np.eye(3), axis=(1, 2)) > 1e-12) == m.n
    np.testing.assert_allclose(np.linalg.det(result.state.Fp.numpy()), 1.0, rtol=2e-12)
    outputs = [result.state.v, result.state.Fp, result.state.stress]
    rng = np.random.default_rng(113)
    seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(outputs, [1.0, 1.0, 2e4])]
    tape.backward(grads={a: m.array(s, a.dtype, False) for a, s in zip(outputs, seeds)})
    for source, scale in [(m.young, 2000.0), (m.yield_stress, 2.0), (initial.affine, 10.0)]:
        original = source.numpy().copy()
        direction = rng.normal(size=original.shape) * scale
        ad = np.sum(source.grad.numpy() * direction)
        for eps in [1e-4, 1e-5]:
            values = []
            for sign in [-1, 1]:
                source.assign(original + sign * eps * direction)
                actual = run()
                values.append([a.numpy() for a in [actual.state.v, actual.state.Fp, actual.state.stress]])
            fd = sum(np.sum((b - a) * s) for a, b, s in zip(*values, seeds)) / (2 * eps)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-8)
        source.assign(original)
    state = initial
    for _ in range(8):
        actual = advance(state, geometry, history, data)

        def energy(Fp):
            return material_energy(
                actual.state.x.numpy(),
                np.zeros((m.n, 3)),
                actual.state.F.numpy(),
                Fp,
                m.mass,
                m.volume,
                float(m.young.numpy()[0]),
                m.config.poisson_ratio,
                gravity=(0, 0, 0),
            )["elastic_j"]

        assert energy(state.Fp.numpy()) - energy(actual.state.Fp.numpy()) >= -1e-18
        state, geometry, history = actual.state, actual.geometry, actual.history
