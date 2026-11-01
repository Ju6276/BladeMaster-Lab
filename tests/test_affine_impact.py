"""Affine impacts: independent projection, kinetic loss and natural derivatives."""

import numpy as np
import pytest
from scipy.optimize import nnls
import warp as wp

from blademaster_warp._cohesive_expansion import E
from blademaster_warp._impact_projection import project
from blademaster_warp.cohesive_geometry import InterfaceGeometry
from blademaster_warp.cohesive_impact import project_affine_state, project_velocity
from blademaster_warp.cohesive_stages import AffineState, advance_affine
from blademaster_warp.compensated_state import CompensatedArray
from test_cohesive_stages import setup


@pytest.mark.parametrize("seed", range(12))
def test_projection_matches_independent_nnls(seed):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(7, 5, 3))
    C = np.concatenate([C, C[:2] * 2, np.zeros_like(C[:1])])
    masses = np.exp(rng.uniform(-4, 4, 5))
    velocity = rng.normal(size=(5, 3))
    roots = np.repeat(np.sqrt(masses), 3)
    weighted = C.reshape(len(C), -1).T / roots[:, None]
    multipliers, _ = nnls(weighted, -roots * velocity.ravel(), maxiter=1000)
    expected = velocity.ravel() + weighted @ multipliers / roots
    result, _, _ = project(E(C), E(masses), E(velocity))
    np.testing.assert_allclose(result.numpy().ravel(), expected, rtol=2e-12, atol=3e-13)
    jump = result.numpy() - velocity
    loss = 0.5 * np.sum(masses[:, None] * (velocity**2 - result.numpy() ** 2))
    np.testing.assert_allclose(loss, 0.5 * np.sum(masses[:, None] * jump**2), rtol=2e-12, atol=3e-13)


@pytest.mark.parametrize("exponent", [-60, -80])
def test_sub_ulp_impact_uses_velocity_remainder(exponent):
    epsilon = 2.0**exponent
    C = wp.array([[[-1, 0, 0], [1, 0, 0]]], dtype=wp.vec3d)
    mass = wp.array([1, 2], dtype=wp.float64)
    high = wp.array([[1, 0, 0], [1, 0, 0]], dtype=wp.vec3d, requires_grad=True)
    velocity = CompensatedArray(high, np.array([[2 * epsilon, 0, 0], [-epsilon, 0, 0]]))
    with wp.Tape() as tape:
        output = project_velocity(C, mass, velocity)
    assert isinstance(output, CompensatedArray)
    difference = output.cpu() - E(np.array([[1, 0, 0], [1, 0, 0]]))
    assert np.max(abs(difference.high)) < 1e-30
    jump = output.cpu() - velocity.cpu()
    loss = 0.5 * (E(np.array([1, 2]))[:, None] * jump * jump).sum()
    np.testing.assert_allclose(loss.numpy(), 3 * epsilon**2, rtol=5e-8, atol=0)
    seed = np.array([[1, 0, 0], [0, 0, 0]], dtype=float)
    tape.backward(grads={output.high: wp.array(seed, dtype=wp.vec3d)})
    np.testing.assert_allclose(high.grad.numpy()[:, 0], [1 / 3, 2 / 3], atol=1e-15)


def prepared():
    m, state, geometry, history, data = setup()
    sign = np.where(m.reference[:, 0] < 0, 1.0, -1.0)
    velocity = sign[:, None] * [0.2, 0.01, -0.02] + [0.03, 0.02, -0.01]
    gradient = np.tile([[0.2, 0.1, 0], [-0.1, 0.3, 0.1], [0, 0.1, -0.2]], (m.n, 1, 1))
    gap = np.array([0, 1e-4, -2e-4])
    position = m.reference - 0.5 * sign[:, None] * gap
    state = AffineState.from_material(m.state(x=position, v=velocity), m.array(gradient, wp.mat33))
    geometry = InterfaceGeometry(m.array(np.tile(gap, (len(history), 1)), wp.vec3), geometry.frame)

    def apply(s=state):
        return project_affine_state(
            data["transfer"], s, geometry, data["reference_normal"], data["separation_map"], data["frame_map"]
        )

    return m, state, geometry, data, apply


def test_affine_impact_energy_momentum_and_retained_state():
    m, state, geometry, data, apply = prepared()
    transfer = data["transfer"]
    result = apply()
    before = transfer.pack(state.v, state.affine).numpy()
    after = transfer.pack(result.state.v, result.state.affine).numpy()
    C = result.constraint.cpu().reshape(len(geometry.gap), -1).numpy()
    np.testing.assert_allclose(C @ after.ravel(), 0, atol=2e-16)
    loss = 0.5 * m.mass * np.sum(before**2 - after**2)
    np.testing.assert_allclose(result.dissipation.cpu().numpy()[0], loss, rtol=2e-13)
    np.testing.assert_allclose(
        m.mass * np.sum(result.state.v.numpy() - state.v.numpy(), axis=0), 0, atol=1e-18
    )
    root = transfer.root.numpy()

    def angular(s):
        arms = np.swapaxes(s.F.numpy() @ root, 1, 2)
        local_velocity = np.swapaxes(s.affine.numpy() @ root, 1, 2)
        return m.mass * (
            np.sum(np.cross(s.x.numpy(), s.v.numpy()), axis=0)
            + np.sum(np.cross(arms, local_velocity), axis=(0, 1))
        )

    np.testing.assert_allclose(angular(result.state), angular(state), atol=2e-20)
    for name in ["x", "F", "Fp", "stress", "yielded"]:
        assert getattr(result.state, name) is getattr(state, name)
    assert np.max(abs(result.state.affine.numpy() - state.affine.numpy())) > 0
    restored = AffineState.restore(result.state.snapshot(), device=m.device)
    np.testing.assert_array_equal(restored.affine.numpy(), result.state.affine.numpy())
    for name, low in result.state.rounding_remainders.items():
        np.testing.assert_array_equal(restored.rounding_remainders[name], low)


def test_affine_impact_state_and_energy_gradients():
    m, state, geometry, data, apply = prepared()
    rng = np.random.default_rng(341)

    def outputs(result):
        return [result.state.v, result.state.affine, result.dissipation.high]

    with wp.Tape() as tape:
        result = apply()
    arrays = outputs(result)
    seeds = [rng.normal(size=a.numpy().shape) * scale for a, scale in zip(arrays, [1, 0.001, 1e4])]
    tape.backward(grads={a: m.array(s, a.dtype, False) for a, s in zip(arrays, seeds)})
    for source, scale in [(state.v, 0.1), (state.affine, 10.0)]:
        baseline = source.numpy().copy()
        direction = rng.normal(size=baseline.shape) * scale
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            results = []
            for sign in [-1, 1]:
                source.assign(baseline + sign * epsilon * direction)
                results.append([a.numpy() for a in outputs(apply())])
            fd = sum(np.sum((b - a) * s) for a, b, s in zip(*results, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-7, atol=1e-9)
        source.assign(baseline)


def test_impact_then_material_stages_preserve_history_and_gradient():
    m, state, geometry, data, apply = prepared()
    history = m.array(np.full(len(geometry.gap), 1e-3), float)
    rate = m.array([2000.0], float)

    def advance():
        impact = apply()
        step = advance_affine(
            state=impact.state,
            geometry=geometry,
            history=history,
            **{key: value for key, value in data.items() if key != "force"},
            damping_rate=rate,
            gravity=(0, 0, 0),
        )
        return step, impact

    def outputs(result):
        step, impact = result
        return [step.state.v, step.state.affine, step.state.F, step.history, impact.dissipation.high]

    with wp.Tape() as tape:
        result = advance()
    np.testing.assert_array_equal(result[0].history.numpy(), history.numpy())
    arrays = outputs(result)
    rng = np.random.default_rng(127)
    seeds = [rng.normal(size=a.numpy().shape) * scale for a, scale in zip(arrays, [1, 0.001, 1, 10, 1e4])]
    tape.backward(grads={a: m.array(s, a.dtype, False) for a, s in zip(arrays, seeds)})
    for source, scale in [(state.v, 0.1), (data["duration"], 2e-7)]:
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
