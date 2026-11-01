"""Analytic small-strain stress and sub-ULP motion across stage checkpoints."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp._increment_stress import compute
from blademaster_warp.cohesive_history import CohesiveHistory
from blademaster_warp.cohesive_stages import AffineState, advance_affine
from blademaster_warp.reference_affine import build_transfer
from test_cohesive_stages import setup


@pytest.mark.parametrize("strain", [1e-4, 2.0**-60, 2.0**-100])
def test_diagonal_increment_stress_and_analytic_derivatives(strain):
    young, poisson = 20000.0, 0.3
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    values = np.zeros((1, 3, 3))
    values[0, 0, 0] = strain
    H = wp.array(values, dtype=wp.mat33d, requires_grad=True)
    Fp = wp.array(np.eye(3)[None], dtype=wp.mat33d)
    modulus = wp.array([young], dtype=wp.float64, requires_grad=True)
    stress = wp.zeros(1, dtype=wp.mat33d, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(compute, 1, inputs=[H, Fp, modulus, poisson], outputs=[stress])
    expected = np.diag([(2 * mu + lam) * strain, lam * strain * (1 + strain), lam * strain * (1 + strain)])
    assert np.linalg.norm(stress.numpy()[0] - expected) <= 2e-14 * np.linalg.norm(expected)
    seed = np.diag([0.7, -0.2, 0.4])
    tape.backward(grads={stress: wp.array(seed[None], dtype=wp.mat33d)})
    derivative = 0.7 * (2 * mu + lam) + 0.2 * lam * (1 + 2 * strain)
    np.testing.assert_allclose(H.grad.numpy()[0, 0, 0], derivative, rtol=2e-14)
    np.testing.assert_allclose(modulus.grad.numpy()[0], np.sum(seed * expected) / young, rtol=2e-14)


def test_sub_ulp_translation_survives_affine_restart_and_backward():
    m, _, _, _, _ = setup()
    retained = CohesiveHistory.empty(m.n, device=m.device)
    transfer = build_transfer(m, m.uncut)
    velocity = np.tile([2.0**-50, 0.0, 0.0], (m.n, 1))
    initial = AffineState.from_material(m.state(x=m.reference + 1.0, v=velocity), m.zeros(m.n, wp.mat33))
    initial.rounding_remainders["x"] = np.full((m.n, 3), 2.0**-80)
    duration = m.array([2.0**-20], float, False)

    def advance(state):
        return advance_affine(
            transfer,
            state,
            retained.geometry,
            retained.history,
            retained.quadrature.normals,
            retained.material,
            m.zeros((0, 4 * m.n), float),
            m.zeros((0, 4 * m.n), wp.vec3),
            retained.quadrature.area,
            duration=duration,
            gravity=(0.0, 0.0, 0.0),
        ).state

    with wp.Tape() as tape:
        first = advance(initial)
        direct = advance(first)
    expected = initial.rounding_remainders["x"] + 2 * duration.numpy()[0] * velocity
    np.testing.assert_array_equal(direct.x.numpy(), initial.x.numpy())
    np.testing.assert_allclose(direct.rounding_remainders["x"], expected, rtol=2e-12, atol=0.0)
    resumed = advance(AffineState.restore(first.snapshot(), device=m.device))
    for name, remainder in direct.rounding_remainders.items():
        np.testing.assert_array_equal(getattr(resumed, name).numpy(), getattr(direct, name).numpy())
        np.testing.assert_array_equal(resumed.rounding_remainders[name], remainder)
    tape.backward(grads={direct.x: m.array(np.ones((m.n, 3)), wp.vec3, False)})
    np.testing.assert_allclose(np.sum(initial.v.grad.numpy()), 6 * m.n * duration.numpy()[0], rtol=2e-12)


def test_initial_strain_remainder_enters_stages_and_total_state_adjoint():
    m, _, _, _, _ = setup()
    retained = CohesiveHistory.empty(m.n, device=m.device)
    transfer = build_transfer(m, m.uncut)
    initial = AffineState.from_material(m.state(), m.zeros(m.n, wp.mat33))
    direction = np.zeros((m.n, 3, 3))
    direction[:, 0, 0] = 1.0
    strain = 2.0**-70
    initial.rounding_remainders["F"] = strain * direction
    assert initial.stress is None

    def advance():
        return advance_affine(
            transfer,
            initial,
            retained.geometry,
            retained.history,
            retained.quadrature.normals,
            retained.material,
            m.zeros((0, 4 * m.n), float),
            m.zeros((0, 4 * m.n), wp.vec3),
            retained.quadrature.area,
            duration=m.duration,
            gravity=(0.0, 0.0, 0.0),
        ).state

    with wp.Tape() as tape:
        result = advance()
    assert np.linalg.norm(result.stress.numpy()) > 0.0
    seed = np.tile(np.diag([0.7, -0.2, 0.4]), (m.n, 1, 1))
    tape.backward(grads={result.stress: m.array(seed, wp.mat33, False)})
    derivative = np.sum(initial.F.grad.numpy() * direction)
    for fraction in [0.25, 0.5]:
        delta = fraction * strain
        values = []
        for sign in [-1.0, 1.0]:
            initial.rounding_remainders["F"] = (strain + sign * delta) * direction
            values.append(np.sum(advance().stress.numpy() * seed))
        np.testing.assert_allclose((values[1] - values[0]) / (2 * delta), derivative, rtol=2e-12)
