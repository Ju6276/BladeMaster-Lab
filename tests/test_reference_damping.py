"""Continuous transfer damping: analytic response, conservation and adjoints."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from blademaster_warp.reference_damping import decay
from test_reference_affine import setup


@pytest.mark.parametrize("spacing", [0.006, 0.004])
@pytest.mark.parametrize("shape,seed", [("block", 0), ("block", 1), ("cylinder", 0)])
def test_conservation_energy_subdivision_and_directional_derivatives(spacing, shape, seed):
    m, _, transfer, _ = setup(spacing, shape, seed)
    rng = np.random.default_rng(152)
    velocity = m.array(rng.normal(size=(m.n, 3)), wp.vec3)
    affine = m.array(rng.normal(size=(m.n, 3, 3)) / spacing, wp.mat33)
    duration = m.array([2e-4], float)
    rate = m.array([2000.0], float)
    with wp.Tape() as tape:
        moments = transfer.pack(velocity, affine)
        result = decay(transfer, moments, duration, rate)
    before = np.sum(transfer.kinetic_energy(moments).numpy())
    after = np.sum(transfer.kinetic_energy(result.moments).numpy())
    dissipated = np.sum(result.energy.numpy())
    assert dissipated > 0.0
    np.testing.assert_allclose(dissipated, before - after, rtol=3e-14)
    divided = moments
    for _ in range(4):
        divided = decay(transfer, divided, m.array([5e-5], float), rate).moments
    np.testing.assert_allclose(divided.numpy(), result.moments.numpy(), rtol=3e-14, atol=2e-15)

    A = np.array([[0.3, -50.0, -20.0], [50.0, -0.2, -10.0], [20.0, 10.0, 0.1]])
    linear = transfer.pack(
        m.array(m.reference @ A.T + [0.1, 0.2, 0.3], wp.vec3),
        m.array(np.tile(A, (m.n, 1, 1)), wp.mat33),
    )
    preserved = decay(transfer, linear, duration, rate)
    np.testing.assert_allclose(preserved.moments.numpy(), linear.numpy(), rtol=3e-14, atol=2e-15)
    # Pairing with every affine field includes linear and reference angular momentum.
    np.testing.assert_allclose(
        np.sum(result.moments.numpy() * linear.numpy()),
        np.sum(moments.numpy() * linear.numpy()),
        rtol=2e-13,
        atol=2e-14,
    )
    moment_seed = rng.normal(size=result.moments.numpy().shape)
    energy_seed = rng.normal(size=m.n) / m.mass
    tape.backward(
        grads={
            result.moments: m.array(moment_seed, wp.vec3, False),
            result.energy: m.array(energy_seed, float, False),
        }
    )
    for source, scale in [(velocity, 0.1), (affine, 10.0), (duration, 2e-5), (rate, 200.0)]:
        original = source.numpy().copy()
        direction = rng.normal(size=original.shape) * scale
        ad = np.sum(source.grad.numpy() * direction)
        for eps in [1e-4, 1e-5]:
            values = []
            for sign in [-1, 1]:
                source.assign(original + sign * eps * direction)
                actual = decay(transfer, transfer.pack(velocity, affine), duration, rate)
                values.append((actual.moments.numpy(), actual.energy.numpy()))
            fd = (
                np.sum(moment_seed * (values[1][0] - values[0][0]))
                + np.sum(energy_seed * (values[1][1] - values[0][1]))
            ) / (2 * eps)
            np.testing.assert_allclose(ad, fd, rtol=2e-6, atol=2e-8)
        source.assign(original)


@pytest.mark.parametrize("time,rate", [(0.0, 2000.0), (2e-4, 0.0), (0.02, 10000.0)])
def test_diagonal_operator_and_zero_parameter_derivatives(time, rate):
    m, _, transfer, _ = setup()
    size = 4 * m.n
    eigenvalues = np.linspace(0.0, 1.0, size)
    transfer = replace(
        transfer,
        grid_map=m.array(np.diag(np.sqrt(1.0 - eigenvalues)), float, False),
        grid_mass=np.full(size, m.mass),
    )
    rng = np.random.default_rng(444)
    values = rng.normal(size=(size, 3))
    z, dt, eta = m.array(values, wp.vec3), m.array([time], float), m.array([rate], float)
    with wp.Tape() as tape:
        result = decay(transfer, z, dt, eta)
    multiplier = np.exp(-time * rate * eigenvalues[:, None])
    expected = multiplier * values
    np.testing.assert_allclose(result.moments.numpy(), expected, rtol=3e-12, atol=1e-14)
    output_seed = rng.normal(size=values.shape)
    energy_seed = rng.normal(size=m.n) / m.mass
    tape.backward(
        grads={
            result.moments: m.array(output_seed, wp.vec3, False),
            result.energy: m.array(energy_seed, float, False),
        }
    )
    weighting = m.mass * np.repeat(energy_seed, 4)[:, None]
    seed = output_seed - weighting * expected
    np.testing.assert_allclose(z.grad.numpy(), multiplier * seed + weighting * values, rtol=2e-12, atol=3e-14)
    power = np.sum(seed * eigenvalues[:, None] * expected)
    np.testing.assert_allclose(dt.grad.numpy()[0], -rate * power, rtol=2e-12, atol=1e-10)
    np.testing.assert_allclose(eta.grad.numpy()[0], -time * power, rtol=2e-12, atol=1e-15)


def test_invalid_damping_inputs():
    m, _, transfer, _ = setup()
    z = m.zeros(4 * m.n, wp.vec3)
    for time, rate in [(-1.0, 1.0), (1.0, -1.0), (np.nan, 1.0), (1.0, np.inf)]:
        with pytest.raises(ValueError, match="finite and nonnegative"):
            decay(transfer, z, m.array([time], float), m.array([rate], float))
    with pytest.raises(ValueError, match="matching float64"):
        decay(transfer, m.zeros(m.n, wp.vec3), m.array([1.0], float), m.array([1.0], float))
    z.fill_(wp.vec3d(np.nan))
    with pytest.raises(ValueError, match="moments must be finite"):
        decay(transfer, z, m.array([1.0], float), m.array([1.0], float))
