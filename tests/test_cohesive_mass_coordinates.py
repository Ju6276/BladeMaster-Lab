"""Mass-coordinate invariance and derivatives of the shared geometry transform."""

import numpy as np
import warp as wp

from blademaster_warp.cohesive_grid_update import mass_coordinates


def fixture():
    rng = np.random.default_rng(823)
    values = [
        rng.normal(size=(3, 5)),
        rng.normal(size=(3, 5, 3)),
        rng.normal(size=(5, 3)),
        np.geomspace(1e-12, 1e3, 5),
    ]
    return [
        wp.array(value, dtype=dtype, requires_grad=True)
        for value, dtype in zip(values, [wp.float64, wp.vec3d, wp.vec3d, wp.float64])
    ]


def test_mass_coordinates_preserve_geometry_action_across_mass_scales():
    from blademaster_warp._compensated import Expansion as E

    mapping, gradient, velocity, mass = fixture()
    B, G, z = [value.cpu() for value in mass_coordinates(mapping, gradient, velocity, mass)]
    original = E(mapping.numpy()) @ E(velocity.numpy())
    error = B @ z - original
    scale = abs(mapping.numpy()) @ abs(velocity.numpy())
    assert np.all(abs(error.high) <= 64 * np.finfo(float).eps ** 2 * scale)
    for axis in range(3):
        original = E(gradient.numpy()[:, :, axis]) @ E(velocity.numpy())
        error = G[:, :, axis] @ z - original
        scale = abs(gradient.numpy()[:, :, axis]) @ abs(velocity.numpy())
        assert np.all(abs(error.high) <= 64 * np.finfo(float).eps ** 2 * scale)


def test_mass_coordinates_pullback_matches_all_input_directions():
    sources = fixture()
    with wp.Tape() as tape:
        outputs = mass_coordinates(*sources)
    rng = np.random.default_rng(824)
    seeds = [rng.normal(size=value.high.numpy().shape) for value in outputs]
    tape.backward(
        grads={
            value.high: wp.array(seed, dtype=value.high.dtype, device=value.high.device)
            for value, seed in zip(outputs, seeds)
        }
    )
    for source in sources:
        initial = source.numpy().copy()
        direction = rng.normal(size=initial.shape) * np.maximum(abs(initial), 1e-12)
        derivative = float(np.sum(source.grad.numpy() * direction))
        for epsilon in [1e-4, 1e-5]:
            trials = []
            for sign in [-1.0, 1.0]:
                source.assign(initial + sign * epsilon * direction)
                trials.append([value.cpu() for value in mass_coordinates(*sources)])
            difference = sum(float(((b - a) * seed).sum().numpy()) for a, b, seed in zip(*trials, seeds)) / (
                2 * epsilon
            )
            np.testing.assert_allclose(derivative, difference, rtol=2e-5, atol=1e-8)
        source.assign(initial)
