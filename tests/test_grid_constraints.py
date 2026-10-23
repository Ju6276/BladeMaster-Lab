"""Shared-node momentum, projection energy and the constrained velocity adjoint."""

import numpy as np
import warp as wp

from blademaster_warp.grid_constraints import project_velocity


def arrays(Q, mass, velocity):
    return [
        wp.array(value, dtype=dtype, requires_grad=True)
        for value, dtype in zip([Q, mass, velocity], [wp.float64, wp.float64, wp.vec3d])
    ]


def test_shared_nodes_match_mass_average_and_projection_energy():
    rng = np.random.default_rng(45)
    mass = rng.uniform(0.01, 0.2, size=6)
    velocity = rng.normal(size=(6, 3))
    Q = np.c_[np.eye(3), -np.eye(3)]
    projected = project_velocity(*arrays(Q, mass, velocity)).numpy()
    average = mass[:3, None] * velocity[:3] + mass[3:, None] * velocity[3:]
    average /= (mass[:3] + mass[3:])[:, None]
    np.testing.assert_allclose(projected, np.tile(average, (2, 1)), atol=1e-14)
    np.testing.assert_allclose(np.sum(mass[:, None] * (projected - velocity), axis=0), 0.0, atol=1e-15)
    kinetic_change = 0.5 * np.sum(mass[:, None] * (projected**2 - velocity**2))
    removed = -0.5 * np.sum(mass[:, None] * (projected - velocity) ** 2)
    np.testing.assert_allclose(kinetic_change, removed, atol=1e-15)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = project_velocity(*arrays(Q, mass, velocity @ rotation.T)).numpy()
    np.testing.assert_allclose(rotated, projected @ rotation.T, atol=1e-14)


def test_projection_derivatives_for_mapping_mass_and_velocity():
    rng = np.random.default_rng(843)
    Q = rng.normal(size=(4, 9))
    Q -= Q.mean(axis=1)[:, None]
    values = [Q, rng.uniform(0.001, 0.01, size=9), rng.normal(size=(9, 3))]
    sources = arrays(*values)
    active = np.array([True, False, True, True])
    with wp.Tape() as tape:
        projected = project_velocity(*sources, active=active)
    seed = rng.normal(size=(9, 3))
    tape.backward(grads={projected: wp.array(seed, dtype=wp.vec3d)})
    np.testing.assert_array_equal(sources[0].grad.numpy()[1], 0.0)
    for source in sources:
        base = source.numpy()
        direction = rng.normal(size=base.shape) * np.maximum(np.abs(base), 1e-3)
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5, 1e-6]:
            trials = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                trials.append(float(np.sum(project_velocity(*sources, active=active).numpy() * seed)))
            np.testing.assert_allclose(ad, (trials[1] - trials[0]) / (2 * epsilon), rtol=2e-6, atol=1e-9)
        source.assign(base)


def test_duplicate_constraints_preserve_solution_and_physical_adjoint():
    rng = np.random.default_rng(34)
    Q = rng.normal(size=(3, 8))
    mass = rng.uniform(0.01, 0.1, size=8)
    velocity = rng.normal(size=(8, 3))
    seed = rng.normal(size=(8, 3))
    results, gradients = [], []
    for copies in [1, 2]:
        sources = arrays(np.repeat(Q, copies, axis=0), mass, velocity)
        with wp.Tape() as tape:
            result = project_velocity(*sources)
        tape.backward(grads={result: wp.array(seed, dtype=wp.vec3d)})
        results.append(result.numpy())
        gradients.append(
            [
                sources[0].grad.numpy().reshape(3, copies, 8).sum(axis=1),
                sources[1].grad.numpy(),
                sources[2].grad.numpy(),
            ]
        )
    np.testing.assert_allclose(*results, atol=1e-14)
    for base, repeated in zip(*gradients):
        np.testing.assert_allclose(base, repeated, rtol=1e-12, atol=1e-12)
