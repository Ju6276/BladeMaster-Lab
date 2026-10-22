"""Constraint work, rigid rotations and derivatives for moving surface normals."""

import numpy as np
import warp as wp

from blademaster_warp.cohesive import cohesive_contact_response
from blademaster_warp.precision import launch_double


def inputs(gap, deformation, normal, multiplier):
    return [wp.array(np.asarray(v), dtype=dtype, requires_grad=True)
            for v, dtype in zip([gap, deformation, normal, multiplier],
                                [wp.vec3d, wp.mat33d, wp.vec3d, wp.float64])]


def response(values):
    outputs = [wp.zeros(len(values[0]), dtype=dtype, requires_grad=True)
               for dtype in [wp.vec3d, wp.mat33d]]
    launch_double(cohesive_contact_response, len(values[0]), inputs=values, outputs=outputs)
    return outputs


def states(count=64):
    rng = np.random.default_rng(4598)
    gap = rng.normal(scale=.001, size=(count, 3))
    deformation = np.eye(3) + rng.normal(scale=.15, size=(count, 3, 3))
    normal = rng.normal(size=(count, 3))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    multiplier = -rng.uniform(1e3, 1e5, count)
    return rng, [gap, deformation, normal, multiplier]


def lagrangian(values):
    gap, deformation, normal, multiplier = values
    cofactor = np.linalg.det(deformation)[:, None, None] * np.linalg.inv(deformation).transpose(0, 2, 1)
    area = np.einsum('nij,nj->ni', cofactor, normal)
    unit = area / np.linalg.norm(area, axis=1)[:, None]
    return multiplier * np.sum(gap * unit, axis=1)


def test_constraint_forces_match_independent_lagrangian_virtual_work():
    rng, values = states()
    traction, stress = [a.numpy() for a in response(inputs(*values))]
    for index, force, scale in [(0, traction, .001), (1, stress, .2)]:
        direction = rng.normal(size=values[index].shape) * scale
        virtual_work = np.sum(force * direction, axis=tuple(range(1, force.ndim)))
        for epsilon in [1e-5, 1e-6]:
            plus, minus = values.copy(), values.copy()
            plus[index] = values[index] + epsilon * direction
            minus[index] = values[index] - epsilon * direction
            difference = (lagrangian(plus) - lagrangian(minus)) / (2 * epsilon)
            np.testing.assert_allclose(difference, virtual_work, rtol=2e-7, atol=2e-8)


def test_constraint_rotation_and_zero_rigid_spin_work():
    rng, values = states()
    gap, deformation, normal, multiplier = values
    traction, stress = [a.numpy() for a in response(inputs(*values))]
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = response(inputs(gap @ q.T, q @ deformation, normal, multiplier))
    np.testing.assert_allclose(rotated[0].numpy(), traction @ q.T, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(rotated[1].numpy(), q @ stress, rtol=1e-12, atol=1e-12)
    omega = np.array([[0., -.3, .2], [.3, 0., -.1], [-.2, .1, 0.]])
    power = np.sum(traction * (gap @ omega.T), axis=1)
    power += np.sum(stress * (omega @ deformation), axis=(1, 2))
    np.testing.assert_allclose(power, 0., atol=1e-13)


def test_constraint_force_gradients_for_all_inputs():
    rng, values = states()
    sources = inputs(*values)
    with wp.Tape() as tape:
        outputs = response(sources)
    seeds = [rng.normal(size=(len(values[0]), 3)) / 5e4,
             rng.normal(size=(len(values[0]), 3, 3)) / 50.]
    tape.backward(grads={a: wp.array(seed, dtype=a.dtype) for a, seed in zip(outputs, seeds)})
    for source, scale in zip(sources, [.001, .2, 1., 5e4]):
        base = source.numpy()
        direction = rng.normal(size=base.shape) * scale
        ad = float(np.sum(source.grad.numpy() * direction))
        for epsilon in [1e-5, 1e-6]:
            trials = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                trials.append([a.numpy() for a in response(sources)])
            fd = sum(np.sum((b - a) * seed) for a, b, seed in zip(*trials, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=1e-7, atol=1e-8)
        source.assign(base)


def test_rank_two_surface_and_zero_pressure():
    gap = np.array([[0., .002, -.001], [0., .002, -.001]])
    deformation = np.tile(np.diag([0., 2., 3.]), (2, 1, 1))
    normal = np.tile([1., 0., 0.], (2, 1))
    multiplier = np.array([-5e4, 0.])
    traction, stress = [a.numpy() for a in response(inputs(gap, deformation, normal, multiplier))]
    np.testing.assert_allclose(traction, [[-5e4, 0., 0.], [0., 0., 0.]], atol=1e-12)
    expected = np.zeros((2, 3, 3))
    expected[0, 0, 1] = 50.
    expected[0, 0, 2] = -50. / 3.
    np.testing.assert_allclose(stress, expected, atol=1e-12)
