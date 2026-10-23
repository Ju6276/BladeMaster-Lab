"""Analytic impacts, redundant contacts, dissipation and input derivatives."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_impact import project_velocity, normal_velocity_map
from blademaster_warp.cohesive_geometry import InterfaceGeometry


def array(value, dtype):
    return wp.array(value, dtype=dtype, requires_grad=True)


@pytest.mark.parametrize("normal", [[1.0, 0.0, 0.0], [0.3, -0.4, 0.5]])
@pytest.mark.parametrize("masses", [[0.001, 0.002], [3.0, 1.0]])
def test_two_body_impact_preserves_momentum_and_tangent(normal, masses):
    n = np.asarray(normal) / np.linalg.norm(normal)
    mass = np.array(masses)
    v = np.array([[0.2, 0.03, -0.01], [-0.3, -0.02, 0.04]])
    relative = (v[1] - v[0]) @ n
    if relative >= 0.0:
        v = -v
        relative = -relative
    C = np.array([[-n, n]])
    output = project_velocity(array(C, wp.vec3d), array(mass, wp.float64), array(v, wp.vec3d)).numpy()
    impulse = -relative / np.sum(1 / mass)
    expected = v + np.array([-n / mass[0], n / mass[1]]) * impulse
    np.testing.assert_allclose(output, expected, atol=2e-15)
    np.testing.assert_allclose(np.sum(mass[:, None] * (output - v), axis=0), 0.0, atol=2e-15)
    loss = 0.5 * np.sum(mass[:, None] * (v * v - output * output))
    np.testing.assert_allclose(loss, 0.5 * relative**2 / np.sum(1 / mass), rtol=2e-14)


@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("scaling", [1.0, 1e-8, 1e8])
def test_simultaneous_three_body_collision(duplicate, scaling):
    mass = np.array([1.0, 2.0, 3.0])
    v = np.array([[3.0, 0.3, 0.0], [0.0, -0.2, 0.1], [-1.0, 0.4, -0.1]])
    C = np.zeros((2, 3, 3))
    C[0, :, 0], C[1, :, 0] = [-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]
    if duplicate:
        C = np.concatenate([C, C[:1], C[1:] * scaling, np.zeros_like(C[:1])])
    else:
        C[0] *= scaling
    output = project_velocity(array(C, wp.vec3d), array(mass, wp.float64), array(v, wp.vec3d)).numpy()
    expected = v.copy()
    expected[:, 0] = mass @ v[:, 0] / mass.sum()
    np.testing.assert_allclose(output, expected, atol=3e-15)
    jump_loss = 0.5 * np.sum(mass[:, None] * (output - v) ** 2)
    energy_loss = 0.5 * np.sum(mass[:, None] * (v * v - output * output))
    np.testing.assert_allclose(jump_loss, energy_loss, rtol=3e-15)


@pytest.mark.parametrize("kind", ["empty", "zero", "separating"])
def test_free_velocity_has_identity_pullback(kind):
    C = np.array([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    if kind == "empty":
        C = C[:0]
    elif kind == "zero":
        C *= 0.0
    velocity = array([[-0.2, 0.0, 0.0], [0.1, 0.0, 0.0]], wp.vec3d)
    with wp.Tape() as tape:
        result = project_velocity(array(C, wp.vec3d), array([1.0, 2.0], wp.float64), velocity)
    seed = np.array([[0.2, -0.3, 0.7], [-0.1, 0.4, 0.3]])
    tape.backward(grads={result: array(seed, wp.vec3d)})
    np.testing.assert_array_equal(result.numpy(), velocity.numpy())
    np.testing.assert_array_equal(velocity.grad.numpy(), seed)


@pytest.mark.parametrize("copies", [1, 3])
def test_mass_velocity_and_contact_geometry_derivatives(copies):
    n = np.array([0.8, 0.3, -0.2])
    C = np.tile(np.array([[-n, n, np.zeros(3)]]), (copies, 1, 1))
    # A separate, separating contact has zero multiplier.
    C = np.concatenate([C, np.array([[np.zeros(3), [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]])])
    values = [
        C,
        np.array([0.001, 0.002, 0.003]),
        np.array([[0.4, 0.03, 0.01], [-0.2, -0.04, -0.01], [0.0, 0.2, 0.0]]),
    ]
    dtypes = [wp.vec3d, wp.float64, wp.vec3d]
    sources = [array(value, dtype) for value, dtype in zip(values, dtypes)]
    with wp.Tape() as tape:
        result = project_velocity(*sources)
    seed = np.random.default_rng(35).normal(size=(3, 3))
    tape.backward(grads={result: array(seed, wp.vec3d)})
    rng = np.random.default_rng(751)
    for index, source in enumerate(sources):
        direction = rng.normal(size=values[index].shape) * [1.0, 0.001, 0.1][index]
        if index == 0:
            # Duplicated quadrature rows describe the same moving face.
            direction[:copies] = direction[0]
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            outputs = []
            for sign in [-1.0, 1.0]:
                trial = [v.copy() for v in values]
                trial[index] += sign * epsilon * direction
                arrays = [array(value, dtype) for value, dtype in zip(trial, dtypes)]
                outputs.append(project_velocity(*arrays).numpy())
            fd = np.sum((outputs[1] - outputs[0]) * seed) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-7, atol=1e-10)


@pytest.mark.parametrize("mass", [[0.0, 1.0], [-1.0, 1.0], [np.nan, 1.0]])
def test_invalid_masses_are_rejected(mass):
    with pytest.raises(ValueError, match="positive masses"):
        project_velocity(
            array([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], wp.vec3d),
            array(mass, wp.float64),
            array(np.zeros((2, 3)), wp.vec3d),
        )


def moving_geometry():
    rng = np.random.default_rng(219)
    values = [
        rng.normal(size=(2, 3)) * 0.001,
        np.eye(3)[None] + rng.normal(size=(2, 3, 3)) * 0.1,
        rng.normal(size=(2, 3)),
        rng.normal(size=(2, 4)),
        rng.normal(size=(2, 4, 3)) * 30.0,
    ]
    dtypes = [wp.vec3d, wp.mat33d, wp.vec3d, wp.float64, wp.vec3d]
    return values, dtypes


def map_from_arrays(sources):
    return normal_velocity_map(InterfaceGeometry(*sources[:2]), *sources[2:])


def test_moving_normal_map_matches_gap_rate():
    values, dtypes = moving_geometry()
    C = map_from_arrays([array(value, dtype) for value, dtype in zip(values, dtypes)]).numpy()
    v = np.random.default_rng(932).normal(size=(4, 3))
    g, F, N, B, G = values
    expected = np.einsum("qpa,pa->q", C, v)
    for epsilon in [1e-5, 1e-6]:
        guards = []
        for sign in [-1.0, 1.0]:
            next_gap = g + sign * epsilon * (B @ v)
            next_frame = F + sign * epsilon * np.einsum("pa,qpb->qab", v, G)
            normal = np.linalg.solve(next_frame.transpose(0, 2, 1), N[..., None])[..., 0]
            normal /= np.linalg.norm(normal, axis=1)[:, None]
            guards.append(np.sum(next_gap * normal, axis=1))
        np.testing.assert_allclose((guards[1] - guards[0]) / (2 * epsilon), expected, rtol=2e-6)


def test_moving_normal_map_input_derivatives():
    values, dtypes = moving_geometry()
    sources = [array(value, dtype) for value, dtype in zip(values, dtypes)]
    with wp.Tape() as tape:
        result = map_from_arrays(sources)
    rng = np.random.default_rng(219)
    seed = rng.normal(size=(2, 4, 3))
    tape.backward(grads={result: array(seed, wp.vec3d)})
    for index, source in enumerate(sources):
        direction = rng.normal(size=values[index].shape) * [0.001, 0.1, 0.1, 0.1, 3.0][index]
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            outputs = []
            for sign in [-1.0, 1.0]:
                trial = [v.copy() for v in values]
                trial[index] += sign * epsilon * direction
                outputs.append(map_from_arrays([array(v, t) for v, t in zip(trial, dtypes)]).numpy())
            fd = np.sum((outputs[1] - outputs[0]) * seed) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-7, atol=1e-10)
