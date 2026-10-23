"""Affine completeness, surface geometry and virtual work of material traces."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.reference_interface import reconstruct


DTYPES = [wp.vec3d, wp.mat33d, wp.vec3d, wp.vec3d, wp.float64, wp.vec3d]


def arrays(values):
    return [wp.array(a, dtype=dtype, requires_grad=True) for a, dtype in zip(values, DTYPES)]


def fixture(seed=710):
    rng = np.random.default_rng(seed)
    reference = rng.uniform(-0.01, 0.01, size=(12, 3))
    points = rng.uniform(-0.004, 0.004, size=(4, 3))
    weights = rng.uniform(0.1, 1.0, size=(4, 2, 12))
    weights[:, 0, 6:] = 0
    weights[:, 1, :6] = 0
    gradients = rng.normal(scale=30.0, size=(4, 2, 12, 3)) * (weights > 0)[..., None]
    deformation = np.eye(3) + rng.normal(scale=0.1, size=(12, 3, 3))
    position = reference + rng.normal(scale=0.001, size=reference.shape)
    return [position, deformation, reference, points, weights, gradients]


@pytest.mark.parametrize("kind", ["translation", "rotation", "stretch", "shear"])
def test_shared_affine_motion_has_zero_jump_and_correct_frame(kind):
    values = fixture()
    F = np.eye(3)
    if kind == "rotation":
        F, _ = np.linalg.qr(np.random.default_rng(59).normal(size=(3, 3)))
    elif kind == "stretch":
        F = np.diag([0.85, 1.3, 1.1])
    elif kind == "shear":
        F[0, 2] = 0.4
        F[1, 0] = -0.2
    translation = np.array([0.23, -0.17, 0.3])
    values[0] = values[2] @ F.T + translation
    values[1] = np.tile(F, (12, 1, 1))
    result = reconstruct(*arrays(values))
    np.testing.assert_allclose(result.gap.numpy(), 0.0, atol=2e-16, rtol=0.0)
    np.testing.assert_allclose(result.frame.numpy(), np.tile(F, (4, 1, 1)), atol=1e-14, rtol=0.0)
    exact = values[3] @ (F - np.eye(3)).T + translation
    np.testing.assert_allclose(
        result.displacement.numpy(), np.repeat(exact[:, None], 2, axis=1), atol=2e-16, rtol=0.0
    )


def test_distinct_affine_faces_and_superposed_rigid_motion():
    values = fixture()
    Fs = [
        np.array([[1.0, 0.2, 0.1], [0.0, 0.8, 0.1], [0.0, 0.0, 1.1]]),
        np.array([[0.9, 0.0, 0.0], [0.1, 1.2, 0.0], [0.0, -0.1, 1.3]]),
    ]
    shifts = [np.array([0.001, -0.002, 0.003]), np.array([-0.002, 0.001, 0.005])]
    for side in range(2):
        section = slice(6 * side, 6 * side + 6)
        values[0][section] = values[2][section] @ Fs[side].T + shifts[side]
        values[1][section] = Fs[side]
    result = reconstruct(*arrays(values))
    exact_gap = values[3] @ (Fs[1] - Fs[0]).T + shifts[1] - shifts[0]
    np.testing.assert_allclose(result.gap.numpy(), exact_gap, atol=2e-17)
    np.testing.assert_allclose(result.frame.numpy(), np.tile(0.5 * (Fs[0] + Fs[1]), (4, 1, 1)), atol=1e-14)
    rotation, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))
    values[0] = values[0] @ rotation.T + [0.1, -0.2, 0.3]
    values[1] = rotation @ values[1]
    transformed = reconstruct(*arrays(values))
    np.testing.assert_allclose(transformed.gap.numpy(), result.gap.numpy() @ rotation.T, atol=2e-16)
    np.testing.assert_allclose(transformed.frame.numpy(), rotation @ result.frame.numpy(), atol=1e-14)


def test_frame_is_spatial_derivative_of_same_trace():
    values = fixture()
    result = reconstruct(*arrays(values))
    for axis in range(3):
        epsilon = 1e-6
        traces = []
        for sign in [-1, 1]:
            shifted = [a.copy() for a in values]
            shifted[3][:, axis] += sign * epsilon
            shifted[4] += sign * epsilon * values[5][..., axis]
            out = reconstruct(*arrays(shifted))
            traces.append(shifted[3] + out.displacement.numpy().mean(axis=1))
        fd = (traces[1] - traces[0]) / (2 * epsilon)
        np.testing.assert_allclose(result.frame.numpy()[:, :, axis], fd, rtol=2e-7, atol=1e-9)


def test_input_derivatives_and_translation_virtual_work():
    rng = np.random.default_rng(1623)
    values = fixture()
    sources = arrays(values)
    with wp.Tape() as tape:
        result = reconstruct(*sources)
    targets = [result.gap, result.frame]
    seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(targets, [0.01, 1.0])]
    tape.backward(grads={a: wp.array(seed, dtype=a.dtype) for a, seed in zip(targets, seeds)})
    np.testing.assert_allclose(sources[0].grad.numpy().sum(axis=0), 0.0, atol=2e-13)
    for index, source in enumerate(sources):
        base = source.numpy()
        direction = rng.normal(size=base.shape) * np.maximum(np.abs(base), 0.01)
        if index == 4:
            direction *= base > 0
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            losses = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                trial = reconstruct(*sources)
                losses.append(
                    sum(np.sum(a.numpy() * seed) for a, seed in zip([trial.gap, trial.frame], seeds))
                )
            fd = (losses[1] - losses[0]) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-6, atol=1e-8)
        source.assign(base)
