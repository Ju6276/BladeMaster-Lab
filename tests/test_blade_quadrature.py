"""Finite blade footprints through sampled domains and blade-pose derivatives."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.geometry import points
from blademaster_warp.reference_cut import blade_quadrature


def arrays(values):
    dtypes = [wp.vec3d, wp.mat33d, wp.vec3d, wp.vec3d, wp.vec3d, wp.quatd, wp.vec3d]
    return [wp.array(a, dtype=d, requires_grad=True) for a, d in zip(values, dtypes)]


def block(spacing=0.006):
    reference, _, _ = points((0.012,) * 3, spacing, "block")
    return [
        reference.copy(),
        np.repeat(np.eye(3)[None], len(reference), axis=0),
        reference,
        np.full_like(reference, spacing / 4),
        np.array([[0.0, 0.0004, 0.006]]),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0005, 0.0037, 0.004]]),
    ]


@pytest.mark.parametrize("spacing", [0.006, 0.004])
@pytest.mark.parametrize("reverse", [False, True])
def test_blade_entry_and_exit_match_rectangular_overlap(spacing, reverse):
    values = block(spacing)
    if reverse:
        values[5][0] = [0.0, 0.0, 1.0, 0.0]
    for x in [0.0, 0.0007]:
        for z in [-0.005, -0.004, -0.003, 0.0, 0.004, 0.006, 0.008, 0.011, 0.012, 0.015, 0.016, 0.017]:
            values[4][0, [0, 2]] = [x, z]
            result = blade_quadrature(*arrays(values))
            lower, upper = max(0.0, z - 0.004), min(0.012, z + 0.004)
            area = 0.0074 * max(0.0, upper - lower)
            np.testing.assert_allclose(result.area.numpy().sum(), area, rtol=2e-14, atol=1e-20)
            if area > 0:
                center = result.area.numpy() @ result.points.numpy() / area
                np.testing.assert_allclose(center, [x, 0.0004, (lower + upper) / 2], atol=1e-17)
            else:
                assert len(result.area) == 0


def test_blade_footprint_is_invariant_under_common_rigid_motion():
    values = block()
    values[4][0] = [0.0007, 0.0004, 0.003]
    original = blade_quadrature(*arrays(values))
    axis = np.array([0.2, 0.6, 0.7])
    axis /= np.linalg.norm(axis)
    angle = 0.83
    cross = np.cross(axis, np.eye(3)).T
    rotation = np.cos(angle) * np.eye(3) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * cross
    shift = np.array([0.02, -0.03, 0.01])
    values[0] = values[0] @ rotation.T + shift
    values[1] = rotation @ values[1]
    values[4] = values[4] @ rotation.T + shift
    values[5][0] = np.r_[axis * np.sin(angle / 2), np.cos(angle / 2)]
    moved = blade_quadrature(*arrays(values))
    np.testing.assert_allclose(moved.points.numpy(), original.points.numpy(), atol=2e-17)
    np.testing.assert_allclose(moved.area.numpy(), original.area.numpy(), rtol=1e-13)
    np.testing.assert_allclose(moved.normals.numpy(), original.normals.numpy(), atol=5e-16)


def test_blade_and_particle_derivatives_of_integrated_measure():
    values = block()
    values[1][:] = [[1.1, 0.1, 0.0], [0.0, 0.9, 0.12], [0.03, 0.0, 1.2]]
    values[4][0] = [0.0007, 0.0005, 0.0027]
    values[5][0] = [0.1, -0.15, 0.2, 1.2]
    arguments = arrays(values)
    scale = 0.012
    linear = np.array([1.2, -0.5, 0.8]) / scale
    normal_seed = np.array([0.2, -0.3, 0.4])

    def density(result):
        return 1 + result.points.numpy() @ linear + result.normals.numpy() @ normal_seed

    with wp.Tape() as tape:
        result = blade_quadrature(*arguments)
    area = result.area.numpy()
    tape.backward(
        grads={
            result.area: wp.array(density(result) / scale**2, dtype=wp.float64),
            result.points: wp.array(area[:, None] * linear / scale**2, dtype=wp.vec3d),
            result.normals: wp.array(area[:, None] * normal_seed / scale**2, dtype=wp.vec3d),
        }
    )
    np.testing.assert_allclose(np.sum(arguments[5].grad.numpy() * values[5]), 0.0, atol=2e-14)
    assert arguments[6].grad.numpy()[0, 0] == 0.0
    rng = np.random.default_rng(941)
    for source in arguments:
        base = source.numpy().copy()
        direction = rng.normal(size=base.shape) * np.maximum(np.abs(base), 0.001)
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-5, 1e-6]:
            losses = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                trial = blade_quadrature(*arguments)
                losses.append(trial.area.numpy() @ density(trial) / scale**2)
            np.testing.assert_allclose(ad, (losses[1] - losses[0]) / (2 * epsilon), rtol=2e-6, atol=1e-8)
        source.assign(base)
