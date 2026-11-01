"""Finite rigid blade geometry against Warp, analytic distances and natural differences."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp._blade_contact_geometry import query
from blademaster_warp._cohesive_expansion import E
from blademaster_warp.contact import blade_query
from blademaster_warp.precision import launch_double


@wp.kernel
def rigid_query(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    origin: wp.array(dtype=wp.vec3),
    rotation: wp.array(dtype=wp.quat),
    velocity: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3),
    half: wp.array(dtype=wp.vec3),
    bevel: wp.array(dtype=float),
    radius: wp.array(dtype=float),
    shifted: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    relative: wp.array(dtype=wp.vec3),
    closest: wp.array(dtype=wp.vec3),
    gap: wp.array(dtype=float),
):
    i = wp.tid()
    local = wp.quat_rotate_inv(rotation[0], x[i] - origin[0])
    near, n, distance = blade_query(local, half[0], bevel[0])
    lever = wp.quat_rotate(rotation[0], near)
    normal[i] = wp.quat_rotate(rotation[0], n)
    shifted[i] = (distance - radius[0]) * normal[i]
    relative[i] = v[i] - velocity[0] - wp.cross(omega[0], lever)
    closest[i] = origin[0] + lever
    gap[i] = distance - radius[0]


def fixture(bevel):
    axis = np.array([0.3, -0.7, 0.2])
    axis /= np.linalg.norm(axis)
    angle = 0.37
    quaternion = np.r_[axis * np.sin(angle / 2), np.cos(angle / 2)]
    return [
        np.array([0.003, -0.002, -0.007]),
        np.array([0.12, 0.03, -0.4]),
        np.array([0.001, -0.001, 0.002]),
        quaternion,
        np.array([0.03, 0.01, -0.2]),
        np.array([0.3, -0.2, 0.7]),
        np.array([0.001, 0.005, 0.01]),
        np.array([bevel]),
        np.array([0.0008]),
    ]


@pytest.mark.parametrize("bevel", [0.0, 0.004])
def test_rigid_geometry_and_tangent_quaternion_derivative_match_warp(bevel):
    rng = np.random.default_rng(2752)
    base = fixture(bevel)
    points = rng.uniform([-0.003, -0.007, -0.012], [0.003, 0.007, 0.012], (24, 3))
    base[0], base[1] = points, rng.normal(size=(len(points), 3)) * 0.2
    kinds = [wp.vec3d, wp.vec3d, wp.vec3d, wp.quatd, wp.vec3d, wp.vec3d, wp.vec3d, wp.float64, wp.float64]
    arrays = [
        wp.array(
            a if i < 2 else a.reshape(1, -1) if i < 7 else a, dtype=kind, device="cuda:0", requires_grad=True
        )
        for i, (a, kind) in enumerate(zip(base, kinds))
    ]
    outputs = [
        wp.zeros(len(points), dtype=kind, device="cuda:0", requires_grad=True)
        for kind in [wp.vec3d] * 4 + [wp.float64]
    ]
    with wp.Tape() as tape:
        launch_double(rigid_query, len(points), inputs=arrays, outputs=outputs, device="cuda:0")
    reference = np.concatenate([a.numpy().reshape(len(points), -1) for a in outputs], axis=1)
    seed = rng.normal(size=reference.shape)
    slices = [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), 12]
    tape.backward(
        grads={a: wp.array(seed[:, span], dtype=a.dtype, device=a.device) for a, span in zip(outputs, slices)}
    )
    total = np.zeros(24)
    for index in range(len(points)):
        result = query(points[index], base[1][index], *base[2:])
        np.testing.assert_allclose(result.values.numpy(), reference[index], rtol=2e-12, atol=2e-14)
        gradient = (result.jacobian.T @ seed[index]).numpy()
        np.testing.assert_allclose(gradient[:3], arrays[0].grad.numpy()[index], rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(gradient[3:6], arrays[1].grad.numpy()[index], rtol=2e-10, atol=2e-10)
        total += gradient
    start = 6
    for i in range(2, 9):
        actual = arrays[i].grad.numpy().ravel()
        expected = total[start : start + actual.size]
        if i == 3:
            q = base[3]
            actual = actual - np.dot(actual, q) * q
            expected = expected - np.dot(expected, q) * q
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-9)
        start += actual.size


@pytest.mark.parametrize("bevel", [0.0, 0.004])
def test_all_rigid_geometry_inputs_against_two_natural_difference_sizes(bevel):
    values = fixture(bevel)
    result = query(*values)
    rng = np.random.default_rng(2753)
    seed = rng.normal(size=13)
    gradient = (result.jacobian.T @ seed).numpy()
    start = 0
    scales = [0.001, 0.1, 0.001, 0.1, 0.1, 0.1, 0.0002, 0.001, 0.0001]
    for index, (source, scale) in enumerate(zip(values, scales)):
        if index == 7 and bevel == 0.0:
            start += 1
            continue
        direction = rng.normal(size=source.shape) * scale
        if index == 3:
            direction -= np.dot(direction, source) * source
        ad = float(np.dot(gradient[start : start + source.size], direction))
        for epsilon in [1e-4, 1e-5]:
            targets = []
            for sign in [-1, 1]:
                changed = list(values)
                changed[index] = source + sign * epsilon * direction
                if index == 3:
                    changed[index] /= np.linalg.norm(changed[index])
                targets.append(query(*changed).values.numpy())
            fd = float(np.dot(seed, targets[1] - targets[0]) / (2 * epsilon))
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-8)
        start += source.size


def test_box_corner_distance_retains_sub_ulp_position_offset():
    half = np.array([2.0**-8, 2.0**-7, 2.0**-6])
    point = E([half[0], half[1], 0.0], [2.0**-70, 2.0**-70, 0.0])
    result = query(point, np.zeros(3), np.zeros(3), [0, 0, 0, 1], np.zeros(3), np.zeros(3), half, [0], [0])
    np.testing.assert_allclose(result.gap.numpy(), np.sqrt(2.0) * 2.0**-70, rtol=2e-15, atol=0)
    np.testing.assert_allclose(result.normal.numpy(), [1 / np.sqrt(2), 1 / np.sqrt(2), 0], rtol=2e-15)
    np.testing.assert_allclose(result.jacobian.numpy()[12, :3], result.normal.numpy(), rtol=2e-15)


def test_rotating_surface_velocity_uses_the_moving_closest_point():
    omega = np.array([0.7, -0.3, 0.2])
    result = query(
        [0.0002, 0.001, 0.011],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0],
        omega,
        [0.001, 0.005, 0.01],
        [0],
        [0.002],
    )
    skew = np.array([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])
    np.testing.assert_allclose(result.jacobian.numpy()[6:9, :3], -skew @ np.diag([1, 1, 0]), atol=1e-15)
    np.testing.assert_allclose(result.relative.numpy(), -np.cross(omega, [0.0002, 0.001, 0.01]), atol=1e-18)
