"""Finite cut-footprint measures, boundary ownership and clipping derivatives."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.reference_cut import quadrature


def sources():
    return [
        np.zeros((1, 3)),
        np.eye(3)[None],
        np.zeros((1, 3)),
        np.full((1, 3), 0.003),
        np.zeros((1, 3)),
        np.array([[1.0, 0.0, 0.0]]),
    ]


def arrays(values):
    dtypes = [wp.vec3d, wp.mat33d] + [wp.vec3d] * 6
    return [wp.array(a, dtype=d, requires_grad=True) for a, d in zip(values, dtypes)]


def rectangle(lower_y, upper_y, lower_z, upper_z):
    return [
        np.array([[0.0, lower_y, 0.0], [0.0, upper_y, 0.0], [0.0, 0.0, lower_z], [0.0, 0.0, upper_z]]),
        np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]),
    ]


def evaluate(values):
    return quadrature(*values[:6], clip_points=values[6], clip_normals=values[7])


def test_rectangle_area_centroid_and_quadratic_moments():
    lower = np.array([-0.001, -0.002])
    upper = np.array([0.002, 0.0015])
    result = evaluate(arrays(sources() + rectangle(lower[0], upper[0], lower[1], upper[1])))
    points, weights = result.points.numpy(), result.area.numpy()
    area = np.prod(upper - lower)
    center = (upper + lower) / 2
    np.testing.assert_allclose(weights.sum(), area, rtol=1e-14)
    np.testing.assert_allclose(weights @ points, area * np.r_[0.0, center], rtol=1e-14, atol=1e-23)
    np.testing.assert_allclose(
        weights @ (points[:, 1:] ** 2), area * (center**2 + (upper - lower) ** 2 / 12), rtol=1e-14
    )


def test_triangular_footprint_with_three_active_edges():
    length = 0.002
    clip_points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, length, 0.0]])
    clip_normals = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
    result = evaluate(arrays(sources() + [clip_points, clip_normals]))
    area = length**2 / 2
    np.testing.assert_allclose(result.area.numpy().sum(), area, rtol=1e-14)
    np.testing.assert_allclose(
        result.area.numpy() @ result.points.numpy(),
        area * np.array([0.0, length / 3, length / 3]),
        rtol=1e-14,
        atol=1e-23,
    )


@pytest.mark.parametrize("lower", [0.003, 0.004])
def test_zero_area_and_disjoint_footprints_return_empty(lower):
    result = evaluate(arrays(sources() + rectangle(lower, 0.006, -0.006, 0.006)))
    assert len(result.area) == len(result.points) == len(result.owners) == 0


@pytest.mark.parametrize("sign", [-1, 1])
def test_clipping_boundary_coincident_with_section_preserves_area(sign):
    result = evaluate(arrays(sources() + [np.zeros((1, 3)), np.array([[sign, 0.0, 0.0]])]))
    np.testing.assert_allclose(result.area.numpy().sum(), 0.006**2, rtol=1e-14)


def test_clipped_reference_geometry_is_affine_invariant():
    values = sources() + rectangle(-0.001, 0.002, -0.002, 0.0015)
    original = evaluate(arrays(values))
    F = np.array([[1.2, 0.3, -0.1], [0.0, 0.85, 0.2], [0.1, 0.0, 1.1]])
    shift = np.array([0.03, -0.02, 0.01])
    values[0] = values[0] @ F.T + shift
    values[1][:] = F
    for k in [4, 6]:
        values[k] = values[k] @ F.T + shift
        values[k + 1] = np.linalg.solve(F.T, values[k + 1].T).T
    transformed = evaluate(arrays(values))
    np.testing.assert_allclose(transformed.points.numpy(), original.points.numpy(), atol=2e-17)
    np.testing.assert_allclose(transformed.area.numpy(), original.area.numpy(), rtol=5e-14)
    np.testing.assert_allclose(transformed.normals.numpy(), original.normals.numpy(), atol=5e-16)


def test_geometry_and_clip_plane_derivatives():
    values = sources()
    values[1][0] = [[1.1, 0.1, 0.0], [0.0, 0.9, 0.12], [0.03, 0.0, 1.2]]
    values[4][0] = [0.0003, 0.0001, 0.0002]
    values[5][0] = [1.0, 0.37, -0.23]
    values += [
        np.array([[0.0, 0.0015, 0.0], [0.0, -0.0015, 0.0], [0.0, 0.0, -0.0012]]),
        np.array([[0.0, 1.0, 0.1], [0.0, -1.0, 0.1], [0.0, 0.0, -1.0]]),
    ]
    arguments = arrays(values)
    linear = np.array([1.2, -0.5, 0.8]) / 0.003
    quadratic = np.array([[0.2, 0.1, 0.0], [0.1, -0.3, 0.1], [0.0, 0.1, 0.4]]) / 0.003**2
    normal_seed = np.array([0.2, -0.3, 0.4])
    scale = 0.003**2

    def density(result):
        x = result.points.numpy()
        return (
            1 + x @ linear + np.einsum("qa,ab,qb->q", x, quadratic, x) + result.normals.numpy() @ normal_seed
        )

    with wp.Tape() as tape:
        result = evaluate(arguments)
    x, area = result.points.numpy(), result.area.numpy()
    tape.backward(
        grads={
            result.area: wp.array(density(result) / scale, dtype=wp.float64),
            result.points: wp.array(area[:, None] * (linear + 2 * x @ quadratic) / scale, dtype=wp.vec3d),
            result.normals: wp.array(area[:, None] * normal_seed / scale, dtype=wp.vec3d),
        }
    )
    rng = np.random.default_rng(749)
    for source in arguments:
        base = source.numpy().copy()
        direction = rng.normal(size=base.shape) * np.maximum(np.abs(base), 0.001)
        ad = float(np.sum(source.grad.numpy() * direction))
        for epsilon in [1e-4, 1e-5]:
            losses = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                trial = evaluate(arguments)
                losses.append(trial.area.numpy() @ density(trial) / scale)
            np.testing.assert_allclose(ad, (losses[1] - losses[0]) / (2 * epsilon), rtol=2e-6, atol=1e-8)
        source.assign(base)
