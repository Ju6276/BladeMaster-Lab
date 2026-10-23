"""Analytic cut measures, affine geometry and cut-quadrature input derivatives."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.reference_cut import quadrature
from blademaster_warp.geometry import points


DTYPES = [wp.vec3d, wp.mat33d, wp.vec3d, wp.vec3d, wp.vec3d, wp.vec3d]


def arrays(values):
    return [wp.array(a, dtype=dtype, requires_grad=True) for a, dtype in zip(values, DTYPES)]


def cube(normal=(1.0, 0.0, 0.0), point=(0.0, 0.0, 0.0), half=0.003):
    return [
        np.zeros((1, 3)),
        np.eye(3)[None],
        np.zeros((1, 3)),
        np.full((1, 3), half),
        np.array([point]),
        np.array([normal]),
    ]


def test_axis_section_area_and_degree_two_moments():
    a, offset = 0.003, 0.0007
    result = quadrature(*arrays(cube(point=(offset, 0.0, 0.0), half=a)))
    x, area = result.points.numpy(), result.area.numpy()
    np.testing.assert_allclose(area.sum(), 4 * a**2, rtol=1e-14)
    np.testing.assert_allclose(x[:, 0], offset, atol=1e-18)
    np.testing.assert_allclose(area @ x, [4 * a**2 * offset, 0.0, 0.0], atol=1e-23)
    np.testing.assert_allclose(area @ (x**2), [4 * a**2 * offset**2, 4 * a**4 / 3, 4 * a**4 / 3], rtol=1e-14)


@pytest.mark.parametrize("offset, factor", [(0.0, 3.0), (0.003, 2.0)])
def test_hexagonal_and_triangular_sections(offset, factor):
    result = quadrature(*arrays(cube(normal=(1.0, 1.0, 1.0), point=(offset, 0.0, 0.0))))
    np.testing.assert_allclose(result.area.numpy().sum(), factor * np.sqrt(3) * 0.003**2, rtol=1e-14)
    np.testing.assert_allclose(result.points.numpy().sum(axis=1), offset, atol=2e-18)
    np.testing.assert_allclose(
        result.normals.numpy(), np.full((len(result.area), 3), 1 / np.sqrt(3)), atol=2e-16
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_shared_face_is_counted_once(sign):
    values = cube(normal=(sign, 0.0, 0.0))
    values[:4] = [np.repeat(a, 2, axis=0) for a in values[:4]]
    values[0][:, 0] = [-0.003, 0.003]
    values[2] = values[0].copy()
    result = quadrature(*arrays(values))
    np.testing.assert_allclose(result.area.numpy().sum(), 4 * 0.003**2, rtol=1e-14)
    np.testing.assert_array_equal(result.owners, 1 if sign == 1 else 0)
    np.testing.assert_allclose(result.points.numpy()[:, 0], 0.0, atol=1e-18)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [-1, 1])
def test_sampled_lattice_sections_preserve_area_at_shared_faces(axis, sign):
    size, spacing = 0.012, 0.006
    reference, _, _ = points((size,) * 3, spacing, "block")
    normal = np.zeros((1, 3))
    normal[0, axis] = sign
    center = np.array([[0.0, 0.0, size / 2]])
    values = [
        reference.copy(),
        np.repeat(np.eye(3)[None], len(reference), axis=0),
        reference,
        np.full_like(reference, spacing / 4),
        center.copy(),
        normal,
    ]
    for offset in [0.0, 0.0003, 0.001, 0.002, 0.003]:
        values[4][:] = center
        values[4][0, axis] += offset
        result = quadrature(*arrays(values))
        area = result.area.numpy()
        np.testing.assert_allclose(area.sum(), size**2, rtol=1e-14)
        np.testing.assert_allclose(result.points.numpy()[:, axis], values[4][0, axis], atol=3e-18)
        centroid = (area @ result.points.numpy()) / area.sum()
        np.testing.assert_allclose(centroid, values[4][0], atol=3e-18)


def test_plane_outside_domain_returns_empty_quadrature():
    result = quadrature(*arrays(cube(point=(0.004, 0.0, 0.0))))
    assert len(result.points) == len(result.area) == len(result.normals) == len(result.owners) == 0


def test_affine_pullback_and_superposed_rigid_motion():
    values = cube(normal=(1.0, 0.37, -0.23), point=(0.0003, 0.0001, 0.0002))
    original = quadrature(*arrays(values))
    F = np.array([[1.2, 0.3, -0.1], [0.0, 0.85, 0.2], [0.1, 0.0, 1.1]])
    shift = np.array([0.03, -0.02, 0.01])
    values[0] = values[0] @ F.T + shift
    values[1][:] = F
    values[4] = values[4] @ F.T + shift
    values[5] = np.linalg.solve(F.T, values[5][0])[None]
    transformed = quadrature(*arrays(values))
    rotation, _ = np.linalg.qr(np.random.default_rng(78).normal(size=(3, 3)))
    values[0] = values[0] @ rotation.T + [0.1, 0.2, 0.3]
    values[1] = rotation @ values[1]
    values[4] = values[4] @ rotation.T + [0.1, 0.2, 0.3]
    values[5] = values[5] @ rotation.T
    rotated = quadrature(*arrays(values))
    for result in [transformed, rotated]:
        np.testing.assert_allclose(result.points.numpy(), original.points.numpy(), atol=2e-16)
        np.testing.assert_allclose(result.area.numpy(), original.area.numpy(), rtol=2e-13)
        np.testing.assert_allclose(result.normals.numpy(), original.normals.numpy(), atol=5e-16)


def test_cut_measure_derivatives_include_plane_and_particle_inputs():
    values = cube(normal=(1.0, 0.37, -0.23), point=(0.0003, 0.0001, 0.0002))
    values[1][0] = [[1.1, 0.1, 0.0], [0.0, 0.9, 0.12], [0.03, 0.0, 1.2]]
    values[3][0] = [0.002, 0.003, 0.004]
    sources = arrays(values)
    linear = np.array([1.2, -0.5, 0.8]) / 0.003
    quadratic = np.array([[0.2, 0.1, 0.0], [0.1, -0.3, 0.1], [0.0, 0.1, 0.4]]) / 0.003**2
    normal_seed = np.array([0.2, -0.3, 0.4])
    scale = 0.003**2

    def potential(result):
        x, n = result.points.numpy(), result.normals.numpy()
        return 1 + x @ linear + np.einsum("qa,ab,qb->q", x, quadratic, x) + n @ normal_seed

    with wp.Tape() as tape:
        result = quadrature(*sources)
    x, area = result.points.numpy(), result.area.numpy()
    tape.backward(
        grads={
            result.area: wp.array(potential(result) / scale, dtype=wp.float64),
            result.points: wp.array(area[:, None] * (linear + 2 * x @ quadratic) / scale, dtype=wp.vec3d),
            result.normals: wp.array(area[:, None] * normal_seed / scale, dtype=wp.vec3d),
        }
    )
    np.testing.assert_allclose(
        sources[0].grad.numpy().sum(axis=0) + sources[4].grad.numpy()[0], 0.0, atol=1e-10
    )
    np.testing.assert_allclose(np.sum(sources[5].grad.numpy() * values[5]), 0.0, atol=1e-12)
    rng = np.random.default_rng(670)
    for source in sources:
        base = source.numpy().copy()
        direction = rng.normal(size=base.shape) * np.maximum(np.abs(base), 0.001)
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            losses = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                trial = quadrature(*sources)
                losses.append(trial.area.numpy() @ potential(trial) / scale)
            np.testing.assert_allclose(ad, (losses[1] - losses[0]) / (2 * epsilon), rtol=2e-6, atol=1e-8)
        source.assign(base)
