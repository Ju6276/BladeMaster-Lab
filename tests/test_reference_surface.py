"""Online material sweeps, polygon unions, and natural geometry derivatives."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.geometry import points
from blademaster_warp.reference_surface import ReferenceSurface, swept_edge as _swept_edge
from blademaster_warp.reference_coordinates import ReferenceCoordinates


DTYPES = [
    wp.vec3d,
    wp.mat33d,
    wp.vec3d,
    wp.mat33d,
    wp.vec3d,
    wp.vec3d,
    wp.quatd,
    wp.vec3d,
    wp.quatd,
    wp.vec3d,
]


def block(spacing=0.006, start=0.015, end=0.003, x=0.0007, seed=0):
    reference, _, _ = points((0.012,) * 3, spacing, "block", seed)
    F = np.repeat(np.eye(3)[None], len(reference), axis=0)
    values = [
        reference.copy(),
        F.copy(),
        reference.copy(),
        F.copy(),
        np.full_like(reference, spacing / 4),
        np.array([[x, 0.0004, start + 0.004]]),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[x, 0.0004, end + 0.004]]),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0005, 0.0037, 0.004]]),
    ]
    return reference, values


def arrays(values):
    return [wp.array(value, dtype=dtype, requires_grad=True) for value, dtype in zip(values, DTYPES)]


def swept_edge(*sources, spacing=0.006, seed=0, **kwargs):
    return _swept_edge(
        *sources,
        reference_coordinates=ReferenceCoordinates.sampled((0.012,) * 3, spacing, "block", seed),
        **kwargs
    )


@pytest.mark.parametrize("spacing", [0.006, 0.004])
@pytest.mark.parametrize("offset", [0.0, 0.0007])
def test_sweep_matches_analytic_material_area_and_moments(spacing, offset):
    reference, values = block(spacing, x=offset)
    result = swept_edge(*arrays(values), spacing=spacing).quadrature(wp.array(reference, dtype=wp.vec3d))
    area = 0.0074 * 0.009
    center = np.array([offset, 0.0004, 0.0075])
    np.testing.assert_allclose(result.area.numpy().sum(), area, rtol=2e-14)
    np.testing.assert_allclose(result.area.numpy() @ result.points.numpy() / area, center, atol=1e-17)
    np.testing.assert_allclose(
        result.area.numpy() @ (result.points.numpy() ** 2) / area,
        center**2 + np.array([0.0, 0.0074**2, 0.009**2]) / 12,
        rtol=2e-14,
        atol=1e-23,
    )


def test_common_rigid_motion_adds_no_material_surface():
    _, values = block(start=0.003, end=0.003)
    axis = np.array([0.2, 0.6, 0.7])
    axis /= np.linalg.norm(axis)
    angle = 0.83
    R = (
        np.cos(angle) * np.eye(3)
        + (1 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * np.cross(axis, np.eye(3)).T
    )
    shift = np.array([0.02, -0.03, 0.01])
    values[2] = values[2] @ R.T + shift
    values[3] = R @ values[3]
    values[7] = values[7] @ R.T + shift
    values[8][0] = np.r_[axis * np.sin(angle / 2), np.cos(angle / 2)]
    result = swept_edge(*arrays(values))
    assert result.polygons == ()


@pytest.mark.parametrize("spacing", [0.006, 0.004])
def test_withdrawal_reentry_and_crossing_preserve_existing_area(spacing):
    reference, _ = block(spacing)
    ref = wp.array(reference, dtype=wp.vec3d)
    surface = ReferenceSurface.empty(len(reference), device=ref.device)
    heights = [0.015, 0.009, 0.003, 0.009, 0.001]
    expected = [0.003, 0.009, 0.009, 0.011]
    old_vertices = None
    for start, end, depth in zip(heights, heights[1:], expected):
        _, values = block(spacing, start=start, end=end)
        candidate = swept_edge(*arrays(values), spacing=spacing)
        fresh = candidate.subtract(surface)
        if end > start:
            # Zero-area boundary cells retain the derivative of a moving front.
            np.testing.assert_array_equal(fresh.quadrature(ref).area.numpy(), 0.0)
        if old_vertices is not None:
            np.testing.assert_array_equal(surface.vertices.numpy(), old_vertices)
        surface = surface.join(fresh)
        old_vertices = surface.vertices.numpy().copy()
        np.testing.assert_allclose(surface.quadrature(ref).area.numpy().sum(), 0.0074 * depth, rtol=3e-14)
    _, values = block(spacing)
    values[6][0] = values[8][0] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    second = swept_edge(*arrays(values), spacing=spacing)
    np.testing.assert_allclose(
        second.subtract(surface).quadrature(ref).area.numpy().sum(),
        second.quadrature(ref).area.numpy().sum(),
        rtol=2e-14,
    )


@pytest.mark.parametrize("width", [2.0**-30, 2.0**-60])
def test_sub_roundoff_positive_sliver_and_area_derivative_survive_clipping(width):
    vertices = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3d, requires_grad=True)
    surface = ReferenceSurface(vertices, ((0, 1, 2),), np.array([0]), np.array([1e-14]))
    point = wp.array([[width, 0, 0]], dtype=wp.vec3d, requires_grad=True)
    normal = wp.array([[1, 0, 0]], dtype=wp.vec3d, requires_grad=True)
    reference = wp.zeros(1, dtype=wp.vec3d)
    with wp.Tape() as tape:
        result = surface.clip(point, normal).quadrature(reference)
    assert np.all(result.area.numpy() >= 0)
    np.testing.assert_allclose(result.area.numpy().sum(), width - width**2 / 2, rtol=3e-16)
    tape.backward(grads={result.area: wp.ones(len(result.area), dtype=wp.float64)})
    np.testing.assert_allclose(point.grad.numpy()[0, 0], 1 - width, rtol=3e-16)
    for epsilon in [width / 8, width / 32]:
        losses = []
        for sign in [-1, 1]:
            point.assign(np.array([[width + sign * epsilon, 0, 0]]))
            losses.append(surface.clip(point, normal).quadrature(reference).area.numpy().sum())
        np.testing.assert_allclose((losses[1] - losses[0]) / (2 * epsilon), 1 - width, rtol=2e-14)


def test_collapsed_clip_retains_one_sided_front_derivative():
    vertices = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3d, requires_grad=True)
    surface = ReferenceSurface(vertices, ((0, 1, 2),), np.array([0]), np.array([0.0]))
    point = wp.zeros(1, dtype=wp.vec3d, requires_grad=True)
    normal = wp.array([[1, 0, 0]], dtype=wp.vec3d)
    reference = wp.zeros(1, dtype=wp.vec3d)
    with wp.Tape() as tape:
        result = surface.clip(point, normal).quadrature(reference)
    assert len(result.area) > 0
    np.testing.assert_array_equal(result.area.numpy(), 0)
    tape.backward(grads={result.area: wp.ones(len(result.area), dtype=wp.float64)})
    np.testing.assert_allclose(point.grad.numpy()[0, 0], 1.0, rtol=2e-15)
    for epsilon in [1e-6, 1e-7]:
        point.assign(np.array([[epsilon, 0, 0]]))
        area = surface.clip(point, normal).quadrature(reference).area.numpy().sum()
        np.testing.assert_allclose(area / epsilon, 1.0, rtol=epsilon)


@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_repeated_oblique_clipping_and_subtraction_have_exactly_zero_covered_area(scale):
    values = scale * np.array([[0.13, -0.2, 0.1], [1.2, 0.27, -0.43], [-0.29, 1.3, 0.6]])
    vertices = wp.array(values, dtype=wp.vec3d, requires_grad=True)
    surface = ReferenceSurface(vertices, ((0, 1, 2),), np.array([0]), np.array([0.0]))
    point = wp.array([values.mean(axis=0)], dtype=wp.vec3d, requires_grad=True)
    normal = wp.array([[0.3, -0.7, 0.2]], dtype=wp.vec3d, requires_grad=True)
    clipped = surface.clip(point, normal)
    for _ in range(4):
        clipped = clipped.clip(point, normal)
    reference = wp.zeros(1, dtype=wp.vec3d)
    area = clipped.subtract(surface).quadrature(reference).area.numpy()
    np.testing.assert_array_equal(area, 0.0)


@pytest.mark.parametrize("front", [0.006, 0.0063])
def test_subtracting_covered_area_differentiates_the_previous_front(front):
    reference, old_values = block(end=front)
    _, new_values = block(start=0.010, end=0.002)
    old, new = arrays(old_values), arrays(new_values)
    ref = wp.array(reference, dtype=wp.vec3d)
    with wp.Tape() as tape:
        previous = swept_edge(*old)
        fresh = swept_edge(*new).subtract(previous).quadrature(ref)
    tape.backward(grads={fresh.area: wp.ones(len(fresh.area), dtype=wp.float64)})
    expected = 0.0074
    np.testing.assert_allclose(old[7].grad.numpy()[0, 2], expected, rtol=2e-13)
    np.testing.assert_allclose(new[7].grad.numpy()[0, 2], -expected, rtol=2e-13)
    for source in [old[7], new[7]]:
        base = source.numpy().copy()
        for epsilon in [1e-7, 1e-8]:
            losses = []
            for sign in [-1, 1]:
                value = base.copy()
                value[0, 2] += sign * epsilon
                source.assign(value)
                trial = swept_edge(*new).subtract(swept_edge(*old)).quadrature(ref)
                losses.append(trial.area.numpy().sum())
            np.testing.assert_allclose(
                source.grad.numpy()[0, 2], (losses[1] - losses[0]) / (2 * epsilon), rtol=2e-7, atol=1e-10
            )
        source.assign(base)


@pytest.mark.parametrize("spacing", [0.006, 0.004])
def test_front_area_derivative_is_counted_once_on_particle_boundaries(spacing):
    for front in [0.003, 0.004, 0.006, 0.009]:
        reference, values = block(spacing, end=front)
        arguments = arrays(values)
        ref = wp.array(reference, dtype=wp.vec3d)
        with wp.Tape() as tape:
            result = swept_edge(*arguments, spacing=spacing).quadrature(ref)
        tape.backward(grads={result.area: wp.ones(len(result.area), dtype=wp.float64)})
        np.testing.assert_allclose(arguments[7].grad.numpy()[0, 2], -0.0074, rtol=2e-13)


@pytest.mark.parametrize("refine_edge", [False, True])
def test_rotating_edge_sweep_converges_to_helicoid_area(refine_edge):
    half, depth, angle = 0.002, 0.006, 0.6
    analytic = half * np.sqrt(depth**2 + angle**2 * half**2) + depth**2 / angle * np.arcsinh(
        angle * half / depth
    )
    if not refine_edge:
        edge = np.linspace(-half, half, 5)
        rate = np.sqrt(depth**2 + angle**2 * edge**2)
        analytic = np.sum((rate[:-1] + rate[1:]) * np.diff(edge) / 2)
    errors = []
    for steps in [4, 8, 16]:
        area = 0.0
        for i in range(steps):
            start, end = i / steps, (i + 1) / steps
            reference, values = block(start=0.009 - depth * start, end=0.009 - depth * end, x=0.0007)
            values[9][0, 1] = half
            values[6][0] = [0, 0, np.sin(angle * start / 2), np.cos(angle * start / 2)]
            values[8][0] = [0, 0, np.sin(angle * end / 2), np.cos(angle * end / 2)]
            result = swept_edge(*arrays(values), edge_segments=steps if refine_edge else 4).quadrature(
                wp.array(reference, dtype=wp.vec3d)
            )
            area += result.area.numpy().sum()
        errors.append(abs(area / analytic - 1))
    assert all(3.8 < errors[i] / errors[i + 1] < 4.2 for i in [0, 1]), errors


@pytest.mark.parametrize("seed", [0, 17])
def test_swept_area_and_normal_derivatives_include_both_particle_states_and_poses(seed):
    reference, values = block(start=0.013, end=0.002, seed=seed)
    values[1][:] = [[1.1, 0.1, 0.0], [0.0, 0.9, 0.12], [0.03, 0.0, 1.2]]
    values[3][:] = [[1.12, 0.09, 0.0], [0.0, 0.91, 0.11], [0.03, 0.0, 1.19]]
    values[6][0] = [0.03, -0.02, 0.04, 1.1]
    values[8][0] = [0.07, -0.05, 0.08, 0.9]
    arguments = arrays(values)
    ref = wp.array(reference, dtype=wp.vec3d, requires_grad=True)
    scale = 0.012
    linear = np.array([1.2, -0.5, 0.8]) / scale
    normal_seed = np.array([0.2, -0.3, 0.4])

    def measure():
        return swept_edge(*arguments, seed=seed).quadrature(ref)

    def density(result):
        return 1 + result.points.numpy() @ linear + result.normals.numpy() @ normal_seed

    with wp.Tape() as tape:
        result = measure()
    area = result.area.numpy()
    tape.backward(
        grads={
            result.area: wp.array(density(result) / scale**2, dtype=wp.float64),
            result.points: wp.array(area[:, None] * linear / scale**2, dtype=wp.vec3d),
            result.normals: wp.array(area[:, None] * normal_seed / scale**2, dtype=wp.vec3d),
        }
    )
    rng = np.random.default_rng(903)
    for source in [*arguments, ref]:
        base = source.numpy().copy()
        direction = rng.normal(size=base.shape) * np.maximum(np.abs(base), 0.001)
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-5, 1e-6]:
            losses = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                trial = measure()
                losses.append(trial.area.numpy() @ density(trial) / scale**2)
            np.testing.assert_allclose(ad, (losses[1] - losses[0]) / (2 * epsilon), rtol=2e-6, atol=1e-8)
        source.assign(base)


@pytest.mark.parametrize("shape", ["block", "cylinder", "banana"])
@pytest.mark.parametrize("seed", [0, 17])
@pytest.mark.parametrize("tilted", [False, True])
def test_exact_retreat_preserves_area_for_shapes_sampling_and_oblique_sweeps(shape, seed, tilted):
    size = (0.12, 0.06, 0.03) if shape == "banana" else (0.012,) * 3
    spacing = 0.006
    coordinates = ReferenceCoordinates.sampled(size, spacing, shape, seed)
    reference = coordinates.head
    F = np.array([[1.03, 0.02, 0.0], [0.0, 0.99, 0.01], [0.0, 0.0, 1.01]])
    deformation = np.repeat(F[None], len(reference), axis=0)
    rotation = np.array([[0.03, -0.02, 0.07, 1.1] if tilted else [0.0, 0.0, 0.0, 1.0]])
    reference_array = wp.array(reference, dtype=wp.vec3d)
    surface = ReferenceSurface.empty(len(reference), device=reference_array.device)
    old_area = None
    for start, end in [(0.03, 0.017), (0.017, 0.003), (0.003, 0.025)]:
        y = -0.0175 if shape == "banana" else 0.0004
        values = [
            reference.copy(),
            deformation,
            reference.copy(),
            deformation,
            np.full_like(reference, spacing / 4),
            np.array([[0.0007, y, start + 0.004]]),
            rotation,
            np.array([[0.0007, y, end + 0.004]]),
            rotation,
            np.array([[0.0005, 0.01, 0.004]]),
        ]
        candidate = _swept_edge(*arrays(values), reference_coordinates=coordinates)
        fresh = candidate.subtract(surface)
        area = fresh.quadrature(reference_array).area.numpy()
        assert np.all(area >= 0)
        if end > start:
            np.testing.assert_array_equal(area, 0.0)
            np.testing.assert_array_equal(surface.quadrature(reference_array).area.numpy(), old_area)
        surface = surface.join(fresh)
        old_area = surface.quadrature(reference_array).area.numpy().copy()
    assert old_area.sum() > 0
