"""Finite crack visibility, crossing-field selection and natural trace derivatives."""

from fractions import Fraction

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_history import CohesiveHistory
from blademaster_warp.reference_coordinates import rational
from blademaster_warp.reference_grid import build_trace
from blademaster_warp.reference_surface import swept_edge
from blademaster_warp.reference_visibility import ReferenceVisibility
from test_reference_grid import setup, nodal_coordinates, polynomial


def plane(y, bottom=0.0):
    a, b, c, d = [[-0.006, y, bottom], [0.006, y, bottom], [0.006, y, 0.012], [-0.006, y, 0.012]]
    return np.array([[a, b, c], [a, c, d]])


@pytest.mark.parametrize("reverse", [False, True])
def test_finite_faces_include_edges_and_exclude_endpoint_and_coplanar_touches(reverse):
    triangles = plane(0.0, bottom=0.006)
    if reverse:
        triangles = triangles[::-1, ::-1].astype(object)
    faces = ReferenceVisibility(triangles, np.ones(2, dtype=int))
    points = np.array([[0, 0.001, 0.003], [0, 0.001, 0.009], [0, 0, 0.009]])
    particles = np.array([[0, -0.001, 0.003], [0, -0.001, 0.009], [0, 0, 0.009]])
    mask = faces.particle_mask(points, particles, np.zeros(3, dtype=int))
    assert mask[0, 0] and not mask[1, 1]
    assert mask[:, 2].all() and mask[2].all()
    np.testing.assert_array_equal(mask, faces.particle_mask(particles, points, np.zeros(3, dtype=int)).T)
    np.testing.assert_array_equal(faces.particle_mask(points, particles, np.ones(3, dtype=int)), True)


def test_sub_float_reference_triangle_width_is_preserved():
    triangles = rational(np.array([[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]]))
    shift = Fraction(1, 2**60)
    triangles[:, :, 0] += shift
    faces = ReferenceVisibility(triangles, np.array([1]))
    points = np.array([[1.0, 0.2, 0.2]])
    particles = np.array([[np.nextafter(1.0, 2.0), 0.2, 0.2]])
    np.testing.assert_array_equal(faces.particle_mask(points, particles, np.array([0])), False)


def determinant(a, b, c):
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        + a[1] * (b[2] * c[0] - b[0] * c[2])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def exact_barycentric_hit(start, end, triangle):
    a, b, c = triangle
    d, s, e1, e2 = end - start, start - a, b - a, c - a
    denominator = determinant(e1, d, e2)
    if denominator == 0:
        return False
    u = determinant(s, d, e2) / denominator
    v = determinant(d, s, e1) / denominator
    t = determinant(e2, s, e1) / denominator
    return 0 < t < 1 and u >= 0 and v >= 0 and u + v <= 1


@pytest.mark.parametrize("scale", [1e-150, 1e-3, 1.0, 1e150])
def test_filtered_predicates_match_independent_rational_barycentric_intersections(scale):
    rng = np.random.default_rng(619)
    triangles = rng.normal(size=(7, 3, 3)) * scale
    points = rng.normal(size=(8, 3)) * scale
    particles = rng.normal(size=(9, 3)) * scale
    faces = ReferenceVisibility(triangles, np.ones(len(triangles), dtype=int))
    expected = np.ones((len(points), len(particles)), dtype=bool)
    for q, start in enumerate(rational(points)):
        for p, end in enumerate(rational(particles)):
            expected[q, p] = not any(
                exact_barycentric_hit(start, end, triangle) for triangle in rational(triangles)
            )
    np.testing.assert_array_equal(
        faces.particle_mask(points, particles, np.zeros(len(points), dtype=int)), expected
    )
    assert np.any(expected) and np.any(~expected)


@pytest.mark.parametrize("offset", [1e12, 1e150])
def test_interval_filter_retains_sub_ulp_faces_at_large_coordinates(offset):
    triangles = rational(np.array([[[offset, 0.0, 0.0], [offset, 1.0, 0.0], [offset, 0.0, 1.0]]]))
    triangles[:, :, 0] += Fraction(1, 2**60)
    faces = ReferenceVisibility(triangles, np.array([1]))
    points = np.array([[offset, 0.2, 0.2]])
    particles = np.array([[np.nextafter(offset, np.inf), 0.2, 0.2]])
    np.testing.assert_array_equal(faces.particle_mask(points, particles, np.array([0])), False)


def test_zero_area_and_zero_length_segments_are_not_crossings():
    triangles = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]])
    faces = ReferenceVisibility(triangles, np.array([1]))
    points = np.array([[0.0, 0.2, 0.0], [-1.0, 0.2, 0.0]])
    particles = np.array([[0.0, 0.2, 0.0], [1.0, 0.2, 0.0]])
    np.testing.assert_array_equal(faces.particle_mask(points, particles, np.array([0, 0])), True)


def crossing():
    m, _, _, _ = setup(0.0007, [1.0, 0.0, 0.0])
    x, y, _ = m.reference.T
    codes = np.where(x < 0.0007, 1, 2) + 3 * np.where(y < 0.0007, 1, 2)
    grid = m.topology(codes)
    points = m.array([[0.0007, 0.0004, 0.0053], [0.0007, 0.0012, 0.0067]], wp.vec3)
    normals = m.array([[1.0, 0.0, 0.0]] * len(points), wp.vec3)
    faces = ReferenceVisibility(plane(0.0007), np.ones(2, dtype=int))

    def evaluate():
        visible = faces.particle_mask(points.numpy(), m.reference, np.zeros(len(points), dtype=int))
        return build_trace(m, grid, points, normals, np.tile([1, 2], (len(points), 1)), visibility=visible)

    return m, grid, codes, points, normals, evaluate


def test_crossing_trace_matches_analytic_regions_with_points_inside_one_owner_cell():
    m, grid, codes, points, normals, evaluate = crossing()
    mapped = evaluate()
    expected = build_trace(m, grid, points, normals, np.array([[4, 5], [7, 8]]))
    np.testing.assert_array_equal(mapped.weights.numpy(), expected.weights.numpy())
    np.testing.assert_array_equal(mapped.gradients.numpy(), expected.gradients.numpy())
    owners = np.argmin(np.sum((points.numpy()[:, None] - m.reference[None]) ** 2, axis=2), axis=1)
    owner_sides = ((codes[owners] // 3) * 3)[:, None] + [1, 2]
    wrong = build_trace(m, grid, points, normals, owner_sides)
    assert np.max(np.abs(wrong.weights.numpy() - expected.weights.numpy())) > 0.5


def test_visible_trace_position_and_normal_derivatives_rebuild_visibility():
    m, grid, _, points, normals, evaluate = crossing()
    rng = np.random.default_rng(117)
    with wp.Tape() as tape:
        result = evaluate()
    seeds = [rng.normal(size=a.numpy().shape) for a in [result.separation_map, result.frame_map]]
    seeds[1] *= m.config.grid_spacing
    tape.backward(
        grads={
            a: m.array(seed, kind, False)
            for a, seed, kind in zip([result.separation_map, result.frame_map], seeds, [float, wp.vec3])
        }
    )
    for source, direction in [
        (points, [[0.0, 0.0001, -0.0002], [0.0, -0.0003, 0.0002]]),
        (normals, [[0.1, -0.2, 0.3], [-0.2, 0.3, 0.1]]),
    ]:
        direction = np.asarray(direction)
        base = source.numpy().copy()
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            losses = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                trial = evaluate()
                losses.append(
                    sum(
                        np.sum(a.numpy() * seed)
                        for a, seed in zip([trial.separation_map, trial.frame_map], seeds)
                    )
                )
            np.testing.assert_allclose((losses[1] - losses[0]) / (2 * epsilon), ad, rtol=2e-6, atol=1e-9)
        source.assign(base)


def natural_history(end, width=0.01):
    m, _, _, _ = setup(0.0007, [1.0, 0.0, 0.0])
    reference = m.array(m.reference, wp.vec3, False)
    F = m.array(np.tile(np.eye(3), (m.n, 1, 1)), wp.mat33, False)
    half = m.array(np.full((m.n, 3), m.config.grid_spacing / 4), wp.vec3, False)
    blade = m.array([[0.0005, width, 0.004]], wp.vec3, False)
    material = m.array(np.tile([5e4, 70.0, 0.0], (m.n, 1)), wp.vec3, False)
    history = CohesiveHistory.empty(m.n, device=m.device)
    for cut, (rotation, bottom) in enumerate(
        [
            ([0.0, 0.0, 0.0, 1.0], 0.0 if width > 0.001 else 0.0065),
            ([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], end),
        ]
    ):
        quaternion = m.array([rotation], wp.quat, False)
        candidate = swept_edge(
            reference,
            F,
            reference,
            F,
            half,
            m.array([[0.0007, 0.0007, 0.018 if width > 0.001 else 0.013]], wp.vec3, False),
            quaternion,
            m.array([[0.0007, 0.0007, bottom + 0.004]], wp.vec3, False),
            quaternion,
            blade,
            reference_coordinates=m.reference_coordinates,
        )
        history = history.append(candidate, reference, F, material, cut_index=cut)
    cuts = history.cut_indices
    n = history.quadrature.normals.numpy()
    positive = np.where(np.where(cuts == 0, n[:, 0], n[:, 1]) > 0, 2, 1)
    x, y, z = m.reference.T
    codes = np.where(x < 0.0007, 1, 2) + 3 * np.where(z < end, 0, np.where(y < 0.0007, 1, 2))
    return m, history, positive, m.topology(codes)


@pytest.mark.parametrize("end", [0.0, 0.006])
def test_actual_sweeps_reproduce_polynomials_through_crossing_and_partial_fronts(end):
    m, history, positive, grid = natural_history(end)
    mapped = history.reference_trace(m, grid, positive)
    nodal, _ = polynomial(nodal_coordinates(m, grid), m.config.grid_spacing)
    expected, gradient = polynomial(history.quadrature.points.numpy(), m.config.grid_spacing)
    for side in [0, 1]:
        np.testing.assert_allclose(mapped.weights.numpy()[:, side] @ nodal, expected, rtol=2e-14, atol=2e-15)
        np.testing.assert_allclose(
            np.einsum("qna,n->qa", mapped.gradients.numpy()[:, side], nodal), gradient, rtol=2e-14, atol=2e-13
        )
    if end == 0:
        cuts, p = history.cut_indices, history.quadrature.points.numpy()
        other = np.where(cuts == 0, 3 * np.where(p[:, 1] < 0.0007, 1, 2), np.where(p[:, 0] < 0.0007, 1, 2))
        sides = (3**cuts)[:, None] * np.stack([3 - positive, positive], axis=1) + other[:, None]
        explicit = build_trace(m, grid, history.quadrature.points, history.quadrature.normals, sides)
        np.testing.assert_array_equal(mapped.weights.numpy(), explicit.weights.numpy())


def test_released_crossing_patches_preserve_free_material_motion_and_restart():
    m, history, positive, grid = natural_history(0.0065, width=0.0003)
    history = history.updated(history.geometry, m.array(np.full(len(history.history), 0.003), float))
    codes = grid.keys.numpy()[grid.slots.numpy()[:, 0]] // m.node_count
    velocity = np.zeros((m.n, 3))
    velocity[:, 0] = np.where(codes % 3 == 1, -0.1, 0.1)
    velocity[:, 1] = np.where(codes // 3 == 1, -0.1, 0.1)
    state = m.state(v=velocity)
    result, updated = history.advance(m, state, topology=grid, positive_sides=positive, gravity=(0, 0, 0))
    free = m.step(state, topology=grid, gravity=(0, 0, 0))
    for name in ["x", "v", "F", "Fp", "stress"]:
        np.testing.assert_allclose(
            getattr(result, name).numpy(), getattr(free, name).numpy(), rtol=2e-13, atol=1e-13
        )
    np.testing.assert_array_equal(updated.history.numpy(), history.history.numpy())
    restored = CohesiveHistory.restore(history.snapshot(), device=m.device)
    direct, replay = history.reference_trace(m, grid, positive), restored.reference_trace(m, grid, positive)
    np.testing.assert_array_equal(direct.weights.numpy(), replay.weights.numpy())
