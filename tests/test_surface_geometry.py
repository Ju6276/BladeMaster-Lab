"""Geometric regressions independent of edge incidence and vertex fitting error."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

from blademaster_warp.geometry import BANANA_DEMO_SIZE_M, points
from blademaster_warp.surface import LinkedSurface
from scripts.surface_accuracy import CUTS, accuracy, banana_volume
from scripts.surface_deformed import geometry_report, intersection_pairs, run, segment_triangle_pairs
from scripts.surface_observations import Observation, compare_observations


def assert_closed_oriented(geometry, components):
    assert geometry["finite_vertices"]
    assert geometry["components"] == components
    for key in (
        "boundary_edges",
        "nonmanifold_edges",
        "nonmanifold_vertices",
        "inconsistent_edge_orientations",
        "degenerate_triangles",
        "unused_vertices",
    ):
        assert geometry[key] == 0, (key, geometry[key])
    assert min(geometry["component_volumes_m3"]) > 0


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3])
def test_segment_triangle_contacts_are_scale_invariant(scale):
    triangle = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    a = np.array(
        [
            [0.2, 0.2, -1],
            [0.8, 0.8, -1],
            [0.2, 0.2, 0],
            [-1, 0.2, 0],
            [0.8, 0.8, 0],
            [1, 0, -1],
            [0.2, 0.2, 0.01],
            [-1, -1, 0],
        ]
    )
    b = np.array(
        [
            [0.2, 0.2, 1],
            [0.8, 0.8, 1],
            [0.3, 0.3, 0],
            [1, 0.2, 0],
            [0.9, 0.9, 0],
            [1, 0, 0],
            [0.3, 0.3, 0.01],
            [-0.2, -0.2, 0],
        ]
    )
    # Includes an interior coplanar segment, coplanar crossing, and endpoint touch.
    expected = [True, False, True, True, False, True, False, False]
    translation = np.array([13, -2, 4]) * scale
    triangles = np.tile(triangle, (len(a), 1, 1))
    for winding in (triangles, triangles[:, ::-1]):
        np.testing.assert_array_equal(
            segment_triangle_pairs(
                a * scale + translation, b * scale + translation, winding * scale + translation
            ),
            expected,
        )
        np.testing.assert_array_equal(
            segment_triangle_pairs(
                a * scale + translation, b * scale + translation, winding * scale + translation, strict=True
            ),
            [True, False, False, False, False, False, False, False],
        )


def test_intersection_witness_agrees_with_independent_linear_solve():
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, -1], [0.2, 0.2, 1], [2, 2, 1]])
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    edges, faces = intersection_pairs(vertices, triangles, chunk=1)
    found = np.flatnonzero((edges == [3, 4]).all(1) & (faces == 0))
    assert len(found) == 1
    for edge, face in zip(edges, faces):
        a, b = vertices[edge]
        p, q, r = vertices[triangles[face]]
        matrix = np.column_stack([b - a, p - q, p - r])
        t, u, v = np.linalg.solve(matrix, p - a)
        assert 0 <= t <= 1 and u >= 0 and v >= 0 and u + v <= 1
    assert len(intersection_pairs(vertices[:3], triangles[:1])[1]) == 0
    empty = geometry_report(np.empty((0, 3)), np.empty((0, 3), int))
    assert empty["components"] == empty["self_intersections"] == 0


def test_vertex_link_catches_pinched_closed_shells_and_signed_volume_catches_inversion():
    v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    valid = geometry_report(v, t)
    assert_closed_oriented(valid, 1)
    assert valid["signed_volume_m3"] == pytest.approx(1 / 6)
    assert geometry_report(v, t[:, ::-1])["signed_volume_m3"] == pytest.approx(-1 / 6)
    # The shells meet only at vertex 0: each edge still has two opposing faces.
    pinched_v = np.r_[v, -v[1:]]
    other = np.where(t == 0, 0, t + 3)[:, ::-1]
    pinched = geometry_report(pinched_v, np.r_[t, other], intersections=False)
    assert pinched["nonmanifold_edges"] == pinched["inconsistent_edge_orientations"] == 0
    assert pinched["nonmanifold_vertices"] == 1

    from blademaster_warp.surface import split_vertex_fans

    faces = np.r_[t, other]
    repaired, source = split_vertex_fans(faces, len(pinched_v))
    assert len(source) == len(pinched_v) + 1
    np.testing.assert_array_equal(pinched_v[source][repaired], pinched_v[faces])
    fixed = geometry_report(pinched_v[source], repaired)
    assert_closed_oriented(fixed, 2)
    assert fixed["signed_volume_m3"] == pinched["signed_volume_m3"]
    assert fixed["inter_component_contacts"] > 0
    assert fixed["inter_component_interior_crossings"] == 0


def test_partial_crack_point_pinch_splits_fans_without_changing_geometry(monkeypatch):
    import blademaster_warp.surface as module

    surface = LinkedSurface(BANANA_DEMO_SIZE_M, 0.003, "banana")
    # Reduced from the third cut at frame 418 of review_fix_sequence_final_20261018.
    # Seven severed x links suffice to create two closed boundary fans in one region.
    for j, k in [(1, 4), (2, 3), (3, 2), (3, 3), (4, 2), (4, 3), (5, 2)]:
        node = np.ravel_multi_index((22, j, k), surface.dims)
        edge = int(np.flatnonzero((surface.edges[:, 0] == node) & (surface.axes == 0))[0])
        surface.broken[edge] = True
        surface.hermite[edge] = (surface.reference[surface.edges[edge]].mean(0), np.array([1.0, 0, 0]))
    with monkeypatch.context() as context:
        context.setattr(
            module, "split_vertex_fans", lambda faces, count, candidates: (faces, np.arange(count))
        )
        before = surface.mesh()
    after = surface.mesh(use_cache=False)
    original = geometry_report(before.reference, before.triangles, intersections=False)
    repaired = geometry_report(after.reference, after.triangles, intersections=False)
    assert original["nonmanifold_vertices"] == 1
    assert repaired["nonmanifold_vertices"] == 0
    assert surface.vertex_splits == 1
    assert len(after.triangles) == len(before.triangles)
    for key in (
        "boundary_edges",
        "nonmanifold_edges",
        "inconsistent_edge_orientations",
        "degenerate_triangles",
    ):
        assert original[key] == repaired[key] == 0
    for name in ("reference", "codes", "cells", "weights"):
        np.testing.assert_array_equal(
            getattr(after, name)[after.triangles], getattr(before, name)[before.triangles]
        )
    # Duplicating a vertex changes the centering sum used by the volume audit.
    np.testing.assert_allclose(
        repaired["signed_volume_m3"], original["signed_volume_m3"], rtol=8 * np.finfo(float).eps, atol=0
    )
    assert repaired["unused_vertices"] == original["unused_vertices"]
    assert surface.mesh() is after


def test_empty_vertex_fans_keep_an_indexable_mapping():
    from blademaster_warp.surface import split_vertex_fans

    faces, source = split_vertex_fans(np.empty((0, 3), dtype=int), 0)
    assert faces.shape == (0, 3)
    assert np.empty((0, 3))[source].shape == (0, 3)


def test_recorded_partial_crack_crossing_is_not_a_seam_contact():
    # Disjoint edge (1156,1157) and face (1189,1161,1158), frame 63 of the same
    # recording, in world and reference metres. Both intersections are interior.
    witnesses = np.array(
        [
            [
                [0.0398879680733818, -0.0045857194803388, 0.0182694588209824],
                [0.0398253441424537, -0.0020982893581123, 0.0195506278302425],
                [0.0401247330060240, -0.0029475242692001, 0.0189437745739089],
                [0.0399057140977536, 0.0008867408955158, 0.0195058063379799],
                [0.0397338057876380, -0.0032619441731029, 0.0190737986521769],
            ],
            [
                [0.0402014054168172, -0.007497, 0.019497],
                [0.0400749617894615, -0.0052579892826032, 0.0212016899438004],
                [0.040503, -0.0059930211694902, 0.0204762513165015],
                [0.0400329500389484, -0.0022746967673233, 0.0216915364778886],
                [0.0400970385157980, -0.0063300814694288, 0.0205389980940894],
            ],
        ]
    )
    for a, b, p, q, r in witnesses:
        matrix = np.column_stack([b - a, p - q, p - r])
        t, u, v = np.linalg.solve(matrix, p - a)
        assert min(t, 1 - t, u, v, 1 - u - v) > 0.01
        assert segment_triangle_pairs(a[None], b[None], np.array([[p, q, r]])).item()
        assert segment_triangle_pairs(a[None], b[None], np.array([[p, q, r]]), strict=True).item()


@pytest.mark.parametrize("spacing", [0.003, 0.0015])
def test_banana_reference_has_no_contacts_or_volume_loss(spacing):
    surface = LinkedSurface(BANANA_DEMO_SIZE_M, spacing, "banana")
    mesh = surface.mesh()
    report = geometry_report(mesh.reference, mesh.triangles)
    assert_closed_oriented(report, 1)
    assert report["self_intersections"] == report["inter_component_contacts"] == 0
    assert 1.0 < report["signed_volume_m3"] / banana_volume() < 1.01
    # Insetting the solve bounds must not move the trilinear coordinate origin.
    np.testing.assert_allclose(
        (surface.reference[mesh.cells] * mesh.weights[:, :, None]).sum(1), mesh.reference, rtol=0, atol=2e-16
    )
    np.testing.assert_allclose(mesh.weights.sum(1), 1.0, rtol=0, atol=1e-14)
    assert np.all(mesh.weights >= 0)


def test_embedded_bends_twist_and_separated_cut_pieces_have_no_contacts():
    results = run(0.003)
    for result in results:
        components = 3 if result["case"].startswith("3 pieces") else 1
        for key in ("geometry", "exact_geometry"):
            geometry = result[key]
            assert_closed_oriented(geometry, components)
            assert geometry["self_intersections"] == geometry["inter_component_contacts"] == 0
        assert abs(result["relative_volume_error"]) < 1e-4
        assert result["max_error_cells"] < 0.025
    assert results[-1]["max_error_cells"] < 1e-10


@pytest.mark.parametrize("case", list(CUTS))
def test_fine_cut_surfaces_preserve_analytic_volume_including_thin_slices(case):
    cuts = CUTS[case]
    result = accuracy(0.0015, cuts)
    assert_closed_oriented(result, len(cuts) + 1)
    assert result["self_intersections"] == 0
    # Opposite crack faces touch before the pieces are moved apart.
    assert result["inter_component_contacts"] > 0
    assert abs(result["relative_volume_error"]) < 0.003
    assert result["max_error_cells"] < 0.5
    assert result["max_face_error_cells"] < 0.3
    if case != "oblique":
        boundaries = [-0.06, *[p[0] for p, _ in cuts], 0.06]
        exact = [
            quad(
                lambda x: np.pi * (0.003 + 0.009 * max(0, 1 - (x / 0.06) ** 2) ** 0.55) ** 2,
                a,
                b,
                epsabs=1e-14,
            )[0]
            for a, b in zip(boundaries, boundaries[1:])
        ]
        # Check each thin slice; conserving only the total could hide a missing slice.
        np.testing.assert_allclose(result["component_volumes_m3"], exact, rtol=0.006, atol=0)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_observation_intervals_preserve_moving_cut_geometry(backend):
    if backend == "cuda":
        import warp as wp

        if not wp.is_cuda_available():
            pytest.skip("CUDA required")
    cfg = SimpleNamespace(
        size=(0.024, 0.012, 0.012), grid_spacing=0.006, shape="block", blade_half_size=(0.0005, 0.02, 0.01)
    )
    ref, origin, _ = points(cfg.size, cfg.grid_spacing, cfg.shape)
    solver = SimpleNamespace(config=cfg, reference=ref, origin=origin, device="cpu")
    # An affine deformation and translation linear in time are exactly represented
    # by both the embedding and the material interpolation between observations.
    times = np.linspace(0, 0.08, 9)
    samples = []
    for i, time in enumerate(times):
        A = np.array([[1, 0, 0.1 * time / times[-1]], [0, 1, 0], [0, 0, 1]])
        shift = np.array([0, 0.002 * time / times[-1], 0])
        pose = np.array([0, 0, 0.03 - 0.03 * time / times[-1], 0, 0, 0, 1.0])
        row = np.r_[time, pose, int(i < len(times) - 1), 0]
        path = row[None] if not samples else np.vstack([samples[-1].tool_path[-1], row])
        samples.append(
            Observation(
                time,
                ref @ A.T + shift,
                np.tile(A, (len(ref), 1, 1)),
                np.zeros(len(ref), np.int64),
                pose,
                path,
            )
        )
    results = compare_observations(
        solver, 0.003, samples, strides=(1, 2, 4), backend=backend, check_frames=(0, 4, 8)
    )
    for result in results:
        assert [check["frame"] for check in result["geometry_checks"]] == [0, 4, 8]
        assert result["broken_links"] > 0
        assert result["topology_mismatch_frames"] == []
        assert result["max_broken_link_difference"] == result["max_node_code_difference"] == 0
        assert result["max_matching_vertex_displacement_m"] < 1e-12
        assert_closed_oriented(result["geometry"], 2)
        assert abs(result.get("relative_volume_difference", 0)) < 1e-12
