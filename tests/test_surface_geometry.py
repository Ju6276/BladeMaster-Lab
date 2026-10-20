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


def test_partial_crack_front_stays_on_the_cut_plane(monkeypatch):
    import blademaster_warp.surface as module

    surface = LinkedSurface(BANANA_DEMO_SIZE_M, 0.003, "banana")
    # First severed link at frame 63; the remaining links still connect both
    # banks in three cells. Their shared rim must lie on this Hermite plane.
    node = np.ravel_multi_index((33, 9, 7), surface.dims)
    edge = int(np.flatnonzero((surface.edges[:, 0] == node) & (surface.axes == 0))[0])
    point = np.array([0.040068095115760365, -0.0045, 0.0195])
    normal = np.array([0.9998799731904567, 0.014900775404051499, 0.004243360108371719])
    surface.broken[edge] = True
    surface.hermite[edge] = point, normal
    # Isolate the front constraint from the independent two-bank pairing.
    monkeypatch.setattr(LinkedSurface, "_pair_cell_regions", lambda self, ci, regions: regions)
    with monkeypatch.context() as context:
        context.setattr(module, "constrained_box_lstsq", lambda *args: None)
        before = surface.mesh()
    assert geometry_report(before.reference, before.triangles)["self_interior_crossings"] == 4
    surface.cell_geometry.clear()
    surface.dirty.update(surface.cells_around(edge))
    after = surface.mesh()
    geometry = geometry_report(after.reference, after.triangles)
    assert_closed_oriented(geometry, 1)
    assert geometry["self_interior_crossings"] == 0
    np.testing.assert_allclose(
        (surface.reference[after.cells] * after.weights[:, :, None]).sum(1),
        after.reference,
        rtol=0,
        atol=2e-16,
    )


def test_recorded_interface_pairs_preserve_front_planes_and_separate_bank_patches():
    surface = LinkedSurface(BANANA_DEMO_SIZE_M, 0.003, "banana")
    # Nine Hermite samples from the first cut, actual951237c frames 150-531.
    # Cell 2038 has two disconnected patches on the same bank. Pairing all
    # three regions makes final-frame edge (1402,1414) pierce face (1416,1478,1476).
    data = [
        (
            26169,
            [0.042804863278028724, 0.0015, 0.0105],
            [0.9851891143295205, 0.11157652047934122, 0.13020402867937078],
        ),
        (
            26201,
            [0.042716015102529536, 0.0045, 0.0075],
            [0.9765049618280566, 0.13203017772295514, 0.1703117485543219],
        ),
        (
            26204,
            [0.04237099853030895, 0.0045, 0.0105],
            [0.9865658487566955, 0.10563453328426767, 0.12461609625082945],
        ),
        (
            26921,
            [0.04372796651317317, 0.0015, 0.0045],
            [0.9745505681134353, 0.17158335223942564, 0.14425790593260895],
        ),
        (
            26922,
            [0.0435, 0.002634903993206865, 0.0045],
            [0.9776572288920822, 0.12145506889171204, 0.17156634004245788],
        ),
        (
            26923,
            [0.0435, 0.0015, 0.0072774873461058836],
            [0.9731191619412387, 0.151800149475791, 0.17319298854719498],
        ),
        (
            26924,
            [0.0435041486917116, 0.0015, 0.0075],
            [0.9737739377393473, 0.14738157854810185, 0.17332913339734757],
        ),
        (
            26925,
            [0.0435, 0.0015922525955474522, 0.0075],
            [0.9729416596573048, 0.15128727883641188, 0.17463300422761682],
        ),
        (
            26926,
            [0.0435, 0.0015, 0.007569922284440133],
            [0.9730241613623943, 0.1509447393831898, 0.17446967374747466],
        ),
    ]
    for edge, point, normal in data:
        surface.broken[edge] = True
        surface.hermite[edge] = np.array(point), np.array(normal)
    cells = {ci: surface._cell_regions(ci) for ci in (1990, 1997, 1998, 2038, 2039)}
    for ci in (1997, 1998, 2039):
        assert len(cells[ci]) == 2
        np.testing.assert_array_equal(cells[ci][0][1], cells[ci][1][1])
        np.testing.assert_array_equal(cells[ci][0][2], cells[ci][1][2])
        assert cells[ci][0][0].isdisjoint(cells[ci][1][0])
    point, normal = surface.hermite[26923]
    assert abs((cells[1997][0][1] - point) @ normal) < 1e-14
    assert [region for region, _, _ in cells[2038]] == [{0}, {1}, {2, 3, 4, 5, 6, 7}]
    assert len(np.unique([p for _, p, _ in cells[2038]], axis=0)) == 3
    a, b = cells[1990][0][1], cells[1997][0][1]
    triangle = np.array([cells[1998][0][1], cells[2039][0][1], cells[2038][1][1]])
    assert not segment_triangle_pairs(a[None], b[None], triangle[None], strict=True).item()
    # The discarded common-QEF result for the three-region cell reproduces
    # the introduced crossing, independently of the global intersection audit.
    triangle[2] = [0.043503, 0.0020678764078308253, 0.006282852495208274]
    assert segment_triangle_pairs(a[None], b[None], triangle[None], strict=True).item()


def test_coincident_cut_banks_keep_codes_and_share_a_nonplanar_quad_diagonal():
    surface = LinkedSurface((0.03, 0.03, 0.03), 0.003, "block")
    surface.sweep(
        [[0, -0.04, 0.04], [0, 0.04, 0.04]],
        [[0, -0.04, -0.01], [0, 0.04, -0.01]],
        normal_world=[1, 0, 0],
    )
    surface.mesh()
    links = np.flatnonzero(surface.broken)
    centres = surface.reference[surface.edges[links]].mean(1)
    link = links[np.argmin(np.linalg.norm(centres - [0, 0, 0.015], axis=1))]
    cells = surface.cells_around(link)
    assert len(cells) == 4
    # Both banks use the same slightly warped quad, with nearly tied normal
    # agreement. Reversing the ring swaps the default and alternate diagonals.
    first = cells[0]
    surface.region_point[first, surface.region_valid[first], 0] += surface.h * 1e-5
    mesh = surface.mesh(use_cache=False)
    selected = np.array(
        [any(np.array_equal(nodes, surface.cells[ci]) for ci in cells) for nodes in mesh.cells]
    )
    banks = []
    for code in (1, 2):
        vertices = np.flatnonzero(selected & (mesh.codes == code))
        assert len(vertices) == 4
        faces = mesh.triangles[np.isin(mesh.triangles, vertices).all(1)]
        assert len(faces) == 2
        banks.append(sorted(tuple(sorted(map(tuple, face))) for face in mesh.reference[faces]))
    assert banks[0] == banks[1]


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3])
def test_crack_plane_constrained_fit_respects_box_and_infeasible_planes(scale):
    from blademaster_warp.surface import constrained_box_lstsq

    lower, upper = np.zeros(3), np.ones(3) * scale
    plane = np.array([[1.0, 1, 0]]) / np.sqrt(2)
    fit = constrained_box_lstsq(
        np.eye(3), np.array([3.0, 2, -1]) * scale, plane, np.array([scale / np.sqrt(2)]), lower, upper
    )
    # Projection of (3,2,-1) onto x+y=1 inside the unit cube is (1,0,0).
    np.testing.assert_allclose(fit / scale, [1, 0, 0], rtol=0, atol=1e-12)
    assert constrained_box_lstsq(np.eye(3), lower, plane, np.array([3 * scale]), lower, upper) is None
    assert (
        constrained_box_lstsq(np.eye(3), lower, np.r_[plane, plane], np.array([0, scale]), lower, upper)
        is None
    )


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
