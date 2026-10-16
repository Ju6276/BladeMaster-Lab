import numpy as np
from blademaster_warp.surface import LinkedSurface, MaterialEmbedding
from blademaster_warp.geometry import points


def test_original_banana_tips_remain_connected_on_refined_surface():
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from blademaster_warp.geometry import BANANA_DEMO_SIZE_M

    s = LinkedSurface(BANANA_DEMO_SIZE_M, 0.003, "banana")
    mesh = s.mesh()
    tri = mesh.triangles
    assert (edge_incidence(mesh) == 2).all()
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(mesh.reference), len(mesh.reference))
    )
    assert connected_components(graph, directed=False, return_labels=False) == 1


def edge_incidence(mesh):
    triangles = mesh.triangles
    edges = np.sort(
        np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1
    )
    return np.unique(edges, axis=0, return_counts=True)[1]


def test_original_banana_surface_has_consistent_outward_winding():
    # QEF vertices on curved/flat boundaries can give the wrong face normal
    # despite a closed mesh. Primal cell ordering must determine its winding.
    from blademaster_warp.benchmark_surface import mesh_violations

    s = LinkedSurface((0.12, 0.06, 0.03), 0.003, "banana")
    for cut in [False, True]:
        if cut:
            s.sweep(
                [[0, -0.04, 0.04], [0, 0.04, 0.04]],
                [[0, -0.04, -0.01], [0, 0.04, -0.01]],
                normal_world=[1, 0, 0],
            )
        assert not any(mesh_violations(s.mesh().triangles).values())


def test_reference_mesh_cache_invalidates_on_cuts_without_freezing_motion():
    s = LinkedSurface((0.06, 0.03, 0.03), 0.006)
    first = s.mesh()
    assert s.mesh() is first
    s.sweep([[0, -0.04, 0.04], [0, 0.04, 0.04]], [[0, -0.04, -0.01], [0, 0.04, -0.01]])
    cut = s.mesh()
    assert cut is not first
    assert s.mesh() is cut
    recomputed = s.mesh(use_cache=False)
    for name in ["reference", "triangles", "codes", "cells", "weights"]:
        np.testing.assert_array_equal(getattr(cut, name), getattr(recomputed, name))
    ref, origin, cells = points(s.size, 0.006, "block")
    emb = MaterialEmbedding(s.reference, ref, 0.006, origin, cells)
    codes = np.where(ref[:, 0] < 0, 1, 2)
    F = np.tile(np.eye(3), (len(ref), 1, 1))
    before = emb.vertices(s.mesh(), ref, F, codes)
    after = emb.vertices(s.mesh(), ref + [0.01, 0.02, 0.03], F, codes)
    np.testing.assert_allclose(after - before, np.tile([0.01, 0.02, 0.03], (len(before), 1)), atol=1.0e-12)


def test_original_surface_and_finite_sweep_are_closed_and_persistent():
    s = LinkedSurface((0.06, 0.03, 0.03), 0.006)
    initial = s.mesh()
    assert len(initial.triangles) > 0
    assert (edge_incidence(initial) == 2).all()
    assert s.sweep([[0.2, -0.04, 0.04], [0.2, 0.04, 0.04]], [[0.2, -0.04, -0.01], [0.2, 0.04, -0.01]]) == 0
    assert not s.broken.any()
    assert s.sweep([[0, -0.04, 0.04], [0, 0.04, 0.04]], [[0, -0.04, -0.01], [0, 0.04, -0.01]]) > 0
    cut = s.mesh()
    assert len(cut.reference) > len(initial.reference)
    assert (edge_incidence(cut) == 2).all()
    old = s.broken.copy()
    s.sweep([[0, -0.04, -0.01], [0, 0.04, -0.01]], [[0, -0.04, 0.04], [0, 0.04, 0.04]])
    np.testing.assert_array_equal(s.broken, old)
    assert len(s.mesh().reference) == len(cut.reference)


def test_affine_exact_embedding_and_cut_faces_separate():
    s = LinkedSurface((0.06, 0.03, 0.03), 0.006)
    ref, _, _ = points(s.size, 0.006, "block")
    emb = MaterialEmbedding(s.reference, ref, 0.006)
    mesh = s.mesh()
    A = np.array([[1, 0.1, 0], [0, 1.2, 0], [0, 0, 0.9]])
    t = np.array([0.1, 0.2, 0.3])
    F = np.tile(A, (len(ref), 1, 1))
    x = ref @ A.T + t
    np.testing.assert_allclose(
        emb.vertices(mesh, x, F, np.zeros(len(ref), int)), mesh.reference @ A.T + t, atol=1.0e-10
    )
    s.sweep([[0, -0.04, -0.01], [0, 0.04, -0.01]], [[0, -0.04, 0.04], [0, 0.04, 0.04]])
    mesh = s.mesh()
    codes = np.where(ref[:, 0] < 0, 1, 2)
    displacement = np.where(codes == 1, -0.003, 0.003)
    x = ref.copy()
    x[:, 0] += displacement
    moved = emb.vertices(mesh, x, np.tile(np.eye(3), (len(ref), 1, 1)), codes)
    for code, dx in [(1, -0.003), (2, 0.003)]:
        selected = mesh.codes == code
        assert selected.any()
        np.testing.assert_allclose(moved[selected, 0], mesh.reference[selected, 0] + dx, atol=1.0e-10)


def test_embedding_uses_material_grid_not_padded_surface_grid():
    size = (0.06, 0.03, 0.03)
    h = 0.006
    s = LinkedSurface(size, h)
    ref, origin, cells = points(size, h, "block")
    emb = MaterialEmbedding(s.reference, ref, h, origin, cells)
    s.sweep(
        [[0, -0.04, 0.04], [0, 0.04, 0.04]], [[0, -0.04, -0.01], [0, 0.04, -0.01]], normal_world=[1, 0, 0]
    )
    mesh = s.mesh()
    codes = np.where(ref[:, 0] < 0, 1, 2)
    x = ref.copy()
    x[:, 0] += np.where(codes == 1, -0.01, 0.01)
    result = emb.vertices(mesh, x, np.tile(np.eye(3), (len(ref), 1, 1)), codes)
    assert np.isfinite(result).all()
    for code, delta in [(1, -0.01), (2, 0.01)]:
        selected = mesh.codes == code
        np.testing.assert_allclose(result[selected, 0], mesh.reference[selected, 0] + delta, atol=1.0e-10)


def test_partial_cut_front_uses_connected_region_hermite():
    s = LinkedSurface((0.06, 0.03, 0.03), 0.006)
    s.sweep(
        [[0, -0.04, 0.04], [0, 0.04, 0.04]], [[0, -0.04, 0.015], [0, 0.04, 0.015]], normal_world=[1, 0, 0]
    )
    mesh = s.mesh()
    assert (edge_incidence(mesh) == 2).all()
    # Shallow intersecting cut; the uncut lower volume remains connected.
    s.sweep(
        [[-0.04, 0, 0.04], [0.04, 0, 0.04]],
        [[-0.04, 0, 0.015], [0.04, 0, 0.015]],
        cut_index=1,
        normal_world=[0, 1, 0],
    )
    assert (edge_incidence(s.mesh()) == 2).all()


def test_moving_material_link_crosses_stationary_finite_blade():
    from blademaster_warp.surface import moving_segment_hits
    import numpy as np

    a = np.array([[-1.0, 0.0, 1.0], [-1.0, 3.0, 1.0]])
    b = np.array([[1.0, 0.0, 1.0], [1.0, 3.0, 1.0]])
    knife = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    hit, alpha, t = moving_segment_hits(a, b, a - [0, 0, 2], b - [0, 0, 2], knife, knife)
    np.testing.assert_array_equal(hit, [True, False])
    assert abs(alpha[0] - 0.5) < 1.0e-12 and abs(t[0] - 0.5) < 1.0e-12


def test_relative_sweep_separates_surface_without_mechanical_labels():
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    s = LinkedSurface((0.06, 0.03, 0.03), 0.006)
    knife = np.array([[0, -0.04, 0], [0, 0.04, 0]])
    s.sweep(
        knife,
        knife,
        s.reference - [0, 0, 0.04],
        normal_world=[1, 0, 0],
        previous_nodes=s.reference + [0, 0, 0.01],
    )
    mesh = s.mesh()
    tri = mesh.triangles
    assert (edge_incidence(mesh) == 2).all()
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    adjacency = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(mesh.reference), len(mesh.reference))
    )
    count, _ = connected_components(adjacency, directed=False)
    assert count == 2


def test_band_labels_keep_skin_near_a_cut_on_its_own_side():
    s = LinkedSurface((0.06, 0.03, 0.03), 0.003)
    above, below = [[0, -0.04, 0.04], [0, 0.04, 0.04]], [[0, -0.04, -0.01], [0, 0.04, -0.01]]
    s.sweep(above, below, normal_world=[1, 0, 0], band=(0.018, 0.006))
    x = s.reference[:, 0]
    near = s.inside & (np.abs(x) <= 0.018)
    np.testing.assert_array_equal(s.codes[near], np.where(x[near] < 0, 1, 2))
    a, b = s.edges.T
    assert not (~s.broken & (s.codes[a] * s.codes[b] == 2)).any()
    mesh = s.mesh()
    assert (mesh.codes[np.abs(mesh.reference[:, 0]) < 0.015] > 0).all()
    # A crossing link the swept-edge test missed is severed on the next update.
    missed = np.flatnonzero(s.broken & (s.codes[a] * s.codes[b] == 2))[0]
    s.broken[missed] = False
    s.sweep(below, below, normal_world=[1, 0, 0], band=(0.018, 0.006))
    assert s.broken[missed]


def loop_mesh(s):
    """The per-link loop assembly the vectorized LinkedSurface.mesh replaced."""
    vertices, lookup = [], {}
    for ci in range(len(s.cells)):
        for slot in np.flatnonzero(s.region_valid[ci]):
            for k in np.flatnonzero(s.corner_region[ci] == slot):
                lookup[ci, s.cells[ci][k]] = len(vertices)
            vertices.append(s.region_point[ci, slot])
    faces = []
    for e in np.flatnonzero(s.boundary | s.broken):
        a, b = s.edges[e]
        for node in (a, b):
            if not s.inside[node]:
                continue
            ring = list(dict.fromkeys(lookup[ci, node] for ci in s.incident[e] if (ci, node) in lookup))
            if len(ring) < 3:
                continue
            if (-1 if s.axes[e] == 1 else 1) * (1 if node == a else -1) < 0:
                ring = ring[::-1]
            faces += [[ring[0], ring[j], ring[j + 1]] for j in range(1, len(ring) - 1)]
    return np.array(vertices), np.array(faces).reshape(-1, 3)


def test_vectorized_mesh_matches_the_loop_assembly_after_crossing_cuts():
    s = LinkedSurface((0.12, 0.06, 0.03), 0.003, "banana")
    s.sweep(
        [[0.01, -0.04, 0.04], [0.01, 0.04, 0.04]], [[0.01, -0.04, -0.01], [0.01, 0.04, -0.01]], cut_index=0
    )
    s.sweep([[-0.05, 0.0, 0.02], [0.05, 0.0, 0.02]], [[-0.05, 0.0, 0.008], [0.05, 0.0, 0.008]], cut_index=1)
    mesh = s.mesh()
    vertices, faces = loop_mesh(s)
    np.testing.assert_array_equal(mesh.reference, vertices)
    np.testing.assert_array_equal(mesh.triangles, faces)
    assert len(faces) and (edge_incidence(mesh) == 2).all()


def test_box_lstsq_matches_scipy_including_rank_deficient_normals():
    from scipy.optimize import lsq_linear
    from blademaster_warp.surface import box_lstsq

    rng = np.random.default_rng(3)
    for trial in range(200):
        rows = rng.integers(2, 7)
        A = rng.normal(size=(rows, 3))
        if trial % 3 == 0:
            A[:] = A[0]  # every Hermite normal equal: a flat cut
        b = rng.normal(size=rows) * 2
        lower, upper = -rng.uniform(0.1, 1, 3), rng.uniform(0.1, 1, 3)
        x = box_lstsq(A, b, lower, upper)
        ref = lsq_linear(A, b, bounds=(lower, upper), tol=1e-12).x
        assert np.all(x >= lower) and np.all(x <= upper)
        assert np.linalg.norm(A @ x - b) <= np.linalg.norm(A @ ref - b) + 1e-9
