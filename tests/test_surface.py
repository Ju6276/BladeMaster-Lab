import numpy as np
from blademaster_warp.surface import QUAD_SPLIT_MARGIN, LinkedSurface, MaterialEmbedding
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


def quad_agreement(vertices, ring):
    q = np.asarray([vertices[i] for i in ring])
    n1, n2 = np.cross(q[1] - q[0], q[2] - q[0]), np.cross(q[2] - q[0], q[3] - q[0])
    return n1 @ n2 / max(np.linalg.norm(n1) * np.linalg.norm(n2), 1.0e-300)


def loop_mesh(s):
    """The per-link loop assembly the vectorized LinkedSurface.mesh replaced."""
    vertices, lookup = [], {}
    for ci in range(len(s.cells)):
        for slot in np.flatnonzero(s.region_valid[ci]):
            for k in np.flatnonzero(s.corner_region[ci] == slot):
                lookup[ci, s.cells[ci][k]] = len(vertices)
            vertices.append(s.region_point[ci, slot])
    polygons = []
    changes = dict(dropped=0, reverted=0, switched=0)
    for e in np.flatnonzero(s.boundary | s.broken):
        a, b = s.edges[e]
        ends = []
        for node in (a, b):
            if not s.inside[node]:
                continue
            ring = list(dict.fromkeys(lookup[ci, node] for ci in s.cells_around(e) if (ci, node) in lookup))
            if len(ring) < 3:
                continue
            if (-1 if s.axes[e] == 1 else 1) * (1 if node == a else -1) < 0:
                ring = ring[::-1]
            other = len(ring) == 4 and quad_agreement(vertices, ring[1:] + ring[:1]) > quad_agreement(
                vertices, ring
            ) + QUAD_SPLIT_MARGIN
            ends.append([ring, other, len(ring) != 4])
        # Both ends in one region all around the link: an internal double wall, no faces.
        if len(ends) == 2 and sorted(ends[0][0]) == sorted(ends[1][0]):
            changes["dropped"] += 1
            continue
        polygons += ends

    def faces_of():
        faces = []
        for ring, other, _ in polygons:
            ring = ring[1:] + ring[:1] if other else ring
            faces += [[ring[0], ring[j], ring[j + 1]] for j in range(1, len(ring) - 1)]
        return faces

    # Rollback: a quad diagonal used by another polygon is non-manifold; take the other
    # diagonal if it is unused, else go back to the default split (once).
    while True:
        uses = {}
        for f in faces_of():
            for i, j in ((0, 1), (1, 2), (2, 0)):
                key = tuple(sorted((f[i], f[j])))
                uses[key] = uses.get(key, 0) + 1
        reverts, switches = [], []
        for k, (ring, other, locked) in enumerate(polygons):
            if locked:
                continue
            alternative = uses.get(tuple(sorted((ring[1], ring[3]))), 0)
            default = uses.get(tuple(sorted((ring[0], ring[2]))), 0)
            if other and alternative != 2:
                reverts.append((k, False, True))
            elif not other and default > 2 and alternative == 0:
                switches.append((k, True, False))
        moves = reverts or switches
        if not moves:
            break
        for k, other, locked in moves:
            polygons[k][1:] = [other, locked]
            changes["switched" if other else "reverted"] += 1
    return np.array(vertices), np.array(faces_of()).reshape(-1, 3), changes


def test_vectorized_mesh_matches_the_loop_assembly_after_crossing_cuts():
    s = LinkedSurface((0.12, 0.06, 0.03), 0.003, "banana")
    s.sweep(
        [[0.01, -0.04, 0.04], [0.01, 0.04, 0.04]], [[0.01, -0.04, -0.01], [0.01, 0.04, -0.01]], cut_index=0
    )
    s.sweep([[-0.05, 0.0, 0.02], [0.05, 0.0, 0.02]], [[-0.05, 0.0, 0.008], [0.05, 0.0, 0.008]], cut_index=1)
    mesh = s.mesh()
    vertices, faces, _ = loop_mesh(s)
    np.testing.assert_array_equal(mesh.reference, vertices)
    np.testing.assert_array_equal(mesh.triangles, faces)
    assert len(faces) and (edge_incidence(mesh) == 2).all()


def test_quad_split_rolls_back_in_both_directions_and_drops_internal_double_walls():
    from blademaster_warp.surface import triangulate

    polygons = [
        ([0, 1, 2, 3], False),  # default diagonal 0-2 also used by the next two triangles
        ([0, 2, 4, -1], False),
        ([2, 0, 5, -1], False),
        ([10, 11, 12, 13], True),  # other diagonal 11-13 also used by the next two triangles
        ([11, 13, 14, -1], False),
        ([13, 11, 15, -1], False),
        ([20, 21, 22, 23], False),  # both diagonals taken: stays
        ([20, 22, 24, -1], False),
        ([22, 20, 25, -1], False),
        ([21, 23, 26, -1], False),
        ([23, 21, 27, -1], False),
    ]
    rings = np.full((len(polygons) + 1, 2, 4), -1)
    others = np.zeros((len(polygons) + 1, 2), bool)
    for k, (ring, other) in enumerate(polygons):
        rings[k, 0], others[k, 0] = ring, other
    # Same quad from both link ends, opposite winding.
    rings[-1] = [[30, 31, 32, 33], [33, 32, 31, 30]]
    faces, changes = triangulate(rings, others)
    assert changes == dict(dropped=1, reverted=1, switched=1)
    faces = {tuple(f) for f in faces.tolist()}
    assert {(1, 2, 3), (1, 3, 0), (10, 11, 12), (10, 12, 13), (20, 21, 22), (20, 22, 23)} <= faces
    assert not any(30 in f for f in faces)


def test_isolated_severed_links_leave_a_closed_manifold_surface():
    s = LinkedSurface((0.03, 0.03, 0.03), 0.003, "block")
    strides = np.array([s.dims[1] * s.dims[2], s.dims[2], 1])
    centre = int((s.dims // 2) @ strides)
    # Two neighbouring x links severed with all the material around them intact, as
    # band labels with opposite codes do (_sever_opposite).
    for node in (centre, centre + strides[1]):
        e = int(np.flatnonzero((s.edges[:, 0] == node) & (s.axes == 0))[0])
        s.broken[e] = True
        s.hermite[e] = (s.reference[s.edges[e]].mean(0), np.array([1.0, 0.0, 0.0]))
    mesh = s.mesh()
    assert s.split_changes["dropped"] == 2
    assert (edge_incidence(mesh) == 2).all()
    vertices, faces, changes = loop_mesh(s)
    assert changes == s.split_changes
    np.testing.assert_array_equal(mesh.reference, vertices)
    np.testing.assert_array_equal(mesh.triangles, faces)


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


def test_cut_pieces_match_the_analytic_pieces_and_stay_closed():
    from scripts.surface_accuracy import CUTS, accuracy

    for cuts in CUTS.values():
        result = accuracy(0.003, cuts)
        assert result["open_edges"] == 0
        assert result["mean_error_cells"] < 0.05 and result["max_error_cells"] < 0.5


def test_bucketed_tent_supports_match_brute_force():
    from blademaster_warp.surface import tent_supports

    rng = np.random.default_rng(3)
    particles = rng.uniform(-0.02, 0.02, (700, 3))
    # Nodes inside, on the edge of and far outside the particle cloud.
    nodes = np.r_[rng.uniform(-0.03, 0.03, (300, 3)), [[0.5, 0.0, 0.0]]]
    for radius in (0.004, 0.008):
        support = tent_supports(nodes, particles, radius, chunk=64)
        assert len(support) == len(nodes)
        for node, (ids, w) in zip(nodes, support):
            expected = np.maximum(1 - np.abs(particles - node) / radius, 0).prod(axis=1)
            want = np.flatnonzero(expected > 1.0e-12)
            np.testing.assert_array_equal(ids, want)
            np.testing.assert_allclose(w, expected[want], rtol=0, atol=1.0e-15)


def test_unassigned_surface_node_follows_one_side_not_the_gap():
    ref, origin, cells = points((0.06, 0.03, 0.03), 0.006, "block")
    s = LinkedSurface((0.06, 0.03, 0.03), 0.003)
    emb = MaterialEmbedding(s.reference, ref, 0.006, origin, cells)
    codes = np.where(ref[:, 0] < 0, 1, 2)
    F = np.tile(np.eye(3), (len(ref), 1, 1))
    x = ref + np.where(ref[:, :1] < 0, -0.03, 0.03) * np.array([1.0, 0.0, 0.0])
    node_codes = np.zeros(len(s.reference), dtype=np.int64)
    nodes, _ = emb.node_state(x, F, codes, node_codes)
    shift = np.abs(nodes[:, 0] - s.reference[:, 0])
    ok = np.isfinite(shift)
    np.testing.assert_allclose(shift[ok], 0.03, atol=1e-12)
