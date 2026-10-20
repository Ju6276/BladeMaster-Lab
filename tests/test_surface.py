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
    vertices, owners, lookup = [], [], {}
    for ci in range(len(s.cells)):
        for slot in np.flatnonzero(s.region_valid[ci]):
            for k in np.flatnonzero(s.corner_region[ci] == slot):
                lookup[ci, s.cells[ci][k]] = len(vertices)
            vertices.append(s.region_point[ci, slot])
            owners.append(ci)
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
            other = False
            if len(ring) == 4:
                default = quad_agreement(vertices, ring)
                alternative = quad_agreement(vertices, ring[1:] + ring[:1])
                if abs(alternative - default) <= QUAD_SPLIT_MARGIN:
                    other = s.broken[e] and min(range(4), key=lambda k: owners[ring[k]]) % 2 == 1
                else:
                    other = alternative > default
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


def _plane_crossing(penetration):
    """One edge through a separate triangle, with the near end `penetration` metres past the plane."""
    from blademaster_warp.surface import OFFSETS, _mesh_labels

    h = 0.003
    lower = np.array([-0.0015, -0.0015, -0.0015])
    holder = type("Holder", (), {})()
    holder.h = h
    holder.reference = lower + OFFSETS * h
    vertices = np.array(
        [
            [-penetration, 0.0, 0.0],
            [0.002, 0.0, 0.0],
            [0.001, 0.002, 0.0],
            [0.0, -0.01, -0.01],
            [0.0, 0.01, -0.01],
            [0.0, 0.0, 0.01],
        ]
    )
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    codes = np.array([1, 1, 1, 2, 2, 2])
    cells = np.tile(np.arange(8), (len(vertices), 1))
    weights = np.zeros((len(vertices), 8))
    labels = _mesh_labels(len(vertices), triangles)
    return holder, vertices, weights, triangles, codes, cells, labels


def test_shallow_clearance_comes_out_of_the_stated_move():
    """0.15 mm is the whole move. A 0.05 mm clearance must not shrink it to 0.10 mm."""
    from blademaster_warp.surface import LinkedSurface, _strict_crossing_counts

    gap, limit = 5.0e-5, 1.5e-4
    holder, vertices, weights, triangles, codes, cells, labels = _plane_crossing(1.1e-4)
    before_self, before_inter = _strict_crossing_counts(vertices, triangles, labels)
    assert before_self == 0 and before_inter >= 1
    accepted = LinkedSurface._accept_shallow_crossing(
        holder, vertices, weights, triangles, codes, cells, labels, gap, limit, True
    )
    assert accepted is not None
    moved, _moved_weights = accepted
    assert abs(moved[0, 0] - vertices[0, 0]) <= limit + 1.0e-12
    assert moved[0, 0] > 0.0
    after_self, after_inter = _strict_crossing_counts(moved, triangles, labels)
    assert after_self == 0 and after_inter < before_inter

    holder, vertices, weights, triangles, codes, cells, labels = _plane_crossing(5.0e-5)
    accepted = LinkedSurface._accept_shallow_crossing(
        holder, vertices, weights, triangles, codes, cells, labels, gap, limit, True
    )
    assert accepted is not None
    np.testing.assert_allclose(accepted[0][0, 0], gap, atol=1.0e-12)

    holder, vertices, weights, triangles, codes, cells, labels = _plane_crossing(1.6e-4)
    assert (
        LinkedSurface._accept_shallow_crossing(
            holder, vertices, weights, triangles, codes, cells, labels, gap, limit, True
        )
        is None
    )


def test_crossed_quad_diagonal_is_replaced_by_the_other_one():
    """The other diagonal of a quad is taken when it is what removes the crossing."""
    from blademaster_warp.surface import _max_edge_incidence, _strict_crossing_counts, flip_crossed_diagonals
    from blademaster_warp.surface import _mesh_labels

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.2, -1.0],
            [0.8, 0.2, 1.0],
            [0.85, 0.2, 0.0],
        ]
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])
    labels = _mesh_labels(len(vertices), triangles)
    before = _strict_crossing_counts(vertices, triangles, labels)
    assert before[1] >= 1 and before[0] == 0
    flipped = flip_crossed_diagonals(vertices, triangles)
    after = _strict_crossing_counts(vertices, flipped, _mesh_labels(len(vertices), flipped))
    assert after == (0, 0)
    assert _max_edge_incidence(flipped) <= 2
    # The same four corners, the other diagonal. Boundary edges stay.
    assert {tuple(sorted(face)) for face in flipped[:2]} == {(0, 1, 3), (1, 2, 3)}
    np.testing.assert_array_equal(flipped[2], triangles[2])
    unchanged = np.array([[0, 1, 2], [2, 0, 3]])
    np.testing.assert_array_equal(flip_crossed_diagonals(vertices[:4], unchanged), unchanged)


def test_one_diagonal_swap_keeps_the_crossings_it_does_not_touch():
    """A second disjoint pierce stays in the count when only the first quad flips."""
    from scripts.surface_deformed import intersection_pairs, segment_triangle_pairs
    from blademaster_warp.surface import (
        _counts_after_quad_flip,
        _mesh_edges,
        _mesh_labels,
        _other_diagonal,
        _strict_crossing_counts,
        flip_crossed_diagonals,
    )

    one = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.2, -1.0],
            [0.8, 0.2, 1.0],
            [0.85, 0.2, 0.0],
        ]
    )
    vertices = np.vstack([one, one + (5.0, 0.0, 0.0)])
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [7, 8, 9], [7, 9, 10], [11, 12, 13]])
    labels = _mesh_labels(len(vertices), triangles)
    before = _strict_crossing_counts(vertices, triangles, labels)
    assert before[1] >= 2 and before[0] == 0
    new = _other_diagonal(triangles[0], triangles[1])
    edges, faces = intersection_pairs(vertices, triangles)
    interior = segment_triangle_pairs(
        vertices[edges[:, 0]], vertices[edges[:, 1]], vertices[triangles[faces]], strict=True
    )
    strict_edges, strict_faces = edges[interior], faces[interior]
    same = labels[strict_edges[:, 0]] == labels[triangles[strict_faces, 0]]
    local = _counts_after_quad_flip(
        vertices,
        triangles,
        labels,
        strict_edges,
        strict_faces,
        same,
        _mesh_edges(triangles),
        0,
        1,
        new[0],
        new[1],
        before,
    )
    trial = triangles.copy()
    trial[0], trial[1] = new
    assert local == _strict_crossing_counts(vertices, trial, labels)
    flipped = flip_crossed_diagonals(vertices, triangles)
    assert _strict_crossing_counts(vertices, flipped, _mesh_labels(len(vertices), flipped)) == (0, 0)


def test_piercing_quad_diagonal_is_replaced_when_the_stabbed_face_has_no_partner():
    # Edge 0-2 is the diagonal of one quad and it passes through a separate
    # triangle. That triangle has no neighbour, so its own edges cannot be
    # flipped. The diagonal's quad can.
    from blademaster_warp.surface import _max_edge_incidence, _strict_crossing_counts, flip_crossed_diagonals
    from blademaster_warp.surface import _mesh_labels

    vertices = np.array(
        [
            [0.0, 0.0, 0.4],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, -0.4],
            [0.0, 0.2, 0.05],
            [-0.05, -0.05, 0.0],
            [0.05, -0.05, 0.0],
            [0.0, 0.02, 0.0],
        ]
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])
    before = _strict_crossing_counts(vertices, triangles, _mesh_labels(len(vertices), triangles))
    assert before == (0, 1)
    flipped = flip_crossed_diagonals(vertices, triangles)
    after = _strict_crossing_counts(vertices, flipped, _mesh_labels(len(vertices), flipped))
    assert after == (0, 0)
    assert _max_edge_incidence(flipped) <= 2
    assert {tuple(sorted(face)) for face in flipped[:2]} == {(0, 1, 3), (1, 2, 3)}
    np.testing.assert_array_equal(flipped[2], triangles[2])


def test_settle_swaps_a_world_diagonal_and_leaves_the_flat_reference():
    """The other diagonal is taken when the reference quad does not cross."""
    from blademaster_warp.surface import (
        _mesh_labels,
        _strict_crossing_counts,
        settle_crossed_diagonals,
    )

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.2, -1.0],
            [0.8, 0.2, 1.0],
            [0.85, 0.2, 0.0],
        ]
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])
    reference = vertices.copy()
    reference[4:] += (3.0, 0.0, 0.0)
    codes = np.ones(len(vertices), np.int64)

    def counts(points, faces):
        return _strict_crossing_counts(points, faces, _mesh_labels(len(points), faces))

    assert counts(vertices, triangles)[1] >= 1
    assert counts(reference, triangles) == (0, 0)
    repaired, faces = settle_crossed_diagonals(vertices, triangles, reference, codes)
    assert counts(repaired, faces) == (0, 0)
    assert counts(reference, faces) == (0, 0)
    assert np.allclose(repaired, vertices)
    assert {tuple(sorted(face)) for face in faces[:2]} == {(0, 1, 3), (1, 2, 3)}


def test_settle_pulls_the_shared_vertex_without_crossing_the_reference():
    """The 0.288 mm fold still returns to the opening, and the flat reference stays clear."""
    from blademaster_warp.surface import (
        BANK_OPENING_M,
        _mesh_labels,
        _strict_crossing_counts,
        settle_crossed_diagonals,
    )

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.003, 0.003, 0.000288],
            [0.002, 0.006, -0.001036],
            [0.006, 0.002, -0.000556],
        ]
    )
    triangles = np.array([[0, 1, 2], [3, 4, 5], [3, 5, 2], [3, 2, 1], [1, 0, 3], [0, 4, 3]])
    reference = vertices.copy()
    reference[:, 2] = 0.0
    codes = np.ones(len(vertices), np.int64)

    def counts(points, faces):
        return _strict_crossing_counts(points, faces, _mesh_labels(len(points), faces))

    assert counts(vertices, triangles) == (2, 0)
    assert counts(reference, triangles) == (0, 0)
    repaired, faces = settle_crossed_diagonals(vertices, triangles, reference, codes)
    assert counts(repaired, faces) == (0, 0)
    assert counts(reference, faces) == (0, 0)
    assert abs(repaired[3, 2] + BANK_OPENING_M) < 1.0e-12


def test_clamped_point_on_an_unsampled_face_returns_to_the_sample_centroid():
    """A box corner that no Hermite sample touches is not that region's cut."""
    from blademaster_warp.surface import _mesh_labels, _strict_crossing_counts, unsampled_face_centroid

    lower, upper = np.zeros(3), np.ones(3)
    samples = np.array([[1.0, 0.0, 0.2], [1.0, 0.3, 0.0], [0.4, 0.0, 0.0]])
    clamped = np.array([0.35, 0.999, 0.999])
    centroid = unsampled_face_centroid(clamped, samples, lower, upper, 1.0e-3)
    np.testing.assert_allclose(centroid, samples.mean(0))
    assert unsampled_face_centroid(np.array([0.999, 0.2, 0.2]), samples, lower, upper, 1.0e-3) is None
    vertices = np.array(
        [
            clamped,
            [0.2, 0.4, 0.999],
            [0.2, 0.999, 0.4],
            [0.05, 0.999, 0.999],
            [0.9, 0.2, 0.2],
            [0.05, 0.949, 0.999],
        ]
    )
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    before = _strict_crossing_counts(vertices, triangles, _mesh_labels(len(vertices), triangles))
    assert sum(before) == 2
    vertices = vertices.copy()
    vertices[0] = centroid
    after = _strict_crossing_counts(vertices, triangles, _mesh_labels(len(vertices), triangles))
    assert after == (0, 0)


def test_adjacent_cell_vertices_meet_on_the_shared_face():
    """The midpoint of a crossing pair can leave one cell; the meeting point cannot."""
    surface = LinkedSurface((0.009, 0.006, 0.006), 0.003, "block")
    centres = surface.reference[surface.cells].mean(1)
    gaps = np.linalg.norm(centres - centres[0], axis=1)
    neighbour = int(np.argmin(np.where(np.abs(gaps - surface.h) < 1.0e-9, gaps, np.inf)))
    shared = 0.5 * (centres[0] + centres[neighbour])
    axis = int(np.argmax(np.abs(centres[neighbour] - centres[0])))
    left, right = shared.copy(), shared.copy()
    left[axis] -= 2.0e-4
    right[axis] += 3.0e-4
    cells = np.stack([surface.cells[0], surface.cells[neighbour]])
    meeting = surface._cells_meet(cells, 0, 1, np.stack([left, right]))
    assert meeting is not None
    assert abs(float(meeting[axis] - shared[axis])) < 1.0e-12
    far = int(np.argmax(gaps))
    assert surface._cells_meet(np.stack([surface.cells[0], surface.cells[far]]), 0, 1, np.stack([left, right])) is None


def test_one_blade_far_from_its_samples_stays_on_the_average_plane():
    """Fanned tangents of one cut meet at the centre of curvature, not in the cell."""
    from blademaster_warp.surface import average_plane_point

    samples = np.array(
        [[-0.0135, -0.010419, 0.0195], [-0.0135, -0.0075, 0.016983], [-0.012312, -0.0075, 0.0195]]
    )
    normals = np.array(
        [[0.9991, 0.0384, -0.0161], [0.9984, 0.0335, -0.0451], [0.9991, 0.0231, -0.0363]]
    )
    lower, upper = np.array([-0.0135, -0.0105, 0.0165]), np.array([-0.0105, -0.0075, 0.0195])
    point = average_plane_point(samples, normals, lower, upper)
    assert point is not None
    assert np.linalg.norm(point - samples.mean(0)) < 0.001
    assert np.all(point >= lower) and np.all(point <= upper)
    # A second blade meets the first outside this cell and stays on the ordinary QEF.
    corner_normals = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    corner_samples = np.array(
        [[0.010, 0.011, 0.011], [0.010, 0.012, 0.011], [0.011, 0.010, 0.011], [0.012, 0.010, 0.012]]
    )
    assert (
        average_plane_point(corner_samples, corner_normals, np.full(3, 0.010), np.full(3, 0.013))
        is None
    )


def test_far_intersection_stays_on_the_average_plane_above_a_tenth():
    """A blade can turn by more than a tenth and still meet only outside the cell."""
    from blademaster_warp.surface import average_plane_point

    samples = np.array([[0.2, 0.2, 0.4], [0.8, 0.3, 0.45], [0.5, 0.75, 0.55]])
    raw = np.array([[0.02, 0.0, 1.0], [-0.04, 0.05, 1.0], [0.22, -0.08, 1.0]])
    normals = raw / np.linalg.norm(raw, axis=1)[:, None]
    singular = np.linalg.svd(normals, compute_uv=False)
    assert singular[1] > 0.1 * singular[0]
    lower, upper = np.zeros(3), np.ones(3)
    point = average_plane_point(samples, normals, lower, upper)
    assert point is not None
    assert np.all(point >= lower) and np.all(point <= upper)
    assert np.linalg.norm(point - samples.mean(0)) < 0.05


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


def _interior_x_edge(surface):
    edge = int(np.flatnonzero(surface.inside[surface.edges].all(1) & (surface.axes == 0))[0])
    ends = tuple(int(v) for v in surface.edges[edge])
    return edge, ends


def test_missing_endpoint_uses_the_other_affine_field():
    # The finite end carries the reference edge. Stretching that end by 2 along
    # the edge puts the plane at a quarter of the reference segment, not halfway.
    surface = LinkedSurface((0.03, 0.03, 0.03), 0.006, "block")
    edge, (i, j) = _interior_x_edge(surface)
    surface.codes[i], surface.codes[j] = 1, 2
    deformation = np.tile(np.eye(3), (len(surface.reference), 1, 1))
    deformation[i] = np.diag([2.0, 1.0, 1.0])
    deformation[j] = np.nan
    positions = surface.reference.copy()
    positions[j] = np.nan
    plane = surface.reference[i] + np.array([0.5 * surface.h, 0.0, 0.0])
    surface._sever_opposite(positions, np.array([plane, plane + [0.0, 0.01, 0.0]]), [1.0, 0.0, 0.0], 0, deformation)
    sample, normal = surface.hermite[edge]
    expected = 0.75 * surface.reference[i] + 0.25 * surface.reference[j]
    np.testing.assert_allclose(sample, expected, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-12)
    assert surface.broken[edge]


def test_identity_field_with_one_sample_matches_the_chord():
    def place(hide_end):
        surface = LinkedSurface((0.03, 0.03, 0.03), 0.006, "block")
        edge, (i, j) = _interior_x_edge(surface)
        surface.codes[i], surface.codes[j] = 1, 2
        positions = surface.reference.copy()
        deformation = np.tile(np.eye(3), (len(surface.reference), 1, 1))
        if hide_end:
            positions[j] = np.nan
            deformation[j] = np.nan
        mid = surface.reference[[i, j]].mean(0)
        surface._sever_opposite(positions, np.array([mid, mid + [0.0, 0.01, 0.0]]), [1.0, 0.0, 0.0], 0, deformation)
        return surface.hermite[edge][0]

    np.testing.assert_allclose(place(False), place(True), atol=1e-12)


def test_missing_both_endpoints_does_not_store_a_nonfinite_sample():
    surface = LinkedSurface((0.03, 0.03, 0.03), 0.006, "block")
    edge, (i, j) = _interior_x_edge(surface)
    surface.codes[i], surface.codes[j] = 1, 2
    positions = surface.reference.copy()
    positions[[i, j]] = np.nan
    deformation = np.tile(np.eye(3), (len(surface.reference), 1, 1))
    deformation[[i, j]] = np.nan
    mid = surface.reference[[i, j]].mean(0)
    surface._sever_opposite(positions, np.array([mid, mid + [0.0, 0.01, 0.0]]), [1.0, 0.0, 0.0], 0, deformation)
    assert surface.broken[edge]
    assert edge not in surface.hermite


def test_band_waits_for_the_anchor_particle_instead_of_the_embedded_side():
    # Geometry would write side 2. An unlabeled anchor particle blocks that,
    # and a particle already on side 1 is copied onto the node.
    surface = LinkedSurface((0.03, 0.03, 0.03), 0.006, "block")
    inside = np.flatnonzero(surface.inside)
    node = int(inside[len(inside) // 2])
    positions = surface.reference.copy()
    edge = np.array(
        [
            [positions[node, 0] - 0.001, positions[node, 1] - 0.02, positions[node, 2]],
            [positions[node, 0] - 0.001, positions[node, 1] + 0.02, positions[node, 2]],
        ]
    )
    normal, band = [1.0, 0.0, 0.0], (0.01, 0.01)
    surface._assign_band(positions, edge, edge, normal, 0, band)
    assert int(surface.codes[node]) == 2
    surface.codes[node] = 0
    waiting = np.zeros(len(surface.reference), np.int64)
    surface._assign_band(positions, edge, edge, normal, 0, band, waiting)
    assert int(surface.codes[node]) == 0
    labeled = waiting.copy()
    labeled[node] = 1
    surface._take_anchor_sides(labeled, 0)
    assert int(surface.codes[node]) == 1
    surface.codes[node] = 0
    missing = np.full(len(surface.reference), np.int64(-1))
    surface._assign_band(positions, edge, edge, normal, 0, band, missing)
    assert int(surface.codes[node]) == 2


def test_anchor_digit_is_the_heaviest_compatible_particle():
    from blademaster_warp.surface import Supports, anchor_cut_digits

    # Node 0 sees particle 0 at weight 0.2 (digit 2) and particle 1 at weight 0.9
    # (digit 1). Node 1's code already conflicts with digit 1, so particle 0 wins.
    # Node 2 has no particle. A lone unlabeled particle stays 0, not "no anchor".
    support = Supports(
        np.array([0, 2, 4, 4]),
        np.array([0, 1, 0, 1]),
        np.array([0.2, 0.9, 0.2, 0.9]),
    )
    digits = anchor_cut_digits(support, np.array([0, 2, 0]), np.array([2, 1]), 0)
    np.testing.assert_array_equal(digits, [1, 2, -1])
    waiting = Supports(np.array([0, 1]), np.array([0]), np.array([1.0]))
    np.testing.assert_array_equal(anchor_cut_digits(waiting, np.array([0]), np.array([0]), 0), [0])


def test_singular_pullback_keeps_a_unit_world_normal():
    surface = LinkedSurface((0.03, 0.03, 0.03), 0.006, "block")
    edge, (i, j) = _interior_x_edge(surface)
    surface.codes[i], surface.codes[j] = 1, 2
    positions = surface.reference.copy()
    deformation = np.tile(np.diag([0.0, 1.0, 1.0]), (len(surface.reference), 1, 1))
    mid = surface.reference[[i, j]].mean(0)
    surface._sever_opposite(positions, np.array([mid, mid + [0.0, 0.01, 0.0]]), [1.0, 0.0, 0.0], 0, deformation)
    sample, normal = surface.hermite[edge]
    np.testing.assert_allclose(sample, mid, atol=1e-12)
    np.testing.assert_allclose(normal, [1.0, 0.0, 0.0], atol=1e-12)
