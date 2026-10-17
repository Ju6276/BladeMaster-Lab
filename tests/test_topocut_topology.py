import numpy as np
from blademaster_warp.geometry import points
from blademaster_warp.topocut_topology import TopologyTracker

SPACING = 0.006
HALF = (0.0005, 0.03, 0.03)


def block(seed=1):
    return points((0.06, 0.03, 0.03), SPACING, "block", seed)[0]


def stroke(x, top=0.07, bottom=0.03, count=20):
    return [([x, 0.0, z], [0, 0, 0, 1]) for z in np.linspace(top, bottom, count)]


def test_uncut_block_is_one_piece():
    x = block()
    t = TopologyTracker(len(x), SPACING, HALF)
    labels = t.reconstruct(x)
    assert len(t.meshes) == 1
    assert (labels == 0).all()


def test_carved_trajectory_splits_the_block_at_the_knife():
    x = block()
    t = TopologyTracker(len(x), SPACING, HALF)
    labels = t.reconstruct(x, stroke(0.0))
    assert len(t.meshes) == 2
    assert np.array_equal(t.pieces(), np.bincount(labels[labels >= 0]))
    left, right = labels[x[:, 0] < -0.002], labels[x[:, 0] > 0.002]
    assert len(set(left)) == 1 and len(set(right)) == 1 and left[0] != right[0]


def test_earlier_pieces_keep_their_ids_after_the_gap_closes():
    x = block()
    t = TopologyTracker(len(x), SPACING, HALF)
    t.reconstruct(x, stroke(-0.012))
    labels = t.reconstruct(x, stroke(0.012))
    assert len(t.pieces()) == 3
    for lo, hi in ((-1, -0.014), (-0.01, 0.01), (0.014, 1)):
        inside = labels[(x[:, 0] > lo) & (x[:, 0] < hi)]
        assert len(set(inside)) == 1


def test_a_partial_stroke_does_not_separate():
    x = block()
    t = TopologyTracker(len(x), SPACING, HALF)
    t.reconstruct(x, stroke(0.0, bottom=0.045))
    assert len(t.pieces()) == 1


def test_a_piece_that_needs_two_cuts_is_carved_by_both():
    x = block()
    # Each knife covers one half of the width, beyond the 3 mm influence radius.
    t = TopologyTracker(len(x), SPACING, (0.0005, 0.0125, 0.03))
    first = [([0.0, -0.0125, z], [0, 0, 0, 1]) for z in np.linspace(0.07, 0.03, 20)]
    second = [([0.0, 0.0125, z], [0, 0, 0, 1]) for z in np.linspace(0.07, 0.03, 20)]
    t.reconstruct(x, first)
    assert len(t.pieces()) == 1
    t.reconstruct(x, second)
    assert len(t.pieces()) == 2
    assert not t.pending


def test_deep_interior_assignment_survives_refinement(monkeypatch):
    import warp as wp
    from blademaster_warp.topocut_topology import TopologyParams, nearest_cluster

    vertices = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    ) * 0.03 + [0, 0, 0.03]
    faces = np.array(
        [
            [0, 3, 2],
            [0, 1, 3],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        dtype=np.int32,
    )
    # Supply a closed reconstructed surface to isolate assignment from meshing.
    from blademaster_warp.topocut_topology import particle_sdf

    class CubeSurface:
        def __init__(self, *args, **kwargs):
            self.verts = wp.array(vertices, dtype=wp.vec3, device="cpu")
            self.indices = wp.array(faces.ravel(), dtype=int, device="cpu")

        def surface(self, field, level):
            pass

    launch = wp.launch

    def assignment_only(kernel, *args, **kwargs):
        if kernel is not particle_sdf:
            return launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "MarchingCubes", CubeSurface)
    monkeypatch.setattr(wp, "launch", assignment_only)
    with wp.ScopedDevice("cpu"):
        mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3),
            indices=wp.array(faces.ravel(), dtype=int),
            support_winding_number=True,
        )
        positions = wp.array([[0, 0, 0.03], [0.025, 0, 0.03], [0.04, 0, 0.03]], dtype=wp.vec3)
        labels = wp.empty(3, dtype=int)
        for spacing in (0.002, 0.001, 0.0005):
            tau = TopologyParams().tau * spacing / 2
            reach = float(np.linalg.norm(vertices.max(0) - vertices.min(0))) + tau
            wp.launch(
                nearest_cluster, 3, [wp.array([mesh.id], dtype=wp.uint64), positions, tau, reach, labels]
            )
            np.testing.assert_array_equal(labels.numpy(), [0, 0, -1])
            tracker = TopologyTracker(3, spacing, HALF, TopologyParams(smoothing_steps=0), device="cpu")
            np.testing.assert_array_equal(tracker.reconstruct(positions.numpy()), [0, 0, -1])
