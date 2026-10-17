"""Surface topology must be independent of the display observation cadence."""

from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from blademaster_warp.geometry import points
from blademaster_warp.surface_runtime import SurfaceRecorder


def scene():
    cfg = SimpleNamespace(
        size=(0.024, 0.012, 0.012), grid_spacing=0.006, shape="block", blade_half_size=(0.0005, 0.02, 0.01)
    )
    ref, origin, _ = points(cfg.size, cfg.grid_spacing, cfg.shape)
    solver = SimpleNamespace(config=cfg, reference=ref, origin=origin, device="cuda:0")
    return solver, np.tile(np.eye(3), (len(ref), 1, 1))


def pose(edge_z, x=0.0, yaw=0.0):
    return np.r_[x, 0, edge_z + 0.01, Rotation.from_euler("z", yaw, degrees=True).as_quat()]


def tool(p):
    return SimpleNamespace(position=p[:3], orientation_xyzw=p[3:])


def count_components(recorder):
    mesh = recorder.surface.mesh()
    edges = np.concatenate([mesh.triangles[:, [0, 1]], mesh.triangles[:, [1, 2]], mesh.triangles[:, [2, 0]]])
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(mesh.reference), len(mesh.reference))
    )
    return connected_components(graph, directed=False, return_labels=False)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_complete_cut_and_withdrawal_between_display_frames(backend):
    if backend == "cuda":
        import warp as wp

        if not wp.is_cuda_available():
            pytest.skip("CUDA required")
    solver, F = scene()
    ref = solver.reference
    surface = SurfaceRecorder(solver, None, embedding_backend=backend, ccd_backend=backend)
    surface.update(0, ref, F, np.zeros(len(ref), np.int64), tool(pose(0.02)), False)
    surface.record_trajectory([0.001, 0.019], np.array([pose(0.02), pose(-0.003)]), True, 0)
    surface.record_trajectory([0.020, 0.039], np.array([pose(-0.003), pose(0.02)]), False, 0)
    codes = np.where(ref[:, 0] < 0, 1, 2)
    surface.update(0.04, ref, F, codes, tool(pose(0.02)), False)
    assert surface.surface.broken.sum() == 16
    assert count_components(surface) == 2
    assert np.any(surface.last_path[:, 8])


def test_two_different_cuts_between_frames_match_observing_every_sample():
    solver, F = scene()
    ref = solver.reference
    codes = np.where(ref[:, 0] < 0, 1, 2) + 3 * np.where(ref[:, 1] < 0, 1, 2)
    dense, sparse = SurfaceRecorder(solver, None), SurfaceRecorder(solver, None)
    for recorder in (dense, sparse):
        recorder.update(0, ref, F, np.zeros(len(ref), np.int64), tool(pose(0.02)), False)
    samples = [
        (0.001, pose(0.02), True, 0),
        (0.010, pose(-0.003), True, 0),
        (0.011, pose(-0.003), False, 0),
        (0.019, pose(0.02), False, 0),
        (0.020, pose(0.02, yaw=90), False, 1),
        (0.021, pose(0.02, yaw=90), True, 1),
        (0.030, pose(-0.003, yaw=90), True, 1),
        (0.031, pose(-0.003, yaw=90), False, 1),
        (0.039, pose(0.02, yaw=90), False, 1),
    ]
    for t, p, active, index in samples:
        sparse.record_trajectory([t], p[None], active, index)
        dense.update(t, ref, F, codes, tool(p), active, index)
    sparse.update(0.04, ref, F, codes, tool(pose(0.02, yaw=90)), False, 1)
    np.testing.assert_array_equal(sparse.surface.broken, dense.surface.broken)
    np.testing.assert_array_equal(sparse.surface.codes, dense.surface.codes)
    assert count_components(sparse) == count_components(dense) == 4


def test_path_around_material_does_not_cut_the_straight_endpoint_shortcut():
    solver, F = scene()
    ref, codes = solver.reference, np.zeros(len(solver.reference), np.int64)
    surface = SurfaceRecorder(solver, None)
    surface.update(0, ref, F, codes, tool(pose(0.0, x=-0.06)), False)
    path = np.array([pose(0.0, x=-0.06), pose(0.06, x=-0.06), pose(0.06, x=0.06), pose(0.0, x=0.06)])
    surface.record_trajectory([0.001, 0.010, 0.020, 0.030], path, True, 0)
    surface.update(0.04, ref, F, codes, tool(path[-1]), False)
    assert not surface.surface.broken.any()
    assert count_components(surface) == 1


def test_compressed_stationary_tool_still_cuts_material_moving_past_it():
    solver, F = scene()
    ref = solver.reference
    surface = SurfaceRecorder(solver, None)
    p = pose(0.02)
    surface.update(0, ref, F, np.zeros(len(ref), np.int64), tool(p), False)
    times = np.linspace(0.001, 0.04, 100)
    surface.record_trajectory(times, np.tile(p, (len(times), 1)), True, 0)
    assert len(surface._tool_samples) <= 3
    codes = np.where(ref[:, 0] < 0, 1, 2)
    surface.update(0.04, ref + [0, 0, 0.025], F, codes, tool(p), False)
    assert surface.surface.broken.any()
    assert count_components(surface) == 2
