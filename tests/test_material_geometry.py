from itertools import product
import numpy as np
from blademaster_warp.check_material_geometry import lattice_tetrahedra, frame_geometry


def test_affine_rotation_and_stretch_preserve_probe_orientation():
    ref = np.array(list(product((0., 1., 2.), repeat=3)))
    tet = lattice_tetrahedra(ref, 1.)
    A = np.array([[0., -1.2, 0.], [.8, 0., 0.], [0., 0., 1.1]])
    result = frame_geometry(ref, ref@A.T+5, np.zeros(len(ref), int), tet, np.tile(A, (len(ref), 1, 1)))
    assert len(tet) == 48
    assert result['inverted_probes'] == 0
    np.testing.assert_allclose(result['signed_volume_ratio_quantiles'], np.linalg.det(A))


def test_position_fold_detected_even_with_identity_stored_F():
    ref = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    moved = ref.copy(); moved[3, 2] = -1
    result = frame_geometry(ref, moved, np.ones(4, int), np.array([[0, 1, 2, 3]]), np.tile(np.eye(3), (4, 1, 1)))
    assert result['inverted_probes'] == result['assigned_inverted_probes'] == 1
    assert result['minimum_signed_volume_ratio'] == -1
    assert result['minimum_det_F'] == 1


def test_probes_across_cut_are_excluded():
    ref = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    moved = ref.copy(); moved[3, 2] = -1
    result = frame_geometry(ref, moved, np.array([1, 1, 1, 2]), np.array([[0, 1, 2, 3]]), np.tile(np.eye(3), (4, 1, 1)))
    assert result['probes'] == result['inverted_probes'] == 0
