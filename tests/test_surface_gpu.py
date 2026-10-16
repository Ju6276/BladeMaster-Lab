import numpy as np
import pytest
import warp as wp
from blademaster_warp.geometry import points
from blademaster_warp.surface import LinkedSurface, MaterialEmbedding
from blademaster_warp.surface_gpu import CudaMaterialEmbedding


@pytest.fixture
def embeddings():
    if not wp.is_cuda_available():
        pytest.skip('CUDA required')
    size = (.06, .03, .03)
    surface = LinkedSurface(size, .003)
    ref, origin, cells = points(size, .006, 'block')
    cpu = MaterialEmbedding(surface.reference, ref, .006, origin, cells)
    return surface, ref, cpu, CudaMaterialEmbedding.from_reference(cpu)


def test_cuda_nodal_fields_match_cpu_for_intersecting_cut_histories(embeddings):
    surface, ref, cpu, gpu = embeddings
    rng = np.random.default_rng(414)
    F = np.tile(np.eye(3), (len(ref), 1, 1))+rng.normal(0, .03, (len(ref), 3, 3))
    for axis in [0, 1]:
        codes = np.where(ref[:, 0] < 0, 1, 2)+3*np.where(ref[:, axis] < .002, 1, 2)
        node_codes = np.where(surface.reference[:, 0] < 0, 1, 2)
        x = ref+rng.normal(0, .002, ref.shape)
        expected = cpu.node_state(x, F, codes, node_codes)
        actual = gpu.node_state(x, F, codes, node_codes)
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(a, b, rtol=1.e-12, atol=1.e-12, equal_nan=True)
        for code in [0, 1, 2, 4, 5, 7, 8]:
            cpu.extension_nodes = gpu.extension_nodes = 0
            np.testing.assert_allclose(cpu.field(x, F, codes, code), gpu.field(x, F, codes, code),
                                       rtol=1.e-12, atol=1.e-12, equal_nan=True)
            assert cpu.extension_nodes == gpu.extension_nodes


def test_cuda_embedding_separates_both_cut_faces_like_cpu(embeddings):
    surface, ref, cpu, gpu = embeddings
    surface.sweep([[0, -.04, .04], [0, .04, .04]], [[0, -.04, -.01], [0, .04, -.01]])
    mesh = surface.mesh()
    codes = np.where(ref[:, 0] < 0, 1, 2)
    x = ref.copy(); x[:, 0] += np.where(codes == 1, -.004, .006)
    F = np.tile(np.eye(3), (len(ref), 1, 1))
    expected = cpu.vertices(mesh, x, F, codes)
    actual = gpu.vertices(mesh, x, F, codes)
    np.testing.assert_allclose(actual, expected, rtol=1.e-12, atol=1.e-12)
    assert gpu.material_grid.extension_nodes == cpu.material_grid.extension_nodes
    for code, shift in [(1, -.004), (2, .006)]:
        select = mesh.codes == code
        np.testing.assert_allclose(actual[select, 0], mesh.reference[select, 0]+shift, atol=1.e-12)


def test_cuda_empty_support_stays_invalid_instead_of_freezing_or_zeroing(embeddings):
    surface, ref, cpu, gpu = embeddings
    x = ref.copy(); F = np.tile(np.eye(3), (len(ref), 1, 1))
    # Code 1 opposes every particle in this deliberately unsupported region.
    codes = np.full(len(ref), 2)
    assert np.isnan(gpu.field(x, F, codes, 1)).all()
    with pytest.raises(ValueError):
        gpu.field(x, F, codes, 3**20)
