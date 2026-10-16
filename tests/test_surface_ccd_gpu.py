import numpy as np
import pytest
import warp as wp
from blademaster_warp.surface import moving_segment_hits
from blademaster_warp.surface_ccd_gpu import CudaSegmentCCD


@pytest.fixture(autouse=True)
def require_cuda():
    if not wp.is_cuda_available():
        pytest.skip('CUDA required')


def compare(endpoints, blade):
    expected = moving_segment_hits(*endpoints, *blade)
    actual = CudaSegmentCCD()(*endpoints, *blade)
    np.testing.assert_array_equal(actual[0], expected[0])
    selected = expected[0]
    np.testing.assert_allclose(actual[1][selected], expected[1][selected], atol=1.e-8, rtol=1.e-8)
    np.testing.assert_allclose(actual[2][selected], expected[2][selected], atol=1.e-8, rtol=1.e-8)


def test_cuda_ccd_random_general_moving_segments():
    rng = np.random.default_rng(471)
    for _ in range(5):
        endpoints = [rng.uniform(-.05, .05, (300, 3)) for _ in range(4)]
        blade = rng.uniform(-.05, .05, (2, 2, 3))
        compare(endpoints, blade)


def test_cuda_ccd_coplanar_motion_and_stationary_blade():
    rng = np.random.default_rng(830)
    endpoints = [rng.uniform(-.05, .05, (300, 3)) for _ in range(4)]
    for v in endpoints:
        v[:, 2] = 0
    blade = np.array([[[0, -.04, 0], [0, .04, 0]], [[0, -.04, 0], [0, .04, 0]]])
    compare(endpoints, blade)
    a = np.array([[-.03, 0, .01], [-.03, .06, .01], [0, -.03, .01]])
    b = np.array([[.03, 0, .01], [.03, .06, .01], [0, .03, .01]])
    compare([a, b, a-[0, 0, .02], b-[0, 0, .02]], blade)


def test_cuda_ccd_nearly_quadratic_cut_does_not_lose_root():
    # Actual finite link/knife motion from the shared banana regression. The
    # cubic coefficient is ~2e-17 while the linear one is ~-2e-7. Closed-form
    # cubic evaluation previously cancelled the in-interval root and missed it.
    endpoints = np.array([
        [-0.002278456830274086, -0.022483942322732757, 0.022480641450338228],
        [0.0007241473390662562, -0.022501063925497878, 0.022478243607065632],
        [-0.0021514921848087448, -0.022501927741807138, 0.02250257799442088],
        [0.0008677817516989291, -0.022518758706787233, 0.022501242707334095],
    ])[:, None, :]
    blade = np.array([
        [[-0.0006557768977572494, -0.04003920669493038, 0.023002451290826013],
         [-0.0006598182774875667, 0.03996079317980191, 0.023000525132880964]],
        [[-0.0006334223876866684, -0.040045556810240746, 0.022228944366594578],
         [-0.000636864261988937, 0.03995444310397632, 0.022227573683540672]],
    ])
    compare(endpoints, blade)
    assert CudaSegmentCCD()(*endpoints, *blade)[0][0]


def test_cuda_ccd_preserves_persistent_cut_surface_topology():
    from blademaster_warp.surface import LinkedSurface
    surfaces = [LinkedSurface((.06,.03,.03),.006) for _ in range(2)]
    for cut, normal, edge in [(0, [1,0,0], [[0,-.04,0],[0,.04,0]]),
                              (1, [0,1,0], [[-.04,0,0],[.04,0,0]])]:
        edge=np.asarray(edge)
        for s, detector in zip(surfaces, [moving_segment_hits, CudaSegmentCCD()]):
            s.sweep(edge,edge,s.reference-[0,0,.04],normal_world=normal,
                    previous_nodes=s.reference+[0,0,.01],cut_index=cut,ccd=detector)
        np.testing.assert_array_equal(surfaces[0].broken,surfaces[1].broken)
        np.testing.assert_array_equal(surfaces[0].codes,surfaces[1].codes)
        meshes=[s.mesh() for s in surfaces]
        np.testing.assert_array_equal(meshes[0].triangles,meshes[1].triangles)
        np.testing.assert_allclose(meshes[0].reference,meshes[1].reference,atol=1.e-12)
