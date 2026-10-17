import numpy as np
import pytest
import warp as wp
from blademaster_warp.surface import moving_segment_hits
from blademaster_warp.surface_ccd_gpu import CudaSegmentCCD, moving_hits


@pytest.fixture
def require_cuda():
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")


def compare(detector, endpoints, blade):
    expected = moving_segment_hits(*endpoints, *blade)
    actual = detector(*endpoints, *blade)
    np.testing.assert_array_equal(actual[0], expected[0])
    selected = expected[0]
    np.testing.assert_allclose(actual[1][selected], expected[1][selected], atol=1.0e-8, rtol=1.0e-8)
    np.testing.assert_allclose(actual[2][selected], expected[2][selected], atol=1.0e-8, rtol=1.0e-8)


def test_ccd_random_general_moving_segments(kernel_detector):
    rng = np.random.default_rng(471)
    for _ in range(5):
        endpoints = [rng.uniform(-0.05, 0.05, (300, 3)) for _ in range(4)]
        blade = rng.uniform(-0.05, 0.05, (2, 2, 3))
        compare(kernel_detector, endpoints, blade)


def test_ccd_coplanar_motion_and_stationary_blade(kernel_detector):
    rng = np.random.default_rng(830)
    endpoints = [rng.uniform(-0.05, 0.05, (300, 3)) for _ in range(4)]
    for v in endpoints:
        v[:, 2] = 0
    blade = np.array([[[0, -0.04, 0], [0, 0.04, 0]], [[0, -0.04, 0], [0, 0.04, 0]]])
    compare(kernel_detector, endpoints, blade)
    a = np.array([[-0.03, 0, 0.01], [-0.03, 0.06, 0.01], [0, -0.03, 0.01]])
    b = np.array([[0.03, 0, 0.01], [0.03, 0.06, 0.01], [0, 0.03, 0.01]])
    compare(kernel_detector, [a, b, a - [0, 0, 0.02], b - [0, 0, 0.02]], blade)


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("reverse_blade", [False, True])
def test_ccd_parallel_material_links_cross_stationary_blade(kernel_detector, scale, reverse_blade):
    a = scale * np.array([[0, -0.03, 0.01], [0, 0.05, 0.01], [0, 0.04, 0.01]])
    b = a + scale * np.array([0, 0.06, 0])
    motion = scale * np.array([0, 0, -0.02])
    edge = scale * np.array([[0, -0.04, 0], [0, 0.04, 0]])
    if reverse_blade:
        edge = edge[::-1].copy()
    hit, alpha, times = kernel_detector(a, b, a + motion, b + motion, edge, edge)
    np.testing.assert_array_equal(hit, [True, False, True])
    np.testing.assert_allclose(times[hit], 0.5, atol=1e-12, rtol=0)
    np.testing.assert_allclose(alpha[hit], 0.0, atol=1e-12, rtol=0)


def test_ccd_nearly_quadratic_cut_does_not_lose_root(kernel_detector):
    # Actual finite link/knife motion from the shared banana regression. The
    # cubic coefficient is ~2e-17 while the linear one is ~-2e-7. Closed-form
    # cubic evaluation previously cancelled the in-interval root and missed it.
    endpoints = np.array(
        [
            [-0.002278456830274086, -0.022483942322732757, 0.022480641450338228],
            [0.0007241473390662562, -0.022501063925497878, 0.022478243607065632],
            [-0.0021514921848087448, -0.022501927741807138, 0.02250257799442088],
            [0.0008677817516989291, -0.022518758706787233, 0.022501242707334095],
        ]
    )[:, None, :]
    blade = np.array(
        [
            [
                [-0.0006557768977572494, -0.04003920669493038, 0.023002451290826013],
                [-0.0006598182774875667, 0.03996079317980191, 0.023000525132880964],
            ],
            [
                [-0.0006334223876866684, -0.040045556810240746, 0.022228944366594578],
                [-0.000636864261988937, 0.03995444310397632, 0.022227573683540672],
            ],
        ]
    )
    compare(kernel_detector, endpoints, blade)
    assert kernel_detector(*endpoints, *blade)[0][0]


def test_ccd_preserves_persistent_cut_surface_topology(kernel_detector):
    from blademaster_warp.surface import LinkedSurface

    surfaces = [LinkedSurface((0.06, 0.03, 0.03), 0.006) for _ in range(2)]
    for cut, normal, edge in [
        (0, [1, 0, 0], [[0, -0.04, 0], [0, 0.04, 0]]),
        (1, [0, 1, 0], [[-0.04, 0, 0], [0.04, 0, 0]]),
    ]:
        edge = np.asarray(edge)
        for s, detector in zip(surfaces, [moving_segment_hits, kernel_detector]):
            s.sweep(
                edge,
                edge,
                s.reference - [0, 0, 0.04],
                normal_world=normal,
                previous_nodes=s.reference + [0, 0, 0.01],
                cut_index=cut,
                ccd=detector,
            )
        np.testing.assert_array_equal(surfaces[0].broken, surfaces[1].broken)
        np.testing.assert_array_equal(surfaces[0].codes, surfaces[1].codes)
        meshes = [s.mesh() for s in surfaces]
        np.testing.assert_array_equal(meshes[0].triangles, meshes[1].triangles)
        np.testing.assert_allclose(meshes[0].reference, meshes[1].reference, atol=1.0e-12)


def test_cuda_ccd_reuses_growing_buffers_without_stale_results(require_cuda):
    detector = CudaSegmentCCD()
    rng = np.random.default_rng(81)
    for n in [1, 65, 3, 240, 0, 12]:
        endpoints = [rng.uniform(-0.05, 0.05, (n, 3)) for _ in range(4)]
        blade = rng.uniform(-0.05, 0.05, (2, 2, 3))
        expected = moving_segment_hits(*endpoints, *blade)
        actual = detector(*endpoints, *blade)
        np.testing.assert_array_equal(actual[0], expected[0])
        mask = expected[0]
        np.testing.assert_allclose(actual[1][mask], expected[1][mask], atol=1e-8)
        np.testing.assert_allclose(actual[2][mask], expected[2][mask], atol=1e-8)


def device_detector(device):
    if device == "cuda":
        if not wp.is_cuda_available():
            pytest.skip("CUDA required")
        return CudaSegmentCCD()

    def detect(a0, b0, a1, b1, edge0, edge1):
        inputs = [wp.array(v, dtype=wp.vec3d, device="cpu") for v in (a0, b0, a1, b1)]
        blade = wp.array(np.concatenate([edge0, edge1]), dtype=wp.vec3d, device="cpu")
        result = wp.empty(len(a0), dtype=wp.vec3d, device="cpu")
        wp.launch(moving_hits, len(a0), [*inputs, blade, result], device="cpu")
        values = result.numpy()
        return values[:, 0].astype(bool), values[:, 1], values[:, 2]

    return detect


@pytest.fixture(params=["warp-cpu", "cuda"])
def kernel_detector(request):
    return device_detector(request.param)


@pytest.fixture(params=["numpy", "warp-cpu", "cuda"])
def segment_detector(request):
    return moving_segment_hits if request.param == "numpy" else device_detector(request.param)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "link,start,end,first_hit",
    [
        ((0, 2), (1, 3), (1, 3), 0.0),
        ((0, 2), (3, 5), (-3, -1), 1 / 6),
        ((0, 2), (3, 5), (4, 6), None),
        ((0, 2), (2, 4), (2, 4), 0.0),
        ((0, 2), (1, 1), (1, 1), 0.0),
        ((0, 2), (3, 3), (-1, -1), 0.25),
        ((1, 1), (0, 2), (0, 2), 0.0),
        ((1, 1), (2, 2), (0, 0), 0.5),
    ],
)
def test_collinear_ccd_finds_bounded_overlap_and_first_contact(
    segment_detector, reverse, link, start, end, first_hit
):
    def endpoints(values):
        values = values[::-1] if reverse else values
        return np.array([[v, 0, 0] for v in values], dtype=float)

    a, b = endpoints(link)[:, None, :]
    edge0, edge1 = endpoints(start), endpoints(end)
    hit, alpha, times = segment_detector(a, b, a, b, edge0, edge1)
    assert hit[0] == (first_hit is not None)
    if first_hit is not None:
        assert times[0] == pytest.approx(first_hit, abs=1e-12)
        assert 0 <= alpha[0] <= 1
        point = ((1 - alpha[0]) * a + alpha[0] * b)[0]
        edge = (1 - times[0]) * edge0 + times[0] * edge1
        assert edge[:, 0].min() - 1e-12 <= point[0] <= edge[:, 0].max() + 1e-12
