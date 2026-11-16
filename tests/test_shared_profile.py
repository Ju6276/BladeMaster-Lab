import numpy as np
import warp as wp
from blademaster_warp.profile_geometry import profile_query
from blademaster_warp.profile_tool import vertex
from blademaster_warp.solver import Config
from blademaster_warp.profile_runtime import profile_kernel
from blademaster_warp import contact

HALF = (0.00175, 0.075, 0.02952)
BEVEL = 0.00004
NECK = 0.00004


@wp.kernel
def query(
    points: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
    dist: wp.array(dtype=float),
    normal: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    p, n, d = profile_query(points[i], wp.vec3(0.00175, 0.075, 0.02952), 0.00004, 0.00004)
    out[i] = p
    normal[i] = n
    dist[i] = d


@wp.kernel
def verts(out: wp.array(dtype=wp.vec3)):
    out[wp.tid()] = vertex(wp.tid(), wp.vec3(0.00175, 0.075, 0.02952), 0.00004, 0.00004)


def test_two_slope_contact_signed_distance_and_normals():
    slope = (HALF[0] - NECK) / (2 * HALF[2] - BEVEL)
    n = np.array([1, 0, -slope])
    n /= np.linalg.norm(n)
    surface = np.array([NECK + slope * HALF[2], 0, BEVEL])
    points = np.array(
        [surface + 0.0002 * n, surface - 0.0001 * n, [0, 0.076, 0], [0, 0, -0.03052], [0, 0, 0.03052]],
        np.float32,
    )
    a = wp.array(points, dtype=wp.vec3, device="cpu")
    out = wp.zeros(5, dtype=wp.vec3, device="cpu")
    dist = wp.zeros(5, device="cpu")
    normal = wp.zeros(5, dtype=wp.vec3, device="cpu")
    wp.launch(query, 5, [a, out, dist, normal], device="cpu")
    np.testing.assert_allclose(dist.numpy(), [0.0002, -0.0001, 0.001, 0.001, 0.001], atol=2e-8)
    np.testing.assert_allclose(normal.numpy()[0], n, atol=1e-5)
    np.testing.assert_allclose(out.numpy()[0], surface, atol=1e-7)
    np.testing.assert_allclose(points - out.numpy(), dist.numpy()[:, None] * normal.numpy(), atol=1e-7)


def test_vertices_include_thin_shoulder_and_full_spine():
    a = wp.zeros(10, dtype=wp.vec3, device="cpu")
    wp.launch(verts, 10, [a], device="cpu")
    v = a.numpy()
    np.testing.assert_allclose(v[[1, 4], 0], [NECK, -NECK], atol=1e-9)
    np.testing.assert_allclose(v[[2, 3], 0], [HALF[0], -HALF[0]], atol=1e-9)
    assert v[:, 2].max() - v[:, 2].min() > 0.059


def test_default_dispatch_is_unchanged_and_profile_is_opt_in():
    cfg = Config()
    replacement, args = profile_kernel(contact.rigid_and_speed, [], cfg)
    assert replacement is None and args == []
    from dataclasses import replace

    cfg = replace(cfg, blade_half_size=HALF, blade_bevel_height=BEVEL, blade_edge_half_width=NECK)
    replacement, args = profile_kernel(contact.rigid_and_speed, [], cfg)
    assert replacement.module.name.endswith("profile_contact") and args == [NECK]


def test_nonconvex_profile_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="convex"):
        Config(blade_half_size=HALF, blade_bevel_height=0.01, blade_edge_half_width=1e-7)
