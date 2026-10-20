"""Physics invariants and FP64 checks of the paper equations, executed on CUDA."""

from dataclasses import replace
import numpy as np
import pytest
import warp as wp
from blademaster_warp.solver import Config, Solver
from blademaster_warp import material as m, contact as c
from blademaster_warp.topology import compatible


@wp.kernel
def query_kernel(
    x: wp.array(dtype=wp.vec3),
    half: wp.vec3,
    bevel: float,
    point: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    distance: wp.array(dtype=float),
):
    p = wp.tid()
    a, b, d = c.blade_query(x[p], half, bevel)
    point[p] = a
    normal[p] = b
    distance[p] = d


@wp.kernel
def friction_kernel(
    v: wp.array(dtype=wp.vec3),
    h: wp.array(dtype=wp.vec3),
    j: wp.array(dtype=wp.vec3),
    hn: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    a, b = c.friction(wp.vec3(0.0, 0.0, 1.0), v[p], 0.003, h[p], 0.02, 0.001, 0.3, 0.2, 0.05, 0.1)
    j[p] = a
    hn[p] = b


def arr(x, dtype):
    return wp.array(np.asarray(x), dtype=dtype, device="cuda:0")


def exchange_reference(codes, masses, states):
    """Paper Eq. 8 at one node: s_a + sum_b compatible (m_a s_b - m_b s_a) / M."""
    total = masses.sum()
    result = states.astype(float).copy()
    for a in range(len(codes)):
        for b in range(len(codes)):
            if a != b and compatible(codes[a], codes[b]):
                result[a] += (masses[a] * states[b] - masses[b] * states[a]) / total
    return result


def wedge_planes(half, bevel):
    """Outward unit normals and offsets of the convex blade, n . x <= offset."""
    hx, hy, hz = (float(v) for v in half)
    planes = [
        ((1, 0, 0), hx),
        ((-1, 0, 0), hx),
        ((0, 1, 0), hy),
        ((0, -1, 0), hy),
        ((0, 0, 1), hz),
    ]
    if bevel == 0.0:
        planes.append(((0, 0, -1), hz))
    else:
        # The edge is the line x = 0, z = -hz; each bevel rises to x = +-hx at z = bevel - hz.
        length = np.hypot(bevel, hx)
        planes += [((bevel / length, 0, -hx / length), hx * hz / length)]
        planes += [((-bevel / length, 0, -hx / length), hx * hz / length)]
    normals = np.array([p[0] for p in planes], float)
    offsets = np.array([p[1] for p in planes], float)
    return normals, offsets


def wedge_vertices(half, bevel):
    hx, hy, hz = (float(v) for v in half)
    if bevel == 0.0:
        return np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = []
    for y in (-hy, hy):
        corners += [[0.0, y, -hz], [hx, y, bevel - hz], [-hx, y, bevel - hz], [hx, y, hz], [-hx, y, hz]]
    return np.array(corners)


def test_partition_and_reference_gradient():
    s = Solver()
    np.testing.assert_allclose(s.weights.numpy().sum(1), 1, atol=2e-7)
    np.testing.assert_allclose(s.gradients.numpy().sum(1), 0, atol=2e-5)


def test_sparse_exchange_matches_equation_and_conserves_momentum():
    s = Solver()
    # One node, several histories: 1 vs 2 separated, 0 bridges, 4 vs 7 conflict.
    codes = np.array([0, 1, 2, 4, 7], np.int32)
    masses = np.array([0.2, 0.5, 0.7, 0.3, 0.4], np.float32)
    states = np.random.default_rng(3).normal(size=(5, 3)).astype(np.float32)
    cap = 16
    keys = np.full(cap, -1, np.int64)
    keys[:5] = codes
    mass = np.zeros(cap, np.float32)
    mass[:5] = masses
    state = np.zeros((cap, 3), np.float32)
    state[:5] = states
    heads = arr([0], int)
    links = arr([1, 2, 3, 4, -1] + [-1] * 11, int)
    result = wp.zeros(cap, dtype=wp.vec3, device="cuda:0")
    s.launch(
        m.couple, cap, [arr(keys, wp.int64), heads, links, arr(mass, float), arr(state, wp.vec3), result, 1]
    )
    expected = exchange_reference(codes, masses.astype(float), states.astype(float))
    np.testing.assert_allclose(result.numpy()[:5], expected, atol=4e-7)
    np.testing.assert_allclose(result.numpy()[:5].sum(0), states.sum(0), atol=8e-7)
    # A common velocity carried by every field is left unchanged.
    common = masses[:, None] * np.array([0.3, -0.1, 0.2], np.float32)
    state[:5] = common
    s.launch(
        m.couple, cap, [arr(keys, wp.int64), heads, links, arr(mass, float), arr(state, wp.vec3), result, 1]
    )
    np.testing.assert_allclose(result.numpy()[:5], common, atol=2e-7)


@pytest.mark.parametrize("bevel", [0.0, 0.005])
def test_blade_distance_and_normal_against_fp64(bevel):
    rng = np.random.default_rng(2)
    x = rng.uniform([-0.003, -0.027, -0.025], [0.003, 0.027, 0.025], (400, 3)).astype(np.float32)
    half = np.array([0.0005, 0.025, 0.02], np.float32)
    pts = wp.zeros(len(x), dtype=wp.vec3, device="cuda:0")
    normals = wp.zeros_like(pts)
    dist = wp.zeros(len(x), dtype=float, device="cuda:0")
    wp.launch(
        query_kernel, len(x), [arr(x, wp.vec3), wp.vec3(*half), bevel, pts, normals, dist], device="cuda:0"
    )
    point, normal, distance = pts.numpy().astype(float), normals.numpy().astype(float), dist.numpy()
    planes, offsets = wedge_planes(half, bevel)
    corners = wedge_vertices(half, bevel)
    x = x.astype(float)
    signed = x @ planes.T - offsets
    outside = (signed > 0).any(axis=1)
    assert outside.sum() >= 20 and (~outside).sum() >= 20
    # Outside: the returned point is on the blade, nearest by the convex
    # optimality condition (x - p) . (v - p) <= 0 for every vertex v.
    gap = x[outside] - point[outside]
    np.testing.assert_allclose(distance[outside], np.linalg.norm(gap, axis=1), atol=6e-9)
    assert (point[outside] @ planes.T - offsets).max() <= 6e-9
    optimality = np.einsum("pi,pvi->pv", gap, corners[None] - point[outside][:, None])
    assert optimality.max() <= 1e-10
    np.testing.assert_allclose(normal[outside], gap / distance[outside, None], atol=5e-4)
    # Inside: depth is the smallest distance to a face, along that face's normal.
    depth = -signed[~outside]
    face = depth.argmin(axis=1)
    np.testing.assert_allclose(-distance[~outside], depth.min(axis=1), atol=6e-9)
    np.testing.assert_allclose(normal[~outside], planes[face], atol=5e-4)


def friction_reference(n, relative, jn, history, me, dt, mus, mud, gk, gc):
    """Paper Eqs. 12-13: tangential spring-dashpot capped by Coulomb friction."""
    vt = relative - (relative @ n) * n
    ht = history - (history @ n) * n
    k = gk * me / dt**2
    damping = 2 * gc * np.sqrt(me * k)
    trial = (-k * ht - damping * vt) * dt
    magnitude = np.linalg.norm(trial)
    if magnitude <= mus * jn:
        return trial, ht + dt * vt
    applied = trial * (mud * jn / magnitude)
    return applied, -(applied / dt + damping * vt) / k


def test_friction_stick_and_slip_history():
    v = np.array([[0.001, 0, 0], [1.0, 0.1, 0.3], [0, 0, 0]], np.float32)
    h = np.array([[0, 0, 0], [0.0002, 0.0003, 0], [0, 0, 0.2]], np.float32)
    j = wp.zeros(3, dtype=wp.vec3, device="cuda:0")
    hn = wp.zeros_like(j)
    wp.launch(friction_kernel, 3, [arr(v, wp.vec3), arr(h, wp.vec3), j, hn], device="cuda:0")
    expected = [
        friction_reference(
            np.array([0.0, 0.0, 1.0]),
            v[i].astype(float),
            0.003,
            h[i].astype(float),
            0.02,
            0.001,
            0.3,
            0.2,
            0.05,
            0.1,
        )
        for i in range(3)
    ]
    np.testing.assert_allclose(j.numpy(), [a[0] for a in expected], atol=1e-9)
    np.testing.assert_allclose(hn.numpy(), [a[1] for a in expected], atol=1e-9)
    assert np.linalg.norm(j.numpy()[1]) == pytest.approx(0.2 * 0.003, rel=2e-6)


def test_beveled_blade_face_normal_remains_exact_near_contact():
    # A segment projection loses z precision near a planar face; dividing that
    # error by a nanometre-scale gap creates spurious tangential normals.
    half = np.array([0.0005, 0.04, 0.02], np.float32)
    x = np.array(
        [
            [sign * (half[0] + gap), y, z]
            for sign in (-1, 1)
            for gap in (1e-8, 1e-7, 1e-6)
            for y in (-0.03, 0.01)
            for z in (-0.01476572, -0.005, 0.00126425, 0.019)
        ],
        np.float32,
    )
    pts = wp.zeros(len(x), dtype=wp.vec3, device="cuda:0")
    normals = wp.zeros_like(pts)
    distance = wp.zeros(len(x), dtype=float, device="cuda:0")
    wp.launch(
        query_kernel,
        len(x),
        [arr(x, wp.vec3), wp.vec3(*half), 0.005, pts, normals, distance],
        device="cuda:0",
    )
    expected_points = x.copy()
    expected_points[:, 0] = np.sign(x[:, 0]) * half[0]
    expected_normals = np.zeros_like(x)
    expected_normals[:, 0] = np.sign(x[:, 0])
    np.testing.assert_array_equal(pts.numpy(), expected_points)
    np.testing.assert_allclose(normals.numpy(), expected_normals, atol=1e-7)
    np.testing.assert_array_equal(distance.numpy(), np.abs(x[:, 0]) - half[0])


@pytest.mark.parametrize("viscosity", [0.0, 2.0])
def test_material_step_fp64_and_momentum(viscosity):
    s = Solver(replace(Config(), dynamic_viscosity=viscosity))
    rng = np.random.default_rng(4)
    vel = rng.normal(scale=0.01, size=(s.n, 3)).astype(np.float32)
    F = np.tile(np.diag([1.01, 0.995, 1.0]), (s.n, 1, 1)).astype(np.float32)
    codes = np.where(s.reference[:, 0] < 0, 1, 2).astype(np.int32)
    impulse = rng.normal(scale=1e-8, size=(s.n, 3)).astype(np.float32)
    s.v.assign(vel)
    s.F.assign(F)
    s.refresh_stress()
    s.codes.assign(codes)
    s.impulses.assign(impulse)
    before = s.mass * vel.astype(float).sum(0)
    s.material_step()
    out = s.state()
    np.testing.assert_allclose(
        s.mass * out["velocity"].astype(float).sum(0), before + impulse.astype(float).sum(0), atol=2e-9
    )


@pytest.mark.parametrize("split", [False, True])
def test_rigid_translation_is_exact(split):
    # Uniform velocity, unstrained material and no external impulse: every
    # field carries the same velocity, so Eqs. 11-16 move x by v dt and leave
    # v and F unchanged, whether or not the block is cut.
    s = Solver()
    velocity = np.array([0.02, -0.01, 0.005], np.float32)
    s.v.assign(np.tile(velocity, (s.n, 1)))
    s.F.assign(np.tile(np.eye(3, dtype=np.float32), (s.n, 1, 1)))
    s.refresh_stress()
    if split:
        s.codes.assign(np.where(s.reference[:, 0] < 0, 1, 2).astype(np.int32))
    s.impulses.zero_()
    x = s.x.numpy().astype(float)
    s.material_step()
    out = s.state()
    np.testing.assert_allclose(out["position"], x + velocity * s.config.dt, atol=1e-8)
    np.testing.assert_allclose(out["velocity"], np.tile(velocity, (s.n, 1)), atol=1e-6)
    np.testing.assert_allclose(out["deformation"], np.tile(np.eye(3), (s.n, 1, 1)), atol=1e-6)


def test_blade_grid_constraint_switch():
    assert Solver(Config(external_tool=True)).blade_constraint is not None
    assert Solver(Config(external_tool=True, blade_grid_constraint=False)).blade_constraint is None


def test_release_is_per_cut_and_per_knife():
    s = Solver(Config(free_blade=True, tool_count=2))
    s.tool_damage[1].fill_(1.0)
    s.assign_sides(0)
    np.testing.assert_array_equal(s.cut_released.numpy(), [0, 1])
    s._set_cut(2)
    np.testing.assert_array_equal(s.cut_released.numpy(), [0, 0])
    s.assign_sides(2)
    np.testing.assert_array_equal(s.cut_released.numpy(), [0, 0])


def test_cut_labels_persist_withdrawal_and_new_cut():
    s = Solver()
    s.damage.fill_(1.0)
    s.tool_pos.assign([[0.0, 0.0, 0.03]])
    s.launch(
        c.assign,
        s.n,
        [
            s.x,
            s.v,
            s.codes,
            s.tool_pos,
            s.tool_vel,
            s.tool_rot,
            s.tool_omega,
            s.half,
            s.config.grid_spacing,
            s.radius,
            s.board_half,
            s.config.dt,
            1,
            1,
            s.damage,
            s.cut_released,
            0,
            1,
            s.topology_dirty,
        ],
    )
    first = s.codes.numpy()
    assert set(np.unique(first)) == {0, 1, 2}
    s.tool_pos.assign([[0.02, 0.0, 0.1]])
    s.launch(
        c.assign,
        s.n,
        [
            s.x,
            s.v,
            s.codes,
            s.tool_pos,
            s.tool_vel,
            s.tool_rot,
            s.tool_omega,
            s.half,
            s.config.grid_spacing,
            s.radius,
            s.board_half,
            s.config.dt,
            1,
            1,
            s.damage,
            s.cut_released,
            0,
            1,
            s.topology_dirty,
        ],
    )
    np.testing.assert_array_equal(s.codes.numpy(), first)
    s.tool_pos.assign([[0.012, 0.0, 0.03]])
    s.launch(
        c.assign,
        s.n,
        [
            s.x,
            s.v,
            s.codes,
            s.tool_pos,
            s.tool_vel,
            s.tool_rot,
            s.tool_omega,
            s.half,
            s.config.grid_spacing,
            s.radius,
            s.board_half,
            s.config.dt,
            3,
            1,
            s.damage,
            s.cut_released,
            0,
            1,
            s.topology_dirty,
        ],
    )
    np.testing.assert_array_equal(s.codes.numpy() % 3, first)
    assert s.codes.numpy().max() > 2


def test_von_mises_projection_preserves_total_and_volume():
    s = Solver()
    total = np.tile(np.diag([1.3, 1 / 1.3, 1.0]), (s.n, 1, 1)).astype(np.float32)
    s.F.assign(total)
    s.launch(m.plastic_project, s.n, [s.F, s.Fp, s.mu, 100.0, s.errors])
    fp = s.Fp.numpy()
    np.testing.assert_array_equal(s.F.numpy(), total)
    np.testing.assert_allclose(np.linalg.det(fp), 1, atol=3e-6)
    fe = total @ np.linalg.inv(fp)
    logs = np.log(np.linalg.svd(fe, compute_uv=False))
    dev = logs - logs.mean(1)[:, None]
    np.testing.assert_allclose(np.linalg.norm(dev, axis=1), 100 / (2 * s.mu) * np.sqrt(2 / 3), atol=2e-6)


@pytest.mark.parametrize("split", [False, True])
def test_pair_recontact_reference_exclusion_and_equal_opposite_impulse(split):
    x = np.array([[0.0, 0.0, 0.05], [0.002, 0.0, 0.05]], np.float32)
    vel = np.array([[0.01, 0.003, 0], [-0.01, -0.003, 0]], np.float32)
    codes = np.array([1, 2] if split else [0, 0], np.int32)
    radius = 0.0015
    spacing = 0.006
    mass = 0.00003
    dt = 0.00025
    xd = arr(x, wp.vec3)
    vd = arr(vel, wp.vec3)
    cd = arr(codes, wp.int64)
    grid = wp.HashGrid(16, 16, 16, device="cuda:0")
    grid.build(xd, 2 * radius)
    keys0 = wp.full(128, -1, dtype=wp.int64, device="cuda:0")
    keys1 = wp.full_like(keys0, -1)
    h0 = wp.zeros(128, dtype=wp.vec3, device="cuda:0")
    h1 = wp.zeros_like(h0)
    out = wp.zeros(2, dtype=wp.vec3, device="cuda:0")
    err = wp.zeros(3, dtype=int, device="cuda:0")
    used = wp.zeros(128, dtype=int, device="cuda:0")
    count = wp.zeros(1, dtype=int, device="cuda:0")
    speed = arr([np.linalg.norm(vel, axis=1).max()], float)
    wp.launch(
        c.pairs,
        2,
        [
            grid.id,
            xd,
            xd,
            vd,
            cd,
            radius,
            spacing,
            mass,
            dt,
            0.1,
            0.3,
            0.2,
            0.05,
            0.1,
            speed,
            keys0,
            h0,
            keys1,
            h1,
            used,
            count,
            128,
            2,
            out,
            err,
        ],
        device="cuda:0",
    )
    impulse = out.numpy().astype(float)
    np.testing.assert_allclose(impulse.sum(0), 0, atol=1e-12)
    # Same-history neighbours within 2 dx in the reference are not in contact
    # (paper SIII-E); opposite cut sides are, and the approaching pair is pushed apart.
    assert bool(np.any(impulse)) == split
    if split:
        assert impulse[0, 0] < 0 < impulse[1, 0]
        # Eq. 11 normal penalty sn m_eff / dt^2 times overlap, with m_eff = m / 2, applied for dt.
        overlap = 2 * radius - (x[1, 0] - x[0, 0])
        normal = 0.1 * (mass / 2) / dt**2 * overlap * dt
        assert abs(impulse[1, 0]) >= normal * (1 - 1e-4)
    filled = np.flatnonzero(keys1.numpy() != -1)
    np.testing.assert_array_equal(np.sort(used.numpy()[: count.numpy()[0]]), filled)
    wp.launch(c.clear_pair_slots, 4, [count, 4, used, keys1], device="cuda:0")
    assert (keys1.numpy() == -1).all()


def test_rigid_reaction_is_equal_opposite_with_rotated_blade():
    cfg = replace(Config(), free_blade=True)
    s = Solver(cfg)
    theta = 0.3
    q = np.array([0, np.sin(theta / 2), 0, np.cos(theta / 2)], np.float32)
    s.tool_rot.assign([q])
    s.tool_pos.assign([[0.0, 0.0, 0.02]])
    s.reaction.zero_()
    s.torque.zero_()
    s.board.zero_()
    s.rigid_contacts(cutting=False)
    gravity = np.array([0.0, 0.0, -9.81 * s.mass * cfg.dt]) * s.n
    material = s.impulses.numpy().astype(float).sum(0)
    np.testing.assert_allclose(material + s.reaction.numpy()[0] - s.board.numpy()[0] - gravity, 0, atol=3e-9)
    assert np.linalg.norm(s.reaction.numpy()) > 0
