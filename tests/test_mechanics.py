"""Physics invariants and FP64 independent-reference comparisons, executed on CUDA.

DexLab must be importable for reference comparisons. No reference solver runs
inside the GPU implementation itself.
"""

from dataclasses import asdict, replace
import numpy as np
import pytest
import warp as wp
from blademaster_warp.solver import Config, Solver
from blademaster_warp import material as m, contact as c
from dexlab.cutting import (
    CutConfig,
    CuttingSimulation,
    couple_grids,
    material_step,
    blade_query,
    friction_impulse,
    particle_contacts,
)


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


def paper_position_reference(x, v, f, *args):
    """FP64 comparison with an explicit correction to the legacy oracle.

    DexLab advects x with blended particle velocity. TLMPM Eq. 13 instead
    uses PIC grid velocity. A disposable pure-PIC call extracts that position;
    the normal call retains the original hybrid momentum and MUSL F update.
    No upstream source is edited, and this adapter is confined to tests.
    """
    xp, vp, fp = x.copy(), v.copy(), f.copy()
    pic_args = list(args)
    pic_args[10] = 0.0  # flip after sides,nodes,weights,gradients,impulse,m,V,mu,lam,dt
    material_step(xp, vp, fp, *pic_args)
    result = material_step(x, v, f, *args)
    x[:] = xp
    return result


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
    dense_m = np.zeros((8, 1))
    dense_s = np.zeros((8, 1, 3))
    dense_m[codes, 0] = masses
    dense_s[codes, 0] = states
    expected = couple_grids(dense_m, dense_s)[codes, 0]
    np.testing.assert_allclose(result.numpy()[:5], expected, atol=4e-7)
    np.testing.assert_allclose(result.numpy()[:5].sum(0), states.sum(0), atol=8e-7)


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
    ref = [blade_query(p.astype(float), np.zeros(3), half.astype(float), bevel) for p in x]
    np.testing.assert_allclose(pts.numpy(), [v[0] for v in ref], atol=6e-9)
    np.testing.assert_allclose(normals.numpy(), [v[1] for v in ref], atol=5e-4)
    np.testing.assert_allclose(dist.numpy(), [v[2] for v in ref], atol=6e-9)


def test_friction_stick_and_slip_history():
    v = np.array([[0.001, 0, 0], [1.0, 0.1, 0.3], [0, 0, 0]], np.float32)
    h = np.array([[0, 0, 0], [0.0002, 0.0003, 0], [0, 0, 0.2]], np.float32)
    j = wp.zeros(3, dtype=wp.vec3, device="cuda:0")
    hn = wp.zeros_like(j)
    wp.launch(friction_kernel, 3, [arr(v, wp.vec3), arr(h, wp.vec3), j, hn], device="cuda:0")
    expected = [
        friction_impulse(
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
    x = s.x.numpy().astype(float)
    v = vel.astype(float)
    f = F.astype(float)
    before = s.mass * v.sum(0)
    paper_position_reference(
        x,
        v,
        f,
        codes.astype(np.int64),
        s.nodes.numpy().astype(np.int64),
        s.weights.numpy().astype(float),
        s.gradients.numpy().astype(float),
        impulse.astype(float),
        s.mass,
        s.volume,
        s.mu,
        s.lam,
        s.config.dt,
        s.config.flip_fraction,
        s.node_count,
        viscosity,
    )
    s.material_step()
    out = s.state()
    np.testing.assert_allclose(out["position"], x, atol=1e-8)
    np.testing.assert_allclose(out["velocity"], v, atol=2e-6)
    np.testing.assert_allclose(out["deformation"], f, atol=2e-6)
    np.testing.assert_allclose(
        s.mass * out["velocity"].astype(float).sum(0), before + impulse.astype(float).sum(0), atol=2e-9
    )


def test_cut_labels_persist_withdrawal_and_new_cut():
    s = Solver()
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
            s.config.dt,
            1,
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
            s.config.dt,
            1,
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
            s.config.dt,
            3,
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


def test_short_full_run_against_corrected_cpu_with_real_contact(monkeypatch):
    import dexlab.cutting

    monkeypatch.setattr(dexlab.cutting, "material_step", paper_position_reference)
    s = Solver()
    values = {k: v for k, v in asdict(s.config).items() if k in CutConfig.__dataclass_fields__}
    cpu = CuttingSimulation(CutConfig(**values))
    # Begin at contact to test feedback, rather than only matching free fall.
    s.tool_pos.assign([[0.0, 0.0, 0.047]])
    cpu.blade_x[:] = s.tool_pos.numpy()[0]
    cpu.x[:] = s.x.numpy()
    cpu.weights[:] = s.weights.numpy()
    cpu.gradients[:] = s.gradients.numpy()
    for _ in range(100):
        s.step((0.0, 0.0, 0.02), cutting=True)
        cpu.step((0.0, 0.0, 0.02), cutting=True)
    out = s.state()
    np.testing.assert_allclose(out["position"], cpu.x, atol=2e-6)
    np.testing.assert_allclose(out["blade_position"][0], cpu.blade_x, atol=2e-6)
    np.testing.assert_allclose(out["velocity"], cpu.velocity, atol=3e-4)
    np.testing.assert_array_equal(out["sides"], cpu.sides)


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
    cd = arr(codes, int)
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
    ref = particle_contacts(
        x.astype(float),
        x.astype(float),
        vel.astype(float),
        codes.astype(np.int64),
        np.array([[0, 1]], np.int64),
        radius,
        spacing,
        mass,
        dt,
        0.1,
        0.3,
        0.2,
        0.05,
        0.1,
        np.zeros((2, 2, 3)),
        np.full((2, 2), -2, np.int32),
        0,
    )[0]
    np.testing.assert_allclose(out.numpy(), ref, atol=1e-10)
    np.testing.assert_allclose(out.numpy().sum(0), 0, atol=1e-12)
    assert bool(np.any(out.numpy())) == split
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
