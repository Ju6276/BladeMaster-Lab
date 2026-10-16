"""External-tool tests use physical invariants, independent of DexLab."""

from dataclasses import replace
import numpy as np
import pytest
from blademaster_warp.solver import Config, Solver
from blademaster_warp.coupling import ExternalToolPort, ToolState, Reaction


def setup():
    s = Solver(replace(Config(), external_tool=True))
    # Move the specimen off the floor: only tool contact and gravity remain.
    x = s.x.numpy()
    x[:, 2] += 0.1
    s.x.assign(x)
    angle = 0.15
    state = ToolState(
        [0.001, 0.0, 0.132],
        [0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)],
        [0.01, 0.0, -0.02],
        [0.0, 0.1, 0.0],
    )
    return s, ExternalToolPort(s), state


def test_tool_is_frozen_and_impulse_is_equal_opposite():
    s, port, state = setup()
    duration = 8 * s.config.dt
    before = s.v.numpy().astype(float).sum(0) * s.mass
    reaction = port.advance(state, duration, graph=False, cutting=False)
    after = s.v.numpy().astype(float).sum(0) * s.mass
    gravity = np.array([0.0, 0.0, -9.81]) * s.mass * s.n * duration
    np.testing.assert_allclose(after - before + reaction.impulse_ns, gravity, atol=3.0e-9)
    for array, value in [
        (s.tool_pos, state.position),
        (s.tool_rot, state.orientation_xyzw),
        (s.tool_vel, state.linear_velocity),
        (s.tool_omega, state.angular_velocity),
    ]:
        np.testing.assert_allclose(array.numpy()[0], value, atol=2.0e-8)
    assert np.linalg.norm(reaction.force_n) > 0.01
    np.testing.assert_allclose(reaction.force_n * duration, reaction.impulse_ns)


def test_interval_accumulation_matches_single_substeps():
    a, pa, state = setup()
    b, pb, _ = setup()
    result = pa.advance(state, 9 * a.config.dt, graph=True)
    pieces = [pb.advance(state, b.config.dt, graph=False) for _ in range(9)]
    np.testing.assert_allclose(result.impulse_ns, sum(r.impulse_ns for r in pieces), atol=2.0e-9)
    np.testing.assert_allclose(
        result.angular_impulse_nms, sum(r.angular_impulse_nms for r in pieces), atol=2.0e-10
    )
    np.testing.assert_allclose(a.x.numpy(), b.x.numpy(), atol=1.0e-7)
    np.testing.assert_array_equal(a.codes.numpy(), b.codes.numpy())
    # Cache hit plus history parity change, without retaining old impulses.
    ra = pa.advance(state, 9 * a.config.dt, graph=True)
    rb = pb.advance(state, 9 * b.config.dt, graph=False)
    np.testing.assert_allclose(ra.impulse_ns, rb.impulse_ns, atol=2.0e-9)


def test_moment_shift_and_power_invariance():
    r = Reaction(0.02, np.array([0.1, 0.2, 0.3]), np.array([0.01, -0.02, 0.03]), np.array([1.0, 2.0, 3.0]))
    origin = np.array([-0.5, 0.3, 2.0])
    shifted = r.about(origin)
    velocity = np.array([0.2, -0.1, 0.7])
    omega = np.array([0.3, 0.5, -0.2])
    v_new = velocity + np.cross(omega, origin - r.origin_world_m)
    assert np.dot(r.force_n, velocity) + np.dot(r.torque_nm, omega) == pytest.approx(
        np.dot(shifted.force_n, v_new) + np.dot(shifted.torque_nm, omega)
    )


def test_external_api_rejects_wrong_ownership_and_time():
    s, port, state = setup()
    with pytest.raises(ValueError):
        s.step()
    with pytest.raises(ValueError):
        s.advance([[0.0, 0.0, 0.0]])
    for duration in [0.0, -1.0, np.nan, 1.5 * s.config.dt]:
        with pytest.raises(ValueError):
            port.advance(state, duration)
    assert s.step_count == 0
    with pytest.raises(ValueError):
        port.advance(ToolState([0, 0, 0], [0, 0, 0, 2], [0, 0, 0], [0, 0, 0]), s.config.dt)
    with pytest.raises(ValueError):
        replace(Config(), external_tool=True, pose_servo=True)


def test_invalid_material_interval_never_returns_a_wrench():
    s, port, state = setup()
    f = s.F.numpy()
    f[0, 0, 0] = -1.0
    s.F.assign(f)
    with pytest.raises(RuntimeError, match="Invalid material coupling interval"):
        port.advance(state, s.config.dt, graph=False)


def test_articulated_effective_mass_in_single_contact():

    s = Solver(replace(Config(), external_tool=True))
    cfg = s.config
    s.tool_pos.assign([[0, 0, 0.15]])
    x = s.x.numpy()
    x[0] = [cfg.blade_half_size[0] + s.radius / 2, 0.01, 0.15]
    s.x.assign(x)
    mobility = np.diag([1000, 2000, 3000, 4000, 5000, 6000.0])

    def impulse(a):
        s.tool_mobility.assign(a.astype(np.float32))
        s.reaction.zero_()
        s.torque.zero_()
        s.histories.zero_()
        s.rigid_contacts(cutting=False)
        return s.reaction.numpy()[0, 0]

    prescribed = impulse(np.zeros((6, 6)))
    articulated = impulse(mobility)
    direction = np.array([1, 0, 0, 0, 0, -0.01])
    expected_ratio = 1 / (1 + s.mass * (direction @ mobility @ direction))
    assert prescribed < 0
    assert articulated / prescribed == pytest.approx(expected_ratio, rel=2.0e-6)


def test_mobility_rejects_invalid_mass_response():
    for invalid in [np.eye(5), -np.eye(6), np.full((6, 6), np.nan), np.triu(np.ones((6, 6)))]:
        with pytest.raises(ValueError, match="Mobility"):
            ToolState([0, 0, 0], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0], invalid).checked()


def test_cut_index_beyond_int32_digits_still_severs():
    from blademaster_warp.presets import banana_config
    from blademaster_warp.topology import components

    s = Solver(replace(banana_config(), external_tool=True))
    assert s.max_cut_index >= 29
    port, half, cut = ExternalToolPort(s), s.config.blade_half_size[2], 25
    for i in range(1900):
        edge = max(0.0, 0.036 - 0.02 * i * 0.001)
        state = ToolState([0.0, 0.0, edge + half], [0, 0, 0, 1], [0.0, 0.0, -0.02], [0, 0, 0])
        port.advance(state, 0.001, cutting=True, cut_index=cut)
    codes = s.codes.numpy()
    assert set(np.unique(codes // 3**cut)) == {0, 1, 2}
    _, sizes = components(s.node_numpy, s.weight_numpy, codes)
    assert len(sizes) == 2 and sizes.min() > 0.3 * s.n


def test_auto_strokes_label_each_cut_plane_with_its_own_digit():
    from blademaster_warp.presets import banana_config
    from blademaster_warp.teleop_input import ContactStrokes
    from blademaster_warp.topology import components

    s = Solver(replace(banana_config(), external_tool=True))
    port, half, strokes, dt = ExternalToolPort(s), s.config.blade_half_size[2], ContactStrokes(), 0.001
    # Edge path: down through x=-25 mm, back out, over, down through x=+25 mm, out.
    legs = [
        (-0.025, 0.0, 0.9),
        (-0.025, 0.0, 0.1),
        (-0.025, 0.04, 0.6),
        (0.025, 0.04, 0.8),
        (0.025, 0.0, 1.0),
        (0.025, 0.0, 0.1),
        (0.025, 0.04, 0.6),
    ]
    path = [(-0.025, 0.036)]
    for x, z, seconds in legs:
        x0, z0 = path[-1]
        for k in range(1, round(seconds / dt) + 1):
            path.append((x0 + (x - x0) * k * dt / seconds, z0 + (z - z0) * k * dt / seconds))
    for (x0, z0), (x1, z1) in zip(path, path[1:]):
        state = ToolState(
            [x1, 0.0, z1 + half], [0, 0, 0, 1], [(x1 - x0) / dt, 0.0, (z1 - z0) / dt], [0, 0, 0]
        )
        port.advance(state, dt, cutting=True, cut_index=strokes.cut_index)
        strokes.update(s.embedded_particles() > 0, dt)
    assert strokes.strokes == 2 and strokes.cut_index == 2
    codes, ref = s.codes.numpy(), s.reference
    band = 7 * s.config.grid_spacing
    assert np.all(abs(ref[codes % 3 != 0, 0] + 0.025) < band)
    assert np.all(abs(ref[(codes // 3) % 3 != 0, 0] - 0.025) < band)
    _, sizes = components(s.node_numpy, s.weight_numpy, codes)
    assert len(sizes) == 3
