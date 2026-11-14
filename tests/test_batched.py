import numpy as np
import warp as wp
from dataclasses import replace
from blademaster_warp.batched import BatchedSolver
from blademaster_warp.solver import Solver
from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.presets import banana_config


def test_world_indexed_contact_and_odd_interval_masking():
    cfg = replace(
        banana_config(),
        grid_storage="compact",
        external_tool=True,
        tool_substep="hold",
        size=(0.03, 0.03, 0.03),
        shape="block",
    )
    cfg = replace(cfg, dt=0.002 / 47)
    b = BatchedSolver(cfg, 2)
    s = Solver(cfg)
    port = ExternalToolPort(s)
    p = np.zeros((2, 49), np.float32)
    p[:, 2] = 0.051
    p[:, 6] = 1
    p[1, 2] = 0.09
    b.save_reset()
    peak = 0.0
    diff = 0.0
    xdiff = 0.0
    for i in range(60):
        p[0, 2] = 0.051 - 0.1 * (i + 1) * 0.002
        p[0, 9] = -0.1
        o = b.advance(p, np.ones(2)).numpy()
        r = port.advance(ToolState(p[0, :3], p[0, 3:7], p[0, 7:10], p[0, 10:13]), 0.002).about(np.zeros(3))
        force = o[0, :6] / 0.002
        ref = np.r_[r.force_n, r.torque_nm]
        assert np.isfinite(o).all() and not o[:, 6:9].any()
        assert np.max(np.abs(o[1, :6])) == 0, "Cross-world tool reaction"
        peak = max(peak, float(np.linalg.norm(force[:3])))
        diff = max(diff, float(abs(force - ref).max()))
        xdiff = max(xdiff, float(abs(b.x.numpy()[0] - s.x.numpy()).max()))
        if i % 10 == 0:
            print(i, "peak", peak, "force_err", diff, "xerr", xdiff, flush=True)
    assert np.array_equal(b.codes.numpy()[0], s.codes.numpy()), "Moving tool labels differ"
    assert (b.codes.numpy()[0] != 0).any(), "No actual side assignment"
    assert peak > 0.01, "No actual contact"
    assert diff < 0.02, "Batched and scalar reactions differ"
    assert xdiff < 2e-5, "Batched and scalar motion differ"
    before = {n: getattr(b, n).numpy().copy() for n in ["x", "v", "F", "Fp", "codes", "damage"]}
    b.advance(p, np.array([0, 1])).numpy()
    for n, v in before.items():
        assert np.array_equal(getattr(b, n).numpy()[0], v[0]), "Inactive changed " + n
    o = b.advance(p, np.array([1, 1])).numpy()
    r = port.advance(ToolState(p[0, :3], p[0, 3:7], p[0, 7:10], p[0, 10:13]), 0.002).about(np.zeros(3))
    assert (
        abs(o[0, :6] / 0.002 - np.r_[r.force_n, r.torque_nm]).max() < 0.02
    ), "Odd-substep resume history differs"
    assert np.array_equal(b.codes.numpy()[0], s.codes.numpy())
    before = {n: getattr(b, n).numpy().copy() for n in ["x", "v", "F", "Fp", "codes", "damage"]}
    b.reset(np.array([1, 0]))
    for n, v in before.items():
        assert np.array_equal(getattr(b, n).numpy()[1], v[1]), "Reset changed other " + n
    assert np.array_equal(b.x.numpy()[0], b.reference.astype(np.float32))


def test_reset_preserves_settled_board_friction_history():
    cfg = replace(
        banana_config(),
        grid_storage="compact",
        external_tool=True,
        tool_substep="hold",
        size=(0.03, 0.03, 0.03),
        shape="block",
        dt=0.002 / 47,
    )
    b = BatchedSolver(cfg, 2)
    packed = np.zeros((2, 49), np.float32)
    packed[:, 2] = 0.09
    packed[:, 6] = 1
    velocity = b.v.numpy()
    velocity[:, :, 0] = 0.03
    b.v.data.assign(velocity)
    for _ in range(10):
        b.advance(packed, np.ones(2), cutting=False).numpy()
    assert np.max(np.abs(b.histories.numpy())) > 0
    b.save_reset()
    settled = {n: getattr(b, n).numpy().copy() for n in ("x", "v", "histories", "impulses")}
    b.advance(packed, np.ones(2), cutting=False).numpy()
    other = {n: getattr(b, n).numpy()[1].copy() for n in settled}
    b.reset([1, 0])
    for name, value in settled.items():
        np.testing.assert_array_equal(getattr(b, name).numpy()[0], value[0])
        np.testing.assert_array_equal(getattr(b, name).numpy()[1], other[name])


def test_generated_kernels_match_the_generator(tmp_path, monkeypatch):
    from pathlib import Path
    from blademaster_warp import generate_batch_kernels as generator

    source = Path(generator.__file__).parent
    for name in generator.NAMES:
        (tmp_path / (name + ".py")).write_text((source / (name + ".py")).read_text())
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    generator.generate()
    for name in generator.NAMES:
        assert (tmp_path / ("batch_" + name + ".py")).read_text() == (
            source / ("batch_" + name + ".py")
        ).read_text()
