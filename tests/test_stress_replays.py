"""CPU regressions for the reusable stress runners and their boundary fixtures."""

from dataclasses import replace
import json

import numpy as np
import pytest
import warp as wp

from blademaster_warp.solver import Config
from scripts.stress_multi_tool import (
    ContactProbe,
    canonical_codes,
    capacity_stress,
    contact_stress,
    cut_code_capacity,
)
from scripts.stress_teleop import INITIAL_EDGE, replay, seeded_replay


@pytest.mark.parametrize("seed", [0, 17, 913])
def test_full_history_survives_restarts_and_json_replay(seed):
    last = cut_code_capacity()["max_cut_index"]
    rows = seeded_replay(seed, extra_cuts=last, restart_cycles=2)
    trace, report = replay(rows)
    assert replay(json.loads(json.dumps(rows))) == (trace, report)
    assert report["sessions"] == 4
    assert report["cuts_per_session"] == [3, last + 1, last + 1, last + 1]
    assert report["rejected_packets"] == 3
    for session in range(1, 4):
        frames = [frame for frame in trace if frame["session"] == session]
        first = frames[0]
        assert first["active"] and first["stale"] and not first["cutting"]
        assert first["cut_index"] == -1 and not first["events"]
        np.testing.assert_array_equal(first["position"], INITIAL_EDGE)
        np.testing.assert_array_equal(first["orientation"], [0, 0, 0, 1])
        begun = [frame["cut_index"] for frame in frames if any(e == "begin_cut" for _, e in frame["events"])]
        assert begun == list(range(last + 1))
        assert frames[-1]["cut_index"] == last and not frames[-1]["cutting"]
    for tick, row in enumerate(rows):
        if row["phase"] == "duplicate":
            assert trace[tick]["seq"] == trace[tick - 1]["seq"]
            assert trace[tick]["cut_index"] == -1 and not trace[tick]["events"]
    assert trace[-1]["cut_index"] == last


@pytest.mark.parametrize("seed", [0, 17, 913])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_production_labels_and_hash_keys_reach_capacity_on_cpu(seed, count):
    report = capacity_stress(seed, tool_count=count)
    capacity = report["capacity"]
    assert capacity["node_count"] == 225
    assert capacity["digits"] == report["labeled_digits"] == 34
    assert report["highest_code"] == 3**34 - 1
    assert report["highest_registered_key"] == 225 * 3**34 - 1
    assert capacity["max_key"] <= np.iinfo(np.int64).max < 225 * 3**35 - 1
    assert report["cut_indices"][-1] == 34 - count
    assert report["sessions"] == 2
    # Thirty-four persistent digits still give two components in this fixture.
    assert report["fixture_component_sizes"] == [2, 2]


@pytest.mark.parametrize("spacing", [0.015, 0.00375])
def test_capacity_tracks_grid_size_instead_of_a_fixed_cut_limit(spacing):
    config = Config(grid_spacing=spacing, free_blade=True, tool_count=3)
    report = capacity_stress(913, config=config)
    capacity = report["capacity"]
    assert capacity["digits"] != 34
    assert capacity["max_key"] <= np.iinfo(np.int64).max
    assert capacity["node_count"] * 3 ** (capacity["digits"] + 1) - 1 > np.iinfo(np.int64).max


@pytest.mark.parametrize("spacing, nodes", [(0.015, 45), (0.0075, 225), (0.00375, 1377)])
def test_budget_uses_the_production_reference_geometry_node_count(spacing, nodes):
    from blademaster_warp.geometry import points, stencil

    cfg = Config(grid_spacing=spacing)
    reference, origin, cells = points(cfg.size, cfg.grid_spacing, cfg.shape, cfg.sampling_seed)
    actual_nodes, _, _ = stencil(reference, origin, cfg.grid_spacing, cells)
    assert int(np.prod(cells + 1)) == nodes
    assert actual_nodes.min() == 0 and actual_nodes.max() + 1 == nodes
    assert cut_code_capacity(cfg)["node_count"] == nodes


@pytest.mark.parametrize("count, spacing, nodes", [(1, 0.015, 45), (2, 0.0075, 225), (3, 0.00375, 1377)])
def test_capacity_uses_actual_solver_allocation_on_cuda(count, spacing, nodes):
    if not wp.is_cuda_available():
        pytest.skip("CUDA Solver allocation required")
    cfg = Config(free_blade=True, tool_count=count, grid_spacing=spacing)
    report = capacity_stress(913, device="cuda:0", config=cfg)
    assert report["capacity_source"] == "solver"
    assert report["capacity"]["node_count"] == nodes


@pytest.mark.parametrize("seed", [0, 17, 913])
@pytest.mark.parametrize("count", [2, 3])
def test_simultaneous_rotating_crossings_preserve_independent_contacts(seed, count):
    report = contact_stress(seed, tool_count=count, motion="crossing")
    assert report["permutations"] == (2 if count == 2 else 6)
    assert report["crossed_tool_pairs"] == count * (count - 1) // 2
    assert report["max_rotation_change_rad"] > 0.03
    assert report["shared_damaged_particles"] == 4
    assert report["max_momentum_residual_kg_m_s"] < 1e-9
    for errors in report["independent_max_abs_error_by_tool"].values():
        np.testing.assert_array_equal(errors, 0)


@pytest.mark.parametrize("count", [2, 3])
def test_permuted_high_digits_round_trip_without_float_conversion(count):
    capacity = cut_code_capacity(replace(Config(), free_blade=True, tool_count=count))
    strokes = capacity["complete_strokes"]
    rng = np.random.default_rng(913)
    digits = rng.integers(0, 3, (4, strokes, count))
    digits[-1] = 2
    original = np.array(
        [sum(int(digit) * 3**index for index, digit in enumerate(row.flat)) for row in digits], dtype=np.int64
    )
    order = list(reversed(range(count)))
    permuted = canonical_codes(original, order, strokes)
    expected = np.array(
        [sum(int(digit) * 3**index for index, digit in enumerate(row[:, order].flat)) for row in digits],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(permuted, expected)
    np.testing.assert_array_equal(canonical_codes(permuted, order, strokes), original)
    assert permuted.dtype == np.int64
    assert permuted.max() == 3 ** (strokes * count) - 1
    assert permuted.max() > 2**31
    if count == 2:
        assert permuted.max() > 2**53


@pytest.mark.parametrize("value", [0.5, 1.0, float("nan"), float("inf"), True, np.bool_(True), "1", None])
def test_invalid_cut_index_type_is_rejected_before_history_mutation(value):
    probe = ContactProbe(Config(free_blade=True, tool_count=3), [[0, 0, 0.041]])
    probe.codes.assign([13])
    for slot, damage in enumerate(probe.tool_damage):
        damage.fill_(0.125 * (slot + 1))
    with pytest.raises(ValueError, match="integer"):
        probe._set_cut(value)
    assert probe.cut_index == 0
    np.testing.assert_array_equal(probe.codes.numpy(), [13])
    for slot, damage in enumerate(probe.tool_damage):
        np.testing.assert_array_equal(damage.numpy(), [0.125 * (slot + 1)])


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_numpy_cut_indices_normalize_to_python_int_before_power(dtype):
    probe = ContactProbe(Config(), [[0, 0, 0.041]])
    probe.damage.fill_(0.75)
    index = probe._set_cut(dtype(probe.max_cut_index))
    assert type(index) is int and type(probe.cut_index) is int
    assert 3**index == 3**33
    np.testing.assert_array_equal(probe.damage.numpy(), [0])
    probe.damage.fill_(0.75)
    assert probe._set_cut(dtype(index)) == index
    np.testing.assert_array_equal(probe.damage.numpy(), [0.75])


@pytest.mark.parametrize("method", ["step", "advance"])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_high_numpy_indices_reach_kernel_dispatch_as_python_int(method, dtype, monkeypatch):
    probe = ContactProbe(Config(), [[0, 0, 0.041]])
    observed = []

    def capture(target, cutting, cut_index, **kwargs):
        assert type(cut_index) is int
        observed.append(3**cut_index)

    monkeypatch.setattr(probe, "_step_kernels", capture)
    if method == "step":
        probe.step(cut_index=dtype(33))
    else:
        probe.advance([[0, 0, 0.041]], cut_index=dtype(33), graph=False)
    assert observed == [3**33]


@pytest.mark.parametrize("invalid", [-1, 34, np.int64(34), 2**80])
def test_out_of_range_cut_indices_preserve_history(invalid):
    probe = ContactProbe(Config(), [[0, 0, 0.041]])
    probe._set_cut(33)
    probe.codes.assign([3**34 - 1])
    probe.damage.fill_(0.75)
    with pytest.raises(ValueError, match="monotonic"):
        probe._set_cut(invalid)
    assert probe.cut_index == 33
    np.testing.assert_array_equal(probe.codes.numpy(), [3**34 - 1])
    np.testing.assert_array_equal(probe.damage.numpy(), [0.75])


@pytest.mark.parametrize(
    "orientations",
    [
        [[0, 0, 1]],
        [[0, 0, 0, 0]],
        [[0, 0, float("nan"), 1]],
        [[0, 0, float("inf"), 1]],
    ],
)
def test_malformed_advance_orientation_preserves_previous_cut_history(orientations):
    probe = ContactProbe(Config(), [[0, 0, 0.041]])
    probe.codes.assign([2])
    probe.damage.fill_(0.75)
    with pytest.raises(ValueError, match="orientations|quaternion"):
        probe.advance([[0, 0, 0.041]], cut_index=1, graph=False, orientations=orientations)
    assert probe.cut_index == 0
    np.testing.assert_array_equal(probe.codes.numpy(), [2])
    np.testing.assert_array_equal(probe.damage.numpy(), [0.75])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"extra_cuts": -1},
        {"extra_cuts": 34},
        {"extra_cuts": 0.5},
        {"restart_cycles": -1},
        {"restart_cycles": 0.5},
        {"restart_cycles": True},
    ],
)
def test_invalid_generated_history_is_rejected(kwargs):
    with pytest.raises(ValueError):
        seeded_replay(**kwargs)


def test_seed_batch_exports_independently_replayable_streams(tmp_path, capsys):
    from scripts.stress_teleop import main

    output = tmp_path / "teleop"
    main(["--seeds", "0", "913", "--extra-cuts", "1", "--restart-cycles", "1", "--output", str(output)])
    aggregate = json.loads((output / "report.json").read_text())
    assert aggregate == json.loads(capsys.readouterr().out)
    assert [run["seed"] for run in aggregate["runs"]] == [0, 913]
    for seed in (0, 913):
        directory = output / f"seed-{seed}"
        rows = json.loads((directory / "commands.json").read_text())
        trace, report = replay(rows)
        assert json.loads(json.dumps(trace)) == json.loads((directory / "trace.json").read_text())
        saved = json.loads((directory / "report.json").read_text())
        assert all(saved[key] == value for key, value in report.items())
        assert saved["cuts_per_session"] == [3, 2, 2]
    replay_output = tmp_path / "replayed"
    main(["--replay", str(output / "seed-913" / "commands.json"), "--output", str(replay_output)])
    assert json.loads((replay_output / "trace.json").read_text()) == json.loads(
        (output / "seed-913" / "trace.json").read_text()
    )


def test_multi_tool_seed_batch_records_each_scenario(tmp_path, capsys):
    from scripts.stress_multi_tool import main

    output = tmp_path / "multi.json"
    main(["--seeds", "0", "913", "--tools", "2", "--output", str(output)])
    report = json.loads(output.read_text())
    assert capsys.readouterr().out.endswith(json.dumps(report, indent=2) + "\n")
    assert [run["seed"] for run in report["runs"]] == [0, 913]
    for run in report["runs"]:
        assert run["contacts"]["motion"] == "fixed"
        assert run["crossing_contacts"]["crossed_tool_pairs"] == 1
        assert run["capacity"]["labeled_digits"] == 34


def test_single_seed_output_format_is_retained(tmp_path, capsys):
    from scripts.stress_teleop import main

    output = tmp_path / "single"
    main(["--seed", "17", "--output", str(output)])
    report = json.loads(capsys.readouterr().out)
    assert report["seed"] == 17 and "runs" not in report
    assert report["cuts_per_session"] == [3, 1]
    assert (output / "commands.json").is_file()
