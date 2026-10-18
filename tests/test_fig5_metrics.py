from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from blademaster_warp.fig5_metrics import local_volume, measure_actual_partition
from blademaster_warp.fig5 import (
    Fig5Config,
    commands,
    cut_completion_indices,
    evaluate,
    initial_left_velocity,
    moving_piece_ids,
    piece_ids,
    protocol,
    reference_particles,
)


@pytest.fixture
def moving_fixture():
    cfg = Fig5Config(size=(0.03, 0.02, 0.02), grid_spacing=0.005, cut_x=(-0.005, 0.005))
    ref = reference_particles(cfg)
    t = np.arange(len(commands(cfg)) + 1) * cfg.interval
    x = np.broadcast_to(ref, (len(t), *ref.shape)).copy()
    first, second = cut_completion_indices(cfg)
    # Two particle columns of horizontal drift before the second full-depth cut.
    x[first + 1 :, :, 0] += cfg.grid_spacing
    pieces = moving_piece_ids(cfg, x)
    index = round(protocol(cfg)[1]["withdrawn"] / cfg.interval)
    v = np.zeros_like(x)
    v[index, :, 0] = -0.004
    after = initial_left_velocity(v[index], pieces, cfg.left_velocity)
    v[index + 1 :] = after
    x[index + 1 :] += (t[index + 1 :] - t[index])[:, None, None] * after
    codes = np.array([4, 5, 8])[pieces]
    data = {
        "reference": ref,
        "time": t,
        "position": x,
        "velocity": v,
        "blade_impulse_ns": np.zeros((len(t), 3)),
        "board_impulse_ns": np.zeros((len(t), 3)),
        "pair_impulse_by_piece_ns": np.zeros((len(t), 3, 3)),
        "velocity_before_kick": v[index].copy(),
        "velocity_after_kick": after,
        "measurement_piece_ids": pieces,
        "topology_event_names": np.array(list(protocol(cfg)[1])),
        "component_labels_by_event": np.tile(pieces, (4, 1)),
        "cut_codes_by_event": np.tile(codes, (4, 1)),
    }
    return cfg, data


def test_actual_partition_follows_drift_and_freezes_the_first_cut(moving_fixture):
    cfg, data = moving_fixture
    ref = data["reference"]
    actual = moving_piece_ids(cfg, data["position"])
    fixed = piece_ids(ref, cfg.cut_x)
    assert np.bincount(actual).tolist() == [256, 128, 384]
    assert np.bincount(fixed).tolist() == [256, 256, 256]
    np.testing.assert_array_equal(actual == 0, fixed == 0)
    result = evaluate(cfg, data)
    assert result["fixture_validity"]["status"] == "matched"
    assert result["actual_partition"]["status"] == "matched"
    assert result["middle_volume"]["valid_for_fragment_volume"]
    assert 0 < result["middle_volume"]["reference_coverage_fraction"] < 1
    assert result["full_material_volume"]["after_cuts"]["middle"]["reference_coverage_fraction"] == 1
    assert result["middle_volume"]["contraction_percent"] == pytest.approx(0, abs=1e-10)
    assert result["middle_volume"]["full_reference_material_m3"] == pytest.approx(
        128 * (cfg.grid_spacing / 2) ** 3
    )
    middle = result["manipulation"]["middle"]
    assert middle["com_displacement_m"][0] == pytest.approx(-0.0016)
    assert middle["displacement_beyond_ballistic_m"][0] == pytest.approx(0, abs=1e-12)
    assert result["horizontal_exchange_isolated"]


def test_initial_region_guard_remains_invalid_for_legacy_archive(moving_fixture):
    cfg, data = moving_fixture
    data.pop("measurement_piece_ids")
    result = evaluate(cfg, data)
    assert result["fixture_validity"]["status"] == "mismatched"
    assert result["middle_volume"]["contraction_percent"] is None
    assert not result["horizontal_exchange_isolated"]
    assert result["actual_partition"]["status"] == "matched"
    assert result["full_material_volume"]["after_cuts"]["middle"]["valid"]


def test_saved_partition_cannot_override_observed_cut_membership(moving_fixture):
    cfg, data = moving_fixture
    data["measurement_piece_ids"] = piece_ids(data["reference"], cfg.cut_x)
    with pytest.raises(ValueError, match="moving cut membership"):
        evaluate(cfg, data)


@pytest.mark.parametrize("extra_drift", [0.0025, 0.005])
def test_empty_or_one_column_middle_is_invalid_serializable_result(moving_fixture, extra_drift):
    cfg, data = moving_fixture
    first, _ = cut_completion_indices(cfg)
    data["position"][first + 1 :, :, 0] += extra_drift
    pieces = moving_piece_ids(cfg, data["position"])
    data["measurement_piece_ids"] = pieces
    data["component_labels_by_event"] = np.tile(pieces, (4, 1))
    data["cut_codes_by_event"] = np.tile(np.array([4, 5, 8])[pieces], (4, 1))
    result = evaluate(cfg, data)
    assert result["middle_volume"]["tetrahedra"] == 0
    assert result["middle_volume"]["after_cuts_m3"] is None
    assert not result["full_material_volume"]["after_cuts"]["middle"]["valid"]
    json.dumps(result, allow_nan=False)
    if not np.any(pieces == 1):
        assert result["manipulation"]["middle"]["com_displacement_m"] is None


def test_actual_components_and_internal_cracks_still_gate_full_volume(moving_fixture):
    cfg, data = moving_fixture
    middle = np.flatnonzero(data["measurement_piece_ids"] == 1)
    data["cut_codes_by_event"][:, middle[0]] = 8
    result = evaluate(cfg, data)
    assert result["actual_partition"]["status"] == "invalid"
    assert not result["middle_volume"]["valid_for_fragment_volume"]
    assert result["middle_volume"]["after_cuts_m3"] is None


def test_shared_geometry_does_not_invent_cressim_mechanical_components(moving_fixture):
    cfg, data = moving_fixture
    for key in ("topology_event_names", "component_labels_by_event", "cut_codes_by_event"):
        data.pop(key)
    result = evaluate(cfg, data)
    assert result["actual_partition"]["status"] == "geometric"
    assert result["fixture_validity"]["status"] == "unverified"
    assert result["middle_volume"]["after_cuts_m3"] is None
    assert result["full_material_volume"]["after_cuts"]["middle"]["contraction_percent"] == pytest.approx(0)
    assert not result["horizontal_exchange_isolated"]


def test_short_blade_rejects_spreading_parent_instead_of_assuming_full_cut(moving_fixture):
    cfg, data = moving_fixture
    first, _ = cut_completion_indices(cfg)
    data["position"][first, :, 1] *= 4
    _, partition, volumes = measure_actual_partition(cfg, data)
    assert partition["status"] == "invalid"
    assert not partition["cut_geometry"][0]["blade_spans_parent"]
    assert volumes["after_cuts"]["middle"]["current_m3"] is None


@pytest.mark.parametrize("spacing", [0.005, 0.0025])
def test_full_volume_covers_outer_cells_and_is_affine_exact_per_fragment(spacing):
    cfg = replace(Fig5Config(), grid_spacing=spacing)
    ref = reference_particles(cfg)
    pieces = piece_ids(ref, cfg.cut_x)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transforms = [rotation, np.array([[0.9, 0.3, 0], [0, 0.8, 0.2], [0, 0, 1.1]]), np.eye(3)]
    x = ref.copy()
    for i, transform in enumerate(transforms):
        x[pieces == i] = np.einsum("ij,kj->ik", ref[pieces == i], transform) + [2 * i, -i, 0.1]
    volumes = local_volume(ref, x, pieces, (spacing / 2) ** 3)
    assert sum(v["reference_m3"] for v in volumes.values()) == pytest.approx(np.prod(cfg.size))
    for (name, volume), transform in zip(volumes.items(), transforms):
        assert volume["valid"], name
        assert volume["reference_coverage_fraction"] == 1
        assert volume["current_m3"] == pytest.approx(volume["reference_m3"] * np.linalg.det(transform))
        assert volume["excluded_neighbor_links"] > 0


@pytest.mark.parametrize("failure", ["inverted", "thin", "crack"])
def test_volume_never_hides_unsupported_or_inverted_material(failure):
    cfg = Fig5Config()
    ref = reference_particles(cfg)
    pieces = piece_ids(ref, cfg.cut_x)
    x = ref.copy()
    codes = np.array([4, 5, 8])[pieces]
    if failure == "inverted":
        x[pieces == 1, 0] *= -1
    elif failure == "thin":
        pieces[pieces == 1] = 2
        pieces[ref[:, 0] == np.unique(ref[:, 0])[8]] = 1
        codes = None
    else:
        codes[np.flatnonzero(pieces == 1)[0]] = 8
    volume = local_volume(ref, x, pieces, (cfg.grid_spacing / 2) ** 3, codes=codes)["middle"]
    assert not volume["valid"]
    assert volume["current_m3"] is None
    assert volume["contraction_percent"] is None
    assert volume["reference_coverage_fraction"] == 1


@pytest.mark.skipif(not os.environ.get("FIG5_CRESSIM_BUILD"), reason="Requires upstream build")
def test_cressim_guard_catches_constrained_stencil_before_old_one_cell_guard(tmp_path):
    executable = Path(os.environ["FIG5_CRESSIM_BUILD"]) / "fig5_cressim_cpu"
    # x=8 mm passes the old +/-15 mm particle check; its stencil reaches a fixed
    # node in the +/-20 mm domain (safe interval is +/-7.5 mm).
    header = [
        1,
        2,
        1,
        0.005,
        0.00025,
        0.01,
        6e6,
        0.4,
        0.06,
        0.025,
        0.00025,
        0,
        0,
        1,
        -0.01,
        -0.03,
        -0.02,
        -0.02,
        -0.02,
        0.02,
        0.02,
        0.02,
        1,
    ]
    source = tmp_path / "input.txt"
    source.write_text(" ".join(map(str, header)) + "\n.008 0 .001\n")
    states = tmp_path / "states.bin"
    result = subprocess.run(
        [str(executable), str(source), str(states), str(tmp_path / "kick.bin")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "grid support boundary guard frame=0 substep=0" in result.stderr
    assert states.stat().st_size == 0
