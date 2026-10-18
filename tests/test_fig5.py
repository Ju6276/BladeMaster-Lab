from dataclasses import asdict, replace
import json
import os
import subprocess

import numpy as np
import pytest

from blademaster_warp.fig5 import (
    Fig5Config,
    commands,
    evaluate,
    half_depth_gap,
    initial_left_velocity,
    middle_tetrahedra,
    piece_ids,
    protocol,
    reference_particles,
    signed_tetra_volumes,
)
from blademaster_warp.fig5_cressim import (
    CRESSIM_REVISION,
    impulses_to_si,
    material_in_solver_units,
    validate_build,
)
from blademaster_warp.fig5_run import compare, run, write_json


@pytest.fixture
def cfg():
    return Fig5Config(size=(0.02, 0.02, 0.02), grid_spacing=0.01, dt=0.001, cut_x=(-0.005, 0.005))


@pytest.fixture
def trajectory(cfg):
    ref = reference_particles(cfg)
    times = np.arange(len(commands(cfg)) + 1) * cfg.interval
    pieces = piece_ids(ref, cfg.cut_x)
    position = np.broadcast_to(ref, (len(times), *ref.shape)).copy()
    velocity = np.zeros_like(position)
    withdrawn = protocol(cfg)[1]["withdrawn"]
    index = round(withdrawn / cfg.interval)
    velocity[index + 1 :, pieces == 0, 0] = cfg.left_velocity
    position[index + 1 :, pieces == 0, 0] += (times[index + 1 :] - withdrawn)[:, None] * cfg.left_velocity
    return {
        "reference": ref,
        "time": times,
        "position": position,
        "velocity": velocity,
        "blade_impulse_ns": np.zeros((len(times), 3)),
        "board_impulse_ns": np.zeros((len(times), 3)),
        "pair_impulse_by_piece_ns": np.zeros((len(times), 3, 3)),
        "velocity_before_kick": velocity[index].copy(),
        "velocity_after_kick": initial_left_velocity(velocity[index], pieces, cfg.left_velocity),
        "topology_event_names": np.array(list(protocol(cfg)[1])),
        "component_labels_by_event": np.tile(pieces, (4, 1)),
        "cut_codes_by_event": np.tile(np.array([1, 5, 8])[pieces], (4, 1)),
    }


def test_protocol_exact_half_depth_two_cuts_and_withdrawal(cfg):
    rows, events = protocol(cfg)
    cmds = commands(cfg)
    half = cmds[round(events["first_half_depth"] / cfg.interval) - 1]
    assert half[3] == 0
    assert half[1][0] == cfg.cut_x[0]
    assert half[1][2] - cfg.blade_half_height == pytest.approx(cfg.size[2] / 2)
    assert {r[3] for r in cmds} == {-1, 0, 1}
    assert 2 * (cfg.blade_thickness / 2) == 0.0005
    withdrawn = round(events["withdrawn"] / cfg.interval)
    for _, p, v, cut in cmds[withdrawn:]:
        assert p[2] - cfg.blade_half_height == pytest.approx(cfg.size[2] + cfg.clearance)
        np.testing.assert_array_equal(v, 0)
        assert cut == -1
    assert rows[-1][0] - events["withdrawn"] == pytest.approx(cfg.manipulation_s)


@pytest.mark.parametrize(
    "change",
    [
        dict(dt=0.0003),
        dict(interval=0.003),
        dict(blade_thickness=0.001),
        dict(left_velocity=0.1),
        dict(cut_x=(0.005, -0.005)),
        dict(grid_spacing=0.009),
        dict(dt=float("nan")),
        dict(cressim_drag=0),
        dict(cressim_mass_unit_kg=0),
        dict(cressim_mass_unit_kg=-0.001),
        dict(cressim_mass_unit_kg=float("inf")),
    ],
)
def test_invalid_fixture_fails_before_simulation(cfg, change):
    with pytest.raises(ValueError):
        replace(cfg, **change)


def test_kick_changes_only_reference_left_and_preserves_other_velocities(cfg):
    ref = reference_particles(cfg)
    pieces = piece_ids(ref, cfg.cut_x)
    before = np.random.default_rng(42).normal(size=ref.shape).astype(np.float32)
    snapshot = before.copy()
    after = initial_left_velocity(before, pieces, cfg.left_velocity)
    np.testing.assert_array_equal(before, snapshot)
    np.testing.assert_array_equal(after[pieces != 0], before[pieces != 0])
    np.testing.assert_allclose(
        after[pieces == 0], np.tile([cfg.left_velocity, 0, 0], ((pieces == 0).sum(), 1))
    )
    integer_zero = initial_left_velocity(np.zeros(ref.shape, dtype=int), pieces, cfg.left_velocity)
    assert integer_zero[pieces == 0, 0].mean() == pytest.approx(cfg.left_velocity)


def test_opening_reports_increment_and_raw_spacing_and_excludes_below_tip(cfg):
    ref = reference_particles(cfg)
    moved = ref + [0.17, -0.12, 0.29]
    use = ref[:, 0] >= cfg.cut_x[0]
    moved[use, 0] += 0.0007
    moved[use & (ref[:, 2] < cfg.size[2] / 2), 0] += 0.1
    result = half_depth_gap(ref, moved, cfg.cut_x[0], cfg.size[2] / 2)
    assert result["mean_opening_increment_mm"] == pytest.approx(0.7)
    assert result["mean_reference_column_spacing_mm"] == pytest.approx(5)
    assert result["mean_column_distance_mm"] == pytest.approx(5.7)
    assert result["pairs"] == 8


def test_gap_rejects_cut_on_lattice_column_and_missing_lattice(cfg):
    ref = reference_particles(cfg)
    with pytest.raises(ValueError, match="strictly"):
        half_depth_gap(ref, ref, ref[0, 0], 0)
    with pytest.raises(ValueError, match="lattice"):
        middle_tetrahedra(ref[:-1], cfg.cut_x)


def test_tetra_volume_affine_determinant_and_reference_membership(cfg):
    ref = reference_particles(cfg)
    tets = middle_tetrahedra(ref, cfg.cut_x)
    assert np.all(piece_ids(ref, cfg.cut_x)[tets] == 1)
    volumes = signed_tetra_volumes(ref, tets)
    assert (volumes > 0).all()
    assert volumes.sum() == pytest.approx(0.005 * 0.015 * 0.015)
    transform = np.array([[0.9, 0.2, 0], [0, 0.8, 0.1], [0, 0, 1.1]])
    moved = ref @ transform.T + [2, -1, 4]
    assert signed_tetra_volumes(moved, tets).sum() == pytest.approx(volumes.sum() * np.linalg.det(transform))


def test_complete_evaluation_has_correct_units_motion_impulse_and_volume(cfg, trajectory):
    ref = trajectory["reference"]
    i = round(protocol(cfg)[1]["both_cuts_complete"] / cfg.interval)
    trajectory["position"][i] = 0.9 * ref
    trajectory["blade_impulse_ns"][1] = [1e-4, -2e-4, 3e-4]
    result = evaluate(cfg, trajectory)
    assert result["middle_volume"]["contraction_percent"] == pytest.approx(27.1)
    assert result["middle_volume"]["reference_region_coverage_fraction"] == pytest.approx(0.28125)
    assert result["manipulation"]["left"]["com_displacement_m"][0] == pytest.approx(-0.012)
    np.testing.assert_allclose(result["manipulation"]["middle"]["com_displacement_m"], 0)
    np.testing.assert_allclose(result["manipulation"]["right"]["com_displacement_m"], 0)
    np.testing.assert_allclose(result["blade_contact"]["total_impulse_ns"], [1e-4, -2e-4, 3e-4])
    assert result["blade_contact"]["after_withdrawal_sum_interval_norms_ns"] == 0
    assert result["horizontal_exchange_isolated"]


def test_momentum_exchange_excludes_kick_and_separates_pair_contact(cfg, trajectory):
    ids = piece_ids(trajectory["reference"], cfg.cut_x)
    trajectory["velocity"][-1, ids == 1, 0] = -0.01
    trajectory["velocity"][-1, ids == 0, 0] += 0.02
    impulse = -0.01 * (ids == 1).sum() * cfg.density * (cfg.grid_spacing / 2) ** 3
    trajectory["pair_impulse_by_piece_ns"][-1, 1, 0] = impulse / 4
    result = evaluate(cfg, trajectory)
    mid = result["manipulation"]["middle"]
    assert mid["material_exchange_horizontal_ns"][0] == pytest.approx(impulse)
    assert mid["transfer_residual_horizontal_ns"][0] == pytest.approx(3 * impulse / 4)
    assert result["manipulation"]["left"]["momentum_change_ns"] == pytest.approx([-impulse, 0, 0])
    trajectory["blade_impulse_ns"][-1, 0] = 0.001
    contaminated = evaluate(cfg, trajectory)
    assert not contaminated["horizontal_exchange_isolated"]
    assert contaminated["manipulation"]["middle"]["material_exchange_horizontal_ns"] is None


def test_momentum_loss_is_not_misreported_as_interpiece_exchange(cfg, trajectory):
    trajectory["velocity"][-1] = 0
    result = evaluate(cfg, trajectory)
    assert result["momentum_balance"]["horizontal_external_contacts_clear"]
    assert not result["momentum_balance"]["horizontal_balance_closed"]
    assert not result["horizontal_exchange_isolated"]
    assert result["manipulation"]["left"]["material_exchange_horizontal_ns"] is None


@pytest.mark.parametrize("corruption", ["time", "nan", "kick", "reference"])
def test_evaluation_rejects_misalignment_nonfinite_and_incorrect_kick(cfg, trajectory, corruption):
    if corruption == "time":
        trajectory["time"][350] += 0.001
    elif corruption == "nan":
        trajectory["position"][-1, 0, 0] = np.nan
    elif corruption == "kick":
        trajectory["velocity_after_kick"][-1, 0] = -0.03
    else:
        trajectory["reference"][0, 0] += 0.00001
    with pytest.raises(ValueError):
        evaluate(cfg, trajectory)


def test_inverted_tets_are_counted(cfg, trajectory):
    i = round(protocol(cfg)[1]["both_cuts_complete"] / cfg.interval)
    trajectory["position"][i, :, 0] *= -1
    result = evaluate(cfg, trajectory)["middle_volume"]
    assert result["inverted_or_degenerate_tetrahedra"] == result["tetrahedra"]


def test_topology_membership_ignores_component_label_order(cfg, trajectory):
    trajectory["component_labels_by_event"] = np.array([19, 4, 8])[trajectory["component_labels_by_event"]]
    result = evaluate(cfg, trajectory)
    assert result["fixture_validity"]["status"] == "matched"
    assert result["middle_volume"]["valid_for_fragment_volume"]
    assert result["initial_velocity_assignment"]["selected_complete_component"]


def test_equal_component_sizes_do_not_prove_reference_membership(cfg, trajectory):
    pieces = piece_ids(trajectory["reference"], cfg.cut_x)
    left, middle = np.flatnonzero(pieces == 0)[0], np.flatnonzero(pieces == 1)[0]
    labels = trajectory["component_labels_by_event"]
    labels[:, [left, middle]] = labels[:, [middle, left]]
    result = evaluate(cfg, trajectory)
    assert result["fixture_validity"]["status"] == "mismatched"
    cut = result["fixture_validity"]["events"]["both_cuts_complete"]
    assert cut["component_sizes"] == np.bincount(pieces).tolist()
    assert cut["middle_tetrahedra_crossing_components"] > 0
    assert result["middle_volume"]["contraction_percent"] is None
    assert result["middle_volume"]["inverted_or_degenerate_tetrahedra"] is None
    assert result["middle_volume"]["reference_lattice_contraction_percent"] == pytest.approx(0)
    assert not result["initial_velocity_assignment"]["selected_complete_component"]
    assert not result["horizontal_exchange_isolated"]


def test_cut_codes_detect_crack_even_within_one_component(cfg, trajectory):
    middle = np.flatnonzero(piece_ids(trajectory["reference"], cfg.cut_x) == 1)
    trajectory["cut_codes_by_event"][:, middle[0]] = 8
    result = evaluate(cfg, trajectory)
    assert (
        result["fixture_validity"]["events"]["both_cuts_complete"]["middle_tetrahedra_crossing_cut_codes"] > 0
    )
    assert not result["middle_volume"]["valid_for_fragment_volume"]
    assert result["middle_volume"]["after_cuts_m3"] is None


@pytest.mark.parametrize("sizes, status", [([32, 16, 16], "unverified"), ([31, 17, 16], "mismatched")])
def test_legacy_component_sizes_gate_without_inventing_labels(cfg, trajectory, sizes, status):
    for key in ("topology_event_names", "component_labels_by_event", "cut_codes_by_event"):
        trajectory.pop(key)
    # This small fixture has 16/32/16 reference particles.
    result = evaluate(cfg, trajectory, component_sizes_by_event={"both_cuts_complete": sizes})
    assert result["fixture_validity"]["status"] == status
    assert result["middle_volume"]["contraction_percent"] is None
    assert result["initial_velocity_assignment"]["selected_complete_component"] is None


def test_missing_topology_array_is_rejected(cfg, trajectory):
    trajectory.pop("cut_codes_by_event")
    with pytest.raises(ValueError, match="Incomplete topology"):
        evaluate(cfg, trajectory)


def test_failed_backend_records_failure_and_does_not_overwrite(cfg, tmp_path):
    path = tmp_path / "failed"
    with pytest.raises(ValueError, match="requires"):
        run(cfg, path, "cressim_cpu")
    assert json.loads((path / "manifest.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        run(cfg, path, "cressim_cpu")


def test_comparison_rejects_own_backend_as_original(cfg, trajectory, tmp_path):
    paths = [tmp_path / name for name in ("bm", "cr")]
    for path in paths:
        path.mkdir()
        write_json(
            path / "manifest.json",
            {
                "experiment": "fig5_parallel_cuts",
                "status": "complete",
                "backend": "blademaster",
                "fixture": asdict(cfg),
            },
        )
        np.savez(path / "states.npz", **trajectory)
    with pytest.raises(ValueError, match="independent"):
        compare(*paths)


def test_build_validation_checks_source_and_exact_driver(tmp_path, monkeypatch):
    from blademaster_warp import fig5_cressim as adapter

    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    # A successful build.json alone cannot override the source checkout's actual revision.
    write_json(
        tmp_path / "build.json",
        {"status": "complete", "commit": adapter.CRESSIM_REVISION, "source": str(source)},
    )
    with pytest.raises(subprocess.CalledProcessError):
        validate_build(tmp_path)
    monkeypatch.setattr(adapter, "source_revision", lambda source: adapter.CRESSIM_REVISION)
    (tmp_path / "fig5_cressim_runner.cpp").write_text("stale driver")
    with pytest.raises(ValueError, match="changed since"):
        validate_build(tmp_path)


@pytest.mark.parametrize("mass_unit", [1.0, 1e-2, 1e-3, 1e-5, 1e-6])
def test_internal_mass_units_preserve_volume_wave_speed_and_impulse(cfg, mass_unit):
    material = material_in_solver_units(replace(cfg, cressim_mass_unit_kg=mass_unit))
    assert material["particle_mass"] * mass_unit == pytest.approx(cfg.density * (cfg.grid_spacing / 2) ** 3)
    assert material["particle_mass"] / material["density"] == pytest.approx((cfg.grid_spacing / 2) ** 3)
    assert material["young_modulus"] / material["density"] == pytest.approx(cfg.young_modulus / cfg.density)
    velocity_change = np.array([0.01, -0.02, 0.03])
    measured = impulses_to_si(material["particle_mass"] * velocity_change, mass_unit)
    np.testing.assert_allclose(measured, cfg.density * (cfg.grid_spacing / 2) ** 3 * velocity_change)


@pytest.mark.parametrize("fix", [False, True])
def test_comparison_matches_physics_across_internal_mass_units(cfg, trajectory, tmp_path, fix):
    paths = [tmp_path / name for name in ("bm", "cr")]
    for path, backend, unit in zip(paths, ("blademaster", "cressim_cpu"), (1.0, 0.001)):
        path.mkdir()
        write_json(
            path / "manifest.json",
            {
                "experiment": "fig5_parallel_cuts",
                "status": "complete",
                "backend": backend,
                "fixture": asdict(replace(cfg, cressim_mass_unit_kg=unit)),
                "protocol": protocol(cfg)[0],
                "source_build": {"commit": CRESSIM_REVISION, "fix_corotated_dispatch": fix},
            },
        )
        np.savez(path / "states.npz", **trajectory)
    result = compare(*paths)
    assert result["cressim_mass_unit_kg"] == 0.001
    assert result["cressim_fix_corotated_dispatch"] == fix
    assert result["fixture_comparison_status"] == "matched"
    assert "cressim_mass_unit_kg" not in result["fixture"]
    trajectory["component_labels_by_event"][:, 0] = 99
    np.savez(paths[0] / "states.npz", **trajectory)
    assert compare(*paths)["fixture_comparison_status"] == "invalid"
    meta = json.loads((paths[1] / "manifest.json").read_text())
    meta["fixture"]["young_modulus"] *= 2
    write_json(paths[1] / "manifest.json", meta)
    with pytest.raises(ValueError, match="identical fixture"):
        compare(*paths)


@pytest.mark.skipif(
    not os.environ.get("FIG5_CRESSIM_BUILD"), reason="Set FIG5_CRESSIM_BUILD for upstream integration"
)
def test_upstream_constitutive_volume_and_impulse_unit_invariance():
    from pathlib import Path

    build = Path(os.environ["FIG5_CRESSIM_BUILD"])
    meta = validate_build(build)
    executable = build / "fig5_cressim_cpu"
    values = [
        json.loads(subprocess.check_output([str(executable), "--check-units", str(unit)], text=True))
        for unit in (1, 0.01, 0.001, 1e-5, 1e-6)
    ]
    assert values[0]["volume_m3"] > 0
    assert np.linalg.norm(values[0]["stress_diagonal_pa"]) > 0
    fixed = meta["fix_corotated_dispatch"]
    assert values[0]["fix_corotated_dispatch"] == fixed
    expected = "corotated_stress_diagonal_pa" if fixed else "linear_stress_diagonal_pa"
    other = "linear_stress_diagonal_pa" if fixed else "corotated_stress_diagonal_pa"
    np.testing.assert_array_equal(values[0]["stress_diagonal_pa"], values[0][expected])
    assert not np.allclose(values[0]["stress_diagonal_pa"], values[0][other])
    for value in values[1:]:
        for key in values[0]:
            np.testing.assert_allclose(value[key], values[0][key], rtol=3e-6, atol=1e-14)


@pytest.mark.skipif(
    not os.environ.get("FIG5_CRESSIM_BUILD"), reason="Set FIG5_CRESSIM_BUILD for upstream integration"
)
@pytest.mark.parametrize("corruption", ["original", "compiled", "flag", "identity"])
def test_build_rejects_changed_header_mode_and_executable_identity(tmp_path, corruption):
    import hashlib
    from pathlib import Path
    import shutil
    from blademaster_warp.fig5_cressim import constitutive_snapshot

    build = tmp_path / "build"
    shutil.copytree(os.environ["FIG5_CRESSIM_BUILD"], build)
    meta = validate_build(build)
    original = (build / "constitutive_models.original.h").read_bytes()
    compiled = (build / "constitutive_models.h").read_bytes()
    assert original == (Path(meta["source"]) / meta["constitutive_source"]).read_bytes()
    if meta["fix_corotated_dispatch"]:
        assert (
            compiled.replace(
                b"            break;\n        case ParticleMaterialType::eLinearElastic:",
                b"        case ParticleMaterialType::eLinearElastic:",
                1,
            )
            == original
        )
    else:
        assert compiled == original
    if corruption in ("original", "compiled"):
        key = "original_header_snapshot" if corruption == "original" else "compiled_header_snapshot"
        (build / meta[key]).write_bytes(b"stale snapshot")
    else:
        meta["fix_corotated_dispatch"] = not meta["fix_corotated_dispatch"]
        if corruption == "identity":
            changed = constitutive_snapshot(original, meta["fix_corotated_dispatch"])
            (build / "constitutive_models.h").write_bytes(changed)
            meta["snapshot_integrity"]["compiled_header_sha256"] = hashlib.sha256(changed).hexdigest()
        write_json(build / "build.json", meta)
    with pytest.raises(ValueError, match="identity" if corruption == "identity" else "snapshots"):
        validate_build(build)


@pytest.mark.skipif(
    not os.environ.get("FIG5_CRESSIM_BUILD"), reason="Set FIG5_CRESSIM_BUILD for upstream integration"
)
@pytest.mark.parametrize("mass_unit", [1.0, 0.001])
def test_cressim_executes_complete_protocol(cfg, tmp_path, mass_unit):
    cfg = replace(cfg, dt=0.00025, cressim_mass_unit_kg=mass_unit, cressim_domain_margin=0.08)
    result = run(cfg, tmp_path / "original", "cressim_cpu", build_dir=os.environ["FIG5_CRESSIM_BUILD"])
    assert result["initial_velocity_assignment"]["changed_middle_right_particles"] == 0
    assert result["first_half_depth_opening"]["pairs"] == 8
    assert result["blade_contact"]["after_withdrawal_sum_interval_norms_ns"] == 0
    assert result["board_contact"]["after_withdrawal_impulse_on_material_ns"][2] > 0
    assert result["momentum_balance"]["horizontal_external_contacts_clear"]
    assert result["fixture_validity"]["status"] == "unverified"
    assert result["middle_volume"]["contraction_percent"] is None
    path = tmp_path / "original"
    meta = json.loads((path / "manifest.json").read_text())
    expected = (
        "computeCoRotationalStress"
        if meta["source_build"]["fix_corotated_dispatch"]
        else "computeLinearElasticStress"
    )
    assert meta["solver_config"]["stress_function"] == expected
    assert meta["solver_config"]["grid_mass_cutoff_kg"] == pytest.approx(1e-4 * mass_unit)
    guard = meta["solver_config"]["grid_support_guard"]
    assert guard["checked_every_substep"]
    assert guard["minimum_support_margin_m"] > 0
    header = (path / "cressim_input.txt").read_text().splitlines()[0].split()
    assert float(header[5]) == pytest.approx(cfg.density * (cfg.grid_spacing / 2) ** 3 / mass_unit)
    assert float(header[6]) == pytest.approx(cfg.young_modulus / mass_unit)
    ref = reference_particles(cfg)
    raw = np.fromfile(path / "cressim_states.bin", dtype=np.float32).reshape(-1, 6 * len(ref) + 6)
    with np.load(path / "states.npz") as states:
        np.testing.assert_array_equal(states["position"], raw[:, :-6].reshape(-1, len(ref), 6)[..., :3])
        np.testing.assert_allclose(states["board_impulse_ns"], raw[:, -3:].astype(float) * mass_unit)
        np.testing.assert_allclose(states["blade_impulse_ns"], raw[:, -6:-3].astype(float) * mass_unit)
