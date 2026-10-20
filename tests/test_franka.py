"""Dynamics checks independent of a successful-looking cutting video."""

import numpy as np
import pytest
import mujoco as mj
from blademaster_warp.franka import Franka


def test_banana_material_and_blade_match_standalone_driver():
    from dataclasses import asdict
    from types import SimpleNamespace
    from blademaster_warp.run import preset
    from blademaster_warp.franka_run import material_config
    from blademaster_warp.franka import HALF, BEVEL

    standalone = asdict(preset("paper-material"))
    robot = asdict(
        material_config(SimpleNamespace(material="paper-banana", dt=None, spacing=None, cutting_energy=None))
    )
    differences = {k for k in robot if robot[k] != standalone[k]}
    assert differences == {"external_tool", "board_half_extent"}
    assert robot["external_tool"] and not standalone["external_tool"]
    assert HALF == robot["blade_half_size"] and BEVEL == robot["blade_bevel_height"] == 0.005


@pytest.mark.parametrize("material,scene", [("fig4-cake", "fig4"), ("fig7-tofu", "fig7-shallow"), ("fig8-luncheon", "fig8")])
def test_teleop_paper_materials_use_their_scene(material, scene):
    from dataclasses import asdict
    from types import SimpleNamespace
    from blademaster_warp.franka_run import material_config
    from blademaster_warp.scenarios import scene_config

    teleop = asdict(
        material_config(
            SimpleNamespace(material=material, motion="teleop", dt=None, spacing=None, cutting_energy=None, interval=0.001)
        )
    )
    paper = asdict(
        material_config(
            SimpleNamespace(material="elastic-block", motion=scene if scene != "fig4" else "slice-lift",
                            dt=None, spacing=None, cutting_energy=None, interval=0.001)
        )
    )
    assert teleop == paper
    reference = scene_config(scene)
    for key in ("size", "shape", "grid_spacing", "young_modulus", "poisson_ratio", "density", "blade_half_size"):
        assert teleop[key] == asdict(reference)[key]


def test_piece_colors_keep_material_for_piece_zero_only():
    from blademaster_warp.material_colors import material_colormap
    from blademaster_warp.topology import piece_rgb

    reference = np.array([[0, 0, 0.0], [0, 0, 0.05], [0, 0, 0.01], [0, 0, 0.04]])
    base = material_colormap("fig4-cake", reference)(reference)
    rgb = piece_rgb(np.array([0, 0, 1, 2]), base)
    np.testing.assert_array_equal(rgb[:2], base[:2])
    assert not np.allclose(rgb[2], rgb[3]) and not np.allclose(rgb[2], base[2])


def test_explicit_material_overrides_do_not_change_other_parameters():
    from dataclasses import asdict
    from types import SimpleNamespace
    from blademaster_warp.franka_run import material_config

    base = asdict(
        material_config(SimpleNamespace(material="paper-banana", dt=None, spacing=None, cutting_energy=None))
    )
    changed = asdict(
        material_config(
            SimpleNamespace(
                material="paper-banana", dt=0.0000125, spacing=0.003, cutting_energy=750.0, interval=0.001
            )
        )
    )
    assert {k for k in base if base[k] != changed[k]} == {"dt", "grid_spacing", "cutting_energy"}


def test_banana_step_follows_grid_spacing_unless_given():
    from types import SimpleNamespace
    from blademaster_warp.franka_run import material_config

    def dt(spacing, given=None):
        args = SimpleNamespace(
            material="paper-banana", dt=given, spacing=spacing, cutting_energy=None, interval=0.001
        )
        return material_config(args).dt

    assert dt(None) == dt(0.006) == pytest.approx(62.5e-6)
    assert dt(0.003) == pytest.approx(40e-6)
    assert dt(0.003, given=31.25e-6) == 31.25e-6


def test_teleop_resolution_defaults_and_explicit_overrides():
    from blademaster_warp.franka_run import material_config, parser

    def config(*options):
        return material_config(parser().parse_args(["--material", "paper-banana", *options]))

    teleop = config("--motion", "teleop")
    assert teleop.grid_spacing == 0.006
    assert teleop.dt == pytest.approx(62.5e-6)
    assert teleop.sampling_seed == 0 and teleop.blade_contact_radius == 0
    # Resolution selection keeps physical contact stiffness and material parameters.
    fine = config("--motion", "teleop", "--spacing", "0.003")
    assert fine.grid_spacing == 0.003
    assert fine.dt == pytest.approx(40e-6)
    for field in ("young_modulus", "yield_stress", "contact_reference_dt", "sampling_seed", "flip_fraction"):
        assert getattr(teleop, field) == getattr(fine, field)
    custom = config("--motion", "teleop", "--spacing", "0.0045", "--dt", "0.000025")
    assert custom.grid_spacing == 0.0045 and custom.dt == 25e-6
    assert config("--motion", "sequence").grid_spacing == 0.006
    old = config("--motion", "teleop", "--sampling-seed", "1", "--blade-contact-radius", "0.0005")
    assert old.sampling_seed == 1 and old.blade_contact_radius == 0.0005
    with pytest.raises(ValueError, match="blade_contact_radius"):
        config("--blade-contact-radius", "-0.001")


def test_banana_breakthrough_keeps_contact_without_a_lateral_kick(tmp_path):
    from blademaster_warp.franka_run import parser
    from scripts.check_cut_release import CutReleaseRun, summarize

    output = tmp_path / "release"
    args = parser().parse_args(
        [
            "--material",
            "paper-banana",
            "--motion",
            "teleop",
            "--cut-events",
            "auto",
            "--spacing",
            "0.006",
            "--duration",
            "5.4",
            "--record",
            "--output",
            str(output),
        ]
    )
    run = CutReleaseRun(args)
    run.execute()
    result = summarize(output)
    assert result["peak_force_n"] > 0.1  # A blade passing through a sampling gap is not a valid fix.
    assert max(p["peak_lateral_speed_m_s"] for p in result["pieces"]) < 0.02
    assert max(abs(p["displacement_at_hold_m"][0]) for p in result["pieces"]) < 0.003


def test_jacobian_finite_difference():
    r = Franka()
    analytic = r.jacobian()[:3, :7]
    q = r.data.qpos.copy()
    numerical = np.zeros((3, 7))
    for j in range(7):
        r.data.qpos[:] = q
        r.data.qpos[j] += 1.0e-6
        mj.mj_forward(r.model, r.data)
        plus = r.position
        r.data.qpos[:] = q
        r.data.qpos[j] -= 1.0e-6
        mj.mj_forward(r.model, r.data)
        numerical[:, j] = (plus - r.position) / 2.0e-6
    np.testing.assert_allclose(analytic, numerical, atol=1.0e-8)


def test_wrench_virtual_work_and_effective_mass():
    r = Franka()
    wrench = np.array([2.0, -3.0, 1.0, 0.05, -0.1, 0.02])
    j = r.jacobian()
    generalized = np.zeros(r.model.nv)
    mj.mj_applyFT(r.model, r.data, wrench[:3], wrench[3:], r.position, r.body, generalized)
    np.testing.assert_allclose(generalized, j.T @ wrench, atol=1.0e-12)
    velocity = np.arange(r.model.nv) * 0.03
    np.testing.assert_allclose(velocity @ generalized, (j @ velocity) @ wrench, atol=1.0e-12)
    mobility = j @ np.linalg.solve(r.mass(), j.T)
    assert np.linalg.eigvalsh(mobility).min() > 0
    np.testing.assert_allclose(mobility @ wrench, j @ np.linalg.solve(r.mass(), generalized), atol=1.0e-12)


@pytest.mark.parametrize("controller", ["libfranka", "serl"])
def test_bias_hold_and_external_force_changes_motion(controller):
    free = Franka(controller=controller)
    held = Franka(controller=controller)
    target = free.position
    for _ in range(500):
        free.step(target)
        held.step(target, np.array([0, 0, 2.0, 0, 0, 0]))
    assert np.linalg.norm(free.position - target) < 1.0e-4
    # Reaction is not discarded and the end effector is not teleported.
    assert held.position[2] - free.position[2] > 0.002
    assert np.linalg.norm(held.data.qpos[:7] - free.data.qpos[:7]) > 0.005
    assert np.max(np.abs(held.data.ctrl[:7])) < 87


def test_frictional_grasp_lifts_and_releases_free_knife():
    r = Franka(attachment="grasp")
    assert r.model.nq == 16 and r.model.nv == 15
    assert r.model.jnt_type[r.model.joint("knife_free").id] == mj.mjtJoint.mjJNT_FREE
    # Only the upstream finger synchronizer remains; no tool weld.
    assert r.model.neq == 1
    for _ in range(500):
        r.step(r.initial_position)
    settled = r.position.copy()
    relative = r.position - r.control_position
    for i in range(1000):
        r.step(r.initial_position + [0, 0, 0.03 * min(i / 500, 1)])
    assert r.position[2] - settled[2] > 0.02
    assert np.linalg.norm(r.position - r.control_position - relative) < 0.003
    height = r.position[2]
    r.release()
    for _ in range(200):
        r.step(r.initial_position + [0, 0, 0.03])
    assert r.position[2] < height - 0.015
