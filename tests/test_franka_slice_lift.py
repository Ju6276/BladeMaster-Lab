import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from blademaster_warp.franka_task import SliceLift, SliceLiftMetrics, SliceLiftSettings


def task():
    return SliceLift([0, 0, 0.116], np.eye(3), np.zeros(3))


def edges(position, rotation):
    return np.array([[0, -0.05, -0.04], [0, 0.05, -0.04]]) @ rotation.T + position


def follow(controller, until=65):
    position = controller.start_position.copy()
    rotation = controller.start_rotation.copy()
    cuts = set()
    minimum = np.inf
    corners = np.array([[x, y, z] for x in (-0.0005, 0.0005) for y in (-0.05, 0.05) for z in (-0.04, 0.04)])
    for t in np.arange(0, until, 0.01):
        position, rotation = controller.update(t, position, rotation, edges(position, rotation))
        minimum = min(minimum, (corners @ rotation.T + position)[:, 2].min())
        if controller.cutting:
            cuts.add(controller.cut_index)
        elif controller.phase in ("turn", "lower", "insert", "support", "lift", "verify"):
            assert not controller.cutting
        if controller.done or controller.failed:
            break
    return cuts, minimum


def test_slice_lift_path_cuts_twice_and_keeps_the_flat_knife_above_board():
    controller = task()
    cuts, minimum = follow(controller)
    assert controller.done and not controller.failed
    assert cuts == {0, 1}
    assert minimum >= controller.settings.board_clearance_m - 1e-6
    assert [e["phase"] for e in controller.events].count("descend") == 2
    support = next(segment for segment in controller.segments if segment.phase == "support")
    corners = np.array([[0, y, z] for y in (-0.05, 0.05) for z in (-0.04, 0.04)])
    footprint = corners @ support.rotation.T + support.position
    # The support rectangle stays inside the wedge, clear of the remaining body.
    assert np.all(np.abs(footprint[:, 1]) <= np.sqrt(3) * footprint[:, 0])
    np.testing.assert_allclose(footprint[0], [0.005, 0, 0.001], atol=1e-12)
    # Keep a 20 mm COM margin even after 5 mm drift along either horizontal axis.
    for dx in (-0.005, 0.005):
        for dy in (-0.005, 0.005):
            com = (np.array([0.04415 + dx, dy, 0.0275]) - support.position) @ support.rotation
            assert min(0.05 - abs(com[1]), 0.04 - abs(com[2])) > 0.02
    np.testing.assert_allclose(controller.segments[-1].position, support.position + [0, 0, 0.04])


def test_approach_waits_for_orientation_and_times_out_without_cutting():
    controller = task()
    actual, rotation = controller.start_position.copy(), np.eye(3)
    for t in np.arange(0, 15, 0.01):
        controller.update(t, actual, rotation, edges(actual, rotation))
        actual = controller.segments[controller.index].position.copy()
        if controller.failed:
            break
    assert controller.failed
    assert controller.failure_reason == "pose_not_reached:approach"
    assert not controller.cutting
    assert not any(e["phase"] == "descend" for e in controller.events)


def test_descent_rejects_one_high_edge_even_with_an_arrived_origin():
    controller = task()
    actual, rotation = controller.start_position.copy(), np.eye(3)
    for t in np.arange(0, 25, 0.01):
        measured_edges = edges(actual, rotation)
        if controller.phase == "descend":
            measured_edges[1, 2] += 0.004
        actual, rotation = controller.update(t, actual, rotation, measured_edges)
        if controller.failed:
            break
    assert controller.failure_reason == "pose_not_reached:descend"
    assert not any(e["phase"] == "withdraw" for e in controller.events)


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan])
def test_invalid_task_settings(value):
    with pytest.raises(ValueError):
        SliceLiftSettings(angular_speed_rad_s=value)


def measurements(mode="success"):
    # Uniform-density cylinder-like reference points, with a known 120-degree wedge.
    angle = np.linspace(-np.pi, np.pi, 90, endpoint=False)
    reference = np.array([[0.05 * np.cos(a), 0.05 * np.sin(a), z] for a in angle for z in (0.01, 0.03)])
    meter = SliceLiftMetrics(reference)
    labels = meter.slice.astype(int)
    x = reference.copy()
    speed = np.zeros_like(x)
    meter.observe(0, "settle", x, np.zeros(len(x), int), speed)
    x[meter.slice, 2] += 0.04
    if mode == "attached":
        labels[:] = 0
    if mode == "whole_body":
        x[~meter.slice, 2] += 0.04
    if mode == "wrong_piece":
        x = reference.copy()
        x[~meter.slice, 2] += 0.04
    if mode == "mostly_low":
        x[np.flatnonzero(meter.slice)[:15], 2] -= 0.04
    if mode == "moving":
        speed[meter.slice, 2] = 0.1
    for t in np.arange(1, 2.01, 0.04):
        meter.observe(t, "verify", x, labels, speed)
    return meter


def test_success_requires_a_separated_sustained_slice_lift_and_stationary_remainder():
    result = measurements().evaluate(True)
    assert result["success"]
    assert result["final"]["slice_median_lift_m"] == pytest.approx(0.04)
    assert not measurements().evaluate(False)["success"]


@pytest.mark.parametrize("mode", ["attached", "whole_body", "wrong_piece", "mostly_low", "moving"])
def test_material_metrics_reject_false_success(mode):
    assert not measurements(mode).evaluate(True)["success"]


def test_transient_lift_or_missing_baseline_cannot_pass():
    meter = measurements()
    meter.observations = meter.observations[-2:]
    assert not meter.evaluate(True)["success"]
    meter = measurements()
    meter.baseline = None
    assert not meter.evaluate(True)["success"]


def test_particle_order_or_label_number_does_not_choose_the_wrong_piece():
    meter = measurements()
    x = meter.reference.copy()
    x[meter.slice, 2] += 0.04
    labels = np.where(meter.slice, 42, 100)
    obs = meter.observe(3, "done", x, labels, np.zeros_like(x))
    assert obs["slice_component_recall"] == obs["slice_component_purity"] == 1


def test_fig4_robot_material_config_preserves_geometry_and_contact_time_scale():
    pytest.importorskip("mujoco")
    from blademaster_warp.franka_run import material_config, parser
    from blademaster_warp.scenarios import scene_config

    args = parser().parse_args(["--motion", "slice-lift"])
    cfg = material_config(args)
    source = scene_config("fig4")
    assert cfg.shape == "cylinder"
    assert cfg.blade_half_size == source.blade_half_size
    assert cfg.size == source.size
    assert cfg.dt == 0.001 and cfg.contact_reference_dt == source.dt
    assert cfg.contact_scales()[0] == pytest.approx(0.25)
    assert cfg.board_half_extent == (0.17, 0.16)


def test_full_torque_robot_path_reaches_both_cuts_and_flat_lift(tmp_path):
    pytest.importorskip("mujoco")
    from blademaster_warp.franka_run import FrankaCuttingRun, parser

    run = FrankaCuttingRun(
        parser().parse_args(
            [
                "--motion",
                "slice-lift",
                "--no-material",
                "--output",
                str(tmp_path / "dry"),
            ]
        )
    )
    q0 = run.robot.data.qpos.copy()
    result = run.execute()
    assert result["motion_status"] == "completed", result
    assert result["final_phase"] == "done"
    assert not result["task_evaluation"]["success"]  # Reaching the pose alone does not lift material.
    assert np.max(np.abs(run.robot.data.qpos - q0)) > 0.5
    np.testing.assert_allclose(run.robot.position, run.cycle.segments[-1].position, atol=0.001)
    assert Rotation.from_matrix(
        run.robot.rotation @ run.cycle.segments[-1].rotation.T
    ).magnitude() < np.deg2rad(1)
    assert not any(run.robot.data.warning.number)
    assert run.robot.blade_half_size == run.cfg.blade_half_size
    assert run.robot.model.body("knife").mass[0] == pytest.approx(run.cfg.blade_mass + 0.03)
    assert result["saturated_steps"] == 0


@pytest.mark.parametrize("sag, completed", [(0.004, True), (0.008, False)])
def test_loaded_arrival_tolerance_accepts_spring_sag_but_not_a_missed_lift(sag, completed):
    controller = task()
    actual, rotation = controller.start_position.copy(), np.eye(3)
    for t in np.arange(0, 65, 0.01):
        actual, rotation = controller.update(t, actual, rotation, edges(actual, rotation))
        if controller.phase in ("lift", "verify"):
            actual = actual - [0, 0, sag]
        if controller.done or controller.failed:
            break
    assert controller.done == completed
    if not completed:
        assert controller.failure_reason == "pose_not_reached:lift"


def test_gpu_slice_lift_separates_and_holds_fig4_wedge(tmp_path):
    pytest.importorskip("mujoco")
    wp = pytest.importorskip("warp")
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    from blademaster_warp.franka_run import FrankaCuttingRun, parser
    from blademaster_warp.live import LiveState

    class SupportObserver(LiveState):
        def initialize(self, robot, solver, surface, material=None):
            ref = solver.reference
            self.slice = (ref[:, 0] > 0) & (abs(ref[:, 1]) < np.sqrt(3) * ref[:, 0])
            self.crossed = np.zeros(len(ref), bool)
            self.previous_support = None
            super().initialize(robot, solver, surface, material)

        def publish(self, robot, solver, surface, x, codes, wall_s, phase, wrench, **kwargs):
            if x is not None and phase in ("insert", "support", "lift", "verify", "done"):
                local = (x - (robot.position - robot.material_origin)) @ robot.rotation
                hx, hy, hz = robot.blade_half_size
                normal_height = -local[:, 0]
                # Stay a particle radius away from the bevel and every lateral
                # edge: crossing here is penetration, not slipping off the blade.
                margin = np.minimum.reduce(
                    [
                        hy - abs(local[:, 1]),
                        local[:, 2] + hz - robot.blade_bevel_height,
                        hz - local[:, 2],
                    ]
                )
                if self.previous_support is not None:
                    previous_height, previous_margin = self.previous_support
                    self.crossed |= (
                        self.slice
                        & (previous_height > 0)
                        & (normal_height <= 0)
                        & (previous_margin > solver.radius)
                        & (margin > solver.radius)
                    )
                self.previous_support = normal_height, margin
            super().publish(robot, solver, surface, x, codes, wall_s, phase, wrench, **kwargs)

    live = SupportObserver()
    live.autostart = True
    run = FrankaCuttingRun(
        parser().parse_args(
            [
                "--motion",
                "slice-lift",
                "--surface",
                "--surface-embedding",
                "cuda",
                "--surface-ccd",
                "cuda",
                "--output",
                str(tmp_path / "lift"),
            ]
        ),
        observer=live,
    )
    report = run.execute()
    assert report["motion_status"] == "completed", report
    assert report["task_evaluation"]["success"], report["task_evaluation"]
    assert report["peak_material_force_n"] > 1.0
    assert len(report["components"]) == 2
    assert report["min_det_F"] > 0.5
    assert report["gpu_errors"] == [0, 0, 0]
    assert report["saturated_steps"] == 0
    assert live.snapshot["phase"] == "done"
    assert live.snapshot["report"]["task_evaluation"]["success"]
    assert np.linalg.norm(live.snapshot["wrench"][:3]) > 1.0
    assert len(live.snapshot["particles"]) == 3 * report["particle_count"]
    assert len(live.snapshot["vertices"]) > 0
    assert not live.crossed.any(), f"{live.crossed.sum()} slice particles crossed the blade interior"
    assert not run.data.particles


def test_loaded_lift_to_hold_preserves_command_continuity():
    controller = task()
    actual, rotation = controller.start_position.copy(), np.eye(3)
    previous = actual.copy()
    previous_phase = controller.phase
    saw_hold = False
    for t in np.arange(0, 65, 0.01):
        target, rotation = controller.update(t, actual, rotation, edges(actual, rotation))
        if controller.phase == "verify" or previous_phase == "verify":
            assert np.linalg.norm(target - previous) <= 0.000401
        actual = target.copy()
        if controller.phase in ("lift", "verify"):
            actual -= [0, 0, 0.004]
        saw_hold |= controller.phase == "verify"
        previous = target
        previous_phase = controller.phase
        if controller.done or controller.failed:
            break
    assert saw_hold and controller.done


@pytest.mark.parametrize(
    "options, message",
    [
        (["--attachment", "grasp"], "attachment fixed"),
        (["--layout", "side"], "layout paper"),
        (["--observation-interval", "0.5"], "observation interval"),
    ],
)
def test_slice_lift_rejects_unsupported_setup_before_creating_output(tmp_path, options, message):
    pytest.importorskip("mujoco")
    from blademaster_warp.franka_run import FrankaCuttingRun, parser

    out = tmp_path / "bad"
    with pytest.raises(ValueError, match=message):
        FrankaCuttingRun(parser().parse_args(["--motion", "slice-lift", "--output", str(out), *options]))
    assert not out.exists()


def test_gpu_slice_lift_surface_live_preview_uses_fig4_geometry(tmp_path):
    pytest.importorskip("mujoco")
    wp = pytest.importorskip("warp")
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    from blademaster_warp.franka_run import FrankaCuttingRun, parser
    from blademaster_warp.live import LiveState

    live = LiveState()
    live.autostart = True
    run = FrankaCuttingRun(
        parser().parse_args(
            [
                "--motion",
                "slice-lift",
                "--duration",
                ".042",
                "--surface",
                "--surface-embedding",
                "cuda",
                "--surface-ccd",
                "cuda",
                "--output",
                str(tmp_path / "preview"),
            ]
        ),
        observer=live,
    )
    report = run.execute()
    assert report["status"] == "completed"
    assert not report["task_evaluation"]["success"]
    assert live.snapshot["phase"] == "settle"
    assert len(live.snapshot["vertices"]) > 0
    assert len(live.snapshot["triangles"]) > 0
    assert live.asset["material_grid_spacing_m"] == 0.01
