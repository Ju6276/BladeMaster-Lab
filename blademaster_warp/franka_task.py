"""Measured-pose slice lifting and material-space success checks (SI units)."""

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SliceLiftSettings:
    travel_speed_m_s: float = 0.04
    descent_speed_m_s: float = 0.02
    lift_speed_m_s: float = 0.02
    angular_speed_rad_s: float = 0.5
    position_tolerance_m: float = 0.001
    # Insertion and lifting carry 3-4 N through a 1000 N/m spring.
    loaded_position_tolerance_m: float = 0.005
    orientation_tolerance_rad: float = np.deg2rad(1.0)
    arrival_hold_s: float = 0.15
    final_hold_s: float = 1.0
    tracking_timeout_s: float = 8.0
    board_clearance_m: float = 0.0005
    lift_height_m: float = 0.04
    minimum_slice_lift_m: float = 0.025
    minimum_slice_p10_lift_m: float = 0.020
    maximum_rest_motion_m: float = 0.008
    minimum_piece_fraction: float = 0.9
    maximum_hold_speed_m_s: float = 0.02

    def __post_init__(self):
        values = np.array(list(vars(self).values()))
        if not np.isfinite(values).all() or np.min(values) <= 0:
            raise ValueError("Slice-lift settings must be finite and positive")
        if not 0.5 < self.minimum_piece_fraction <= 1:
            raise ValueError("minimum_piece_fraction must be in (0.5, 1]")
        if (
            self.minimum_slice_p10_lift_m > self.minimum_slice_lift_m
            or self.minimum_slice_lift_m > self.lift_height_m
        ):
            raise ValueError("Lift thresholds must fit the commanded lift")


@dataclass(frozen=True)
class PoseSegment:
    phase: str
    position: np.ndarray
    rotation: np.ndarray
    speed: float
    cut_index: int = 0
    hold_s: float = 0.15


class SliceLift:
    phases = (
        "settle",
        "approach",
        "descend",
        "withdraw",
        "outside",
        "turn",
        "lower",
        "insert",
        "support",
        "lift",
        "verify",
        "done",
        "failed",
    )
    expected_pieces = 2

    def __init__(self, initial, initial_rotation, material_origin, settings=None):
        from .scenarios import (
            FIG4_RADIUS,
            FIG4_HEIGHT,
            FIG4_KNIFE_HALF_SIZE,
            FIG4_CUT_ANGLES_DEG,
            FIG4_CUT_CENTRE_RADIUS,
        )

        self.settings = s = settings or SliceLiftSettings()
        self.origin = np.asarray(material_origin, float)
        self.half = np.array(FIG4_KNIFE_HALF_SIZE)
        self.segments = []
        self.start_position = np.array(initial, float, copy=True)
        self.start_rotation = np.array(initial_rotation, float, copy=True)
        if (
            self.origin.shape != (3,)
            or self.start_position.shape != (3,)
            or self.start_rotation.shape != (3, 3)
            or not all(np.isfinite(v).all() for v in (self.origin, self.start_position, self.start_rotation))
        ):
            raise ValueError("Expected finite initial pose and material origin")
        self.target_rotation = self.start_rotation.copy()
        top = max(FIG4_HEIGHT + 0.02 + self.half[2], initial[2] - self.origin[2])
        bottom = self.half[2] + s.board_clearance_m

        def go(phase, local, rotation, speed=None, cut=0, hold=None):
            self.segments.append(
                PoseSegment(
                    phase,
                    self.origin + local,
                    np.asarray(rotation),
                    speed or s.travel_speed_m_s,
                    cut,
                    s.arrival_hold_s if hold is None else hold,
                )
            )

        go("settle", self.start_position - self.origin, self.start_rotation, hold=0.3)
        for index, angle in enumerate(FIG4_CUT_ANGLES_DEG):
            a = np.deg2rad(angle)
            xy = FIG4_CUT_CENTRE_RADIUS * np.array([np.cos(a), np.sin(a)])
            # A half-turn about z leaves this symmetric blade unchanged; choose
            # the equivalent yaw within +/-90 degrees to avoid winding the wrist.
            yaw = angle % 180 - 90
            rotation = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
            go("approach", np.r_[xy, top], rotation, cut=index)
            go("descend", np.r_[xy, bottom], rotation, s.descent_speed_m_s, index)
            go("withdraw", np.r_[xy, top], rotation, cut=index)
        outside = FIG4_RADIUS + self.half[2] + 0.03
        go("outside", [outside, 0, top], rotation, cut=1)
        flat = Rotation.from_euler("ZY", [-45, 90], degrees=True).as_matrix()
        go("turn", [outside, 0, top], flat, cut=1)
        # The robot knife collides with the board. Its bottom face must stay
        # above z=0, unlike the prescribed Fig4 knife with its TOP face at z=0.
        low = self.half[0] + s.board_clearance_m
        go("lower", [outside, 0, low], flat, cut=1)
        # A rectangle's 90-degree corner fits inside the 120-degree wedge.
        # Point that corner towards the apex, with both supporting edges inside
        # the radial cuts, rather than leaving a 30 mm cantilever at the apex.
        support = np.array([0.005, 0, low]) - flat @ [0, -self.half[1], -self.half[2]]
        go("insert", support, flat, s.descent_speed_m_s, 1)
        go("support", support, flat, cut=1, hold=0.3)
        go("lift", support + [0, 0, s.lift_height_m], flat, s.lift_speed_m_s, 1)
        go("verify", support + [0, 0, s.lift_height_m], flat, cut=1, hold=s.final_hold_s)
        self.index = 0
        self.phase = "settle"
        self.cut_index = 0
        self.segment_start = 0.0
        self.arrived_since = None
        self.failure_reason = None
        self.events = [dict(time_s=0.0, phase="settle", cut_index=0)]
        self.max_position_error_m = 0.0
        self.max_orientation_error_rad = 0.0
        self.last_time = -np.inf

    @property
    def done(self):
        return self.phase == "done"

    @property
    def failed(self):
        return self.phase == "failed"

    @property
    def cutting(self):
        return self.phase == "descend"

    def update(self, t, actual, rotation, edge_points):
        s = self.settings
        actual, rotation, edges = map(np.asarray, (actual, rotation, edge_points))
        if (
            not np.isfinite(t)
            or t < self.last_time
            or actual.shape != (3,)
            or rotation.shape != (3, 3)
            or edges.shape != (2, 3)
            or not all(np.isfinite(v).all() for v in (actual, rotation, edges))
        ):
            raise ValueError("Expected finite measured pose/edge points and monotonic time")
        self.last_time = t
        if self.done or self.failed:
            return self.segments[self.index].position.copy(), self.target_rotation.copy()
        segment = self.segments[self.index]
        delta = Rotation.from_matrix(segment.rotation @ self.start_rotation.T).as_rotvec()
        duration = max(
            np.linalg.norm(segment.position - self.start_position) / segment.speed,
            np.linalg.norm(delta) / s.angular_speed_rad_s,
        )
        elapsed = t - self.segment_start
        fraction = min(elapsed / duration, 1.0) if duration else 1.0
        target = self.start_position + fraction * (segment.position - self.start_position)
        self.target_rotation = Rotation.from_rotvec(fraction * delta).as_matrix() @ self.start_rotation
        position_error = np.linalg.norm(actual - target)
        angle_error = Rotation.from_matrix(self.target_rotation @ rotation.T).magnitude()
        self.max_position_error_m = max(self.max_position_error_m, float(position_error))
        self.max_orientation_error_rad = max(self.max_orientation_error_rad, float(angle_error))
        position_tolerance = (
            s.loaded_position_tolerance_m
            if self.phase in ("insert", "support", "lift", "verify")
            else s.position_tolerance_m
        )
        arrived = (
            elapsed >= duration
            and position_error <= position_tolerance
            and angle_error <= s.orientation_tolerance_rad
        )
        if self.phase == "descend":
            arrived = (
                arrived
                and np.max(np.abs(edges[:, 2] - self.origin[2] - s.board_clearance_m))
                <= s.position_tolerance_m
            )
        if not arrived:
            self.arrived_since = None
        elif self.arrived_since is None:
            self.arrived_since = t
        if self.arrived_since is not None and t - self.arrived_since >= segment.hold_s:
            self.index += 1
            if self.index == len(self.segments):
                self.index -= 1
                self.phase = "done"
            else:
                self.phase = self.segments[self.index].phase
                self.cut_index = self.segments[self.index].cut_index
                # Preserve the spring preload when a loaded motion becomes a
                # hold. Restarting at the sagged measured pose drops the slice.
                loaded = segment.phase in ("insert", "support", "lift")
                self.start_position = segment.position.copy() if loaded else actual.copy()
                self.start_rotation = segment.rotation.copy() if loaded else rotation.copy()
                self.segment_start = t
                self.arrived_since = None
            self.events.append(dict(time_s=float(t), phase=self.phase, cut_index=self.cut_index))
        elif elapsed > duration + segment.hold_s + s.tracking_timeout_s:
            self.failure_reason = f"pose_not_reached:{self.phase}"
            self.phase = "failed"
            self.events.append(dict(time_s=float(t), phase=self.phase, reason=self.failure_reason))
        return target, self.target_rotation.copy()


class SliceLiftMetrics:
    """Track a reference-space wedge, preserving particle identity through cuts."""

    def __init__(self, reference, settings=None):
        self.settings = settings or SliceLiftSettings()
        self.reference = np.array(reference, float, copy=True)
        x, y = self.reference[:, :2].T
        self.slice = (x > 0) & (np.abs(y) < np.sqrt(3) * x)
        if not self.slice.any() or self.slice.all():
            raise ValueError("Slice and remaining body must both contain particles")
        self.baseline = None
        self.observations = []

    def observe(self, t, phase, positions, labels, velocities):
        x, labels, velocities = map(np.asarray, (positions, labels, velocities))
        if x.shape != self.reference.shape or labels.shape != (len(x),) or velocities.shape != x.shape:
            raise ValueError("Task observations must preserve reference particle identity")
        if not np.isfinite(x).all() or not np.isfinite(velocities).all():
            raise ValueError("Nonfinite slice-lift observation")
        # Freeze the settled configuration before the first cut; insertion itself
        # can move the slice and must not redefine zero lift.
        if phase == "settle":
            self.baseline = x.copy()
        component, count = np.unique(labels[self.slice], return_counts=True)
        target_id = component[np.argmax(count)]
        target = labels == target_id
        overlap = np.count_nonzero(target & self.slice)
        recall = overlap / np.count_nonzero(self.slice)
        purity = overlap / np.count_nonzero(target)
        _, sizes = np.unique(labels, return_counts=True)
        major_pieces = int(np.count_nonzero(sizes >= 0.05 * len(x)))
        delta = x - (self.baseline if self.baseline is not None else self.reference)
        observation = dict(
            time_s=float(t),
            phase=phase,
            major_pieces=major_pieces,
            slice_component_recall=float(recall),
            slice_component_purity=float(purity),
            slice_median_lift_m=float(np.median(delta[self.slice, 2])),
            slice_p10_lift_m=float(np.quantile(delta[self.slice, 2], 0.1)),
            rest_p90_motion_m=float(np.quantile(np.linalg.norm(delta[~self.slice], axis=1), 0.9)),
            slice_p90_speed_m_s=float(np.quantile(np.linalg.norm(velocities[self.slice], axis=1), 0.9)),
        )
        self.observations.append(observation)
        return observation

    def evaluate(self, completed):
        s = self.settings
        samples = [o for o in self.observations if o["phase"] in ("verify", "done")]
        span = samples[-1]["time_s"] - samples[0]["time_s"] if samples else 0.0
        # Ignore the initial settling half of the final hold, but require a
        # sustained observation window so a transient bounce cannot pass.
        held = (
            [o for o in samples if o["time_s"] >= samples[0]["time_s"] + s.final_hold_s / 2]
            if samples
            else []
        )
        observed = self.baseline is not None and len(held) >= 3 and span >= 0.9 * s.final_hold_s
        separated = bool(held) and all(
            o["major_pieces"] == 2
            and min(o["slice_component_recall"], o["slice_component_purity"]) >= s.minimum_piece_fraction
            for o in held
        )
        lifted = bool(held) and all(
            o["slice_median_lift_m"] >= s.minimum_slice_lift_m
            and o["slice_p10_lift_m"] >= s.minimum_slice_p10_lift_m
            for o in held
        )
        rest_stable = bool(held) and all(o["rest_p90_motion_m"] <= s.maximum_rest_motion_m for o in held)
        held_stable = bool(held) and all(o["slice_p90_speed_m_s"] <= s.maximum_hold_speed_m_s for o in held)
        return dict(
            success=bool(completed and observed and separated and lifted and rest_stable and held_stable),
            cycle_completed=bool(completed),
            material_observed=bool(self.observations),
            hold_observed=bool(observed),
            slice_separated=separated,
            slice_lifted=lifted,
            remainder_stable=rest_stable,
            slice_held_stably=held_stable,
            verification_samples=len(held),
            verification_span_s=span,
            final=self.observations[-1] if self.observations else None,
            thresholds=dict(
                slice_median_lift_m=s.minimum_slice_lift_m,
                slice_p10_lift_m=s.minimum_slice_p10_lift_m,
                rest_p90_motion_m=s.maximum_rest_motion_m,
                piece_purity_and_recall=s.minimum_piece_fraction,
                slice_p90_speed_m_s=s.maximum_hold_speed_m_s,
                hold_s=s.final_hold_s,
            ),
        )


@dataclass(frozen=True)
class PaperTaskSettings:
    """Franka tracking limits for the prescribed Fig. 7 and Fig. 8 paths."""

    travel_speed_m_s: float = 0.04
    descent_speed_m_s: float = 0.02
    angular_speed_rad_s: float = 0.5
    # A few newtons through the 1000 N/m spring sags several millimetres.
    position_tolerance_m: float = 0.006
    orientation_tolerance_rad: float = np.deg2rad(3.0)
    arrival_hold_s: float = 0.15
    final_hold_s: float = 1.0
    tracking_timeout_s: float = 12.0
    minimum_cut_particles: int = 1
    minimum_piece_fraction: float = 0.8
    # Inner slices lean on the ones already tipped, so their median shift is a few millimetres.
    minimum_slice_shift_m: float = 0.002

    def __post_init__(self):
        values = np.array(list(vars(self).values()))
        if not np.isfinite(values).all() or np.min(values) <= 0:
            raise ValueError("Paper-task settings must be finite and positive")


class PaperTask:
    """Follow a prescribed Fig. 7 or Fig. 8 keyframe path with the measured Franka pose.

    Cutting stays active for the whole motion that leaves a keyframe whose cut
    index is set, matching ``scenarios.command_at``.
    """

    def __init__(self, name, initial, initial_rotation, material_origin, settings=None):
        from .scenarios import keyframes

        if name not in ("fig7-shallow", "fig7-deep", "fig8"):
            raise ValueError(name)
        self.name = name
        self.settings = s = settings or PaperTaskSettings()
        self.expected_pieces = 5 if name == "fig8" else 1
        self.phases = ("settle", "move", "cut", "hold", "verify", "done", "failed")
        self.origin = np.asarray(material_origin, float)
        self.start_position = np.array(initial, float, copy=True)
        self.start_rotation = np.array(initial_rotation, float, copy=True)
        if (
            self.origin.shape != (3,)
            or self.start_position.shape != (3,)
            or self.start_rotation.shape != (3, 3)
            or not all(np.isfinite(v).all() for v in (self.origin, self.start_position, self.start_rotation))
        ):
            raise ValueError("Expected finite initial pose and material origin")
        self.segments = [
            PoseSegment("settle", self.start_position.copy(), self.start_rotation.copy(), s.travel_speed_m_s, 0, 0.3)
        ]
        keys = keyframes(name)
        last_cut = 0
        for index in range(len(keys) - 1):
            source, dest = keys[index], keys[index + 1]
            local = np.asarray(dest[1], float)
            rotation = Rotation.from_euler("ZYX", dest[2], degrees=True).as_matrix()
            if source[3] >= 0:
                last_cut = int(source[3])
                phase = "cut"
            elif np.allclose(local, np.asarray(source[1], float), atol=1e-9):
                phase = "hold"
            else:
                phase = "move"
            drop = self.origin[2] + float(source[1][2]) - (self.origin[2] + local[2])
            speed = s.descent_speed_m_s if drop > 1e-4 else s.travel_speed_m_s
            self.segments.append(
                PoseSegment(phase, self.origin + local, rotation, speed, last_cut, s.arrival_hold_s)
            )
        self.segments[-1] = PoseSegment(
            "verify",
            self.segments[-1].position,
            self.segments[-1].rotation,
            self.segments[-1].speed,
            self.segments[-1].cut_index,
            s.final_hold_s,
        )
        self.index = 0
        self.phase = "settle"
        self.cut_index = 0
        self.segment_start = 0.0
        self.arrived_since = None
        self.failure_reason = None
        self.events = [dict(time_s=0.0, phase="settle", cut_index=0)]
        self.max_position_error_m = 0.0
        self.max_orientation_error_rad = 0.0
        self.last_time = -np.inf
        self.target_rotation = self.start_rotation.copy()

    @property
    def done(self):
        return self.phase == "done"

    @property
    def failed(self):
        return self.phase == "failed"

    @property
    def cutting(self):
        return self.phase == "cut"

    def update(self, t, actual, rotation, edge_points):
        s = self.settings
        actual, rotation, edges = map(np.asarray, (actual, rotation, edge_points))
        if (
            not np.isfinite(t)
            or t < self.last_time
            or actual.shape != (3,)
            or rotation.shape != (3, 3)
            or edges.shape != (2, 3)
            or not all(np.isfinite(v).all() for v in (actual, rotation, edges))
        ):
            raise ValueError("Expected finite measured pose/edge points and monotonic time")
        self.last_time = t
        if self.done or self.failed:
            return self.segments[self.index].position.copy(), self.target_rotation.copy()
        segment = self.segments[self.index]
        delta = Rotation.from_matrix(segment.rotation @ self.start_rotation.T).as_rotvec()
        duration = max(
            np.linalg.norm(segment.position - self.start_position) / segment.speed,
            np.linalg.norm(delta) / s.angular_speed_rad_s,
        )
        elapsed = t - self.segment_start
        fraction = min(elapsed / duration, 1.0) if duration else 1.0
        target = self.start_position + fraction * (segment.position - self.start_position)
        self.target_rotation = Rotation.from_rotvec(fraction * delta).as_matrix() @ self.start_rotation
        position_error = np.linalg.norm(actual - target)
        angle_error = Rotation.from_matrix(self.target_rotation @ rotation.T).magnitude()
        self.max_position_error_m = max(self.max_position_error_m, float(position_error))
        self.max_orientation_error_rad = max(self.max_orientation_error_rad, float(angle_error))
        arrived = (
            elapsed >= duration
            and position_error <= s.position_tolerance_m
            and angle_error <= s.orientation_tolerance_rad
        )
        if not arrived:
            self.arrived_since = None
        elif self.arrived_since is None:
            self.arrived_since = t
        if self.arrived_since is not None and t - self.arrived_since >= segment.hold_s:
            self.index += 1
            if self.index == len(self.segments):
                self.index -= 1
                self.phase = "done"
            else:
                self.phase = self.segments[self.index].phase
                self.cut_index = self.segments[self.index].cut_index
                self.start_position = actual.copy()
                self.start_rotation = rotation.copy()
                self.segment_start = t
                self.arrived_since = None
            self.events.append(dict(time_s=float(t), phase=self.phase, cut_index=int(self.cut_index)))
        elif elapsed > duration + segment.hold_s + s.tracking_timeout_s:
            self.failure_reason = f"pose_not_reached:{self.phase}"
            self.phase = "failed"
            self.events.append(dict(time_s=float(t), phase=self.phase, reason=self.failure_reason))
        return target, self.target_rotation.copy()


class PaperTaskMetrics:
    """Success checks for a Franka Fig. 7 cross score or Fig. 8 slice sequence."""

    def __init__(self, name, reference, settings=None):
        if name not in ("fig7-shallow", "fig7-deep", "fig8"):
            raise ValueError(name)
        self.name = name
        self.settings = settings or PaperTaskSettings()
        self.reference = np.array(reference, float, copy=True)
        self.ncuts = 4 if name == "fig8" else 2
        self.expected_pieces = 5 if name == "fig8" else 1
        self.baseline = None
        self.observations = []
        if name == "fig8":
            from .scenarios import FIG8_CUTS, FIG8_SIZE, FIG8_SLICE

            edge = FIG8_SIZE[0] / 2
            bounds = [edge - index * FIG8_SLICE for index in range(FIG8_CUTS + 1)]
            x = self.reference[:, 0]
            self.slices = [(x < bounds[index]) & (x >= bounds[index + 1]) for index in range(FIG8_CUTS)]
            self.remainder = x < bounds[-1]
            if any(not band.any() for band in self.slices) or not self.remainder.any():
                raise ValueError("Fig. 8 slice bands and the remainder must all contain particles")

    def observe(self, t, phase, positions, labels, velocities, codes):
        x, labels, velocities, codes = map(np.asarray, (positions, labels, velocities, codes))
        if x.shape != self.reference.shape or labels.shape != (len(x),) or velocities.shape != x.shape:
            raise ValueError("Task observations must preserve reference particle identity")
        if codes.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(velocities).all():
            raise ValueError("Nonfinite paper-task observation")
        if phase == "settle":
            self.baseline = x.copy()
        _, sizes = np.unique(labels, return_counts=True)
        major = int(np.count_nonzero(sizes >= 0.05 * len(x)))
        cuts = [int(np.count_nonzero((codes // 3**index) % 3)) for index in range(self.ncuts)]
        observation = dict(
            time_s=float(t),
            phase=phase,
            major_pieces=major,
            cut_particles=cuts,
            lowest_particle_m=float(x[:, 2].min()),
        )
        if self.name == "fig8" and self.baseline is not None:
            delta = x - self.baseline
            observation["slice_median_shift_m"] = [
                float(np.median(delta[band, 0])) for band in self.slices
            ]
            observation["slice_purity"] = []
            for band in self.slices:
                chosen, count = np.unique(labels[band], return_counts=True)
                target = chosen[np.argmax(count)]
                member = labels == target
                overlap = np.count_nonzero(member & band)
                observation["slice_purity"].append(
                    float(min(overlap / band.sum(), overlap / member.sum()))
                )
        self.observations.append(observation)
        return observation

    def evaluate(self, completed):
        s = self.settings
        samples = [o for o in self.observations if o["phase"] in ("verify", "done")]
        span = samples[-1]["time_s"] - samples[0]["time_s"] if samples else 0.0
        held = (
            [o for o in samples if o["time_s"] >= samples[0]["time_s"] + s.final_hold_s / 2]
            if samples
            else []
        )
        observed = self.baseline is not None and len(held) >= 3 and span >= 0.9 * s.final_hold_s
        above_board = bool(held) and all(o["lowest_particle_m"] > -0.003 for o in held)
        cuts_present = bool(held) and all(
            all(count >= s.minimum_cut_particles for count in o["cut_particles"]) for o in held
        )
        if self.name == "fig8":
            separated = bool(held) and all(o["major_pieces"] >= self.expected_pieces for o in held)
            shifted = bool(held) and all(
                all(shift >= s.minimum_slice_shift_m for shift in o["slice_median_shift_m"]) for o in held
            )
            pure = bool(held) and all(
                all(purity >= s.minimum_piece_fraction for purity in o["slice_purity"]) for o in held
            )
            success = bool(completed and observed and above_board and cuts_present and separated and shifted and pure)
        else:
            separated = bool(held) and all(o["major_pieces"] == 1 for o in held)
            shifted = pure = None
            success = bool(completed and observed and above_board and cuts_present and separated)
        return dict(
            success=success,
            cycle_completed=bool(completed),
            material_observed=bool(self.observations),
            hold_observed=bool(observed),
            above_board=above_board,
            cuts_present=cuts_present,
            pieces_match=separated,
            slices_shifted=shifted,
            slices_pure=pure,
            verification_samples=len(held),
            verification_span_s=span,
            final=self.observations[-1] if self.observations else None,
            thresholds=dict(
                expected_pieces=self.expected_pieces,
                minimum_cut_particles=s.minimum_cut_particles,
                minimum_slice_shift_m=s.minimum_slice_shift_m,
                piece_purity=s.minimum_piece_fraction,
                hold_s=s.final_hold_s,
            ),
        )
