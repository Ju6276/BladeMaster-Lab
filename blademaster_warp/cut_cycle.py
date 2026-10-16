"""Descend / withdraw sequencing from the measured knife pose."""

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class CycleSettings:
    descent_speed_m_s: float = 0.02
    withdrawal_speed_m_s: float = 0.025
    position_tolerance_m: float = 0.0005
    initial_hold_s: float = 0.3
    bottom_hold_s: float = 0.1
    final_hold_s: float = 0.5

    def __post_init__(self):
        values = np.array(list(asdict(self).values()))
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Cycle speeds, tolerances and hold times must be positive and finite")


class CutCycle:
    phases = ("settle", "descend", "withdraw", "verify", "done")
    cut_index = 0
    expected_pieces = 2

    def __init__(self, initial, board_z, half_height, settings=None):
        self.settings = settings or CycleSettings()
        self.initial = np.array(initial, dtype=float, copy=True)
        if self.initial.shape != (3,) or not np.isfinite(self.initial).all():
            raise ValueError("Expected finite initial tool position")
        if not np.isfinite([board_z, half_height]).all() or half_height <= 0:
            raise ValueError("Expected finite board height and positive blade half height")
        self.board_z = float(board_z)
        self.bottom = self.initial.copy()
        self.bottom[2] = board_z + half_height
        if self.initial[2] <= self.bottom[2]:
            raise ValueError("Initial knife must be above the board")
        self.phase = "settle"
        self.phase_start = 0.0
        self.arrived_since = None
        self.last_time = None
        self.events = [dict(time_s=0.0, phase=self.phase)]
        self.withdraw_from = self.bottom.copy()

    @property
    def done(self):
        return self.phase == "done"

    @property
    def cutting(self):
        return self.phase == "descend"

    def transition(self, phase, t, actual):
        self.phase = phase
        self.phase_start = t
        self.arrived_since = None
        self.events.append(dict(time_s=float(t), phase=phase, actual_tool_position_m=actual.tolist()))

    def held(self, arrived, t, duration):
        if not arrived:
            self.arrived_since = None
            return False
        if self.arrived_since is None:
            self.arrived_since = t
        return t - self.arrived_since >= duration - 1.0e-10

    def update(self, t, actual, edge_points):
        """Return a tool-origin target; actual pose is never overwritten.

        Both ends of the finite edge must reach the board-height tolerance,
        preventing a tilted edge with only one low corner from passing.
        If resisted above the goal, hold the endpoint and eventually let the
        runner report timeout. Elapsed time alone never triggers withdrawal.
        """
        actual = np.asarray(actual, dtype=float)
        edges = np.asarray(edge_points, dtype=float)
        if (
            not np.isfinite(t)
            or actual.shape != (3,)
            or edges.shape != (2, 3)
            or not np.isfinite(actual).all()
            or not np.isfinite(edges).all()
            or (self.last_time is not None and t < self.last_time)
        ):
            raise ValueError("Invalid/nonmonotonic cycle observation")
        self.last_time = t
        s = self.settings
        if self.phase == "settle":
            if self.held(
                np.linalg.norm(actual - self.initial) <= s.position_tolerance_m, t, s.initial_hold_s
            ):
                self.transition("descend", t, actual)
        elif self.phase == "descend":
            arrived = (
                np.linalg.norm(actual - self.bottom) <= s.position_tolerance_m
                and np.max(np.abs(edges[:, 2] - self.board_z)) <= s.position_tolerance_m
            )
            if self.held(arrived, t, s.bottom_hold_s):
                self.withdraw_from = actual.copy()
                self.transition("withdraw", t, actual)
        elif self.phase == "withdraw":
            if np.linalg.norm(actual - self.initial) <= s.position_tolerance_m:
                self.transition("verify", t, actual)
        elif self.phase == "verify":
            if self.held(np.linalg.norm(actual - self.initial) <= s.position_tolerance_m, t, s.final_hold_s):
                self.transition("done", t, actual)
        if self.phase == "descend":
            distance = self.initial[2] - self.bottom[2]
            # Endpoint interpolation defines the finite cut task, not a clamp
            # on robot state or on general teleoperation workspace.
            fraction = min((t - self.phase_start) * s.descent_speed_m_s / distance, 1.0)
            return (1 - fraction) * self.initial + fraction * self.bottom
        if self.phase == "withdraw":
            distance = np.linalg.norm(self.initial - self.withdraw_from)
            fraction = min((t - self.phase_start) * s.withdrawal_speed_m_s / distance, 1.0)
            return (1 - fraction) * self.withdraw_from + fraction * self.initial
        return self.initial.copy()


def evaluate_cut(observations, cycle_done, expected_pieces=2):
    """Independent post-withdrawal check, without visual-surface shortcuts."""
    verification = [r for r in observations if r["phase"] in ("verify", "done")]
    separated = bool(verification) and all(len(r["component_sizes"]) == expected_pieces for r in verification)
    cleared = bool(verification) and all(r["knife_clear_of_material"] for r in verification)
    return dict(
        success=bool(cycle_done and separated and cleared),
        cycle_completed=bool(cycle_done),
        expected_pieces=expected_pieces,
        pieces_persist_after_withdrawal=separated,
        knife_clear_after_withdrawal=cleared,
        verification_samples=len(verification),
        criteria="cycle completes; the expected number of mechanical pieces and no material above the retracted blade edge throughout the final hold",
    )


@dataclass(frozen=True)
class SequenceSettings:
    # Cut positions along the banana (x) relative to the start pose, cut in this order.
    cut_offsets_m: tuple = (0.04, 0.02, 0.0)
    descent_speed_m_s: float = 0.02
    travel_speed_m_s: float = 0.04
    # After each cut the knife, still in the cut, moves +x and pushes the released piece.
    push_distance_m: float = 0.012
    push_speed_m_s: float = 0.02
    # The push starts only after the measured knife reaches the board and stays there.
    position_tolerance_m: float = 0.001
    bottom_hold_s: float = 0.1
    initial_hold_s: float = 0.3
    final_hold_s: float = 0.5

    def __post_init__(self):
        if not self.cut_offsets_m or len(self.cut_offsets_m) > 18:
            raise ValueError("Between 1 and 18 cuts")
        speeds = [self.descent_speed_m_s, self.travel_speed_m_s, self.push_speed_m_s]
        if min(speeds) <= 0 or self.push_distance_m < 0 or min(self.initial_hold_s, self.final_hold_s) < 0:
            raise ValueError("Speeds must be positive, distances and holds non-negative")


class CutSequence:
    """Paper Fig. 1 style task: several vertical cuts, each followed by pushing the cut piece aside.

    A waypoint schedule for the tool origin. Each segment is timed, except that a descent
    ends only once the measured knife has reached the board, so no cut is left unfinished.
    """

    phases = ("settle", "move", "descend", "push", "withdraw", "verify", "done")

    def __init__(self, initial, board_z, half_height, settings=None):
        self.settings = s = settings or SequenceSettings()
        self.initial = np.array(initial, dtype=float, copy=True)
        bottom_z = board_z + half_height
        if self.initial[2] <= bottom_z:
            raise ValueError("Initial knife must be above the board")
        self.expected_pieces = len(s.cut_offsets_m) + 1
        # Segments: (phase, start, end, duration, cut index)
        self.segments = []
        here = self.initial.copy()

        def go(phase, target, speed, cut_index):
            nonlocal here
            target = np.asarray(target, dtype=float)
            duration = np.linalg.norm(target - here) / speed
            self.segments.append((phase, here.copy(), target, duration, cut_index))
            here = target

        self.segments.append(("settle", here.copy(), here.copy(), s.initial_hold_s, 0))
        for k, dx in enumerate(s.cut_offsets_m):
            above = self.initial + [dx, 0.0, 0.0]
            go("move", above, s.travel_speed_m_s, k)
            go("descend", [above[0], above[1], bottom_z], s.descent_speed_m_s, k)
            go("push", here + [s.push_distance_m, 0.0, 0.0], s.push_speed_m_s, k)
            go("withdraw", [here[0], here[1], self.initial[2]], s.descent_speed_m_s, k)
        self.segments.append(("verify", here.copy(), here.copy(), s.final_hold_s, len(s.cut_offsets_m) - 1))
        # Lower bound: descents can last longer while the knife is held up by the material.
        self.duration_s = sum(segment[3] for segment in self.segments)
        self.index = 0
        self.segment_start = 0.0
        self.arrived_since = None
        self.phase = "settle"
        self.cut_index = 0
        self.events = [dict(time_s=0.0, phase=self.phase, cut_index=0)]

    @property
    def done(self):
        return self.phase == "done"

    @property
    def cutting(self):
        return self.phase == "descend"

    def update(self, t, actual, edge_points):
        """Tool-origin target at time t, given the measured tool origin."""
        s = self.settings
        actual = np.asarray(actual, dtype=float)
        while self.index < len(self.segments):
            phase, a, b, duration, cut_index = self.segments[self.index]
            elapsed = t - self.segment_start
            if elapsed < duration:
                target = a + (elapsed / duration) * (b - a)
                break
            target = b
            if phase == "descend":
                if np.linalg.norm(actual - b) > s.position_tolerance_m:
                    self.arrived_since = None
                    break
                self.arrived_since = t if self.arrived_since is None else self.arrived_since
                if t - self.arrived_since < s.bottom_hold_s:
                    break
                self.arrived_since = None
                self.segment_start = t
            else:
                self.segment_start += duration
            self.index += 1
        else:
            phase, cut_index, target = "done", self.cut_index, self.segments[-1][2].copy()
        if phase != self.phase or cut_index != self.cut_index:
            self.events.append(
                dict(
                    time_s=float(t),
                    phase=phase,
                    cut_index=int(cut_index),
                    actual_tool_position_m=np.asarray(actual).tolist(),
                )
            )
        self.phase, self.cut_index = phase, cut_index
        return target
