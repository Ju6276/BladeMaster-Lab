"""Measured-pose sequencing around the paper's two-way physics interface.

BladeMaster does not publish a robot phase controller. These exposed task
settings are an independent adapter, NOT paper parameters. Only robot/tool
pose drives transitions; material topology is an evaluator, never a command.
There is no teleport, force override, added workspace box or material anchor.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class CycleSettings:
    descent_speed_m_s: float = .02
    withdrawal_speed_m_s: float = .025
    position_tolerance_m: float = .0005
    initial_hold_s: float = .3
    bottom_hold_s: float = .1
    final_hold_s: float = .5

    def __post_init__(self):
        values = np.array(list(asdict(self).values()))
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('Cycle speeds, tolerances and hold times must be positive and finite')


class CutCycle:
    phases = ('settle', 'descend', 'withdraw', 'verify', 'done')

    def __init__(self, initial, board_z, half_height, settings=None):
        self.settings = settings or CycleSettings()
        self.initial = np.array(initial, dtype=float, copy=True)
        if self.initial.shape != (3,) or not np.isfinite(self.initial).all():
            raise ValueError('Expected finite initial tool position')
        if not np.isfinite([board_z, half_height]).all() or half_height <= 0:
            raise ValueError('Expected finite board height and positive blade half height')
        self.board_z = float(board_z)
        self.bottom = self.initial.copy()
        self.bottom[2] = board_z + half_height
        if self.initial[2] <= self.bottom[2]:
            raise ValueError('Initial knife must be above the board')
        self.phase = 'settle'
        self.phase_start = 0.
        self.arrived_since = None
        self.last_time = None
        self.events = [dict(time_s=0., phase=self.phase)]
        self.withdraw_from = self.bottom.copy()

    @property
    def done(self):
        return self.phase == 'done'

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
        return t - self.arrived_since >= duration - 1.e-10

    def update(self, t, actual, edge_points):
        """Return a tool-origin target; actual pose is never overwritten.

        Both ends of the finite edge must reach the board-height tolerance,
        preventing a tilted edge with only one low corner from passing.
        If resisted above the goal, hold the endpoint and eventually let the
        runner report timeout. Elapsed time alone never triggers withdrawal.
        """
        actual = np.asarray(actual, dtype=float)
        edges = np.asarray(edge_points, dtype=float)
        if (not np.isfinite(t) or actual.shape != (3,) or edges.shape != (2, 3)
                or not np.isfinite(actual).all() or not np.isfinite(edges).all()
                or (self.last_time is not None and t < self.last_time)):
            raise ValueError('Invalid/nonmonotonic cycle observation')
        self.last_time = t
        s = self.settings
        if self.phase == 'settle':
            if self.held(np.linalg.norm(actual-self.initial) <= s.position_tolerance_m,
                         t, s.initial_hold_s):
                self.transition('descend', t, actual)
        elif self.phase == 'descend':
            arrived = (np.linalg.norm(actual-self.bottom) <= s.position_tolerance_m
                       and np.max(np.abs(edges[:, 2]-self.board_z)) <= s.position_tolerance_m)
            if self.held(arrived, t, s.bottom_hold_s):
                self.withdraw_from = actual.copy()
                self.transition('withdraw', t, actual)
        elif self.phase == 'withdraw':
            if np.linalg.norm(actual-self.initial) <= s.position_tolerance_m:
                self.transition('verify', t, actual)
        elif self.phase == 'verify':
            if self.held(np.linalg.norm(actual-self.initial) <= s.position_tolerance_m,
                         t, s.final_hold_s):
                self.transition('done', t, actual)
        if self.phase == 'descend':
            distance = self.initial[2]-self.bottom[2]
            # Endpoint interpolation defines the finite cut task, not a clamp
            # on robot state or on general teleoperation workspace.
            fraction = min((t-self.phase_start)*s.descent_speed_m_s/distance, 1.)
            return (1-fraction)*self.initial+fraction*self.bottom
        if self.phase == 'withdraw':
            distance = np.linalg.norm(self.initial-self.withdraw_from)
            fraction = min((t-self.phase_start)*s.withdrawal_speed_m_s/distance, 1.)
            return (1-fraction)*self.withdraw_from+fraction*self.initial
        return self.initial.copy()


def evaluate_cut(observations, cycle_done):
    """Independent post-withdrawal check, without visual-surface shortcuts."""
    verification = [r for r in observations if r['phase'] in ('verify', 'done')]
    separated = bool(verification) and all(len(r['component_sizes']) == 2 for r in verification)
    cleared = bool(verification) and all(r['knife_clear_of_material'] for r in verification)
    return dict(success=bool(cycle_done and separated and cleared),
                cycle_completed=bool(cycle_done),
                two_components_persist_after_withdrawal=separated,
                knife_clear_after_withdrawal=cleared,
                verification_samples=len(verification),
                criteria='pose-driven cycle completes; exactly two mechanical components and no material above the retracted blade edge throughout final hold; simulation diagnostic, not an author benchmark')
