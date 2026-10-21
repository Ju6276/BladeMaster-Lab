"""Locate geometric cut events and integrate the material on either side of them."""

from dataclasses import replace

import numpy as np
import warp as wp

from .differentiable_contact import Tool
from .differentiable_cutting import FixedCutContact
from .differentiable_natural import CutInstruction, NaturalCutContact


@wp.func
def side_guard(x: wp.vec3, position: wp.vec3, rotation: wp.quat, half: wp.vec3,
               spacing: float, board: wp.vec2, kind: int, plane: int):
    edge = position + wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, -half[2]))
    local = wp.quat_rotate_inv(rotation, x - edge)
    value = float(-1.0)
    if plane < 4:
        axis = plane // 2
        sign = float(1.0)
        if plane % 2 == 1:
            sign = -1.0
        bound = 6.0 * spacing
        if axis == 1:
            bound = half[1]
        value = sign * local[axis] - bound
    elif kind == 0:
        value = -local[2]
        if plane == 5:
            value = local[2] - 2.0 * spacing
    else:
        edge_here = edge + wp.quat_rotate(rotation, wp.vec3(0.0, local[1], 0.0))
        if plane == 4:
            value = edge_here[2] - spacing / 4.0
        elif plane == 5:
            value = x[2] - edge_here[2] - spacing / 4.0
        else:
            axis = (plane - 6) // 2
            if board[axis] > 0.0:
                sign = float(1.0)
                if plane % 2 == 1:
                    sign = -1.0
                value = sign * x[axis] - board[axis]
    return value


@wp.kernel
def candidate_events(
    before: wp.array(dtype=wp.vec3), after: wp.array(dtype=wp.vec3), codes: wp.array(dtype=wp.int64),
    position0: wp.array(dtype=wp.vec3), rotation0: wp.array(dtype=wp.quat),
    position1: wp.array(dtype=wp.vec3), rotation1: wp.array(dtype=wp.quat),
    half: wp.vec3, spacing: float, board: wp.vec2, label_board: int, digit: wp.int64,
    fractions: wp.array(dtype=float), kinds: wp.array(dtype=int), planes: wp.array(dtype=int),
):
    p = wp.tid()
    best = float(2.0)
    best_kind, best_plane = int(-1), int(-1)
    if (codes[p] // digit) % wp.int64(3) == wp.int64(0):
        for kind in range(1 + label_board):
            enter, leave = float(0.0), float(1.0)
            entering_plane = int(-1)
            count = int(6)
            if kind == 1:
                count = 10
            for plane in range(count):
                a = side_guard(before[p], position0[0], rotation0[0], half, spacing, board, kind, plane)
                b = side_guard(after[p], position1[0], rotation1[0], half, spacing, board, kind, plane)
                if a > 0.0:
                    if b >= a:
                        enter = 2.0
                    else:
                        crossing = a / (a - b)
                        if crossing > enter:
                            enter, entering_plane = crossing, plane
                elif b > 0.0:
                    leave = wp.min(leave, a / (a - b))
            if enter <= leave and enter < best:
                best, best_kind, best_plane = enter, kind, entering_plane
    fractions[p] = best
    kinds[p] = best_kind
    planes[p] = best_plane


@wp.kernel
def evaluate_guard(x: wp.array(dtype=wp.vec3), position: wp.array(dtype=wp.vec3),
                   rotation: wp.array(dtype=wp.quat), half: wp.vec3, spacing: float, board: wp.vec2,
                   particle: int, kind: int, plane: int, value: wp.array(dtype=float)):
    value[0] = side_guard(x[particle], position[0], rotation[0], half, spacing, board, kind, plane)


@wp.kernel
def region_residual(x: wp.array(dtype=wp.vec3), position: wp.array(dtype=wp.vec3),
                    rotation: wp.array(dtype=wp.quat), half: wp.vec3, spacing: float, board: wp.vec2,
                    particle: int, kind: int, value: wp.array(dtype=float)):
    count = int(6)
    if kind == 1:
        count = 10
    residual = float(-1.0e30)
    for plane in range(count):
        residual = wp.max(residual, side_guard(x[particle], position[0], rotation[0], half,
                                              spacing, board, kind, plane))
    value[0] = residual


@wp.kernel
def advance_pose(position: wp.array(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3),
                 rotation: wp.array(dtype=wp.quat), omega: wp.array(dtype=wp.vec3),
                 elapsed: wp.array(dtype=float), next_position: wp.array(dtype=wp.vec3),
                 next_rotation: wp.array(dtype=wp.quat)):
    t = elapsed[0]
    w = omega[0]
    a2 = wp.length_sq(w) * t * t / 4.0
    sine_factor = t / 2.0 * (1.0 - a2 / 6.0 + a2 * a2 / 120.0)
    cosine = 1.0 - a2 / 2.0 + a2 * a2 / 24.0
    if a2 >= 1.0e-8:
        a = wp.sqrt(a2)
        sine_factor = t / 2.0 * wp.sin(a) / a
        cosine = wp.cos(a)
    increment = wp.quat(sine_factor * w[0], sine_factor * w[1], sine_factor * w[2], cosine)
    next_position[0] = position[0] + t * velocity[0]
    next_rotation[0] = wp.normalize(increment * rotation[0])


@wp.kernel
def clock_difference(end: wp.array(dtype=float), start: wp.array(dtype=float), duration: wp.array(dtype=float)):
    duration[0] = end[0] - start[0]


@wp.kernel
def secant_time(lo: wp.array(dtype=float), hi: wp.array(dtype=float),
                glo: wp.array(dtype=float), ghi: wp.array(dtype=float), time: wp.array(dtype=float)):
    time[0] = (lo[0] * ghi[0] - hi[0] * glo[0]) / (ghi[0] - glo[0])


@wp.kernel
def assign_event(x: wp.array(dtype=wp.vec3), position: wp.array(dtype=wp.vec3),
                 rotation: wp.array(dtype=wp.quat), particle: int, digit: wp.int64,
                 edge_entry: int, codes: wp.array(dtype=wp.int64),
                 edge_departed: wp.array(dtype=int)):
    local = wp.quat_rotate_inv(rotation[0], x[particle] - position[0])
    side = wp.int64(2)
    if local[0] < 0.0:
        side = wp.int64(1)
    codes[particle] += digit * side
    if edge_entry != 0:
        edge_departed[particle] = 1


@wp.kernel
def accumulate_segment(force: wp.array(dtype=wp.vec3), work: wp.array(dtype=float),
                       duration: wp.array(dtype=float), total_dt: wp.array(dtype=float),
                       average_force: wp.array(dtype=wp.vec3), total_work: wp.array(dtype=float)):
    p = wp.tid()
    wp.atomic_add(total_work, p, work[p])
    if p == 0:
        wp.atomic_add(average_force, 0, duration[0] / total_dt[0] * force[0])


class GeometryEventContact(NaturalCutContact):
    """Bracketed cut-region entry with midpoint or symplectic TLMPM flow.

    The prescribed knife has constant linear/angular velocity within a macro
    step. Bracketed secant iterations solve the selected guard on the actual
    pre-event material flow; all times and flow evaluations participate in AD.
    """

    def __init__(self, trajectory, *, release="integrated", friction_update="projected",
                 integration="midpoint"):
        super().__init__(trajectory, release=release, friction_update=friction_update)
        if integration not in ("symplectic", "midpoint"):
            raise ValueError("Geometry integration must be symplectic or midpoint")
        if integration == "midpoint" and friction_update != "projected":
            raise ValueError("Midpoint integration requires projected friction histories")
        self.integration = integration

    def pose(self, tool, elapsed):
        m = self.model
        position, rotation = m.zeros(1, wp.vec3), m.zeros(1, wp.quat)
        m.launch(advance_pose, 1,
                 inputs=[tool.position, tool.velocity, tool.rotation, tool.omega, elapsed],
                 outputs=[position, rotation], device=m.device)
        return Tool(position, tool.velocity, rotation, tool.omega)

    def interval(self, end, start):
        duration = self.model.zeros(1, float)
        self.model.launch(clock_difference, 1, inputs=[end, start], outputs=[duration], device=self.model.device)
        return duration

    def candidates(self, state, trial, history, tool0, tool1, index):
        m, cfg = self.model, self.model.config
        fractions, kinds, planes = m.zeros(m.n, float), m.zeros(m.n, int, False), m.zeros(m.n, int, False)
        m.launch(candidate_events, m.n,
                 inputs=[state.x, trial.x, history.codes, tool0.position, tool0.rotation,
                         tool1.position, tool1.rotation, cfg.blade_half_size, cfg.grid_spacing,
                         cfg.board_half_extent, int(cfg.label_board_layer), 3 ** index],
                 outputs=[fractions, kinds, planes], device=m.device)
        values = fractions.numpy()
        particle = int(np.argmin(values))
        return particle, float(values[particle]), int(kinds.numpy()[particle]), int(planes.numpy()[particle])

    def guard(self, state, tool, particle, kind, plane):
        m, cfg = self.model, self.model.config
        value = m.zeros(1, float)
        m.launch(evaluate_guard, 1,
                 inputs=[state.x, tool.position, tool.rotation, cfg.blade_half_size,
                         cfg.grid_spacing, cfg.board_half_extent, particle, kind, plane],
                 outputs=[value], device=m.device)
        return value

    def advance_segment(self, state, history, tool, duration, cutting):
        if self.integration == "midpoint":
            from .differentiable_midpoint import advance_midpoint
            return advance_midpoint(self, state, history, tool, duration, cutting)
        keys = self.contact_keys(state, history.codes, float(duration.numpy()[0]))
        branch = self.freeze(history.codes.numpy(), keys, history.pair_keys,
                             history.damage.numpy() >= 1, cutting=cutting)
        result, updated, force = FixedCutContact.step(
            self, state, history, tool, branch, duration=duration, edge_departed=history.edge_departed,
        )
        updated.codes, updated.pair_keys, updated.cut_index = history.codes, keys, history.cut_index
        return result, updated, force

    def locate(self, state, history, tool, tool0, elapsed, end, trial, trial_tool, particle, kind, plane):
        m = self.model
        lo, hi = elapsed, end
        glo = self.guard(state, tool0, particle, kind, plane)
        ghi = self.guard(trial, trial_tool, particle, kind, plane)
        if float(glo.numpy()[0]) <= 0 or float(ghi.numpy()[0]) > 0:
            raise RuntimeError('Geometric event lost its outside/inside bracket')
        tolerance = self.root_tolerance(state, trial_tool)
        for _ in range(48):
            time = m.zeros(1, float)
            m.launch(secant_time, 1, inputs=[lo, hi, glo, ghi], outputs=[time], device=m.device)
            duration = self.interval(time, elapsed)
            if float(duration.numpy()[0]) <= 0:
                raise RuntimeError('Nonpositive geometric event interval')
            before, h, force = self.advance_segment(state, history, tool0, duration, True)
            at_event = self.pose(tool, time)
            value = self.guard(before, at_event, particle, kind, plane)
            residual = float(value.numpy()[0])
            if abs(residual) <= tolerance:
                region = m.zeros(1, float, False)
                cfg = m.config
                m.launch(region_residual, 1,
                         inputs=[before.x, at_event.position, at_event.rotation, cfg.blade_half_size,
                                 cfg.grid_spacing, cfg.board_half_extent, particle, kind],
                         outputs=[region], device=m.device)
                if float(region.numpy()[0]) > tolerance:
                    raise RuntimeError('Geometric root lies outside the cut region; reduce the macro step')
                return time, before, h, force, at_event, tolerance
            if residual > 0:
                lo, glo = time, value
            else:
                hi, ghi = time, value
        raise RuntimeError(f'Geometric event root did not converge: residual={residual:g} m')

    def root_tolerance(self, state, tool):
        x = state.x.numpy()
        scale = max(self.model.config.grid_spacing, float(np.max(np.abs(x))),
                    float(np.max(np.abs(tool.position.numpy()))))
        return 32 * np.finfo(x.dtype).eps * scale

    def step(self, state, history, tool, instruction=CutInstruction()):
        m, cfg = self.model, self.model.config
        if history.codes is None or history.pair_keys is None:
            raise ValueError('Geometric cutting history requires side codes and pair keys')
        self._check_cut_index(instruction.cut_index, history.cut_index)
        if instruction.cut_index != history.cut_index:
            history = replace(history, damage=m.zeros(m.n, float), cut_index=int(instruction.cut_index),
                              edge_departed=None)
        elapsed = m.array([0.], float, False)
        end = (m.duration if instruction.duration is None
               else m.array([instruction.duration], float, False))
        end_tool = self.pose(tool, end)
        reaction, work = m.zeros(1, wp.vec3), m.zeros(m.n, float)
        events = []
        for _ in range(m.n + 1):
            start_tool = self.pose(tool, elapsed)
            remaining = self.interval(end, elapsed)
            trial, trial_h, trial_force = self.advance_segment(state, history, start_tool, remaining,
                                                               instruction.cutting)
            selected = (-1, 2., -1, -1)
            if instruction.cutting:
                selected = self.candidates(state, trial, history, start_tool, end_tool, instruction.cut_index)
            particle, fraction, kind, plane = selected
            if fraction > 1:
                m.launch(accumulate_segment, m.n,
                         inputs=[trial_force, trial_h.damage_work, remaining, end],
                         outputs=[reaction, work], device=m.device)
                trial_h.damage_work = work
                trial_h.geometry_events = tuple(events)
                return trial, trial_h, reaction
            if fraction > 0:
                bound, bound_state, bound_tool = end, trial, end_tool
                for _ in range(m.n + 1):
                    time, before, h, force, event_tool, tolerance = self.locate(
                        state, history, tool, start_tool, elapsed, bound, bound_state, bound_tool,
                        particle, kind, plane,
                    )
                    earlier = self.candidates(state, before, history, start_tool, event_tool, instruction.cut_index)
                    if earlier[0] == particle or earlier[1] >= 1.0:
                        break
                    particle, _, kind, plane = earlier
                    bound, bound_state, bound_tool = time, before, event_tool
                else:
                    raise RuntimeError('Geometric event ordering did not converge')
                duration = self.interval(time, elapsed)
                m.launch(accumulate_segment, m.n,
                         inputs=[force, h.damage_work, duration, end],
                         outputs=[reaction, work], device=m.device)
                state, history, elapsed = before, h, time
                start_tool = event_tool
            codes = wp.clone(history.codes, requires_grad=False)
            # At a located edge entry, the next flow uses the post-event side
            # even when the root residual has the opposite roundoff sign.
            departed = (m.zeros(m.n, int, False) if history.edge_departed is None
                        else wp.clone(history.edge_departed, requires_grad=False))
            edge_entry = kind == 0 and plane == 4
            if kind == 0 and plane < 0:
                edge_guard = self.guard(state, start_tool, particle, kind, 4)
                edge_entry = abs(float(edge_guard.numpy()[0])) <= self.root_tolerance(state, start_tool)
            m.launch(assign_event, 1,
                     inputs=[state.x, start_tool.position, start_tool.rotation, particle,
                             3 ** instruction.cut_index, int(edge_entry)],
                     outputs=[codes, departed], device=m.device)
            history = replace(history, codes=codes, edge_departed=departed)
            events.append((particle, elapsed))
            if float(elapsed.numpy()[0]) >= float(end.numpy()[0]):
                history.damage_work = work
                history.geometry_events = tuple(events)
                return state, history, reaction
        raise RuntimeError('More side-assignment events than unlabelled particles')
