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


@wp.func
def guard_polynomial(x0: wp.vec3, xm: wp.vec3, x1: wp.vec3,
                     p0: wp.vec3, pm: wp.vec3, p1: wp.vec3,
                     r0: wp.quat, rm: wp.quat, r1: wp.quat,
                     half: wp.vec3, spacing: float, board: wp.vec2, kind: int, plane: int):
    g0 = side_guard(x0, p0, r0, half, spacing, board, kind, plane)
    gm = side_guard(xm, pm, rm, half, spacing, board, kind, plane)
    g1 = side_guard(x1, p1, r1, half, spacing, board, kind, plane)
    a = 2.0 * (g0 + g1 - 2.0 * gm)
    return wp.vec3(a, g1 - g0 - a, g0)


@wp.func
def polynomial_roots(poly: wp.vec3, tolerance: float):
    a, b, c = poly[0], poly[1], poly[2]
    roots = wp.vec2(2.0)
    if wp.abs(a) <= tolerance:
        if wp.abs(b) > tolerance:
            roots[0] = -c / b
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            sign = float(1.0)
            if b < 0.0:
                sign = -1.0
            q = -0.5 * (b + sign * wp.sqrt(discriminant))
            if q != 0.0:
                roots[0] = q / a
                roots[1] = c / q
            else:
                roots[0] = -b / (2.0 * a)
    return roots


@wp.kernel
def interpolate_position(start: wp.array(dtype=wp.vec3), middle: wp.array(dtype=wp.vec3),
                         end: wp.array(dtype=wp.vec3), fraction: float, value: wp.array(dtype=wp.vec3)):
    p = wp.tid()
    t = fraction
    value[p] = ((2.0 * t * t - 3.0 * t + 1.0) * start[p]
                + (4.0 * t - 4.0 * t * t) * middle[p] + (2.0 * t * t - t) * end[p])


@wp.kernel
def curved_candidate_events(
    before: wp.array(dtype=wp.vec3), middle: wp.array(dtype=wp.vec3), after: wp.array(dtype=wp.vec3),
    codes: wp.array(dtype=wp.int64),
    position0: wp.array(dtype=wp.vec3), rotation0: wp.array(dtype=wp.quat),
    positionm: wp.array(dtype=wp.vec3), rotationm: wp.array(dtype=wp.quat),
    position1: wp.array(dtype=wp.vec3), rotation1: wp.array(dtype=wp.quat),
    half: wp.vec3, spacing: float, board: wp.vec2, label_board: int, digit: wp.int64, tolerance: float,
    fractions: wp.array(dtype=float), kinds: wp.array(dtype=int), planes: wp.array(dtype=int),
    lower: wp.array(dtype=float), upper: wp.array(dtype=float),
):
    p = wp.tid()
    best, best_lo, best_hi = float(2.0), float(0.0), float(1.0)
    best_kind, best_plane = int(-1), int(-1)
    if (codes[p] // digit) % wp.int64(3) == wp.int64(0):
        for kind in range(1 + label_board):
            count = int(6)
            if kind == 1:
                count = 10
            initial_inside = bool(True)
            for plane in range(count):
                initial_inside = initial_inside and side_guard(
                    before[p], position0[0], rotation0[0], half, spacing, board, kind, plane) <= 0.0
            if initial_inside and best > 0.0:
                best, best_kind, best_plane = 0.0, kind, -1
            else:
                for plane in range(count):
                    poly = guard_polynomial(before[p], middle[p], after[p],
                                             position0[0], positionm[0], position1[0],
                                             rotation0[0], rotationm[0], rotation1[0],
                                             half, spacing, board, kind, plane)
                    roots = polynomial_roots(poly, tolerance)
                    for index in range(2):
                        entry = roots[index]
                        if entry > 0.0 and entry <= 1.0 and entry < best and 2.0 * poly[0] * entry + poly[1] < 0.0:
                            leave = float(1.0)
                            lower_bound = float(0.0)
                            if poly[2] <= 0.0:
                                lower_bound = 0.5 * (roots[1 - index] + entry)
                            for other in range(count):
                                test = guard_polynomial(before[p], middle[p], after[p],
                                                         position0[0], positionm[0], position1[0],
                                                         rotation0[0], rotationm[0], rotation1[0],
                                                         half, spacing, board, kind, other)
                                exits = polynomial_roots(test, tolerance)
                                for j in range(2):
                                    if exits[j] > entry:
                                        leave = wp.min(leave, exits[j])
                            probe = 0.5 * (entry + leave)
                            inside = bool(True)
                            for other in range(count):
                                test = guard_polynomial(before[p], middle[p], after[p],
                                                         position0[0], positionm[0], position1[0],
                                                         rotation0[0], rotationm[0], rotation1[0],
                                                         half, spacing, board, kind, other)
                                inside = inside and (test[0] * probe + test[1]) * probe + test[2] <= tolerance
                            if inside:
                                best, best_kind, best_plane = entry, kind, plane
                                best_lo, best_hi = lower_bound, probe
    fractions[p] = best
    kinds[p] = best_kind
    planes[p] = best_plane
    lower[p] = best_lo
    upper[p] = best_hi


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
def bracket_time(start: wp.array(dtype=float), end: wp.array(dtype=float), fraction: float,
                 time: wp.array(dtype=float)):
    time[0] = start[0] + fraction * (end[0] - start[0])


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

    def candidates(self, state, trial, history, tool0, tool1, index, trial_history=None, duration=None):
        m, cfg = self.model, self.model.config
        fractions, kinds, planes = m.zeros(m.n, float), m.zeros(m.n, int, False), m.zeros(m.n, int, False)
        lower, upper = m.zeros(m.n, float, False), m.zeros(m.n, float, False)
        if trial_history is not None and trial_history.path_midpoint is not None:
            dt = float(duration.numpy()[0])
            angular_span = float(np.linalg.norm(tool0.omega.numpy()[0])) * dt
            # Query separate arcs so a revolution cannot collapse to equal endpoint poses.
            intervals = max(1, int(np.ceil(angular_span / (np.pi / 2))))
            for interval in range(intervals):
                start_fraction = interval / intervals
                fractions_in_step = [start_fraction, (interval + .5) / intervals, (interval + 1) / intervals]
                positions, poses = [], []
                for fraction in fractions_in_step:
                    position = m.zeros(m.n, wp.vec3, False)
                    m.launch(interpolate_position, m.n,
                             inputs=[state.x, trial_history.path_midpoint, trial.x, fraction],
                             outputs=[position], device=m.device)
                    positions.append(position)
                    poses.append(self.pose(tool0, m.array([dt * fraction], float, False)))
                m.launch(curved_candidate_events, m.n,
                         inputs=[*positions, history.codes, poses[0].position, poses[0].rotation,
                                 poses[1].position, poses[1].rotation, poses[2].position, poses[2].rotation,
                                 cfg.blade_half_size, cfg.grid_spacing, cfg.board_half_extent,
                                 int(cfg.label_board_layer), 3 ** index, self.root_tolerance(state, tool0)],
                         outputs=[fractions, kinds, planes, lower, upper], device=m.device)
                values = fractions.numpy()
                particle = int(np.argmin(values))
                if values[particle] == 0 and interval > 0:
                    m.launch(candidate_events, m.n,
                             inputs=[state.x, positions[0], history.codes, tool0.position, tool0.rotation,
                                     poses[0].position, poses[0].rotation, cfg.blade_half_size,
                                     cfg.grid_spacing, cfg.board_half_extent, int(cfg.label_board_layer), 3 ** index],
                             outputs=[fractions, kinds, planes], device=m.device)
                    values = fractions.numpy()
                    particle = int(np.argmin(values))
                    return (particle, start_fraction * float(values[particle]),
                            int(kinds.numpy()[particle]), int(planes.numpy()[particle]), 0., start_fraction)
                if values[particle] <= 1:
                    return (particle, start_fraction + float(values[particle]) / intervals,
                            int(kinds.numpy()[particle]), int(planes.numpy()[particle]),
                            start_fraction + float(lower.numpy()[particle]) / intervals,
                            start_fraction + float(upper.numpy()[particle]) / intervals)
            return particle, 2., -1, -1, 0., 1.
        else:
            upper.fill_(1.)
            m.launch(candidate_events, m.n,
                     inputs=[state.x, trial.x, history.codes, tool0.position, tool0.rotation,
                             tool1.position, tool1.rotation, cfg.blade_half_size, cfg.grid_spacing,
                             cfg.board_half_extent, int(cfg.label_board_layer), 3 ** index],
                     outputs=[fractions, kinds, planes], device=m.device)
        values = fractions.numpy()
        particle = int(np.argmin(values))
        return (particle, float(values[particle]), int(kinds.numpy()[particle]), int(planes.numpy()[particle]),
                float(lower.numpy()[particle]), float(upper.numpy()[particle]))

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

    def locate(self, state, history, tool, tool0, elapsed, end, trial, trial_tool, particle, kind, plane,
               lower_fraction=0., upper_fraction=1.):
        m = self.model
        lo, hi = elapsed, end
        glo = self.guard(state, tool0, particle, kind, plane)
        ghi = self.guard(trial, trial_tool, particle, kind, plane)
        if lower_fraction > 0:
            lo = m.zeros(1, float)
            m.launch(bracket_time, 1, inputs=[elapsed, end, lower_fraction], outputs=[lo], device=m.device)
            low_state, _, _ = self.advance_segment(state, history, tool0, self.interval(lo, elapsed), True)
            glo = self.guard(low_state, self.pose(tool, lo), particle, kind, plane)
        if upper_fraction < 1:
            hi = m.zeros(1, float)
            m.launch(bracket_time, 1, inputs=[elapsed, end, upper_fraction], outputs=[hi], device=m.device)
            high_state, _, _ = self.advance_segment(state, history, tool0, self.interval(hi, elapsed), True)
            ghi = self.guard(high_state, self.pose(tool, hi), particle, kind, plane)
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
            selected = (-1, 2., -1, -1, 0., 1.)
            if instruction.cutting:
                selected = self.candidates(state, trial, history, start_tool, end_tool, instruction.cut_index,
                                           trial_h, remaining)
            particle, fraction, kind, plane, lower, upper = selected
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
                        particle, kind, plane, lower, upper,
                    )
                    earlier = self.candidates(state, before, history, start_tool, event_tool, instruction.cut_index,
                                              h, self.interval(time, elapsed))
                    if earlier[0] == particle or earlier[1] >= 1.0:
                        break
                    particle, _, kind, plane, lower, upper = earlier
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
