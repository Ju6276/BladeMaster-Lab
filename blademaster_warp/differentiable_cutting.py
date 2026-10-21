"""Prescribed cutting with recorded topology and optional within-step damage release."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import warp as wp

from .contact import blade_query, penalty_impulse
from .differentiable_contact import PrescribedContact
from .differentiable_material import FixedTopology, PlasticState
from .differentiable_events import crossing_fraction
from . import differentiable_constraint as grid_constraint


@wp.func
def contact_mode(gap: float, relative: wp.vec3, normal: wp.vec3, history: wp.vec3,
                 mass: float, dt: float, sn: float, mus: float, gk: float, gc: float,
                 project_history: int):
    predicted = gap + dt * wp.min(wp.dot(relative, normal), 0.0)
    if predicted >= 0.0:
        return 0
    jn = sn * mass / dt * (-predicted)
    vt = relative - wp.dot(relative, normal) * normal
    ht = history - wp.dot(history, normal) * normal
    if project_history != 0:
        ht = ht + dt * vt
    k = gk * mass / (dt * dt)
    damping = 2.0 * gc * wp.sqrt(mass * k)
    mode = int(2)
    if wp.length((-k * ht - damping * vt) * dt) <= mus * jn and (project_history == 0 or mus > 0.0):
        mode = 1
    if wp.dot(relative, normal) < 0.0:
        mode += 10
    return mode


@wp.func
def branch_penalty(gap: float, relative: wp.vec3, normal: wp.vec3, mass: float, history: wp.vec3,
                   dt: float, sn: float, mus: float, mud: float, gk: float, gc: float, mode: int,
                   project_history: int):
    if mode < 0:
        if project_history == 0:
            return penalty_impulse(gap, relative, normal, mass, history, dt, sn, mus, mud, gk, gc)
        mode = contact_mode(gap, relative, normal, history, mass, dt, sn, mus, gk, gc, project_history)
    if mode == 0:
        return wp.vec3(0.0), wp.vec3(0.0)
    closing = float(0.0)
    if mode >= 10:
        closing = wp.dot(relative, normal)
    jn = -sn * mass / dt * (gap + dt * closing)
    vt = relative - wp.dot(relative, normal) * normal
    ht = history - wp.dot(history, normal) * normal
    k = gk * mass / (dt * dt)
    damping = 2.0 * gc * wp.sqrt(mass * k)
    force_history = ht
    if project_history != 0:
        force_history = ht + dt * vt
    trial = (-k * force_history - damping * vt) * dt
    if mode % 10 == 1:
        return jn * normal + trial, ht + dt * vt
    applied = wp.vec3(0.0)
    magnitude = wp.length(trial)
    if magnitude > 0.0:
        applied = trial * (mud * jn / magnitude)
    return jn * normal + applied, -(applied / dt + damping * vt) / k


@wp.kernel
def rigid_step(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    history: wp.array2d(dtype=wp.vec3),
    damage: wp.array(dtype=float),
    released: wp.array(dtype=int),
    edge_departed: wp.array(dtype=int),
    fixed_modes: wp.array2d(dtype=int),
    position: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
    rotation: wp.array(dtype=wp.quat),
    omega: wp.array(dtype=wp.vec3),
    friction: wp.array(dtype=float),
    half: wp.vec3,
    bevel: float,
    radius: float,
    blade_offset: float,
    board_half: wp.vec2,
    mass: float,
    duration: wp.array(dtype=float),
    reference_dt: float,
    sn: float,
    board_sn: float,
    gk: float,
    gc: float,
    band_width: float,
    fracture_energy: wp.array(dtype=float),
    threshold_factor: float,
    integrate_release: int,
    project_history: int,
    cutting: int,
    cut_hold: int,
    cut_advance: int,
    next_history: wp.array2d(dtype=wp.vec3),
    next_damage: wp.array(dtype=float),
    trial_damage: wp.array(dtype=float),
    damage_work: wp.array(dtype=float),
    impulse: wp.array(dtype=wp.vec3),
    reaction: wp.array(dtype=wp.vec3),
    modes: wp.array2d(dtype=int),
):
    p = wp.tid()
    dt = duration[0]
    ratio = dt / reference_dt
    sn = sn * ratio * ratio
    board_sn = board_sn * ratio * ratio
    gk = gk * ratio * ratio
    is_released = released[p]
    if integrate_release != 0:
        is_released = int(damage[p] >= 1.0)
    xp = x[p]
    board_gap = xp[2] - radius
    if board_half[0] > 0.0 and wp.abs(xp[0]) > board_half[0]:
        board_gap = 1.0e30
    if board_half[1] > 0.0 and wp.abs(xp[1]) > board_half[1]:
        board_gap = 1.0e30
    board, board_history = branch_penalty(
        board_gap, v[p], wp.vec3(0.0, 0.0, 1.0), mass, history[0, p],
        dt, board_sn, friction[0], friction[1], gk, gc, fixed_modes[0, p], project_history,
    )
    modes[0, p] = contact_mode(board_gap, v[p], wp.vec3(0.0, 0.0, 1.0), history[0, p],
                               mass, dt, board_sn, friction[0], gk, gc, project_history)
    q = rotation[0]
    local = wp.quat_rotate_inv(q, xp - position[0])
    closest, normal, distance = blade_query(local, half, bevel)
    front = -half[2] - local[2]
    modes[3, p] = 0
    if (cutting != 0 and cut_hold != 0 and is_released == 0 and edge_departed[p] == 0
            and wp.abs(local[0]) <= half[0] + blade_offset
            and wp.abs(local[1]) <= half[1] and front >= 0.0):
        closest = wp.vec3(local[0], local[1], -half[2])
        normal = wp.vec3(0.0, 0.0, -1.0)
        distance = front
        modes[3, p] = 1
    lever = wp.quat_rotate(q, closest)
    n = wp.quat_rotate(q, normal)
    relative = v[p] - velocity[0] - wp.cross(omega[0], lever)
    blade, blade_history = branch_penalty(
        distance - blade_offset, relative, n, mass, history[1, p],
        dt, sn, friction[0], friction[1], gk, gc, fixed_modes[1, p], project_history,
    )
    modes[1, p] = contact_mode(distance - blade_offset, relative, n, history[1, p],
                               mass, dt, sn, friction[0], gk, gc, project_history)
    feature = int(0)
    for axis in range(3):
        digit = int(0)
        if normal[axis] > 0.0:
            digit = 1
        elif normal[axis] < 0.0:
            digit = 2
        feature = 3 * feature + digit
    if distance < 0.0:
        feature += 27
    modes[2, p] = feature
    modes[4, p] = 0
    updated_damage = damage[p]
    trial = damage[p]
    work = float(0.0)
    if cutting != 0 and wp.length_sq(blade) > 0.0:
        edge_lever = wp.quat_rotate(q, wp.vec3(0.0, 0.0, -half[2]))
        edge = position[0] + edge_lever
        down = wp.quat_rotate(q, wp.vec3(0.0, 0.0, -1.0))
        edge_velocity = velocity[0] + wp.cross(omega[0], edge_lever)
        edge_front = wp.dot(xp - edge, down)
        advance = wp.dot(edge_velocity - v[p], down)
        if cut_advance != 0:
            advance = wp.dot(edge_velocity, down)
        local_edge = wp.quat_rotate_inv(q, xp - edge)
        if (edge_departed[p] == 0 and edge_front >= 0.0 and edge_front <= 2.0 * blade_offset
                and wp.abs(local_edge[0]) <= band_width and wp.abs(local_edge[1]) <= half[1]):
            push = wp.max(wp.dot(blade, down), 0.0)
            modes[4, p] = 1
            if is_released != 0:
                blade = blade - push * down
                modes[4, p] = 2
            elif advance > 0.0:
                threshold = fracture_energy[0] * threshold_factor
                work = push * advance
                trial = damage[p] + work / threshold
                updated_damage = wp.min(1.0, trial)
                modes[4, p] = 3
                if updated_damage >= 1.0:
                    modes[4, p] = 4
                    if integrate_release != 0:
                        fraction = crossing_fraction(damage[p], trial, 1.0)
                        blade = blade - (1.0 - fraction) * push * down
                        work = (1.0 - damage[p]) * threshold
    next_history[0, p] = board_history
    next_history[1, p] = blade_history
    next_damage[p] = updated_damage
    trial_damage[p] = trial
    damage_work[p] = work
    impulse[p] = wp.vec3(0.0, 0.0, -9.81 * mass * dt) + board + blade
    wp.atomic_add(reaction, 0, -blade / dt)


@wp.kernel
def pair_step(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    pairs: wp.array2d(dtype=int),
    previous: wp.array(dtype=int),
    fixed_modes: wp.array(dtype=int),
    history: wp.array(dtype=wp.vec3),
    friction: wp.array(dtype=float),
    diameter: float,
    mass: float,
    duration: wp.array(dtype=float),
    reference_dt: float,
    sn: float,
    gk: float,
    gc: float,
    project_history: int,
    next_history: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=wp.vec3),
    modes: wp.array(dtype=int),
):
    index = wp.tid()
    dt = duration[0]
    ratio = dt / reference_dt
    sn = sn * ratio * ratio
    gk = gk * ratio * ratio
    p, q = pairs[index, 0], pairs[index, 1]
    delta = x[p] - x[q]
    distance = wp.length(delta)
    normal = delta / distance
    old = wp.vec3(0.0)
    if previous[index] >= 0:
        old = history[previous[index]]
    applied, updated = branch_penalty(distance - diameter, v[p] - v[q], normal,
                                     mass / 2.0, old, dt, sn, friction[0], friction[1], gk, gc,
                                     fixed_modes[index], project_history)
    next_history[index] = updated
    modes[index] = contact_mode(distance - diameter, v[p] - v[q], normal, old,
                               mass / 2.0, dt, sn, friction[0], gk, gc, project_history)
    wp.atomic_add(impulse, p, applied)
    wp.atomic_add(impulse, q, -applied)


@wp.kernel
def add_impulses(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    p = wp.tid()
    out[p] = a[p] + b[p]


@dataclass
class CutHistory:
    rigid: wp.array
    pairs: wp.array
    damage: wp.array
    rigid_modes: wp.array | None = None
    pair_modes: wp.array | None = None
    side: wp.array | None = None
    grid_batches: int = 0
    grid_residual: float = 0.0
    trial_damage: wp.array | None = None
    damage_work: wp.array | None = None
    codes: wp.array | None = None
    pair_keys: np.ndarray | None = None
    cut_index: int = 0
    geometry_events: tuple = ()
    edge_departed: wp.array | None = None


@dataclass
class FrozenStep:
    topology: FixedTopology
    pairs: wp.array
    previous: wp.array
    released: wp.array
    cutting: bool
    rigid_modes: wp.array
    pair_modes: wp.array
    yielded: wp.array | None
    grid_batches: int | None = None


def pair_snapshot(solver):
    """Sorted pair keys and friction histories at the current step boundary."""
    buffer = solver.step_count % 2
    count = int(solver.pair_count[buffer].numpy()[0])
    slots = solver.pair_used[buffer].numpy()[:count]
    keys = solver.pair_keys[buffer].numpy()[slots]
    order = np.argsort(keys)
    return keys[order], solver.pair_history[buffer].numpy()[slots][order]


class FixedCutContact(PrescribedContact):
    """Fixed release flags and pair identities; pose and particle states remain differentiable."""

    def __init__(self, trajectory, *, friction_update='lagged'):
        super().__init__(trajectory, allow_grid_constraint=True)
        if friction_update not in ('lagged', 'projected'):
            raise ValueError('Friction history update must be lagged or projected')
        self.friction_update = friction_update
        cfg = trajectory.config
        self.friction = trajectory.array([cfg.friction_static, cfg.friction_dynamic], float)
        self.fracture_energy = trajectory.array([cfg.cutting_energy], float)
        self.integrate_release = False
        self._no_edge_departure = trajectory.zeros(trajectory.n, int, False)
        self._topologies = {}

    def history(self, rigid=None, pairs=None, damage=None, side=None):
        m = self.model
        rigid = np.zeros((2, m.n, 3)) if rigid is None else rigid
        pairs = np.zeros((0, 3)) if pairs is None else pairs
        damage = np.zeros(m.n) if damage is None else damage
        return CutHistory(m.array(rigid, wp.vec3), m.array(pairs, wp.vec3), m.array(damage, float),
                          side=m.zeros(m.n, float, False) if side is None else m.array(side, float, False))

    def freeze(self, codes, pair_keys, previous_keys, released, cutting=True,
               rigid_modes=None, pair_modes=None, yielded=None, grid_batches=None):
        m = self.model
        codes = np.asarray(codes, dtype=np.int64)
        key = codes.tobytes()
        if key not in self._topologies:
            self._topologies[key] = m.topology(codes)
        old = {int(key): i for i, key in enumerate(previous_keys)}
        previous = [old.get(int(key), -1) for key in pair_keys]
        pairs = np.column_stack((np.asarray(pair_keys) // m.n, np.asarray(pair_keys) % m.n))
        rigid_modes = np.full((5, m.n), -1) if rigid_modes is None else rigid_modes
        pair_modes = np.full(len(pairs), -1) if pair_modes is None else pair_modes
        return FrozenStep(self._topologies[key], m.array(pairs, int, False),
                          m.array(previous, int, False), m.array(released, int, False), bool(cutting),
                          m.array(rigid_modes, int, False), m.array(pair_modes, int, False),
                          None if yielded is None else m.array(yielded, int, False), grid_batches)

    def step(self, state, history, tool, frozen, *, duration=None, edge_departed=None):
        m, cfg = self.model, self.model.config
        duration = m.duration if duration is None else duration
        edge_departed = self._no_edge_departure if edge_departed is None else edge_departed
        sn, gk = cfg.contact_scales()
        radius = cfg.grid_spacing / 4
        blade_offset = max(cfg.blade_contact_radius or radius, radius - cfg.blade_half_size[0])
        threshold_factor = m.volume ** (2 / 3) * (1 - cfg.sharpness) / cfg.sharpness
        updated = CutHistory(m.zeros((2, m.n), wp.vec3), m.zeros(frozen.pairs.shape[0], wp.vec3),
                             m.zeros(m.n, float), m.zeros((5, m.n), int, False),
                             m.zeros(frozen.pairs.shape[0], int, False))
        rigid = m.zeros(m.n, wp.vec3)
        pairs = m.zeros(m.n, wp.vec3)
        impulse = m.zeros(m.n, wp.vec3)
        reaction = m.zeros(1, wp.vec3)
        updated.trial_damage = m.zeros(m.n, float)
        updated.damage_work = m.zeros(m.n, float)
        m.launch(rigid_step, m.n,
                  inputs=[state.x, state.v, history.rigid, history.damage, frozen.released, edge_departed,
                          frozen.rigid_modes,
                          tool.position, tool.velocity, tool.rotation, tool.omega, self.friction,
                          cfg.blade_half_size, cfg.blade_bevel_height, radius, blade_offset,
                          cfg.board_half_extent, m.mass, duration, cfg.dt, sn, cfg.board_normal_scale or sn,
                          gk, cfg.tangent_damping, cfg.cut_band_cells * cfg.grid_spacing,
                          self.fracture_energy, threshold_factor, int(self.integrate_release),
                          int(self.friction_update == 'projected'),
                          int(frozen.cutting), int(cfg.cut_hold), int(cfg.cut_work == "advance")],
                  outputs=[updated.rigid, updated.damage, updated.trial_damage, updated.damage_work, rigid, reaction,
                           updated.rigid_modes], device=m.device)
        if frozen.pairs.shape[0]:
            m.launch(pair_step, frozen.pairs.shape[0],
                      inputs=[state.x, state.v, frozen.pairs, frozen.previous, frozen.pair_modes,
                              history.pairs, self.friction,
                              2 * radius, m.mass, duration, cfg.dt, sn, gk, cfg.tangent_damping,
                              int(self.friction_update == 'projected')],
                      outputs=[updated.pairs, pairs, updated.pair_modes], device=m.device)
        m.launch(add_impulses, m.n, inputs=[rigid, pairs], outputs=[impulse], device=m.device)
        constraint_result = None

        def constrain(grid, mass, velocity):
            nonlocal constraint_result
            constraint_result = grid_constraint.solve(
                m, grid, mass, velocity, state.x, tool, history.side, frozen.grid_batches, duration=duration,
            )
            return constraint_result.velocity

        result = m.step(state, impulse=impulse, topology=frozen.topology, fixed_yielded=frozen.yielded,
                        grid_transform=constrain if cfg.blade_grid_constraint else None, duration=duration)
        updated.side = history.side
        if constraint_result is not None:
            updated.side = constraint_result.side
            updated.grid_batches = constraint_result.batches
            updated.grid_residual = constraint_result.residual
            combined = m.zeros(1, wp.vec3)
            m.launch(grid_constraint.add_reaction, 1,
                      inputs=[reaction, constraint_result.linear_impulse, duration],
                      outputs=[combined], device=m.device)
            reaction = combined
        return result, updated, reaction

    def checkpoint_vjp(self, initial, history, frozen, segment_steps, tool_at_step, objective,
                       parameters=(), timings=None):
        """Recompute segments with plastic stress and all continuous contact histories.

        frozen and tool_at_step use global step indices. The objective adds a
        per-step loss; it can select the last index for a terminal objective.
        Entry-side labels and recorded solve schedules remain discrete.
        """
        m = self.model
        steps = len(frozen)
        if not m.requires_grad or steps < 1 or segment_steps < 1:
            raise ValueError("Cut checkpoint VJP requires gradients and positive step counts")
        parameters = tuple(dict.fromkeys((*parameters, m.young, m.yield_stress,
                                         self.friction, self.fracture_energy)))

        def arrays(state, contact_history):
            result = dict(x=state.x, v=state.v, F=state.F, Fp=state.Fp,
                          rigid=contact_history.rigid, pairs=contact_history.pairs, damage=contact_history.damage)
            if state.stress is not None:
                result['stress'] = state.stress
            return result

        if any(a.grad is None for a in (*arrays(initial, history).values(), *parameters)):
            raise ValueError("Continuous states and parameters must have gradient buffers")
        parameter_gradients = {a: np.zeros(a.numpy().shape, dtype=float) for a in parameters}

        def now():
            wp.synchronize_device(m.device)
            return perf_counter()

        measured = dict(forward_s=0.0, recompute_s=0.0, backward_s=0.0, recompute_max_abs={}, adjoint_trace=[])
        begun = now()
        loss = m.zeros(1, float)
        checkpoints = [(0, initial, history)]
        current, h = initial, history
        for step in range(steps):
            current, h, reaction = self.step(current, h, tool_at_step(step), frozen[step])
            objective(step, current, h, reaction, loss)
            if (step + 1) % segment_steps == 0 or step + 1 == steps:
                checkpoints.append((step + 1, current, h))
        terminal, terminal_history = current, h
        measured['forward_s'] = now() - begun
        gradient = {name: m.zeros(a.shape, a.dtype, False) for name, a in arrays(current, h).items()}
        seed = m.array([1.0], float, False)
        for index in reversed(range(len(checkpoints) - 1)):
            begun = now()
            first, start, start_history = checkpoints[index]
            last, expected, expected_history = checkpoints[index + 1]
            start_arrays = arrays(start, start_history)
            for a in (*start_arrays.values(), *parameters):
                a.grad.zero_()
            local_loss = m.zeros(1, float)
            with wp.Tape() as tape:
                current, h = start, start_history
                for step in range(first, last):
                    current, h, reaction = self.step(current, h, tool_at_step(step), frozen[step])
                    objective(step, current, h, reaction, local_loss)
            current_arrays = arrays(current, h)
            if timings is not None:
                for name, expected_array in arrays(expected, expected_history).items():
                    delta = np.abs(current_arrays[name].numpy() - expected_array.numpy())
                    error = float(delta.max()) if delta.size else 0.0
                    measured['recompute_max_abs'][name] = max(measured['recompute_max_abs'].get(name, 0.0), error)
                if h.codes is not None:
                    if (not np.array_equal(h.codes.numpy(), expected_history.codes.numpy())
                            or not np.array_equal(h.pair_keys, expected_history.pair_keys)
                            or h.cut_index != expected_history.cut_index):
                        raise RuntimeError("Checkpoint recomputation changed the cutting topology")
                actual_boundary = h.edge_departed
                expected_boundary = expected_history.edge_departed
                if ((actual_boundary is None) != (expected_boundary is None)
                        or (actual_boundary is not None
                            and not np.array_equal(actual_boundary.numpy(), expected_boundary.numpy()))):
                    raise RuntimeError("Checkpoint recomputation changed the event boundary state")
            finished = now()
            measured['recompute_s'] += finished - begun
            seeds = {a: gradient[name] for name, a in current_arrays.items()}
            seeds[local_loss] = seed
            tape.backward(grads=seeds)
            gradient = {name: wp.clone(a.grad) for name, a in start_arrays.items()}
            if timings is not None:
                measured['adjoint_trace'].append(dict(step=first, maximum={
                    name: float(np.max(np.abs(a.numpy()))) if a.size else 0.0
                    for name, a in gradient.items()}))
            for a in parameters:
                parameter_gradients[a] += a.grad.numpy()
            tape.zero()
            measured['backward_s'] += now() - finished
            del tape
        if timings is not None:
            timings.update(measured)
        state_gradient = PlasticState(gradient['x'], gradient['v'], gradient['F'], gradient['Fp'],
                                      stress=gradient.get('stress'))
        history_gradient = CutHistory(gradient['rigid'], gradient['pairs'], gradient['damage'])
        return CutGradient(terminal, terminal_history, loss, state_gradient, history_gradient, parameter_gradients)



class EventCutContact(FixedCutContact):
    """Integrate the edge impulse up to damage release with step-constant force and advance.

    For threshold Q [J], edge force f [N] and positive advance u [m/s],
    the intact duration is min(dt, (1-D)*Q/(f*u)). Side codes and pair IDs
    are supplied by the caller; release flags follow the evolving damage.
    """

    def __init__(self, trajectory, *, friction_update='lagged'):
        if trajectory.config.cut_hold:
            raise ValueError('Event release requires cut_hold=False: the contact normal must persist across release')
        super().__init__(trajectory, friction_update=friction_update)
        self.integrate_release = True


@dataclass
class CutGradient:
    terminal: PlasticState
    history: CutHistory
    loss: wp.array
    initial_gradient: PlasticState
    history_gradient: CutHistory
    parameter_gradients: dict
