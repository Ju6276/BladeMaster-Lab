"""Prescribed cutting along recorded side codes, release events and pair sets."""

from dataclasses import dataclass

import numpy as np
import warp as wp

from .contact import blade_query, penalty_impulse
from .differentiable_contact import PrescribedContact
from .differentiable_material import FixedTopology


@wp.func
def contact_mode(gap: float, relative: wp.vec3, normal: wp.vec3, history: wp.vec3,
                 mass: float, dt: float, sn: float, mus: float, gk: float, gc: float):
    predicted = gap + dt * wp.min(wp.dot(relative, normal), 0.0)
    if predicted >= 0.0:
        return 0
    jn = sn * mass / dt * (-predicted)
    vt = relative - wp.dot(relative, normal) * normal
    ht = history - wp.dot(history, normal) * normal
    k = gk * mass / (dt * dt)
    damping = 2.0 * gc * wp.sqrt(mass * k)
    mode = int(2)
    if wp.length((-k * ht - damping * vt) * dt) <= mus * jn:
        mode = 1
    if wp.dot(relative, normal) < 0.0:
        mode += 10
    return mode


@wp.func
def branch_penalty(gap: float, relative: wp.vec3, normal: wp.vec3, mass: float, history: wp.vec3,
                   dt: float, sn: float, mus: float, mud: float, gk: float, gc: float, mode: int):
    if mode < 0:
        return penalty_impulse(gap, relative, normal, mass, history, dt, sn, mus, mud, gk, gc)
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
    trial = (-k * ht - damping * vt) * dt
    if mode % 10 == 1:
        return jn * normal + trial, ht + dt * vt
    applied = trial * (mud * jn / wp.length(trial))
    return jn * normal + applied, -(applied / dt + damping * vt) / k


@wp.kernel
def rigid_step(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    history: wp.array2d(dtype=wp.vec3),
    damage: wp.array(dtype=float),
    released: wp.array(dtype=int),
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
    dt: float,
    sn: float,
    board_sn: float,
    gk: float,
    gc: float,
    band_width: float,
    threshold: float,
    cutting: int,
    cut_hold: int,
    cut_advance: int,
    next_history: wp.array2d(dtype=wp.vec3),
    next_damage: wp.array(dtype=float),
    impulse: wp.array(dtype=wp.vec3),
    reaction: wp.array(dtype=wp.vec3),
    modes: wp.array2d(dtype=int),
):
    p = wp.tid()
    xp = x[p]
    board_gap = xp[2] - radius
    if board_half[0] > 0.0 and wp.abs(xp[0]) > board_half[0]:
        board_gap = 1.0e30
    if board_half[1] > 0.0 and wp.abs(xp[1]) > board_half[1]:
        board_gap = 1.0e30
    board, board_history = branch_penalty(
        board_gap, v[p], wp.vec3(0.0, 0.0, 1.0), mass, history[0, p],
        dt, board_sn, friction[0], friction[1], gk, gc, fixed_modes[0, p],
    )
    modes[0, p] = contact_mode(board_gap, v[p], wp.vec3(0.0, 0.0, 1.0), history[0, p],
                               mass, dt, board_sn, friction[0], gk, gc)
    q = rotation[0]
    local = wp.quat_rotate_inv(q, xp - position[0])
    closest, normal, distance = blade_query(local, half, bevel)
    front = -half[2] - local[2]
    modes[3, p] = 0
    if (cutting != 0 and cut_hold != 0 and released[p] == 0
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
        dt, sn, friction[0], friction[1], gk, gc, fixed_modes[1, p],
    )
    modes[1, p] = contact_mode(distance - blade_offset, relative, n, history[1, p],
                               mass, dt, sn, friction[0], gk, gc)
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
        if (edge_front >= 0.0 and edge_front <= 2.0 * blade_offset
                and wp.abs(local_edge[0]) <= band_width and wp.abs(local_edge[1]) <= half[1]):
            push = wp.max(wp.dot(blade, down), 0.0)
            modes[4, p] = 1
            if released[p] != 0:
                blade = blade - push * down
                modes[4, p] = 2
            elif advance > 0.0:
                updated_damage = wp.min(1.0, damage[p] + push * advance / threshold)
                modes[4, p] = 3
                if updated_damage >= 1.0:
                    modes[4, p] = 4
    next_history[0, p] = board_history
    next_history[1, p] = blade_history
    next_damage[p] = updated_damage
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
    dt: float,
    sn: float,
    gk: float,
    gc: float,
    next_history: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=wp.vec3),
    modes: wp.array(dtype=int),
):
    index = wp.tid()
    p, q = pairs[index, 0], pairs[index, 1]
    delta = x[p] - x[q]
    distance = wp.length(delta)
    normal = delta / distance
    old = wp.vec3(0.0)
    if previous[index] >= 0:
        old = history[previous[index]]
    applied, updated = branch_penalty(distance - diameter, v[p] - v[q], normal,
                                     mass / 2.0, old, dt, sn, friction[0], friction[1], gk, gc,
                                     fixed_modes[index])
    next_history[index] = updated
    modes[index] = contact_mode(distance - diameter, v[p] - v[q], normal, old,
                               mass / 2.0, dt, sn, friction[0], gk, gc)
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

    def __init__(self, trajectory):
        super().__init__(trajectory)
        cfg = trajectory.config
        self.friction = trajectory.array([cfg.friction_static, cfg.friction_dynamic], float)
        self._topologies = {}

    def history(self, rigid=None, pairs=None, damage=None):
        m = self.model
        rigid = np.zeros((2, m.n, 3)) if rigid is None else rigid
        pairs = np.zeros((0, 3)) if pairs is None else pairs
        damage = np.zeros(m.n) if damage is None else damage
        return CutHistory(m.array(rigid, wp.vec3), m.array(pairs, wp.vec3), m.array(damage, float))

    def freeze(self, codes, pair_keys, previous_keys, released, cutting=True,
               rigid_modes=None, pair_modes=None, yielded=None):
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
                          None if yielded is None else m.array(yielded, int, False))

    def step(self, state, history, tool, frozen):
        m, cfg = self.model, self.model.config
        sn, gk = cfg.contact_scales()
        radius = cfg.grid_spacing / 4
        blade_offset = max(cfg.blade_contact_radius or radius, radius - cfg.blade_half_size[0])
        threshold = cfg.cutting_energy * m.volume ** (2 / 3) * (1 - cfg.sharpness) / cfg.sharpness
        updated = CutHistory(m.zeros((2, m.n), wp.vec3), m.zeros(frozen.pairs.shape[0], wp.vec3),
                             m.zeros(m.n, float), m.zeros((5, m.n), int, False),
                             m.zeros(frozen.pairs.shape[0], int, False))
        rigid = m.zeros(m.n, wp.vec3)
        pairs = m.zeros(m.n, wp.vec3)
        impulse = m.zeros(m.n, wp.vec3)
        reaction = m.zeros(1, wp.vec3)
        wp.launch(rigid_step, m.n,
                  inputs=[state.x, state.v, history.rigid, history.damage, frozen.released, frozen.rigid_modes,
                          tool.position, tool.velocity, tool.rotation, tool.omega, self.friction,
                          wp.vec3(*cfg.blade_half_size), cfg.blade_bevel_height, radius, blade_offset,
                          wp.vec2(*cfg.board_half_extent), m.mass, cfg.dt, sn, cfg.board_normal_scale or sn,
                          gk, cfg.tangent_damping, cfg.cut_band_cells * cfg.grid_spacing, threshold,
                          int(frozen.cutting), int(cfg.cut_hold), int(cfg.cut_work == "advance")],
                  outputs=[updated.rigid, updated.damage, rigid, reaction, updated.rigid_modes], device=m.device)
        if frozen.pairs.shape[0]:
            wp.launch(pair_step, frozen.pairs.shape[0],
                      inputs=[state.x, state.v, frozen.pairs, frozen.previous, frozen.pair_modes,
                              history.pairs, self.friction,
                              2 * radius, m.mass, cfg.dt, sn, gk, cfg.tangent_damping],
                      outputs=[updated.pairs, pairs, updated.pair_modes], device=m.device)
        wp.launch(add_impulses, m.n, inputs=[rigid, pairs], outputs=[impulse], device=m.device)
        result = m.step(state, impulse=impulse, topology=frozen.topology, fixed_yielded=frozen.yielded)
        return result, updated, reaction

    def checkpoint_vjp(self, *args, **kwargs):
        raise NotImplementedError("Fixed-cut trajectories currently retain the full Tape")
