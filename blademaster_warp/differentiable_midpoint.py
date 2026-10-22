"""Explicit midpoint stages with endpoint return maps for contact histories."""

from dataclasses import replace

import numpy as np
import warp as wp

from .contact import blade_query
from .differentiable_cutting import FixedCutContact


@wp.kernel
def half_duration(duration: wp.array(dtype=float), half: wp.array(dtype=float)):
    half[0] = 0.5 * duration[0]


@wp.kernel
def position_midpoint(start: wp.array(dtype=wp.vec3), predictor: wp.array(dtype=wp.vec3),
                      end: wp.array(dtype=wp.vec3), middle: wp.array(dtype=wp.vec3)):
    p = wp.tid()
    middle[p] = 0.25 * start[p] + 0.5 * predictor[p] + 0.25 * end[p]


@wp.func
def tangent(relative: wp.vec3, normal: wp.vec3):
    return relative - wp.dot(relative, normal) * normal


@wp.func
def project_history(gap: float, relative: wp.vec3, normal: wp.vec3, history: wp.vec3,
                    kn: float, kt: float, damping: float, mus: float, mud: float):
    if gap >= 0.0:
        return wp.vec3(0.0)
    vt = tangent(relative, normal)
    ht = tangent(history, normal)
    trial = -kt * ht - damping * vt
    magnitude = wp.length(trial)
    normal_force = kn * (-gap)
    if magnitude <= mus * normal_force:
        return ht
    force = wp.vec3(0.0)
    if magnitude > 0.0:
        force = trial * (mud * normal_force / magnitude)
    return -(force + damping * vt) / kt


@wp.func
def board_gap(x: wp.vec3, radius: float, half: wp.vec2):
    gap = x[2] - radius
    if (half[0] > 0.0 and wp.abs(x[0]) > half[0]) or (half[1] > 0.0 and wp.abs(x[1]) > half[1]):
        gap = 1.0e30
    return gap


@wp.func
def blade_contact(x: wp.vec3, v: wp.vec3, position: wp.vec3, velocity: wp.vec3,
                  rotation: wp.quat, omega: wp.vec3, half: wp.vec3, bevel: float,
                  offset: float, hold: int, damage: float, departed: int):
    local = wp.quat_rotate_inv(rotation, x - position)
    closest, normal, distance = blade_query(local, half, bevel)
    front = -half[2] - local[2]
    if (hold != 0 and damage < 1.0 and departed == 0
            and wp.abs(local[0]) <= half[0] + offset and wp.abs(local[1]) <= half[1] and front >= 0.0):
        closest = wp.vec3(local[0], local[1], -half[2])
        normal = wp.vec3(0.0, 0.0, -1.0)
        distance = front
    relative = v - velocity - wp.cross(omega, wp.quat_rotate(rotation, closest))
    return distance - offset, wp.quat_rotate(rotation, normal), relative


@wp.kernel
def rigid_history_endpoint(
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
    stage_x: wp.array(dtype=wp.vec3), stage_v: wp.array(dtype=wp.vec3),
    base_history: wp.array2d(dtype=wp.vec3), damage: wp.array(dtype=float),
    stage_damage: wp.array(dtype=float), departed: wp.array(dtype=int),
    position: wp.array(dtype=wp.vec3), rotation: wp.array(dtype=wp.quat),
    stage_position: wp.array(dtype=wp.vec3), stage_rotation: wp.array(dtype=wp.quat),
    velocity: wp.array(dtype=wp.vec3), omega: wp.array(dtype=wp.vec3),
    half: wp.vec3, bevel: float, radius: float, offset: float, board_half: wp.vec2,
    hold: int, kn: float, board_kn: float, kt: float, damping: float,
    friction: wp.array(dtype=float), duration: wp.array(dtype=float),
    result: wp.array2d(dtype=wp.vec3),
):
    p = wp.tid()
    up = wp.vec3(0.0, 0.0, 1.0)
    h = tangent(base_history[0, p], up) + duration[0] * tangent(stage_v[p], up)
    result[0, p] = project_history(board_gap(x[p], radius, board_half), v[p], up, h,
                                  board_kn, kt, damping, friction[0], friction[1])
    gap, normal, relative = blade_contact(
        x[p], v[p], position[0], velocity[0], rotation[0], omega[0],
        half, bevel, offset, hold, damage[p], departed[p],
    )
    stage_gap, stage_normal, stage_relative = blade_contact(
        stage_x[p], stage_v[p], stage_position[0], velocity[0], stage_rotation[0], omega[0],
        half, bevel, offset, hold, stage_damage[p], departed[p],
    )
    h = tangent(base_history[1, p], stage_normal) + duration[0] * tangent(stage_relative, stage_normal)
    result[1, p] = project_history(gap, relative, normal, h, kn, kt, damping, friction[0], friction[1])


@wp.kernel
def pair_history_endpoint(
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
    stage_x: wp.array(dtype=wp.vec3), stage_v: wp.array(dtype=wp.vec3),
    pairs: wp.array2d(dtype=int), previous: wp.array(dtype=int), base_history: wp.array(dtype=wp.vec3),
    diameter: float, kn: float, kt: float, damping: float, friction: wp.array(dtype=float),
    duration: wp.array(dtype=float), result: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    p, q = pairs[i, 0], pairs[i, 1]
    normal = wp.normalize(x[p] - x[q])
    stage_normal = wp.normalize(stage_x[p] - stage_x[q])
    h = wp.vec3(0.0)
    if previous[i] >= 0:
        h = tangent(base_history[previous[i]], stage_normal)
    h += duration[0] * tangent(stage_v[p] - stage_v[q], stage_normal)
    result[i] = project_history(wp.length(x[p] - x[q]) - diameter, v[p] - v[q], normal,
                                h, kn, kt, damping, friction[0], friction[1])


def project_endpoint(contact, state, history, stage, base_history, stage_tool, end_tool, duration, branch,
                     departed):
    m, cfg = contact.model, contact.model.config
    radius = cfg.grid_spacing / 4
    offset = max(cfg.blade_contact_radius or radius, radius - cfg.blade_half_size[0])
    sn, gk = cfg.contact_scales()
    kn = sn * m.mass / cfg.dt**2
    kt = gk * m.mass / cfg.dt**2
    damping = 2 * cfg.tangent_damping * np.sqrt(m.mass * kt)
    rigid = m.zeros((2, m.n), wp.vec3)
    m.launch(rigid_history_endpoint, m.n,
             inputs=[state.x, state.v, stage.x, stage.v, base_history.rigid, history.damage,
                     base_history.damage, departed, end_tool.position, end_tool.rotation,
                     stage_tool.position, stage_tool.rotation, stage_tool.velocity, stage_tool.omega,
                     cfg.blade_half_size, cfg.blade_bevel_height, radius, offset, cfg.board_half_extent,
                     int(branch.cutting and cfg.cut_hold), kn,
                     (cfg.board_normal_scale or sn) * m.mass / cfg.dt**2, kt, damping,
                     contact.friction, duration], outputs=[rigid], device=m.device)
    history.rigid = rigid
    if branch.pairs.shape[0]:
        pairs = m.zeros(branch.pairs.shape[0], wp.vec3)
        m.launch(pair_history_endpoint, branch.pairs.shape[0],
                 inputs=[state.x, state.v, stage.x, stage.v, branch.pairs, branch.previous,
                         base_history.pairs, 2 * radius, kn / 2, kt / 2, damping / 2,
                         contact.friction, duration], outputs=[pairs], device=m.device)
        history.pairs = pairs
    return history


def advance_midpoint(contact, state, history, tool, duration, cutting):
    m = contact.model
    keys = contact.contact_keys(state, history.codes, float(duration.numpy()[0]))
    branch = contact.freeze(history.codes.numpy(), keys, history.pair_keys,
                            history.damage.numpy() >= 1, cutting=cutting)
    half = m.zeros(1, float)
    m.launch(half_duration, 1, inputs=[duration], outputs=[half], device=m.device)
    departed = contact._no_edge_departure if history.edge_departed is None else history.edge_departed
    mid_tool = contact.pose(tool, half)
    end_tool = contact.pose(tool, duration)
    mid, mid_history, _ = FixedCutContact.step(
        contact, state, history, tool, branch, duration=half, edge_departed=departed,
        stage_base=state, stage_history=history,
    )
    mid_history = project_endpoint(contact, mid, mid_history, state, history, tool, mid_tool,
                                    half, branch, departed)
    mid_branch = replace(branch, previous=m.array(np.arange(len(keys)), int, False))
    result, updated, force = FixedCutContact.step(
        contact, mid, mid_history, mid_tool, mid_branch, duration=duration, edge_departed=departed,
        stage_base=state, stage_history=history,
    )
    updated = project_endpoint(contact, result, updated, mid, history, mid_tool, end_tool,
                                duration, branch, departed)
    updated.path_midpoint = m.zeros(m.n, wp.vec3, False)
    m.launch(position_midpoint, m.n, inputs=[state.x, mid.x, result.x],
             outputs=[updated.path_midpoint], device=m.device)
    updated.codes, updated.pair_keys, updated.cut_index = history.codes, keys, history.cut_index
    return result, updated, force
