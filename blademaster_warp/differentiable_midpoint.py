"""Midpoint material stages and constitutive contact-history increments."""

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
def transport_history(gap: float, normal: wp.vec3, history: wp.vec3):
    if gap >= 0.0:
        return wp.vec3(0.0)
    return tangent(history, normal)


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
    history: wp.array2d(dtype=wp.vec3), damage: wp.array(dtype=float),
    departed: wp.array(dtype=int), position: wp.array(dtype=wp.vec3),
    rotation: wp.array(dtype=wp.quat), velocity: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3), half: wp.vec3, bevel: float,
    radius: float, offset: float, board_half: wp.vec2, hold: int,
    result: wp.array2d(dtype=wp.vec3),
):
    p = wp.tid()
    up = wp.vec3(0.0, 0.0, 1.0)
    result[0, p] = transport_history(board_gap(x[p], radius, board_half), up, history[0, p])
    gap, normal, relative = blade_contact(
        x[p], v[p], position[0], velocity[0], rotation[0], omega[0],
        half, bevel, offset, hold, damage[p], departed[p],
    )
    result[1, p] = transport_history(gap, normal, history[1, p])


@wp.kernel
def pair_history_endpoint(x: wp.array(dtype=wp.vec3), pairs: wp.array2d(dtype=int),
                          history: wp.array(dtype=wp.vec3), diameter: float,
                          result: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    delta = x[pairs[i, 0]] - x[pairs[i, 1]]
    result[i] = transport_history(wp.length(delta) - diameter, wp.normalize(delta), history[i])


def project_endpoint(contact, state, history, end_tool, branch, departed):
    m, cfg = contact.model, contact.model.config
    radius = cfg.grid_spacing / 4
    offset = max(cfg.blade_contact_radius or radius, radius - cfg.blade_half_size[0])
    rigid = m.zeros((2, m.n), wp.vec3)
    m.launch(rigid_history_endpoint, m.n,
             inputs=[state.x, state.v, history.rigid, history.damage, departed,
                     end_tool.position, end_tool.rotation, end_tool.velocity, end_tool.omega,
                     cfg.blade_half_size, cfg.blade_bevel_height, radius, offset, cfg.board_half_extent,
                     int(branch.cutting and cfg.cut_hold)], outputs=[rigid], device=m.device)
    history.rigid = rigid
    if branch.pairs.shape[0]:
        pairs = m.zeros(branch.pairs.shape[0], wp.vec3)
        m.launch(pair_history_endpoint, branch.pairs.shape[0],
                 inputs=[state.x, branch.pairs, history.pairs, 2 * radius], outputs=[pairs], device=m.device)
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
    # Material transport uses interpolated grid velocities, so an Euler-stage
    # contact can be absent from a search using the initial particle velocities.
    stage_keys = np.union1d(keys, contact.contact_keys(mid, history.codes, 0.))
    mid_branch = contact.freeze(history.codes.numpy(), stage_keys, history.pair_keys,
                                history.damage.numpy() >= 1, cutting=cutting)
    result, updated, force = FixedCutContact.step(
        contact, mid, mid_history, mid_tool, mid_branch, duration=duration, edge_departed=departed,
        stage_base=state, stage_history=history,
    )
    updated = project_endpoint(contact, result, updated, end_tool, mid_branch, departed)
    updated.path_midpoint = m.zeros(m.n, wp.vec3, False)
    m.launch(position_midpoint, m.n, inputs=[state.x, mid.x, result.x],
             outputs=[updated.path_midpoint], device=m.device)
    updated.codes, updated.pair_keys, updated.cut_index = history.codes, stage_keys, history.cut_index
    return result, updated, force
