"""Substep kernels that run independent per-particle and per-slot work in one launch.

A 1.5k-particle substep is bound by launch latency, not arithmetic, so merging
independent stages saves time without changing any result.
"""

import warp as wp
from .contact import assign_at, rigid_at
from .material import clear_grid_slots, update_F_at


@wp.kernel
def begin_substep(
    n: int,
    speed: wp.array(dtype=float),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    tool_pos: wp.array(dtype=wp.vec3),
    tool_vel: wp.array(dtype=wp.vec3),
    tool_rot: wp.array(dtype=wp.quat),
    tool_omega: wp.array(dtype=wp.vec3),
    half: wp.vec3,
    bevel: float,
    radius: float,
    spacing: float,
    mass: float,
    blade_mass: float,
    inertia_inv: wp.vec3,
    mobility: wp.array2d(dtype=float),
    tool_model: int,
    board_half: wp.vec2,
    dt: float,
    sn: float,
    mus: float,
    mud: float,
    gk: float,
    gc: float,
    histories: wp.array2d(dtype=wp.vec3),
    damage: wp.array(dtype=float),
    threshold: float,
    cutting: int,
    impulse: wp.array(dtype=wp.vec3),
    reaction: wp.array(dtype=wp.vec3),
    torque: wp.array(dtype=wp.vec3),
    board: wp.array(dtype=wp.vec3),
    work: wp.array(dtype=float),
    active_count: wp.array(dtype=int),
    grid_threads: int,
    grid_mass: wp.array(dtype=float),
    momentum: wp.array(dtype=wp.vec3),
    grid_impulse: wp.array(dtype=wp.vec3),
    remapped: wp.array(dtype=wp.vec3),
    pair_keys: wp.array(dtype=wp.int64),
):
    """Rigid contacts and maximum speed (per particle), cleared compact grid and next pair table."""
    tid = wp.tid()
    if tid < n:
        wp.atomic_max(speed, 0, wp.length(v[tid]))
        rigid_at(
            tid,
            x,
            v,
            tool_pos,
            tool_vel,
            tool_rot,
            tool_omega,
            half,
            bevel,
            radius,
            spacing,
            mass,
            blade_mass,
            inertia_inv,
            mobility,
            tool_model,
            board_half,
            dt,
            sn,
            mus,
            mud,
            gk,
            gc,
            histories,
            damage,
            threshold,
            cutting,
            impulse,
            reaction,
            torque,
            board,
            work,
        )
    if tid < grid_threads:
        clear_grid_slots(tid, active_count, grid_threads, grid_mass, momentum, grid_impulse, remapped)
    if tid < pair_keys.shape[0]:
        pair_keys[tid] = wp.int64(-1)


@wp.kernel
def update_F_and_assign(
    F: wp.array(dtype=wp.mat33),
    Fp: wp.array(dtype=wp.mat33),
    slots: wp.array2d(dtype=int),
    gradients: wp.array2d(dtype=wp.vec3),
    grid_mass: wp.array(dtype=float),
    momentum: wp.array(dtype=wp.vec3),
    dt: float,
    mu: float,
    lam: float,
    yield_stress: float,
    stress: wp.array(dtype=wp.mat33),
    errors: wp.array(dtype=int),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    codes: wp.array(dtype=int),
    tool_pos: wp.array(dtype=wp.vec3),
    tool_vel: wp.array(dtype=wp.vec3),
    tool_rot: wp.array(dtype=wp.quat),
    tool_omega: wp.array(dtype=wp.vec3),
    half: wp.vec3,
    spacing: float,
    assign_dt: float,
    digit: int,
    cutting: int,
    topology_dirty: wp.array(dtype=int),
):
    """Deformation update and side labels; valid only while the tool is fixed within the substep."""
    p = wp.tid()
    update_F_at(p, F, Fp, slots, gradients, grid_mass, momentum, dt, mu, lam, yield_stress, stress, errors)
    assign_at(
        p,
        x,
        v,
        codes,
        tool_pos,
        tool_vel,
        tool_rot,
        tool_omega,
        half,
        spacing,
        dt,
        digit,
        cutting,
        topology_dirty,
    )
