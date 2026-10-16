"""Tool and material contact, cutting resistance and side labels (paper Eqs. 3-6, 10-16).

Symbols follow the paper:
    sn   normal penalty factor, contact stiffness k_n = sn * me / dt^2
    gk   tangential stiffness factor, k_t = gk * me / dt^2
    gc   tangential damping ratio
    mus  static friction coefficient, mud dynamic friction coefficient
    me   reduced (effective) mass of the contact pair
    jn   normal impulse magnitude, jt tangential impulse
"""

import warp as wp
from .material import compatible, insert_key


@wp.func
def friction(
    n: wp.vec3,
    relative: wp.vec3,
    jn: float,
    history: wp.vec3,
    me: float,
    dt: float,
    mus: float,
    mud: float,
    gk: float,
    gc: float,
):
    vt = relative - wp.dot(relative, n) * n
    ht = history - wp.dot(history, n) * n
    k = gk * me / (dt * dt)
    damping = 2.0 * gc * wp.sqrt(me * k)
    trial = (-k * ht - damping * vt) * dt
    magnitude = wp.length(trial)
    if magnitude <= mus * jn:
        return trial, ht + dt * vt
    applied = trial * (mud * jn / magnitude)
    return applied, -(applied / dt + damping * vt) / k


@wp.func
def blade_query(local: wp.vec3, half: wp.vec3, bevel: float):
    closest = wp.vec3(
        wp.clamp(local[0], -half[0], half[0]),
        wp.clamp(local[1], -half[1], half[1]),
        wp.clamp(local[2], -half[2], half[2]),
    )
    delta = local - closest
    distance = wp.length(delta)
    normal = wp.vec3(0.0)
    if bevel == 0.0:
        if distance > 1.0e-12:
            normal = delta / distance
        else:
            margin = half - wp.abs(local)
            axis = int(0)
            if margin[1] < margin[axis]:
                axis = 1
            if margin[2] < margin[axis]:
                axis = 2
            normal[axis] = 1.0
            if local[axis] < 0.0:
                normal[axis] = -1.0
            closest[axis] = normal[axis] * half[axis]
            distance = -margin[axis]
    else:
        # Convex beveled prism, polygon in the local x-z plane.
        best = float(1.0e30)
        inside = bool(True)
        near = wp.vec2(0.0)
        out = wp.vec2(0.0)
        q = wp.vec2(local[0], local[2])
        for i in range(5):
            a = wp.vec2(0.0, -half[2])
            b = wp.vec2(half[0], -half[2] + bevel)
            if i == 1:
                a = b
                b = wp.vec2(half[0], half[2])
            elif i == 2:
                a = wp.vec2(half[0], half[2])
                b = wp.vec2(-half[0], half[2])
            elif i == 3:
                a = wp.vec2(-half[0], half[2])
                b = wp.vec2(-half[0], -half[2] + bevel)
            elif i == 4:
                a = wp.vec2(-half[0], -half[2] + bevel)
                b = wp.vec2(0.0, -half[2])
            edge = b - a
            offset = q - a
            if edge[0] * offset[1] - edge[1] * offset[0] < 0.0:
                inside = False
            t = wp.clamp(wp.dot(offset, edge) / wp.dot(edge, edge), 0.0, 1.0)
            candidate = a + t * edge
            dist = wp.length(q - candidate)
            if dist < best:
                best = dist
                near = candidate
                out = wp.normalize(wp.vec2(edge[1], -edge[0]))
        closest = wp.vec3(near[0], wp.clamp(local[1], -half[1], half[1]), near[1])
        if inside and wp.abs(local[1]) <= half[1]:
            normal = wp.vec3(out[0], 0.0, out[1])
            distance = -best
            if half[1] - wp.abs(local[1]) < best:
                sign = float(1.0)
                if local[1] < 0.0:
                    sign = -1.0
                closest = local
                closest[1] = sign * half[1]
                normal = wp.vec3(0.0, sign, 0.0)
                distance = -(half[1] - wp.abs(local[1]))
        else:
            if inside:
                closest[0] = local[0]
                closest[2] = local[2]
            delta = local - closest
            distance = wp.length(delta)
            normal = wp.vec3(out[0], 0.0, out[1])
            if distance > 1.0e-12:
                normal = delta / distance
    return closest, normal, distance


@wp.func
def penalty_impulse(
    gap: float,
    relative: wp.vec3,
    n: wp.vec3,
    me: float,
    history: wp.vec3,
    dt: float,
    sn: float,
    mus: float,
    mud: float,
    gk: float,
    gc: float,
):
    """Predictive normal penalty plus history-based friction for one contact.

    Returns (impulse, new tangential history). Both are zero when the contact
    stays separated over the step.
    """
    predicted = gap + dt * wp.min(wp.dot(relative, n), 0.0)
    if predicted >= 0.0:
        return wp.vec3(0.0), wp.vec3(0.0)
    jn = sn * me / dt * (-predicted)
    jt, ht = friction(n, relative, jn, history, me, dt, mus, mud, gk, gc)
    return jn * n + jt, ht


@wp.func
def blade_effective_mass(
    mass: float,
    lever: wp.vec3,
    n: wp.vec3,
    q: wp.quat,
    blade_mass: float,
    inertia_inv: wp.vec3,
    mobility: wp.array2d(dtype=float),
    tool_model: int,
):
    """Reduced mass of a particle against the blade along normal n.

    tool_model -1: articulated tool, inverse mass u^T (J M^-1 J^T) u (paper Eq. 15).
    tool_model  0: translating blade with mass blade_mass.
    tool_model  1: free rigid blade, adds the rotational term.
    """
    inverse = 1.0 / blade_mass
    if tool_model < 0:
        rn = wp.cross(lever, n)
        u = wp.spatial_vector(n[0], n[1], n[2], rn[0], rn[1], rn[2])
        inverse = 0.0
        for row in range(6):
            for col in range(6):
                inverse += u[row] * mobility[row, col] * u[col]
    elif tool_model > 0:
        rn = wp.quat_rotate_inv(q, wp.cross(lever, n))
        inverse += wp.dot(rn, wp.cw_mul(inertia_inv, rn))
    return 1.0 / (1.0 / mass + inverse)


@wp.func
def resist_cut(
    p: int,
    j: wp.vec3,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    origin: wp.vec3,
    tool_velocity: wp.vec3,
    q: wp.quat,
    tool_angular: wp.vec3,
    half: wp.vec3,
    radius: float,
    spacing: float,
    damage: wp.array(dtype=float),
    threshold: float,
    work: wp.array(dtype=float),
):
    """Cutting resistance at the blade edge.

    While a particle just ahead of the edge is intact, the edge-direction
    push does work that accumulates into its damage. Once damage reaches 1
    that push component is removed and the edge passes through.
    """
    edge = origin + wp.quat_rotate(q, wp.vec3(0.0, 0.0, -half[2]))
    down = wp.quat_rotate(q, wp.vec3(0.0, 0.0, -1.0))
    edge_velocity = tool_velocity + wp.cross(tool_angular, edge - origin)
    front = wp.dot(x[p] - edge, down)
    advance = wp.dot(edge_velocity - v[p], down)
    local_edge = wp.quat_rotate_inv(q, x[p] - edge)
    ahead_of_edge = (
        front >= 0.0
        and front <= 2.0 * radius
        and advance > 0.0
        and wp.abs(local_edge[0]) <= 6.0 * spacing
        and wp.abs(local_edge[1]) <= half[1]
    )
    if not ahead_of_edge:
        return j
    push = wp.max(wp.dot(j, down), 0.0)
    if damage[p] >= 1.0:
        return j - push * down
    delta = push * advance
    wp.atomic_add(work, 0, delta)
    damage[p] = wp.min(1.0, damage[p] + delta / threshold)
    return j


@wp.func
def rigid_at(
    p: int,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    tool_pos: wp.array(dtype=wp.vec3),
    tool_vel: wp.array(dtype=wp.vec3),
    tool_rot: wp.array(dtype=wp.quat),
    tool_omega: wp.array(dtype=wp.vec3),
    half: wp.vec3,
    bevel: float,
    radius: float,
    blade_offset: float,
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
):
    """Gravity, board contact and blade contact impulse on particle p."""
    xp = x[p]
    total = wp.vec3(0.0, 0.0, -9.81 * mass * dt)

    # Board: the plane z = 0 inside the board rectangle (a zero half extent means unbounded).
    board_gap = xp[2] - radius
    if board_half[0] > 0.0 and wp.abs(xp[0]) > board_half[0]:
        board_gap = 1.0e30
    if board_half[1] > 0.0 and wp.abs(xp[1]) > board_half[1]:
        board_gap = 1.0e30
    j_board, h_board = penalty_impulse(
        board_gap, v[p], wp.vec3(0.0, 0.0, 1.0), mass, histories[0, p], dt, sn, mus, mud, gk, gc
    )
    histories[0, p] = h_board
    if wp.length_sq(j_board) > 0.0:
        total += j_board
        wp.atomic_add(board, 0, j_board)

    # Blade: closest point on the (beveled) box, relative velocity at that point.
    q = tool_rot[0]
    origin = tool_pos[0]
    # Bounding-sphere test: the particle cannot reach the blade within this step.
    blade_radius = wp.length(half)
    max_closing = wp.length(v[p]) + wp.length(tool_vel[0]) + wp.length(tool_omega[0]) * blade_radius
    if wp.length(xp - origin) - blade_radius - blade_offset > dt * max_closing:
        histories[1, p] = wp.vec3(0.0)
        impulse[p] = total
        return
    closest_local, normal_local, distance = blade_query(wp.quat_rotate_inv(q, xp - origin), half, bevel)
    lever = wp.quat_rotate(q, closest_local)
    n = wp.quat_rotate(q, normal_local)
    relative = v[p] - tool_vel[0] - wp.cross(tool_omega[0], lever)
    if distance - blade_offset + dt * wp.min(wp.dot(relative, n), 0.0) >= 0.0:
        histories[1, p] = wp.vec3(0.0)
        impulse[p] = total
        return
    me = blade_effective_mass(mass, lever, n, q, blade_mass, inertia_inv, mobility, tool_model)
    j_blade, h_blade = penalty_impulse(
        distance - blade_offset, relative, n, me, histories[1, p], dt, sn, mus, mud, gk, gc
    )
    histories[1, p] = h_blade
    if wp.length_sq(j_blade) > 0.0:
        if cutting != 0:
            j_blade = resist_cut(
                p,
                j_blade,
                x,
                v,
                origin,
                tool_vel[0],
                q,
                tool_omega[0],
                half,
                blade_offset,
                spacing,
                damage,
                threshold,
                work,
            )
        total += j_blade
        wp.atomic_add(reaction, 0, -j_blade)
        wp.atomic_add(torque, 0, -wp.cross(lever, j_blade))
    impulse[p] = total


@wp.kernel
def rigid_and_speed(
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
    blade_offset: float,
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
):
    """Rigid contacts for every particle, and the maximum particle speed for the pair search."""
    p = wp.tid()
    wp.atomic_max(speed, 0, wp.length(v[p]))
    rigid_at(
        p,
        x,
        v,
        tool_pos,
        tool_vel,
        tool_rot,
        tool_omega,
        half,
        bevel,
        radius,
        blade_offset,
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


@wp.func
def find_key(keys: wp.array(dtype=wp.int64), key: wp.int64, capacity: int):
    slot = int((wp.uint64(key) * wp.uint64(2654435761)) % wp.uint64(capacity))
    result = int(-1)
    for probe in range(capacity):
        if keys[slot] == wp.int64(-1):
            break
        if keys[slot] == key:
            result = slot
            break
        slot = (slot + 1) % capacity
    return result


@wp.func
def pair_at(
    p: int,
    q: int,
    x: wp.array(dtype=wp.vec3),
    ref: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    codes: wp.array(dtype=wp.int64),
    radius: float,
    spacing: float,
    mass: float,
    dt: float,
    sn: float,
    mus: float,
    mud: float,
    gk: float,
    gc: float,
    old_keys: wp.array(dtype=wp.int64),
    old_history: wp.array(dtype=wp.vec3),
    new_keys: wp.array(dtype=wp.int64),
    new_history: wp.array(dtype=wp.vec3),
    new_used: wp.array(dtype=int),
    new_count: wp.array(dtype=int),
    capacity: int,
    n: int,
    impulses: wp.array(dtype=wp.vec3),
    errors: wp.array(dtype=int),
):
    delta = x[p] - x[q]
    dist = wp.length(delta)
    if dist >= 2.0 * radius + dt * wp.length(v[p] - v[q]):
        return
    if compatible(codes[p], codes[q]) and wp.length(ref[p] - ref[q]) < 2.0 * spacing:
        return
    if dist < 1.0e-12:
        wp.atomic_add(errors, 2, 1)
        return
    normal = delta / dist
    relative = v[p] - v[q]
    predicted = 2.0 * radius - dist - dt * wp.min(wp.dot(relative, normal), 0.0)
    if predicted <= 0.0:
        return
    key = wp.int64(p) * wp.int64(n) + wp.int64(q)
    old = find_key(old_keys, key, capacity)
    history = wp.vec3(0.0)
    if old >= 0:
        history = old_history[old]
    new = insert_key(new_keys, key, capacity)
    if new < 0:
        wp.atomic_add(errors, 0, 1)
        return
    jn = sn * (mass / 2.0) / dt * predicted
    jt, ht = friction(normal, relative, jn, history, mass / 2.0, dt, mus, mud, gk, gc)
    new_history[new] = ht
    new_used[wp.atomic_add(new_count, 0, 1)] = new
    impulse = jn * normal + jt
    wp.atomic_add(impulses, p, impulse)
    wp.atomic_add(impulses, q, -impulse)


@wp.kernel
def clear_pair_slots(
    count: wp.array(dtype=int),
    threads: int,
    used: wp.array(dtype=int),
    keys: wp.array(dtype=wp.int64),
):
    """Empty only the hash slots filled two steps ago, instead of the whole table."""
    for a in range(wp.tid(), count[0], threads):
        keys[used[a]] = wp.int64(-1)


@wp.kernel
def pairs(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    ref: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    codes: wp.array(dtype=wp.int64),
    radius: float,
    spacing: float,
    mass: float,
    dt: float,
    sn: float,
    mus: float,
    mud: float,
    gk: float,
    gc: float,
    speed: wp.array(dtype=float),
    old_keys: wp.array(dtype=wp.int64),
    old_history: wp.array(dtype=wp.vec3),
    new_keys: wp.array(dtype=wp.int64),
    new_history: wp.array(dtype=wp.vec3),
    new_used: wp.array(dtype=int),
    new_count: wp.array(dtype=int),
    capacity: int,
    n: int,
    impulses: wp.array(dtype=wp.vec3),
    errors: wp.array(dtype=int),
):
    p = wp.tid()
    query = wp.hash_grid_query(grid, x[p], 2.0 * radius + 2.0 * speed[0] * dt)
    for q in query:
        if q > p:
            pair_at(
                p,
                q,
                x,
                ref,
                v,
                codes,
                radius,
                spacing,
                mass,
                dt,
                sn,
                mus,
                mud,
                gk,
                gc,
                old_keys,
                old_history,
                new_keys,
                new_history,
                new_used,
                new_count,
                capacity,
                n,
                impulses,
                errors,
            )


@wp.kernel
def pairs_all(
    x: wp.array(dtype=wp.vec3),
    ref: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    codes: wp.array(dtype=wp.int64),
    radius: float,
    spacing: float,
    mass: float,
    dt: float,
    sn: float,
    mus: float,
    mud: float,
    gk: float,
    gc: float,
    old_keys: wp.array(dtype=wp.int64),
    old_history: wp.array(dtype=wp.vec3),
    new_keys: wp.array(dtype=wp.int64),
    new_history: wp.array(dtype=wp.vec3),
    new_used: wp.array(dtype=int),
    new_count: wp.array(dtype=int),
    capacity: int,
    n: int,
    lanes: int,
    speed: wp.array(dtype=float),
    impulses: wp.array(dtype=wp.vec3),
    errors: wp.array(dtype=int),
):
    """All-pairs search over (particle, lane) threads; lane k takes q = p+1+k, p+1+k+lanes, ...

    Pairs farther apart than 2 radius + 2 dt max_speed cannot touch this step
    and are rejected before the exact test in pair_at.
    """
    p, k = wp.tid()
    xp = x[p]
    reach = 2.0 * radius + 2.0 * dt * speed[0]
    for q in range(p + 1 + k, n, lanes):
        if wp.length_sq(x[q] - xp) >= reach * reach:
            continue
        pair_at(
            p,
            q,
            x,
            ref,
            v,
            codes,
            radius,
            spacing,
            mass,
            dt,
            sn,
            mus,
            mud,
            gk,
            gc,
            old_keys,
            old_history,
            new_keys,
            new_history,
            new_used,
            new_count,
            capacity,
            n,
            impulses,
            errors,
        )


@wp.kernel
def assign(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    codes: wp.array(dtype=wp.int64),
    tool_pos: wp.array(dtype=wp.vec3),
    tool_vel: wp.array(dtype=wp.vec3),
    tool_rot: wp.array(dtype=wp.quat),
    tool_omega: wp.array(dtype=wp.vec3),
    half: wp.vec3,
    spacing: float,
    dt: float,
    digit: wp.int64,
    cutting: int,
    topology_dirty: wp.array(dtype=int),
):
    p = wp.tid()
    if cutting == 0 or (codes[p] // digit) % wp.int64(3) != wp.int64(0):
        return
    edge = tool_pos[0] + wp.quat_rotate(tool_rot[0], wp.vec3(0.0, 0.0, -half[2]))
    local = wp.quat_rotate_inv(tool_rot[0], x[p] - edge)
    front = -local[2]
    edge_v = tool_vel[0] + wp.cross(tool_omega[0], edge - tool_pos[0])
    d = wp.quat_rotate(tool_rot[0], wp.vec3(0.0, 0.0, -1.0))
    advance = wp.dot(edge_v - v[p], d)
    back = 2.0 * spacing + wp.max(advance * dt, 0.0)
    if wp.abs(local[0]) <= 6.0 * spacing and wp.abs(local[1]) <= half[1] and front >= -back and front <= 0.0:
        side = int(2)
        if local[0] < 0.0:
            side = 1
        codes[p] += digit * wp.int64(side)
        wp.atomic_max(topology_dirty, 0, 1)


@wp.kernel
def count_embedded(
    x: wp.array(dtype=wp.vec3),
    tool_pos: wp.array(dtype=wp.vec3),
    tool_rot: wp.array(dtype=wp.quat),
    half: wp.vec3,
    spacing: float,
    count: wp.array(dtype=int),
):
    """Particles beside the blade body (above the edge) inside the side-labelling band."""
    p = wp.tid()
    local = wp.quat_rotate_inv(tool_rot[0], x[p] - tool_pos[0])
    if wp.abs(local[0]) <= 6.0 * spacing and wp.abs(local[1]) <= half[1] and wp.abs(local[2]) <= half[2]:
        wp.atomic_add(count, 0, 1)


@wp.kernel
def clear_step(
    reaction: wp.array(dtype=wp.vec3),
    torque: wp.array(dtype=wp.vec3),
    board: wp.array(dtype=wp.vec3),
    speed: wp.array(dtype=float),
):
    reaction[0] = wp.vec3(0.0)
    torque[0] = wp.vec3(0.0)
    board[0] = wp.vec3(0.0)
    speed[0] = 0.0
