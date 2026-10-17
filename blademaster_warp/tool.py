"""Rigid knife drive, knife/board contact and external-tool step finalization."""

import warp as wp


@wp.kernel
def accumulate_reaction(
    reaction: wp.array(dtype=wp.vec3),
    torque: wp.array(dtype=wp.vec3),
    position: wp.array(dtype=wp.vec3),
    linear: wp.array(dtype=wp.vec3),
    angular: wp.array(dtype=wp.vec3),
):
    """Sum impulses; angular impulse is about the fixed world origin."""
    linear[0] += reaction[0]
    angular[0] += torque[0] + wp.cross(position[0], reaction[0])


@wp.kernel
def load_external_state(
    packed: wp.array(dtype=float),
    tool_pos: wp.array(dtype=wp.vec3),
    tool_rot: wp.array(dtype=wp.quat),
    tool_vel: wp.array(dtype=wp.vec3),
    tool_omega: wp.array(dtype=wp.vec3),
    mobility: wp.array2d(dtype=float),
    linear: wp.array(dtype=wp.vec3),
    angular: wp.array(dtype=wp.vec3),
):
    """packed = [position 3, quaternion xyzw 4, velocity 3, angular velocity 3, mobility 36]."""
    tool_pos[0] = wp.vec3(packed[0], packed[1], packed[2])
    tool_rot[0] = wp.quat(packed[3], packed[4], packed[5], packed[6])
    tool_vel[0] = wp.vec3(packed[7], packed[8], packed[9])
    tool_omega[0] = wp.vec3(packed[10], packed[11], packed[12])
    for i in range(6):
        for j in range(6):
            mobility[i, j] = packed[13 + 6 * i + j]
    linear[0] = wp.vec3(0.0)
    angular[0] = wp.vec3(0.0)


@wp.kernel
def record_pose(
    position: wp.array(dtype=wp.vec3),
    rotation: wp.array(dtype=wp.quat),
    index: int,
    out: wp.array(dtype=float),
):
    offset = 9 + 7 * index
    for i in range(3):
        out[offset + i] = position[0][i]
    for i in range(4):
        out[offset + 3 + i] = rotation[0][i]


@wp.kernel
def store_interval(
    linear: wp.array(dtype=wp.vec3),
    angular: wp.array(dtype=wp.vec3),
    errors: wp.array(dtype=int),
    out: wp.array(dtype=float),
):
    """out = [impulse 3, angular impulse about the world origin 3, error flags 3]."""
    for i in range(3):
        out[i] = linear[0][i]
        out[3 + i] = angular[0][i]
        out[6 + i] = float(errors[i])


@wp.kernel
def finish_external_step(
    reaction: wp.array(dtype=wp.vec3),
    torque: wp.array(dtype=wp.vec3),
    board: wp.array(dtype=wp.vec3),
    speed: wp.array(dtype=float),
    tool_pos: wp.array(dtype=wp.vec3),
    tool_vel: wp.array(dtype=wp.vec3),
    tool_rot: wp.array(dtype=wp.quat),
    tool_omega: wp.array(dtype=wp.vec3),
    mobility: wp.array2d(dtype=float),
    predict: int,
    dt: float,
    linear: wp.array(dtype=wp.vec3),
    angular: wp.array(dtype=wp.vec3),
):
    """Accumulate the step impulse, optionally advance the tool, reset sums.

    predict=0 holds pose and twist over the coupling interval (paper III-G).
    predict=1 applies the step impulse through the external body's mobility
    and integrates the pose at the resulting twist; the external engine
    resets the state at the start of the next interval.
    """
    j = reaction[0]
    tau = torque[0]
    linear[0] += j
    angular[0] += tau + wp.cross(tool_pos[0], j)
    if predict != 0:
        u = wp.spatial_vector(j[0], j[1], j[2], tau[0], tau[1], tau[2])
        dv = wp.vec3(0.0)
        dw = wp.vec3(0.0)
        for row in range(3):
            for col in range(6):
                dv[row] += mobility[row, col] * u[col]
                dw[row] += mobility[row + 3, col] * u[col]
        tool_vel[0] = tool_vel[0] + dv
        w = tool_omega[0] + dw
        tool_omega[0] = w
        tool_pos[0] = tool_pos[0] + dt * tool_vel[0]
        rate = wp.length(w)
        if rate > 1.0e-12:
            tool_rot[0] = wp.normalize(wp.quat_from_axis_angle(w / rate, dt * rate) * tool_rot[0])
    reaction[0] = wp.vec3(0.0)
    torque[0] = wp.vec3(0.0)
    board[0] = wp.vec3(0.0)
    speed[0] = 0.0


@wp.func
def vertex(i: int, half: wp.vec3, bevel: float):
    j = i % 5
    x = float(0.0)
    z = -half[2]
    if j == 1:
        x = half[0]
        z += bevel
    elif j == 2:
        x = half[0]
        z = half[2]
    elif j == 3:
        x = -half[0]
        z = half[2]
    elif j == 4:
        x = -half[0]
        z += bevel
    y = -half[1]
    if i >= 5:
        y = half[1]
    return wp.vec3(x, y, z)


@wp.func
def inv_inertia(q: wp.quat, ii: wp.vec3, torque: wp.vec3):
    return wp.quat_rotate(q, wp.cw_mul(ii, wp.quat_rotate_inv(q, torque)))


@wp.kernel
def step(
    tool_pos: wp.array(dtype=wp.vec3),
    tool_vel: wp.array(dtype=wp.vec3),
    tool_rot: wp.array(dtype=wp.quat),
    tool_omega: wp.array(dtype=wp.vec3),
    targets: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.quat),
    index: int,
    reaction: wp.array(dtype=wp.vec3),
    torque: wp.array(dtype=wp.vec3),
    table_impulse: wp.array(dtype=wp.vec3),
    dt: float,
    mass: float,
    half: wp.vec3,
    bevel: float,
    stiffness: float,
    damping: float,
    force_limit: float,
    rotational_stiffness: float,
    rotational_damping: float,
    inertia_inverse: wp.vec3,
    friction: float,
    free: int,
    lock_rotation: int,
    board_half: wp.vec2,
):
    """Drive (pose servo, guided translation or none) plus knife/board contact.

    Every internally integrated knife uses this one board contact: the ten blade
    vertices against the board top z = 0 inside its rectangle (zero half extent:
    unbounded), the same rectangle the particles and the MuJoCo board use.
    lock_rotation keeps the guided knife translation-only.
    """
    x = tool_pos[0] + wp.vec3(0.0)
    q = wp.normalize(tool_rot[0])
    v = tool_vel[0] + wp.vec3(0.0)
    w = tool_omega[0] + wp.vec3(0.0)
    force = wp.vec3(0.0, 0.0, -9.81 * mass)
    servo_torque = wp.vec3(0.0)
    if free == 0:
        # Gravity-compensated pose servo, matching the legacy translational
        # drive. Target denotes the blade-edge centre, not body centre.
        target_q = rotations[index]
        target_center = targets[index] + wp.quat_rotate(target_q, wp.vec3(0.0, 0.0, half[2]))
        force = stiffness * (target_center - x) - damping * v
        norm = wp.length(force)
        if norm > force_limit:
            force *= force_limit / norm
        dq = wp.normalize(target_q * wp.quat_inverse(q))
        if dq[3] < 0.0:
            dq = -dq
        vector = wp.vec3(dq[0], dq[1], dq[2])
        s = wp.length(vector)
        error = 2.0 * vector
        if s > 1.0e-8:
            error = vector * (2.0 * wp.atan2(s, dq[3]) / s)
        servo_torque = rotational_stiffness * error - rotational_damping * w
    v += (force * dt + reaction[0]) / mass
    ii = inertia_inverse
    if lock_rotation != 0:
        ii = wp.vec3(0.0)
        w = wp.vec3(0.0)
    else:
        local_w = wp.quat_rotate_inv(q, w)
        inertia = wp.vec3(1.0 / ii[0], 1.0 / ii[1], 1.0 / ii[2])
        gyroscopic = wp.quat_rotate(q, wp.cross(local_w, wp.cw_mul(inertia, local_w)))
        w += inv_inertia(q, ii, torque[0] + dt * (servo_torque - gyroscopic))
    normal = wp.vec3(0.0, 0.0, 1.0)
    total = wp.vec3(0.0)
    # Sequential velocity-constraint solve. Six iterations are an exposed
    # numerical choice, not a workspace boundary or a command overwrite.
    for sweep in range(6):
        for i in range(10):
            r = wp.quat_rotate(q, vertex(i, half, bevel))
            corner = x + r
            if (board_half[0] > 0.0 and wp.abs(corner[0]) > board_half[0]) or (
                board_half[1] > 0.0 and wp.abs(corner[1]) > board_half[1]
            ):
                continue
            gap = corner[2]
            vn = wp.dot(v + wp.cross(w, r), normal)
            desired = -gap / dt
            if gap < 0.0:
                desired *= 0.2
            if vn < desired:
                rn = wp.cross(r, normal)
                effective = 1.0 / mass + wp.dot(rn, inv_inertia(q, ii, rn))
                jn = (desired - vn) / effective
                impulse = jn * normal
                v += impulse / mass
                w += inv_inertia(q, ii, wp.cross(r, impulse))
                vt = v + wp.cross(w, r)
                vt -= wp.dot(vt, normal) * normal
                speed = wp.length(vt)
                if speed > 1.0e-10:
                    direction = -vt / speed
                    rt = wp.cross(r, direction)
                    wt = 1.0 / mass + wp.dot(rt, inv_inertia(q, ii, rt))
                    jt = wp.min(speed / wt, friction * jn)
                    tangent = jt * direction
                    impulse += tangent
                    v += tangent / mass
                    w += inv_inertia(q, ii, wp.cross(r, tangent))
                total += impulse
    x += dt * v
    speed = wp.length(w)
    if speed > 1.0e-10:
        q = wp.normalize(wp.quat_from_axis_angle(w / speed, dt * speed) * q)
    tool_pos[0] = x
    tool_vel[0] = v
    tool_rot[0] = q
    tool_omega[0] = w
    table_impulse[0] = total
