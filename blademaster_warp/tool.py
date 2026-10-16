"""Six-DOF rigid tool servo and unilateral convex-tool/plane contact.

This is a documented local control/interface, not an author-supplied robot.
Material reactions remain equal-and-opposite impulses. The table uses normal
velocity constraints and Coulomb friction; there is no target-height clipping.
"""
import warp as wp


@wp.kernel
def accumulate_reaction(reaction: wp.array(dtype=wp.vec3),
                        torque: wp.array(dtype=wp.vec3),
                        position: wp.array(dtype=wp.vec3),
                        linear: wp.array(dtype=wp.vec3),
                        angular: wp.array(dtype=wp.vec3)):
    """Sum impulses; angular impulse is about the fixed world origin."""
    linear[0] += reaction[0]
    angular[0] += torque[0] + wp.cross(position[0], reaction[0])


@wp.kernel
def finish_external_step(reaction: wp.array(dtype=wp.vec3),
                         torque: wp.array(dtype=wp.vec3),
                         board: wp.array(dtype=wp.vec3),
                         speed: wp.array(dtype=float),
                         bx: wp.array(dtype=wp.vec3), bv: wp.array(dtype=wp.vec3),
                         bq: wp.array(dtype=wp.quat), bw: wp.array(dtype=wp.vec3),
                         mobility: wp.array2d(dtype=float), predict: int, dt: float,
                         linear: wp.array(dtype=wp.vec3),
                         angular: wp.array(dtype=wp.vec3)):
    """Accumulate the step impulse, optionally advance the tool, reset sums.

    predict=0 holds pose and twist over the coupling interval (paper III-G).
    predict=1 applies the step impulse through the external body's mobility
    and integrates the pose at the resulting twist; the external engine
    resets the state at the start of the next interval.
    """
    j = reaction[0]
    tau = torque[0]
    linear[0] += j
    angular[0] += tau + wp.cross(bx[0], j)
    if predict != 0:
        u = wp.spatial_vector(j[0], j[1], j[2], tau[0], tau[1], tau[2])
        dv = wp.vec3(0.0)
        dw = wp.vec3(0.0)
        for row in range(3):
            for col in range(6):
                dv[row] += mobility[row, col]*u[col]
                dw[row] += mobility[row+3, col]*u[col]
        bv[0] = bv[0] + dv
        w = bw[0] + dw
        bw[0] = w
        bx[0] = bx[0] + dt*bv[0]
        rate = wp.length(w)
        if rate > 1.e-12:
            bq[0] = wp.normalize(wp.quat_from_axis_angle(w/rate, dt*rate)*bq[0])
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
def step(bx: wp.array(dtype=wp.vec3), bv: wp.array(dtype=wp.vec3),
         bq: wp.array(dtype=wp.quat), bw: wp.array(dtype=wp.vec3),
         targets: wp.array(dtype=wp.vec3), rotations: wp.array(dtype=wp.quat), index: int,
         reaction: wp.array(dtype=wp.vec3), torque: wp.array(dtype=wp.vec3),
         table_impulse: wp.array(dtype=wp.vec3), dt: float, mass: float, half: wp.vec3,
         bevel: float, stiffness: float, damping: float, force_limit: float,
         rotational_stiffness: float, rotational_damping: float, inertia_inverse: wp.vec3,
         friction: float, free: int):
    x = bx[0] + wp.vec3(0.0)
    q = wp.normalize(bq[0])
    v = bv[0] + wp.vec3(0.0)
    w = bw[0] + wp.vec3(0.0)
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
        if s > 1.e-8:
            error = vector * (2.0 * wp.atan2(s, dq[3]) / s)
        servo_torque = rotational_stiffness * error - rotational_damping * w
    v += (force * dt + reaction[0]) / mass
    local_w = wp.quat_rotate_inv(q, w)
    inertia = wp.vec3(1.0/inertia_inverse[0], 1.0/inertia_inverse[1], 1.0/inertia_inverse[2])
    gyroscopic = wp.quat_rotate(q, wp.cross(local_w, wp.cw_mul(inertia, local_w)))
    w += inv_inertia(q, inertia_inverse, torque[0] + dt * (servo_torque - gyroscopic))
    normal = wp.vec3(0.0, 0.0, 1.0)
    total = wp.vec3(0.0)
    # Sequential velocity-constraint solve. Six iterations are an exposed
    # numerical choice, not a workspace boundary or a command overwrite.
    for sweep in range(6):
        for i in range(10):
            r = wp.quat_rotate(q, vertex(i, half, bevel))
            gap = (x + r)[2]
            vn = wp.dot(v + wp.cross(w, r), normal)
            desired = -gap / dt
            if gap < 0.0:
                desired *= 0.2
            if vn < desired:
                rn = wp.cross(r, normal)
                effective = 1.0/mass + wp.dot(rn, inv_inertia(q, inertia_inverse, rn))
                jn = (desired - vn) / effective
                impulse = jn * normal
                v += impulse/mass
                w += inv_inertia(q, inertia_inverse, wp.cross(r, impulse))
                vt = v + wp.cross(w, r)
                vt -= wp.dot(vt, normal)*normal
                speed = wp.length(vt)
                if speed > 1.e-10:
                    direction = -vt/speed
                    rt = wp.cross(r, direction)
                    wt = 1.0/mass + wp.dot(rt, inv_inertia(q, inertia_inverse, rt))
                    jt = wp.min(speed/wt, friction*jn)
                    tangent = jt*direction
                    impulse += tangent
                    v += tangent/mass
                    w += inv_inertia(q, inertia_inverse, wp.cross(r, tangent))
                total += impulse
    x += dt*v
    speed = wp.length(w)
    if speed > 1.e-10:
        q = wp.normalize(wp.quat_from_axis_angle(w/speed, dt*speed)*q)
    bx[0] = x
    bv[0] = v
    bq[0] = q
    bw[0] = w
    table_impulse[0] = total
