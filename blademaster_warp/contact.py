"""BladeMaster Eqs. 3--7, 10--16 in Warp. Persistent history, no force rescaling."""
import warp as wp
from .material import compatible, insert_key


@wp.func
def friction(n: wp.vec3, relative: wp.vec3, jn: float, history: wp.vec3,
             me: float, dt: float, mus: float, mud: float, gk: float, gc: float):
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
    closest = wp.vec3(wp.clamp(local[0], -half[0], half[0]),
                      wp.clamp(local[1], -half[1], half[1]),
                      wp.clamp(local[2], -half[2], half[2]))
    delta = local - closest
    distance = wp.length(delta)
    normal = wp.vec3(0.0)
    if bevel == 0.0:
        if distance > 1.e-12:
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
        best = float(1.e30)
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
            if edge[0]*offset[1] - edge[1]*offset[0] < 0.0:
                inside = False
            t = wp.clamp(wp.dot(offset, edge)/wp.dot(edge, edge), 0.0, 1.0)
            candidate = a + t*edge
            dist = wp.length(q-candidate)
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
            if distance > 1.e-12:
                normal = delta / distance
    return closest, normal, distance


@wp.func
def rigid_at(p: int, x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
          bx: wp.array(dtype=wp.vec3), bv: wp.array(dtype=wp.vec3),
          bq: wp.array(dtype=wp.quat), bw: wp.array(dtype=wp.vec3),
          half: wp.vec3, bevel: float, radius: float, spacing: float,
          mass: float, blade_mass: float, inertia_inv: wp.vec3,
          mobility: wp.array2d(dtype=float), free_rotation: int,
          dt: float, sn: float, mus: float, mud: float, gk: float, gc: float,
          histories: wp.array2d(dtype=wp.vec3), damage: wp.array(dtype=float),
          threshold: float, cutting: int, impulse: wp.array(dtype=wp.vec3),
          reaction: wp.array(dtype=wp.vec3), torque: wp.array(dtype=wp.vec3),
          board: wp.array(dtype=wp.vec3), work: wp.array(dtype=float)):
    result = wp.vec3(0.0, 0.0, -9.81*mass*dt)
    q = bq[0]
    for body in range(2):
        n = wp.vec3(0.0, 0.0, 1.0)
        gap = x[p][2] - radius
        relative = v[p]
        me = mass
        point = wp.vec3(x[p][0], x[p][1], 0.0)
        if body == 1:
            local = wp.quat_rotate_inv(q, x[p]-bx[0])
            point_local, normal_local, distance = blade_query(local, half, bevel)
            point = bx[0] + wp.quat_rotate(q, point_local)
            n = wp.quat_rotate(q, normal_local)
            gap = distance - radius
            relative = v[p] - bv[0] - wp.cross(bw[0], point-bx[0])
            inverse_effective = 1.0/blade_mass
            if free_rotation < 0:
                # External articulated body: u.T J M^-1 J.T u (Eq. 15).
                # Zero mobility retains the prescribed-tool special case.
                rn = wp.cross(point-bx[0], n)
                u = wp.spatial_vector(n[0], n[1], n[2], rn[0], rn[1], rn[2])
                inverse_effective = 0.0
                for row in range(6):
                    for col in range(6):
                        inverse_effective += u[row]*mobility[row,col]*u[col]
            elif free_rotation > 0:
                rn = wp.quat_rotate_inv(q, wp.cross(point-bx[0], n))
                inverse_effective += wp.dot(rn, wp.cw_mul(inertia_inv, rn))
            me = 1.0/(1.0/mass + inverse_effective)
        predicted = gap + dt * wp.min(wp.dot(relative, n), 0.0)
        if predicted >= 0.0:
            histories[body, p] = wp.vec3(0.0)
        else:
            jn = sn*me/dt*(-predicted)
            jt, ht = friction(n, relative, jn, histories[body, p], me, dt, mus, mud, gk, gc)
            histories[body, p] = ht
            j = jn*n + jt
            if body == 1 and cutting != 0:
                edge = bx[0] + wp.quat_rotate(q, wp.vec3(0.0, 0.0, -half[2]))
                d = wp.quat_rotate(q, wp.vec3(0.0, 0.0, -1.0))
                edge_v = bv[0] + wp.cross(bw[0], edge-bx[0])
                front = wp.dot(x[p]-edge, d)
                advance = wp.dot(edge_v-v[p], d)
                local_edge = wp.quat_rotate_inv(q, x[p]-edge)
                if (front >= 0.0 and front <= 2.0*radius and advance > 0.0
                    and wp.abs(local_edge[0]) <= 6.0*spacing and wp.abs(local_edge[1]) <= half[1]):
                    positive = wp.max(wp.dot(j, d), 0.0)
                    if damage[p] >= 1.0:
                        j -= positive*d
                    else:
                        delta = positive*advance
                        wp.atomic_add(work, 0, delta)
                        damage[p] = wp.min(1.0, damage[p] + delta/threshold)
            result += j
            if body == 0:
                wp.atomic_add(board, 0, j)
            else:
                wp.atomic_add(reaction, 0, -j)
                wp.atomic_add(torque, 0, -wp.cross(point-bx[0], j))
    impulse[p] = result


@wp.kernel
def rigid(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
          bx: wp.array(dtype=wp.vec3), bv: wp.array(dtype=wp.vec3),
          bq: wp.array(dtype=wp.quat), bw: wp.array(dtype=wp.vec3),
          half: wp.vec3, bevel: float, radius: float, spacing: float,
          mass: float, blade_mass: float, inertia_inv: wp.vec3,
          mobility: wp.array2d(dtype=float), free_rotation: int,
          dt: float, sn: float, mus: float, mud: float, gk: float, gc: float,
          histories: wp.array2d(dtype=wp.vec3), damage: wp.array(dtype=float),
          threshold: float, cutting: int, impulse: wp.array(dtype=wp.vec3),
          reaction: wp.array(dtype=wp.vec3), torque: wp.array(dtype=wp.vec3),
          board: wp.array(dtype=wp.vec3), work: wp.array(dtype=float)):
    rigid_at(wp.tid(), x, v, bx, bv, bq, bw, half, bevel, radius, spacing, mass, blade_mass, inertia_inv, mobility, free_rotation, dt, sn, mus, mud, gk, gc, histories, damage, threshold, cutting, impulse, reaction, torque, board, work)


@wp.kernel
def rigid_and_speed(speed: wp.array(dtype=float), x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
          bx: wp.array(dtype=wp.vec3), bv: wp.array(dtype=wp.vec3),
          bq: wp.array(dtype=wp.quat), bw: wp.array(dtype=wp.vec3),
          half: wp.vec3, bevel: float, radius: float, spacing: float,
          mass: float, blade_mass: float, inertia_inv: wp.vec3,
          mobility: wp.array2d(dtype=float), free_rotation: int,
          dt: float, sn: float, mus: float, mud: float, gk: float, gc: float,
          histories: wp.array2d(dtype=wp.vec3), damage: wp.array(dtype=float),
          threshold: float, cutting: int, impulse: wp.array(dtype=wp.vec3),
          reaction: wp.array(dtype=wp.vec3), torque: wp.array(dtype=wp.vec3),
          board: wp.array(dtype=wp.vec3), work: wp.array(dtype=float)):
    p = wp.tid()
    wp.atomic_max(speed, 0, wp.length(v[p]))
    rigid_at(p, x, v, bx, bv, bq, bw, half, bevel, radius, spacing, mass, blade_mass, inertia_inv, mobility, free_rotation, dt, sn, mus, mud, gk, gc, histories, damage, threshold, cutting, impulse, reaction, torque, board, work)


@wp.func
def find_key(keys: wp.array(dtype=wp.int64), key: wp.int64, capacity: int):
    slot = int((key*wp.int64(2654435761)) % wp.int64(capacity))
    result = int(-1)
    for probe in range(capacity):
        if keys[slot] == wp.int64(-1):
            break
        if keys[slot] == key:
            result = slot
            break
        slot = (slot+1) % capacity
    return result


@wp.kernel
def max_speed(v: wp.array(dtype=wp.vec3), speed: wp.array(dtype=float)):
    wp.atomic_max(speed, 0, wp.length(v[wp.tid()]))


@wp.kernel
def pairs(grid: wp.uint64, x: wp.array(dtype=wp.vec3), ref: wp.array(dtype=wp.vec3),
          v: wp.array(dtype=wp.vec3), codes: wp.array(dtype=int), radius: float,
          spacing: float, mass: float, dt: float, sn: float, mus: float, mud: float,
          gk: float, gc: float, speed: wp.array(dtype=float),
          old_keys: wp.array(dtype=wp.int64), old_history: wp.array(dtype=wp.vec3),
          new_keys: wp.array(dtype=wp.int64), new_history: wp.array(dtype=wp.vec3),
          capacity: int, n: int, impulses: wp.array(dtype=wp.vec3), errors: wp.array(dtype=int)):
    p = wp.tid()
    query = wp.hash_grid_query(grid, x[p], 2.0*radius + 2.0*speed[0]*dt)
    for q in query:
        if q <= p:
            continue
        if compatible(codes[p], codes[q]) and wp.length(ref[p]-ref[q]) < 2.0*spacing:
            continue
        delta = x[p]-x[q]
        dist = wp.length(delta)
        if dist < 1.e-12:
            wp.atomic_add(errors, 2, 1)
            continue
        normal = delta/dist
        relative = v[p]-v[q]
        predicted = 2.0*radius-dist-dt*wp.min(wp.dot(relative, normal), 0.0)
        if predicted <= 0.0:
            continue
        key = wp.int64(p)*wp.int64(n) + wp.int64(q)
        old = find_key(old_keys, key, capacity)
        history = wp.vec3(0.0)
        if old >= 0:
            history = old_history[old]
        new = insert_key(new_keys, key, capacity)
        if new < 0:
            wp.atomic_add(errors, 0, 1)
            continue
        jn = sn*(mass/2.0)/dt*predicted
        jt, ht = friction(normal, relative, jn, history, mass/2.0, dt, mus, mud, gk, gc)
        new_history[new] = ht
        impulse = jn*normal+jt
        wp.atomic_add(impulses, p, impulse)
        wp.atomic_add(impulses, q, -impulse)


@wp.kernel
def assign(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
           codes: wp.array(dtype=int), bx: wp.array(dtype=wp.vec3),
           bv: wp.array(dtype=wp.vec3), bq: wp.array(dtype=wp.quat), bw: wp.array(dtype=wp.vec3),
           half: wp.vec3, spacing: float, dt: float, digit: int, cutting: int,
           topology_dirty: wp.array(dtype=int)):
    p = wp.tid()
    if cutting == 0 or (codes[p]//digit)%3 != 0:
        return
    edge = bx[0] + wp.quat_rotate(bq[0], wp.vec3(0.0, 0.0, -half[2]))
    local = wp.quat_rotate_inv(bq[0], x[p]-edge)
    front = -local[2]
    edge_v = bv[0] + wp.cross(bw[0], edge-bx[0])
    d = wp.quat_rotate(bq[0], wp.vec3(0.0,0.0,-1.0))
    advance = wp.dot(edge_v-v[p], d)
    back = 2.0*spacing + wp.max(advance*dt, 0.0)
    if wp.abs(local[0]) <= 6.0*spacing and wp.abs(local[1]) <= half[1] and front >= -back and front <= 0.0:
        side = int(2)
        if local[0] < 0.0:
            side = 1
        codes[p] += digit*side
        wp.atomic_max(topology_dirty,0,1)


@wp.func
def integrate_blade(bx: wp.array(dtype=wp.vec3), bv: wp.array(dtype=wp.vec3),
               bq: wp.array(dtype=wp.quat), bw: wp.array(dtype=wp.vec3),
               target: wp.vec3, reaction: wp.array(dtype=wp.vec3), torque: wp.array(dtype=wp.vec3),
               dt: float, mass: float, half: wp.vec3, stiffness: float,
               damping: float, force_limit: float, mode: int, inertia_inv: wp.vec3):
    # mode 0: DexLab guided translation. mode 1: free rigid blade under gravity.
    force = wp.vec3(0.0,0.0,-9.81*mass)
    if mode == 0:
        force = stiffness*(target + wp.vec3(0.0,0.0,half[2])-bx[0])-damping*bv[0]
        norm = wp.length(force)
        if norm > force_limit:
            force *= force_limit/norm
    bv[0] += (dt*force + reaction[0])/mass
    if mode == 1:
        omega = wp.quat_rotate_inv(bq[0], bw[0])
        inertia = wp.vec3(1.0/inertia_inv[0],1.0/inertia_inv[1],1.0/inertia_inv[2])
        angular_j = wp.quat_rotate_inv(bq[0], torque[0])-dt*wp.cross(omega,wp.cw_mul(inertia,omega))
        omega += wp.cw_mul(inertia_inv,angular_j)
        bw[0] = wp.quat_rotate(bq[0],omega)
        speed = wp.length(bw[0])
        if speed > 0.0:
            dq = wp.quat_from_axis_angle(bw[0]/speed,dt*speed)
            bq[0] = wp.normalize(dq*bq[0])
    bx[0] += dt*bv[0]
    if mode == 0 and bx[0][2] < half[2]:
        # Explicit DexLab guide/board unilateral constraint; not the free-blade method.
        pos = bx[0]
        vel = bv[0]
        pos[2] = half[2]
        vel[2] = wp.max(vel[2],0.0)
        bx[0] = pos
        bv[0] = vel


@wp.kernel
def blade_step(bx: wp.array(dtype=wp.vec3), bv: wp.array(dtype=wp.vec3),
               bq: wp.array(dtype=wp.quat), bw: wp.array(dtype=wp.vec3),
               target: wp.vec3, reaction: wp.array(dtype=wp.vec3), torque: wp.array(dtype=wp.vec3),
               dt: float, mass: float, half: wp.vec3, stiffness: float,
               damping: float, force_limit: float, mode: int, inertia_inv: wp.vec3):
    integrate_blade(bx, bv, bq, bw, target, reaction, torque, dt, mass, half,
                    stiffness, damping, force_limit, mode, inertia_inv)


@wp.kernel
def blade_step_buffer(bx: wp.array(dtype=wp.vec3), bv: wp.array(dtype=wp.vec3),
               bq: wp.array(dtype=wp.quat), bw: wp.array(dtype=wp.vec3),
               targets: wp.array(dtype=wp.vec3), index: int,
               reaction: wp.array(dtype=wp.vec3), torque: wp.array(dtype=wp.vec3),
               dt: float, mass: float, half: wp.vec3, stiffness: float,
               damping: float, force_limit: float, mode: int, inertia_inv: wp.vec3):
    integrate_blade(bx, bv, bq, bw, targets[index], reaction, torque, dt, mass, half,
                    stiffness, damping, force_limit, mode, inertia_inv)


@wp.kernel
def clear_step(reaction: wp.array(dtype=wp.vec3), torque: wp.array(dtype=wp.vec3),
               board: wp.array(dtype=wp.vec3), speed: wp.array(dtype=float)):
    reaction[0] = wp.vec3(0.0)
    torque[0] = wp.vec3(0.0)
    board[0] = wp.vec3(0.0)
    speed[0] = 0.0
