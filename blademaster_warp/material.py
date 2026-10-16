"""Topology-aware TLMPM transfers and constitutive update (paper Eqs. 1, 2, 7-9)."""

import warp as wp


@wp.kernel
def validate_interval(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    F: wp.array(dtype=wp.mat33),
    errors: wp.array(dtype=int),
):
    """Do not return a wrench from an interval ending in an invalid state."""
    p = wp.tid()
    determinant = wp.determinant(F[p])
    if determinant <= 0.0 or not wp.isfinite(determinant):
        wp.atomic_add(errors, 1, 1)
    for axis in range(3):
        if not wp.isfinite(x[p][axis]) or not wp.isfinite(v[p][axis]):
            wp.atomic_add(errors, 1, 1)


@wp.func
def compatible(a: wp.int64, b: wp.int64):
    zero, three = wp.int64(0), wp.int64(3)
    result = bool(True)
    while a != zero or b != zero:
        x = a % three
        y = b % three
        if x != zero and y != zero and x != y:
            result = False
        a = a // three
        b = b // three
    return result


@wp.func
def insert_key(keys: wp.array(dtype=wp.int64), key: wp.int64, capacity: int):
    slot = int((key * wp.int64(2654435761)) % wp.int64(capacity))
    result = int(-1)
    for probe in range(capacity):
        previous = wp.atomic_cas(keys, slot, wp.int64(-1), key)
        if previous == wp.int64(-1) or previous == key:
            result = slot
            break
        slot = (slot + 1) % capacity
    return result


@wp.kernel
def register(
    nodes: wp.array2d(dtype=int),
    codes: wp.array(dtype=wp.int64),
    keys: wp.array(dtype=wp.int64),
    slots: wp.array2d(dtype=int),
    node_count: int,
    capacity: int,
    errors: wp.array(dtype=int),
):
    p, a = wp.tid()
    key = codes[p] * wp.int64(node_count) + wp.int64(nodes[p, a])
    s = insert_key(keys, key, capacity)
    slots[p, a] = s
    if s < 0:
        wp.atomic_add(errors, 0, 1)


@wp.kernel
def node_lists(
    keys: wp.array(dtype=wp.int64), heads: wp.array(dtype=int), links: wp.array(dtype=int), node_count: int
):
    s = wp.tid()
    if keys[s] >= wp.int64(0):
        node = int(keys[s] % wp.int64(node_count))
        links[s] = wp.atomic_exch(heads, node, s)


@wp.func
def corotated_stress(
    U: wp.mat33, sigma: wp.vec3, V: wp.mat33, plastic_inverse: wp.mat33, mu: float, lam: float
):
    """First Piola stress of fixed corotated elasticity, from the SVD f = U diag(sigma) V^T
    of the elastic deformation f = F Fp^-1:

        P = (2 mu (f - R) + lam (J - 1) J f^-T) Fp^-T,  R = U V^T, J = det f.
    """
    J = sigma[0] * sigma[1] * sigma[2]
    diagonal = wp.vec3(0.0)
    for i in range(3):
        diagonal[i] = 2.0 * mu * (sigma[i] - 1.0) + lam * (J - 1.0) * J / sigma[i]
    return U * wp.diag(diagonal) * wp.transpose(V) * wp.transpose(plastic_inverse)


@wp.func
def stress_at(
    p: int,
    F: wp.array(dtype=wp.mat33),
    Fp: wp.array(dtype=wp.mat33),
    mu: float,
    lam: float,
    stress: wp.array(dtype=wp.mat33),
    errors: wp.array(dtype=int),
):
    plastic_inverse = wp.inverse(Fp[p])
    f = F[p] * plastic_inverse
    J = wp.determinant(f)
    if J <= 0.0 or not wp.isfinite(J):
        wp.atomic_add(errors, 1, 1)
        stress[p] = wp.mat33(0.0)
        return
    U, sigma, V = wp.svd3(f)
    stress[p] = corotated_stress(U, sigma, V, plastic_inverse, mu, lam)


@wp.kernel
def compute_stress(
    F: wp.array(dtype=wp.mat33),
    Fp: wp.array(dtype=wp.mat33),
    mu: float,
    lam: float,
    stress: wp.array(dtype=wp.mat33),
    errors: wp.array(dtype=int),
):
    stress_at(wp.tid(), F, Fp, mu, lam, stress, errors)


@wp.kernel
def p2g(
    v: wp.array(dtype=wp.vec3),
    stress: wp.array(dtype=wp.mat33),
    impulses: wp.array(dtype=wp.vec3),
    slots: wp.array2d(dtype=int),
    weights: wp.array2d(dtype=float),
    gradients: wp.array2d(dtype=wp.vec3),
    mass: float,
    volume: float,
    dt: float,
    grid_mass: wp.array(dtype=float),
    momentum: wp.array(dtype=wp.vec3),
    grid_impulse: wp.array(dtype=wp.vec3),
):
    """Scatter mass, momentum and stress plus contact impulse to the grid (stress from the previous update).

    One thread per (particle, stencil node).
    """
    p, a = wp.tid()
    s = slots[p, a]
    if s >= 0:
        w = weights[p, a]
        wp.atomic_add(grid_mass, s, mass * w)
        wp.atomic_add(momentum, s, mass * w * v[p])
        wp.atomic_add(grid_impulse, s, w * impulses[p] - dt * volume * stress[p] * gradients[p, a])


@wp.func
def couple_at(
    a: int,
    keys: wp.array(dtype=wp.int64),
    heads: wp.array(dtype=int),
    links: wp.array(dtype=int),
    grid_mass: wp.array(dtype=float),
    state: wp.array(dtype=wp.vec3),
    result: wp.array(dtype=wp.vec3),
    node_count: int,
):
    if keys[a] < wp.int64(0) or grid_mass[a] <= 0.0:
        return
    node = int(keys[a] % wp.int64(node_count))
    code_a = keys[a] // wp.int64(node_count)
    b = heads[node]
    total = float(0.0)
    exchange = wp.vec3(0.0)
    while b >= 0:
        total += grid_mass[b]
        code_b = keys[b] // wp.int64(node_count)
        if b != a and compatible(code_a, code_b):
            exchange += grid_mass[a] * state[b] - grid_mass[b] * state[a]
        b = links[b]
    result[a] = state[a] + exchange / total


@wp.kernel
def couple(
    keys: wp.array(dtype=wp.int64),
    heads: wp.array(dtype=int),
    links: wp.array(dtype=int),
    grid_mass: wp.array(dtype=float),
    state: wp.array(dtype=wp.vec3),
    result: wp.array(dtype=wp.vec3),
    node_count: int,
):
    a = wp.tid()
    couple_at(a, keys, heads, links, grid_mass, state, result, node_count)


@wp.kernel
def couple_compact(
    count: wp.array(dtype=int),
    threads: int,
    keys: wp.array(dtype=wp.int64),
    heads: wp.array(dtype=int),
    links: wp.array(dtype=int),
    grid_mass: wp.array(dtype=float),
    state: wp.array(dtype=wp.vec3),
    result: wp.array(dtype=wp.vec3),
    node_count: int,
):
    for a in range(wp.tid(), count[0], threads):
        couple_at(a, keys, heads, links, grid_mass, state, result, node_count)


@wp.func
def grid_vel_at(
    s: int,
    grid_mass: wp.array(dtype=float),
    momentum: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=wp.vec3),
    old_v: wp.array(dtype=wp.vec3),
    new_v: wp.array(dtype=wp.vec3),
):
    if grid_mass[s] > 0.0:
        old_v[s] = momentum[s] / grid_mass[s]
        new_v[s] = (momentum[s] + impulse[s]) / grid_mass[s]


@wp.kernel
def grid_vel(
    grid_mass: wp.array(dtype=float),
    momentum: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=wp.vec3),
    old_v: wp.array(dtype=wp.vec3),
    new_v: wp.array(dtype=wp.vec3),
):
    s = wp.tid()
    grid_vel_at(s, grid_mass, momentum, impulse, old_v, new_v)


@wp.kernel
def grid_vel_compact(
    count: wp.array(dtype=int),
    threads: int,
    grid_mass: wp.array(dtype=float),
    momentum: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=wp.vec3),
    old_v: wp.array(dtype=wp.vec3),
    new_v: wp.array(dtype=wp.vec3),
):
    for s in range(wp.tid(), count[0], threads):
        grid_vel_at(s, grid_mass, momentum, impulse, old_v, new_v)


@wp.func
def couple_and_velocity_at(
    a: int,
    keys: wp.array(dtype=wp.int64),
    heads: wp.array(dtype=int),
    links: wp.array(dtype=int),
    grid_mass: wp.array(dtype=float),
    momentum: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=wp.vec3),
    old_v: wp.array(dtype=wp.vec3),
    new_v: wp.array(dtype=wp.vec3),
    node_count: int,
):
    """Exactly Eq. 8 for two vectors, sharing the linked-list traversal."""
    if keys[a] < wp.int64(0) or grid_mass[a] <= 0.0:
        return
    node = int(keys[a] % wp.int64(node_count))
    code_a = keys[a] // wp.int64(node_count)
    total = float(0.0)
    dm = wp.vec3(0.0)
    di = wp.vec3(0.0)
    b = heads[node]
    while b >= 0:
        total += grid_mass[b]
        code_b = keys[b] // wp.int64(node_count)
        if b != a and compatible(code_a, code_b):
            dm += grid_mass[a] * momentum[b] - grid_mass[b] * momentum[a]
            di += grid_mass[a] * impulse[b] - grid_mass[b] * impulse[a]
        b = links[b]
    p = momentum[a] + dm / total
    j = impulse[a] + di / total
    old_v[a] = p / grid_mass[a]
    new_v[a] = (p + j) / grid_mass[a]


@wp.kernel
def couple_and_velocity(
    keys: wp.array(dtype=wp.int64),
    heads: wp.array(dtype=int),
    links: wp.array(dtype=int),
    grid_mass: wp.array(dtype=float),
    momentum: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=wp.vec3),
    old_v: wp.array(dtype=wp.vec3),
    new_v: wp.array(dtype=wp.vec3),
    node_count: int,
):
    a = wp.tid()
    couple_and_velocity_at(a, keys, heads, links, grid_mass, momentum, impulse, old_v, new_v, node_count)


@wp.kernel
def couple_and_velocity_compact(
    count: wp.array(dtype=int),
    threads: int,
    keys: wp.array(dtype=wp.int64),
    heads: wp.array(dtype=int),
    links: wp.array(dtype=int),
    grid_mass: wp.array(dtype=float),
    momentum: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=wp.vec3),
    old_v: wp.array(dtype=wp.vec3),
    new_v: wp.array(dtype=wp.vec3),
    node_count: int,
):
    for a in range(wp.tid(), count[0], threads):
        couple_and_velocity_at(a, keys, heads, links, grid_mass, momentum, impulse, old_v, new_v, node_count)


@wp.kernel
def viscous_impulse(
    F: wp.array(dtype=wp.mat33),
    slots: wp.array2d(dtype=int),
    gradients: wp.array2d(dtype=wp.vec3),
    old_v: wp.array(dtype=wp.vec3),
    viscosity: float,
    volume: float,
    dt: float,
    impulse: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    rate = wp.mat33(0.0)
    for a in range(8):
        rate += wp.outer(old_v[slots[p, a]], gradients[p, a])
    inv = wp.inverse(F[p])
    L = rate * inv
    stress = viscosity * wp.determinant(F[p]) * (L + wp.transpose(L)) * wp.transpose(inv)
    for a in range(8):
        wp.atomic_add(impulse, slots[p, a], -dt * volume * stress * gradients[p, a])


@wp.func
def g2p_at(
    p: int,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    slots: wp.array2d(dtype=int),
    weights: wp.array2d(dtype=float),
    old_v: wp.array(dtype=wp.vec3),
    new_v: wp.array(dtype=wp.vec3),
    dt: float,
    flip: float,
):
    pic = wp.vec3(0.0)
    dv = wp.vec3(0.0)
    for a in range(8):
        s = slots[p, a]
        w = weights[p, a]
        pic += w * new_v[s]
        dv += w * (new_v[s] - old_v[s])
    v[p] = (1.0 - flip) * pic + flip * (v[p] + dv)
    # TLMPM (de Vaucorbeil et al., 2020), Eqs. 11--16:
    # Eq. 13 advects position with interpolated grid velocity, NOT the
    # blended particle momentum velocity. FLIP residual modes must not
    # independently advect particles while MUSL updates F on the grid.
    x[p] += dt * pic


@wp.kernel
def g2p(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    slots: wp.array2d(dtype=int),
    weights: wp.array2d(dtype=float),
    old_v: wp.array(dtype=wp.vec3),
    new_v: wp.array(dtype=wp.vec3),
    dt: float,
    flip: float,
):
    g2p_at(wp.tid(), x, v, slots, weights, old_v, new_v, dt, flip)


@wp.kernel
def g2p_remap(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    slots: wp.array2d(dtype=int),
    weights: wp.array2d(dtype=float),
    old_v: wp.array(dtype=wp.vec3),
    new_v: wp.array(dtype=wp.vec3),
    dt: float,
    flip: float,
    mass: float,
    out: wp.array(dtype=wp.vec3),
):
    """G2P followed by the MUSL re-scatter of the same particle."""
    p = wp.tid()
    g2p_at(p, x, v, slots, weights, old_v, new_v, dt, flip)
    for a in range(8):
        wp.atomic_add(out, slots[p, a], mass * weights[p, a] * v[p])


@wp.func
def plastic_at(
    p: int,
    F: wp.array(dtype=wp.mat33),
    Fp: wp.array(dtype=wp.mat33),
    mu: float,
    yield_stress: float,
    errors: wp.array(dtype=int),
):
    """Hencky-strain von Mises return map.

    Returns the SVD (U, sigma, V) of the elastic deformation after projection,
    and False if the material is inverted.
    """
    U, s, V = wp.svd3(F[p] * wp.inverse(Fp[p]))
    if wp.min(s) <= 0.0:
        wp.atomic_add(errors, 1, 1)
        return U, s, V, False
    e = wp.vec3(wp.log(s[0]), wp.log(s[1]), wp.log(s[2]))
    dev = e - wp.vec3((e[0] + e[1] + e[2]) / 3.0)
    norm = wp.length(dev)
    limit = yield_stress / (2.0 * mu) * wp.sqrt(2.0 / 3.0)
    if norm > limit:
        ep = e - (1.0 - limit / norm) * dev
        s = wp.vec3(wp.exp(ep[0]), wp.exp(ep[1]), wp.exp(ep[2]))
        next_f = U * wp.diag(s) * wp.transpose(V)
        Fp[p] = wp.inverse(next_f) * F[p]
    return U, s, V, True


@wp.kernel
def plastic_project(
    F: wp.array(dtype=wp.mat33),
    Fp: wp.array(dtype=wp.mat33),
    mu: float,
    yield_stress: float,
    errors: wp.array(dtype=int),
):
    U, s, V, valid = plastic_at(wp.tid(), F, Fp, mu, yield_stress, errors)


@wp.kernel
def update_F_plastic(
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
):
    """MUSL deformation update, plastic projection, and the stress for the next step."""
    p = wp.tid()
    rate = wp.mat33(0.0)
    for a in range(8):
        s = slots[p, a]
        if grid_mass[s] > 0.0:
            rate += wp.outer(momentum[s] / grid_mass[s], gradients[p, a])
    F[p] += dt * rate
    if yield_stress > 0.0:
        U, sigma, V, valid = plastic_at(p, F, Fp, mu, yield_stress, errors)
        if valid:
            stress[p] = corotated_stress(U, sigma, V, wp.inverse(Fp[p]), mu, lam)
        else:
            stress[p] = wp.mat33(0.0)
    else:
        stress_at(p, F, Fp, mu, lam, stress, errors)


@wp.kernel
def compact_keys(
    keys: wp.array(dtype=wp.int64),
    mapping: wp.array(dtype=int),
    packed: wp.array(dtype=wp.int64),
    count: wp.array(dtype=int),
):
    h = wp.tid()
    if keys[h] >= wp.int64(0):
        a = wp.atomic_add(count, 0, 1)
        mapping[h] = a
        packed[a] = keys[h]


@wp.kernel
def remap_slots(slots: wp.array2d(dtype=int), mapping: wp.array(dtype=int)):
    p, a = wp.tid()
    if slots[p, a] >= 0:
        slots[p, a] = mapping[slots[p, a]]


@wp.kernel
def compact_node_lists(
    count: wp.array(dtype=int),
    threads: int,
    keys: wp.array(dtype=wp.int64),
    heads: wp.array(dtype=int),
    links: wp.array(dtype=int),
    node_count: int,
):
    for a in range(wp.tid(), count[0], threads):
        node = int(keys[a] % wp.int64(node_count))
        links[a] = wp.atomic_exch(heads, node, a)


@wp.kernel
def clear_compact_grid(
    count: wp.array(dtype=int),
    threads: int,
    grid_mass: wp.array(dtype=float),
    momentum: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=wp.vec3),
    remapped: wp.array(dtype=wp.vec3),
):
    for a in range(wp.tid(), count[0], threads):
        grid_mass[a] = 0.0
        momentum[a] = wp.vec3(0.0)
        impulse[a] = wp.vec3(0.0)
        remapped[a] = wp.vec3(0.0)
