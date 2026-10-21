"""Prevent grid transport through the broad faces of an externally held blade.

The original penalty, friction and cutting-work laws run first. The velocity
solve uses W = N^T M^-1 C N with the solver's compatible-field exchange and does
not add restitution. A center that is already inside a broad face is then moved
to that face, and only an inward normal velocity is removed. The external body's
pose is fixed during these material substeps; its measured mobility and the
returned reaction remain in the original coupling.
"""

import warp as wp


@wp.func
def prepare_at(
    p: int,
    x: wp.array(dtype=wp.vec3),
    slots: wp.array2d(dtype=int),
    weights: wp.array2d(dtype=float),
    velocity: wp.array(dtype=wp.vec3),
    position: wp.array(dtype=wp.vec3),
    rotation: wp.array(dtype=wp.quat),
    half: wp.vec3,
    bevel: float,
    radius: float,
    dt: float,
    side: wp.array(dtype=float),
    normal: wp.array(dtype=wp.vec3),
    lever: wp.array(dtype=wp.vec3),
    rhs: wp.array(dtype=float),
    active_weights: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=float),
    extrapolated: wp.array(dtype=float),
    needs_solve: wp.array(dtype=int),
):
    local = wp.quat_rotate_inv(rotation[0], x[p] - position[0])
    normal[p] = wp.vec3(0.0)
    impulse[p] = 0.0
    extrapolated[p] = 0.0
    # Going around a finite edge releases the previous entry side. The bevel
    # remains governed by the original cutting-work/contact calculation.
    if wp.abs(local[1]) >= half[1] or local[2] <= -half[2] + bevel or local[2] >= half[2]:
        side[p] = 0.0
        return
    grid_v = wp.vec3(0.0)
    for a in range(8):
        slot = slots[p, a]
        if slot >= 0:
            grid_v += weights[p, a] * velocity[slot]
    s = side[p]
    if s == 0.0 or wp.abs(local[0]) > half[0] + radius:
        s = 1.0
        if local[0] < 0.0:
            s = -1.0
    n = wp.quat_rotate(rotation[0], wp.vec3(s, 0.0, 0.0))
    gap = s * local[0] - half[0]
    # Include fast arrivals from beyond the proxy radius using the velocity
    # after stress and all existing contacts, not the FLIP particle velocity.
    if gap > radius and gap + dt * wp.dot(grid_v, n) > 0.0:
        side[p] = 0.0
        return
    side[p] = s
    normal[p] = n
    lever[p] = wp.quat_rotate(rotation[0], wp.vec3(s * half[0], local[1], local[2]))
    rhs[p] = -wp.max(gap, 0.0) / dt - wp.dot(grid_v, n)
    for a in range(8):
        slot = slots[p, a]
        if slot >= 0:
            wp.atomic_add(active_weights, slot, wp.vec3(weights[p, a], 0.0, 0.0))
    if rhs[p] > 1.0e-7:
        wp.atomic_max(needs_solve, 0, 1)


@wp.kernel
def prepare(
    x: wp.array(dtype=wp.vec3), slots: wp.array2d(dtype=int), weights: wp.array2d(dtype=float),
    velocity: wp.array(dtype=wp.vec3), position: wp.array(dtype=wp.vec3), rotation: wp.array(dtype=wp.quat),
    half: wp.vec3, bevel: float, radius: float, dt: float,
    side: wp.array(dtype=float), normal: wp.array(dtype=wp.vec3), lever: wp.array(dtype=wp.vec3),
    rhs: wp.array(dtype=float), active_weights: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=float), extrapolated: wp.array(dtype=float), needs_solve: wp.array(dtype=int),
):
    prepare_at(wp.tid(), x, slots, weights, velocity, position, rotation, half, bevel, radius, dt,
               side, normal, lever, rhs, active_weights, impulse, extrapolated, needs_solve)


@wp.kernel
def row_bound(
    slots: wp.array2d(dtype=int),
    weights: wp.array2d(dtype=float),
    mass: wp.array(dtype=float),
    coupled_count: wp.array(dtype=wp.vec3),
    bound: wp.array(dtype=float),
):
    # The exchange operator has nonnegative entries. Ignoring normal dot
    # products therefore bounds the absolute row sum of the contact response,
    # and prevents simultaneous contacts from multiplying a diagonal correction.
    p = wp.tid()
    total = 0.0
    for a in range(8):
        s = slots[p, a]
        if s >= 0 and mass[s] > 0.0:
            total += weights[p, a] * coupled_count[s][0] / mass[s]
    bound[p] = total


@wp.kernel
def scatter(
    slots: wp.array2d(dtype=int),
    weights: wp.array2d(dtype=float),
    normal: wp.array(dtype=wp.vec3),
    lever: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=float),
    grid: wp.array(dtype=wp.vec3),
    linear: wp.array(dtype=wp.vec3),
    angular: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    j = impulse[p] * normal[p]
    for a in range(8):
        s = slots[p, a]
        if s >= 0:
            wp.atomic_add(grid, s, weights[p, a] * j)
    wp.atomic_add(linear, 0, j)
    wp.atomic_add(angular, 0, wp.cross(lever[p], j))


@wp.func
def response_at(
    p: int,
    slots: wp.array2d(dtype=int),
    weights: wp.array2d(dtype=float),
    mass: wp.array(dtype=float),
    coupled: wp.array(dtype=wp.vec3),
):
    response = wp.vec3(0.0)
    for a in range(8):
        s = slots[p, a]
        if s >= 0 and mass[s] > 0.0:
            response += weights[p, a] * coupled[s] / mass[s]
    return response


@wp.kernel
def project(
    slots: wp.array2d(dtype=int),
    weights: wp.array2d(dtype=float),
    mass: wp.array(dtype=float),
    coupled: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    rhs: wp.array(dtype=float),
    bound: wp.array(dtype=float),
    impulse: wp.array(dtype=float),
    extrapolated: wp.array(dtype=float),
    acceleration: float,
):
    p = wp.tid()
    if wp.length_sq(normal[p]) == 0.0:
        return
    response = response_at(p, slots, weights, mass, coupled)
    previous = impulse[p]
    current = wp.max(
        0.0, extrapolated[p] + (rhs[p] - wp.dot(normal[p], response)) / wp.max(bound[p], 1.0e-12)
    )
    impulse[p] = current
    extrapolated[p] = current + acceleration * (current - previous)


@wp.kernel
def record_residual(
    slots: wp.array2d(dtype=int),
    weights: wp.array2d(dtype=float),
    mass: wp.array(dtype=float),
    coupled: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    rhs: wp.array(dtype=float),
    maximum: wp.array(dtype=float),
):
    p = wp.tid()
    if wp.length_sq(normal[p]) > 0.0:
        response = response_at(p, slots, weights, mass, coupled)
        wp.atomic_max(maximum, 0, wp.max(0.0, rhs[p] - wp.dot(normal[p], response)))


@wp.kernel
def finish_refinement(
    residual: wp.array(dtype=float),
    batches: wp.array(dtype=int),
    refine: wp.array(dtype=int),
    maximum: wp.array(dtype=float),
):
    batches[0] += 1
    refine[0] = 0
    if residual[0] > 1.0e-6 and batches[0] < 8:
        refine[0] = 1
    else:
        wp.atomic_max(maximum, 0, residual[0])


@wp.kernel
def apply_grid(
    mass: wp.array(dtype=float),
    coupled: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
):
    s = wp.tid()
    if mass[s] > 0.0:
        velocity[s] += coupled[s] / mass[s]


@wp.kernel
def add_constraint_part(linear: wp.array(dtype=wp.vec3), parts: wp.array(dtype=wp.vec3)):
    parts[3] = parts[3] - linear[0]


@wp.kernel
def apply_reaction(
    linear: wp.array(dtype=wp.vec3),
    angular: wp.array(dtype=wp.vec3),
    reaction: wp.array(dtype=wp.vec3),
    torque: wp.array(dtype=wp.vec3),
):
    reaction[0] -= linear[0]
    torque[0] -= angular[0]


@wp.kernel
def store_residual(maximum: wp.array(dtype=float), result: wp.array(dtype=float)):
    result[result.shape[0] - 1] = maximum[0]


class BladeConstraint:
    """Projected gradient solve of W lambda >= rhs, lambda >= 0.

    All contacts share each iteration's grid response. The absolute-row-sum
    bound is a diagonal majorizer of W; acceleration uses k/(k+3). The maximum
    residual (m/s) controls refinement in batches of 48 iterations, up to 384.
    """

    def __init__(self, solver):
        self.solver = solver

        def zeros(n, dtype):
            return wp.zeros(n, dtype=solver._dtype(dtype), device=solver.device)

        for name in ("side", "rhs", "bound", "impulse", "extrapolated"):
            setattr(self, name, zeros(solver.n, float))
        self.normal = zeros(solver.n, wp.vec3)
        self.lever = zeros(solver.n, wp.vec3)
        self.grid = zeros(solver.grid_capacity, wp.vec3)
        self.coupled = zeros(solver.grid_capacity, wp.vec3)
        self.linear = zeros(1, wp.vec3)
        self.angular = zeros(1, wp.vec3)
        self.needs_solve = zeros(1, int)
        self.max_violation = zeros(1, float)
        self.residual = zeros(1, float)
        self.batches = zeros(1, int)
        self.refine = zeros(1, int)
        module = __name__
        if solver.precision == "float64":
            from .precision import double_kernel
            module = double_kernel(prepare).module.name
        wp.load_module(module=module, device=solver.device)

    def scatter(self, impulses):
        s = self.solver
        self.grid.zero_()
        self.linear.zero_()
        self.angular.zero_()
        s.launch(
            scatter,
            s.n,
            [s.slots, s.weights, self.normal, self.lever, impulses, self.grid, self.linear, self.angular],
        )
        s._couple(self.grid, self.coupled)

    def solve(self):
        s = self.solver
        cfg = s.config
        self.grid.zero_()
        self.coupled.zero_()
        self.needs_solve.zero_()
        s.launch(
            prepare,
            s.n,
            [
                s.x,
                s.slots,
                s.weights,
                s.new_v,
                s.tool_pos,
                s.tool_rot,
                s.half,
                cfg.blade_bevel_height,
                s.blade_offset,
                cfg.dt,
                self.side,
                self.normal,
                self.lever,
                self.rhs,
                self.grid,
                self.impulse,
                self.extrapolated,
                self.needs_solve,
            ],
        )
        wp.capture_if(self.needs_solve, self._iterations)

    def _iterations(self):
        s = self.solver
        s._couple(self.grid, self.coupled)
        s.launch(row_bound, s.n, [s.slots, s.weights, s.grid_mass, self.coupled, self.bound])
        self.batches.zero_()
        self.refine.fill_(1)
        wp.capture_while(self.refine, self._refine)
        s.launch(apply_grid, s.grid_capacity, [s.grid_mass, self.coupled, s.new_v])
        s.launch(apply_reaction, 1, [self.linear, self.angular, s.reaction, s.torque])
        s.launch(add_constraint_part, 1, [self.linear, s.contact_parts])

    def _refine(self):
        s = self.solver
        wp.copy(self.extrapolated, self.impulse)
        for iteration in range(48):
            self.scatter(self.extrapolated)
            s.launch(
                project,
                s.n,
                [
                    s.slots,
                    s.weights,
                    s.grid_mass,
                    self.coupled,
                    self.normal,
                    self.rhs,
                    self.bound,
                    self.impulse,
                    self.extrapolated,
                    float(iteration / (iteration + 3)),
                ],
            )
        self.scatter(self.impulse)
        self.residual.zero_()
        s.launch(
            record_residual,
            s.n,
            [s.slots, s.weights, s.grid_mass, self.coupled, self.normal, self.rhs, self.residual],
        )
        s.launch(finish_refinement, 1, [self.residual, self.batches, self.refine, self.max_violation])
