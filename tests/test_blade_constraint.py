"""Broad-face constraints must survive grid transfer without restitution."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.solver import Config, Solver


@pytest.fixture
def solver():
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    instance = Solver(
        Config(
            size=(0.012, 0.012, 0.012),
            grid_spacing=0.006,
            dt=0.001,
            external_tool=True,
            grid_storage="compact",
        )
    )
    instance.tool_pos.zero_()
    instance.tool_vel.zero_()
    instance.tool_omega.zero_()
    instance.tool_mobility.zero_()
    return instance


def shared_grid(solver, codes):
    """Put all material stencils on eight nodes, retaining distinct cut fields."""
    solver.nodes.assign(np.tile(np.arange(8, dtype=np.int32), (solver.n, 1)))
    solver.weights.assign(np.full((solver.n, 8), 0.125, np.float32))
    solver.codes.assign(np.asarray(codes, np.int64))
    solver._register_topology()
    slots = solver.slots.numpy()
    weights = solver.weights.numpy().astype(np.float64)
    masses = np.zeros(solver.grid_capacity, np.float64)
    np.add.at(masses, slots.ravel(), (solver.mass * weights).ravel())
    solver.grid_mass.assign(masses.astype(np.float32))
    return slots, weights, masses


def advecting_velocity(solver, slots, weights):
    return np.sum(weights[:, :, None] * solver.new_v.numpy()[slots], axis=1)


def solve_with_velocity(solver, velocity):
    solver.new_v.assign(velocity.astype(np.float32))
    solver.reaction.zero_()
    solver.torque.zero_()
    solver.blade_constraint.solve()
    return solver.new_v.numpy().astype(np.float64)


def test_held_blade_stops_shared_cut_fields_without_virtual_motion_or_restitution(solver):
    # Held geometry does not move within this substep, even when the external
    # controller supplies a nonzero twist and articulated mobility.
    coupling = np.array([0.4, -0.2, 0.3, 1.0, -0.5, 0.7], np.float32)
    mobility = np.diag([2.0, 3.0, 4.0, 10.0, 12.0, 8.0]) + np.outer(coupling, coupling)
    solver.tool_mobility.assign(mobility.astype(np.float32))
    solver.tool_vel.assign([[-0.07, 0.02, 0.01]])
    solver.tool_omega.assign([[0.1, 0.2, -0.3]])
    held_state = {
        name: getattr(solver, name).numpy().copy()
        for name in ("tool_pos", "tool_rot", "tool_vel", "tool_omega", "tool_mobility")
    }
    # Codes 1 and 2 are separated; the uncut field 0 is compatible with both.
    codes = np.repeat([0, 1, 2], [16, 24, 24])
    assert len(codes) == solver.n
    slots, weights, masses = shared_grid(solver, codes)
    positions = np.full((solver.n, 3), 1.0, np.float32)
    positive, negative = np.array([16, 17]), np.array([40, 41])
    active = np.concatenate([positive, negative])
    sides = np.array([1.0, 1.0, -1.0, -1.0])
    positions[active] = np.column_stack(
        [sides * solver.config.blade_half_size[0], np.full(4, 0.003), np.full(4, 0.002)]
    )
    solver.x.assign(positions)

    # Build a realizable velocity from independently coupled free momenta.
    # The common field mixes with each cut field, never 1 directly with 2.
    field_mass = np.array([16, 24, 24], np.float64) * solver.mass / 8
    free_velocity = np.array([[0.005, 0.03, 0.02], [-0.12, 0.03, 0.02], [0.08, 0.03, 0.02]])
    free_momentum = field_mass[:, None] * free_velocity
    momentum = free_momentum.copy()
    for left, right in [(0, 1), (0, 2)]:
        exchange = (
            field_mass[left] * free_momentum[right] - field_mass[right] * free_momentum[left]
        ) / field_mass.sum()
        momentum[left] += exchange
        momentum[right] -= exchange
    before = np.zeros((solver.grid_capacity, 3), np.float64)
    for code in range(3):
        before[np.unique(slots[codes == code])] = momentum[code] / field_mass[code]
    incoming = np.sum(weights[active, :, None] * before[slots[active]], axis=1)
    assert np.all(sides * incoming[:, 0] < -0.03)

    after = solve_with_velocity(solver, before)
    final = advecting_velocity(solver, slots, weights)[active]
    # This is the velocity used to advect particles, independently read from the grid.
    assert np.min(sides * final[:, 0]) >= -2e-6
    assert np.max(np.abs(final[:, 0])) < 2e-6
    np.testing.assert_allclose(after[:, 1:], before[:, 1:], atol=2e-8)

    material_impulse = np.sum(masses[:, None] * (after - before), axis=0)
    reaction = solver.reaction.numpy()[0]
    assert abs(material_impulse[0]) > 1e-5
    np.testing.assert_allclose(material_impulse + reaction, 0, atol=2e-9)
    np.testing.assert_allclose(
        solver.torque.numpy()[0], -np.cross([0.0, 0.003, 0.002], material_impulse), atol=2e-10
    )
    energy_before = 0.5 * np.sum(masses[:, None] * before**2)
    energy_after = 0.5 * np.sum(masses[:, None] * after**2)
    assert energy_after <= energy_before + 1e-12
    np.testing.assert_array_equal(solver.x.numpy(), positions)
    for name, expected in held_state.items():
        np.testing.assert_array_equal(getattr(solver, name).numpy(), expected, err_msg=name)
    assert not solver.errors.numpy().any()


def test_embedded_center_is_placed_on_the_entry_face_without_added_speed(solver):
    half = solver.config.blade_half_size[0]
    positions = np.full((solver.n, 3), 1.0, np.float32)
    velocity = np.zeros((solver.n, 3), np.float32)
    side = np.zeros(solver.n, np.float32)
    # Inside the entry face, past the midplane, and already outside.
    positions[:3] = [[0.0, 0.001, 0.002], [-0.0002, 0.001, 0.002], [half + 0.001, 0.001, 0.002]]
    velocity[:3] = [[-0.4, 0.1, 0.2], [-0.3, -0.05, 0.04], [-0.2, 0.0, 0.0]]
    side[:2] = 1.0
    solver.x.assign(positions)
    solver.v.assign(velocity)
    solver.blade_constraint.side.assign(side)

    solver.blade_constraint.separate()
    corrected = solver.x.numpy()
    speed = solver.v.numpy()
    assert corrected[0, 0] == pytest.approx(half, abs=1e-6)
    assert corrected[1, 0] == pytest.approx(half, abs=1e-6)
    np.testing.assert_allclose(corrected[2], positions[2], atol=1e-7)
    np.testing.assert_allclose(corrected[:3, 1:], positions[:3, 1:], atol=1e-7)
    # Inward speed is removed. No outward speed is introduced.
    np.testing.assert_allclose(speed[0], [0.0, 0.1, 0.2], atol=1e-6)
    np.testing.assert_allclose(speed[1], [0.0, -0.05, 0.04], atol=1e-6)
    np.testing.assert_allclose(speed[2], velocity[2], atol=1e-7)
    assert not solver.errors.numpy().any()


def test_entry_side_survives_midplane_crossing_and_resets_after_finite_edge_exit(solver):
    slots, weights, masses = shared_grid(solver, np.zeros(solver.n, np.int64))
    positions = np.full((solver.n, 3), 1.0, np.float32)
    before = np.zeros((solver.grid_capacity, 3), np.float64)
    occupied = masses > 0
    half_x, half_y, _ = solver.config.blade_half_size

    # Enter from +x, then start the next solve just past the midplane.
    positions[0] = [half_x, 0, 0]
    solver.x.assign(positions)
    before[occupied, 0] = -0.1
    solve_with_velocity(solver, before)
    positions[0, 0] = -0.0001
    solver.x.assign(positions)
    after = solve_with_velocity(solver, before)
    assert advecting_velocity(solver, slots, weights)[0, 0] >= -2e-6
    assert solver.reaction.numpy()[0, 0] < 0

    # Crossing around the finite y edge is free; the old face must be forgotten.
    positions[0] = [-half_x, half_y + 0.001, 0]
    solver.x.assign(positions)
    before[occupied, 0] = 0.1
    outside = solve_with_velocity(solver, before)
    np.testing.assert_allclose(outside, before, atol=2e-8)
    np.testing.assert_array_equal(solver.reaction.numpy(), 0)

    # Re-enter on -x and cross the midplane in the opposite direction.
    positions[0] = [-half_x, 0, 0]
    solver.x.assign(positions)
    solve_with_velocity(solver, before)
    positions[0, 0] = 0.0001
    solver.x.assign(positions)
    after = solve_with_velocity(solver, before)
    assert advecting_velocity(solver, slots, weights)[0, 0] <= 2e-6
    assert solver.reaction.numpy()[0, 0] > 0
    material_impulse = np.sum(masses[:, None] * (after - before), axis=0)
    np.testing.assert_allclose(material_impulse + solver.reaction.numpy()[0], 0, atol=2e-9)
    assert not solver.errors.numpy().any()
