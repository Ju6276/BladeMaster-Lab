"""Knife-side transport at interface birth and preservation across later cuts."""

import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation

from blademaster_warp.cohesive_history import CohesiveHistory
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.reference_surface import swept_edge
from blademaster_warp.solver import Config


def fixture(spacing=0.006, transformed=False):
    cfg = Config(
        size=(0.012,) * 3,
        grid_spacing=spacing,
        dt=2e-6,
        young_modulus=20000.0,
        yield_stress=0.0,
        external_tool=True,
    )
    m = MaterialTrajectory(cfg, precision="float64", ordered_transfers=True)
    world = Rotation.from_rotvec([0.2, -0.3, 0.4]) if transformed else Rotation.identity()
    F = np.array([[1.1, 0.2, -0.1], [0.0, 0.9, 0.07], [0.03, 0.0, 1.05]]) if transformed else np.eye(3)
    rotation = world.as_matrix()
    shift = np.array([0.03, -0.02, 0.01]) if transformed else np.zeros(3)
    reference = m.array(m.reference, wp.vec3, False)
    positions = m.array(m.reference @ F.T @ rotation.T + shift, wp.vec3, False)
    deformation = m.array(np.tile(rotation @ F, (m.n, 1, 1)), wp.mat33, False)
    half = m.array(np.full((m.n, 3), spacing / 4), wp.vec3, False)
    blade = m.array([[0.0005, 0.0037, 0.004]], wp.vec3, False)
    material = m.array(np.tile([5e4, 70.0, 0.0], (m.n, 1)), wp.vec3)

    def append(history, start, end, cut=0, yaw=0.0):
        tool_rotation = m.array(
            [(world * Rotation.from_rotvec([0.0, 0.0, yaw])).as_quat() * 2.3], wp.quat, False
        )
        centres = [rotation @ np.array([0.0007, 0.0004, z + 0.004]) + shift for z in [start, end]]
        candidate = swept_edge(
            positions,
            deformation,
            positions,
            deformation,
            half,
            m.array([centres[0]], wp.vec3, False),
            tool_rotation,
            m.array([centres[1]], wp.vec3, False),
            tool_rotation,
            blade,
            reference_coordinates=m.reference_coordinates,
        )
        return history.append(
            candidate, reference, deformation, material, cut_index=cut, tool_rotation=tool_rotation
        )

    return m, append


@pytest.mark.parametrize("spacing", [0.006, 0.004])
@pytest.mark.parametrize("transformed", [False, True])
def test_birth_side_transport_is_objective_under_rotation_and_affine_deformation(spacing, transformed):
    m, append = fixture(spacing, transformed)
    down = append(CohesiveHistory.empty(m.n, device=m.device), 0.016, 0.002)
    up = append(CohesiveHistory.empty(m.n, device=m.device), 0.002, 0.016)
    assert len(down.history) and len(up.history)
    np.testing.assert_array_equal(down.positive_sides, 1)
    np.testing.assert_array_equal(up.positive_sides, 2)


def test_withdrawal_reentry_and_new_knife_keep_old_birth_sides_and_damage():
    m, append = fixture()
    history = append(CohesiveHistory.empty(m.n, device=m.device), 0.015, 0.003)
    history = history.updated(history.geometry, m.array(np.full(len(history.history), 0.001), float))
    old_count = len(history.history)
    extended = append(history, 0.003, 0.009)
    assert len(extended.history) > old_count
    np.testing.assert_array_equal(extended.positive_sides[:old_count], 1)
    np.testing.assert_array_equal(extended.positive_sides[old_count:], 2)
    np.testing.assert_array_equal(extended.quadrature.area.numpy()[old_count:], 0.0)
    saved_sides, saved_damage = extended.positive_sides.copy(), extended.history.numpy().copy()
    crossed = append(extended, 0.015, 0.003, cut=1, yaw=np.pi / 2)
    np.testing.assert_array_equal(crossed.positive_sides[: len(saved_sides)], saved_sides)
    np.testing.assert_array_equal(crossed.history.numpy()[: len(saved_sides)], saved_damage)
    restored = CohesiveHistory.restore(crossed.snapshot(), device=m.device)
    np.testing.assert_array_equal(restored.positive_sides, crossed.positive_sides)


def test_stored_orientation_reproduces_explicit_material_trace_and_velocity_adjoint():
    m, append = fixture()
    history = append(CohesiveHistory.empty(m.n, device=m.device), 0.009, 0.0085)
    grid = m.topology(np.where(m.reference[:, 0] < 0.0007, 1, 2))
    explicit = np.tile([2, 1], (len(history.history), 1))
    impulse = m.array(
        np.where(m.reference[:, 0] < 0.0007, -1.0, 1.0)[:, None]
        * np.array([2e5, 0.0, 0.0])
        * m.mass
        * m.config.dt,
        wp.vec3,
    )
    results, gradients = [], []
    for saved in [False, True]:
        initial = m.state(v=np.tile([0.1, 0.2, 0.3], (m.n, 1)))
        current = CohesiveHistory.restore(history.snapshot(), device=m.device)
        with wp.Tape() as tape:
            state, current = current.advance(
                m,
                initial,
                topology=grid,
                sides=None if saved else explicit,
                impulse=impulse,
                gravity=(0, 0, 0),
            )
        seed = np.random.default_rng(75).normal(size=(m.n, 3))
        tape.backward(grads={state.v: m.array(seed, wp.vec3, False)})
        results.append(
            [getattr(state, name).numpy() for name in ["x", "v", "F", "Fp", "stress"]]
            + [current.history.numpy(), current.geometry.gap.numpy()]
        )
        gradients.append(initial.v.grad.numpy().copy())
    for explicit_state, automatic_state in zip(*results):
        np.testing.assert_array_equal(explicit_state, automatic_state)
    np.testing.assert_allclose(*gradients, rtol=2e-13, atol=1e-13)
