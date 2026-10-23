"""Closed interfaces retain compatible motion and weak independent constraints."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_history import CohesiveHistory
from blademaster_warp.differentiable_cohesive import solve_force
from test_cohesive_orientation import fixture


@pytest.mark.parametrize("spacing", [0.006, 0.004])
@pytest.mark.parametrize("motion", ["translation", "rotation", "nonuniform"])
def test_compatible_partial_fields_match_free_material_and_velocity_adjoint(spacing, motion):
    m, append = fixture(spacing)
    retained = append(CohesiveHistory.empty(m.n, device=m.device), 0.009, 0.0085)
    codes = np.zeros(m.n, dtype=np.int64)
    codes[:3] = 1
    grid = m.topology(codes)
    velocity = np.tile([0.1, -0.2, 0.3], (m.n, 1))
    if motion == "rotation":
        velocity += np.cross([3.0, -5.0, 7.0], m.reference)
    elif motion == "nonuniform":
        velocity += np.random.default_rng(13).normal(scale=0.03, size=velocity.shape)
    frame = np.array([[1.01, 0.02, 0.0], [0.0, 0.99, 0.0], [0.0, 0.0, 1.0]])
    retained.geometry.frame.assign(np.tile(frame, (len(retained.history), 1, 1)))
    results, derivatives = [], []
    for coupled in [False, True]:
        state = m.state(x=m.reference @ frame.T, v=velocity, F=np.tile(frame, (m.n, 1, 1)))
        with wp.Tape() as tape:
            if coupled:
                result, updated = retained.advance(m, state, topology=grid, gravity=(0, 0, 0))
            else:
                result = m.step(state, topology=grid, gravity=(0, 0, 0))
        seed = m.array(np.random.default_rng(91).normal(size=(m.n, 3)), wp.vec3, False)
        tape.backward(grads={result.v: seed})
        results.append([getattr(result, name).numpy() for name in ["x", "v", "F", "Fp", "stress"]])
        derivatives.append(state.v.grad.numpy().copy())
    np.testing.assert_array_equal(updated.history.numpy(), 0.0)
    np.testing.assert_array_equal(updated.geometry.gap.numpy(), 0.0)
    for reference, coupled in zip(*results):
        np.testing.assert_allclose(coupled, reference, rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(derivatives[1], derivatives[0], rtol=1e-10, atol=1e-10)


def test_closed_solve_keeps_weak_independent_rows_and_their_adjoint():
    diagonal = np.array([1.0, 2.0**-20, 2.0**-35])
    motion = np.array([[1.0, -0.5, 0.3], [0.2, 0.3, -0.4], [-0.1, 0.2, 0.7]]) * 1e-7
    b = np.diag(diagonal)
    q = b @ motion
    area = np.array([1.0, 0.3, 0.7])

    def array(values, dtype):
        return wp.array(values, dtype=dtype, requires_grad=True)

    gap = array(q, wp.vec3d)
    mapping = array(b, wp.float64)
    with wp.Tape() as tape:
        response = solve_force(
            gap,
            array(np.tile(np.eye(3), (3, 1, 1)), wp.mat33d),
            array(np.tile([1.0, 0.0, 0.0], (3, 1)), wp.vec3d),
            array(np.zeros(3), wp.float64),
            array(np.tile([1e10, 1e10, 0.0], (3, 1)), wp.vec3d),
            mapping,
            array(np.zeros((3, 3, 3)), wp.vec3d),
            array(area, wp.float64),
            array([1.0], wp.float64),
            array([1.0], wp.float64),
        )
    np.testing.assert_allclose(response.displacement.numpy(), -motion, rtol=1e-13, atol=1e-20)
    seed = np.arange(9).reshape(3, 3) / 10 + 0.1
    tape.backward(grads={response.displacement: wp.array(seed, dtype=wp.vec3d)})
    np.testing.assert_allclose(gap.grad.numpy(), -seed / diagonal[:, None], rtol=1e-12)
    expected = np.einsum("ia,ja->ij", seed / diagonal[:, None], motion)
    np.testing.assert_allclose(mapping.grad.numpy(), expected, rtol=1e-12, atol=1e-20)
