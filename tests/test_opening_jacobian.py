import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_cohesive import _evaluate, _opening_jacobian
from blademaster_warp._cohesive_solve import setup
from test_differentiable_cohesive import arrays, fixture


@pytest.mark.parametrize("scenario", ["closed", "opening", "mixed", "opened"])
@pytest.mark.parametrize("integrated", [False, True])
@pytest.mark.parametrize("constrain_normal", [False, True])
@pytest.mark.parametrize("compensated", [False, True])
def test_colored_opening_rows_match_individual_reverse_seeds(
    scenario, integrated, constrain_normal, compensated
):
    values = fixture(scenario)
    inputs = arrays(values)
    rng = np.random.default_rng(241)
    u = wp.array(rng.normal(scale=1e-6, size=(8, 3)), dtype=wp.vec3d, requires_grad=True)
    pressure = np.array([[-1e4, 2e4, -3e4], [6e4, 1e4, 0.], [0., -1e4, 0.]])
    traction = wp.array(
        pressure * values[7][:, None] if integrated else pressure,
        dtype=wp.vec3d, requires_grad=True,
    )
    compliance = setup(values, integrated_force=integrated)[2]
    remainder = None
    maps = None
    if compensated:
        remainder = tuple(rng.normal(size=shape) * 2.0**-60 for shape in [(3, 3), (3, 3, 3), (8, 3)])
        maps = tuple(rng.normal(size=values[index].shape) * 2.0**-60 for index in [5, 6])
    with wp.Tape() as tape:
        _, _, opening = _evaluate(
            inputs, u, traction, compliance, integrated_force=integrated,
            remainders=remainder, map_remainders=maps, constrain_normal=constrain_normal,
        )
    basis = np.linalg.qr(rng.normal(size=(24, 5)))[0]
    expected = []
    for row in range(9):
        tape.zero()
        seed = np.zeros((3, 3))
        seed.ravel()[row] = 1.0
        tape.backward(grads={opening: wp.array(seed, dtype=wp.vec3d)})
        expected.append(np.r_[basis.T @ u.grad.numpy().ravel(), traction.grad.numpy().ravel() / compliance])
    actual = _opening_jacobian(tape, inputs, traction, opening, basis, compliance, maps)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-15)
