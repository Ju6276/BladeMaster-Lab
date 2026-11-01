"""Root convergence across residual units, nonlinearities and domain boundaries."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp._cohesive_solve import solve_primal
from test_cohesive_stages import setup

from blademaster_warp._compensated import Expansion as E
from blademaster_warp._compensated_linalg import solve
from blademaster_warp._cohesive_globalization import TrustRegion
from blademaster_warp._cohesive_expansion import norm, scale_power_two


@pytest.mark.parametrize("exponent", [-1000, -600, 0, 600, 1000])
def test_equilibrium_norm_preserves_small_and_large_states(exponent):
    actual = norm(scale_power_two(E([3.0, 4.0]), exponent))
    normalized = scale_power_two(actual, -exponent)
    np.testing.assert_array_equal(normalized.high, 5.0)
    np.testing.assert_array_equal(normalized.low, 0.0)


@pytest.mark.parametrize("exponent", [-900, -600, 0, 600, 900])
@pytest.mark.parametrize("method", ["newton", "cauchy", "dogleg", "weak_wolfe"])
def test_search_preserves_zero_root_across_state_scales(exponent, method):
    matrix = E([[2.0, -1.0], [1.0, 3.0]])

    def at(x):
        return matrix @ x, matrix, None

    state = scale_power_two(E([1.0, 2.0]), exponent)
    residual, jacobian, _ = at(state)
    direction = solve(jacobian, -residual)
    search = TrustRegion(at, jacobian)
    fraction = 0.99 if method == "dogleg" else 0.3
    if method in ("cauchy", "dogleg"):
        search.radius = fraction * float(norm(direction).high)
    entry = search.newton_search if method == "weak_wolfe" else search.step
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        result, trace = entry(state, direction, residual, jacobian)
    normalized = scale_power_two(result, -exponent)
    assert len(trace) == 1
    if method in ("cauchy", "dogleg"):
        original_norm = float(norm(matrix @ E([1.0, 2.0])).high)
        assert float(norm(matrix @ normalized).high) < original_norm
        step_length = float(norm(scale_power_two(result - state, -exponent)).high)
        np.testing.assert_allclose(step_length, fraction * np.sqrt(5.0), rtol=2e-14)
    else:
        np.testing.assert_allclose(normalized.high, 0.0, atol=1e-30)


@pytest.mark.parametrize("first", [2.0**-300, 1.0, 2.0**300])
@pytest.mark.parametrize("second", [2.0**-300, 1.0, 2.0**300])
def test_linear_equilibrium_accepts_exact_newton_step(first, second):
    matrix = E(np.array([first, second])[:, None] * np.array([[2.0, -1.0], [1.0, 3.0]]))
    target = E([0.25, -0.125])

    def at(x):
        return matrix @ (x - target), matrix, None

    initial = E([1.0, 2.0])
    residual, jacobian, _ = at(initial)
    direction = solve(jacobian, -residual)
    result, trace = TrustRegion(at, jacobian).step(initial, direction, residual, jacobian)
    np.testing.assert_allclose(result.high, target.high, rtol=0.0, atol=1e-30)
    assert len(trace) == 1


@pytest.mark.parametrize("function", ["cubic", "logarithm", "nonsmooth"])
def test_nonlinear_root_and_invalid_trials(function):
    def at(x):
        if function == "cubic":
            return x * x * x - 1.0, (3.0 * x * x).reshape(1, 1), None
        if function == "nonsmooth":
            coefficient = 10.0 if x.high[0] < 1.0 else 1.0
            return coefficient * (x - 1.0), E([[coefficient]]), None
        with np.errstate(invalid="ignore"):
            value = np.log(x.high[0])
        return E([value]), E([[1.0 / x.high[0]]]), None

    x = E([0.1 if function == "cubic" else 4.0])
    search = TrustRegion(at, at(x)[1])
    for _ in range(40):
        residual, jacobian, _ = at(x)
        if np.max(abs(residual.high)) <= 64 * np.finfo(float).eps ** 2:
            break
        x, _ = search.step(x, solve(jacobian, -residual), residual, jacobian)
    np.testing.assert_allclose(x.high, [1.0], rtol=0.0, atol=64 * np.finfo(float).eps ** 2)


@pytest.mark.parametrize("scales", [(1.0, 1.0), (2.0**-200, 2.0**200), (2.0**200, 2.0**-200)])
def test_nonlinear_root_with_different_residual_units(scales):
    def at(x):
        first = 10.0 * (x[1] - x[0] * x[0])
        second = 1.0 - x[0]
        value = E([first.high, second.high], [first.low, second.low])
        coefficient = -20.0 * x[0]
        jacobian = E([[coefficient.high, 10.0], [-1.0, 0.0]], [[coefficient.low, 0.0], [0.0, 0.0]])
        return E(scales) * value, E(scales)[:, None] * jacobian, None

    x = E([-1.2, 1.0])
    search = TrustRegion(at, at(x)[1])
    for _ in range(60):
        if np.max(abs((x - 1.0).high)) <= 64 * np.finfo(float).eps ** 2:
            break
        residual, jacobian, _ = at(x)
        x, _ = search.step(x, solve(jacobian, -residual), residual, jacobian)
    np.testing.assert_allclose(x.high, [1.0, 1.0], rtol=0.0, atol=64 * np.finfo(float).eps ** 2)


@pytest.mark.parametrize("exponent", [-600, 0, 600])
def test_small_trust_radius_does_not_stall_a_resolved_newton_direction(exponent):
    matrix = E([[1.0, 0.0], [0.0, 1e-12]])

    def at(x):
        return matrix @ x, matrix, None

    state = scale_power_two(E([0.0, 1.0]), exponent)
    search = TrustRegion(at, E(np.eye(2)))
    search.radius = np.ldexp(1e-12, exponent)
    for _ in range(4):
        residual, jacobian, _ = at(state)
        if np.max(abs(scale_power_two(state, -exponent).high)) < 1e-28:
            break
        state, _ = search.step(state, solve(jacobian, -residual), residual, jacobian)
    np.testing.assert_allclose(scale_power_two(state, -exponent).high, 0, atol=1e-28)


def near_closed_values(seed, maximum):
    m, state, geometry, history, data = setup("rigid", 2e-6)
    transfer = data["transfer"]
    B, G = data["separation_map"].high.numpy(), data["frame_map"].high.numpy()
    _, singular, right = np.linalg.svd(B, full_matrices=False)
    rank = np.count_nonzero(singular > singular[0] * max(B.shape) * np.finfo(float).eps)
    z = transfer.pack(state.v, state.affine).numpy()
    z += np.random.default_rng(seed).normal(scale=0.03, size=z.shape)
    z -= right[:rank].T @ (right[:rank] @ z)
    v, affine = transfer.unpack(m.array(z, wp.vec3))
    z = transfer.pack(v, affine).numpy()
    tau = (1 - 1 / np.sqrt(2)) * 2e-6
    P = transfer.grid_map.numpy()
    L = np.eye(len(z)) - P.T @ (transfer.grid_mass[:, None] / m.mass * P)
    eigen, vectors = np.linalg.eigh(L)
    S = (vectors / np.sqrt(1 + tau * 2000 * eigen)) @ vectors.T
    mapped = B @ S
    gradient = np.einsum("qic,ij->qjc", G, S)
    whitened = S @ z
    gap = tau * (mapped.astype(np.longdouble) @ whitened.astype(np.longdouble))
    frame = geometry.frame.numpy().astype(np.longdouble) + tau * np.einsum(
        "pa,qpb->qab", whitened.astype(np.longdouble), gradient.astype(np.longdouble)
    )
    return [
        gap.astype(float),
        frame.astype(float),
        data["reference_normal"].numpy(),
        np.full(len(history), maximum),
        data["material"].numpy(),
        mapped,
        gradient,
        data["area"].numpy(),
        np.array([m.mass]),
        np.array([tau]),
    ]


@pytest.mark.parametrize("seed", [218, 306])
@pytest.mark.parametrize("maximum", [1e-13, 1e-15])
def test_near_closed_equilibrium_is_independent_of_proximal_scale(seed, maximum):
    values = near_closed_values(seed, maximum)
    u, force, _, _ = solve_primal(values, return_traction=True, integrated_force=True)
    other_u, other_force, _, _ = solve_primal(values, 0.25, return_traction=True, integrated_force=True)
    np.testing.assert_allclose(other_u, u, rtol=2e-8, atol=1e-20)
    np.testing.assert_allclose(other_force, force, rtol=2e-8, atol=1e-12)
