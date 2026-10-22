"""Incremental energy, yield consistency and isochoric plastic return."""

import numpy as np
import pytest
from scipy.optimize import minimize, root
import warp as wp

from blademaster_warp.plasticity import corotated_projection, corotated_return


@wp.kernel
def project(trial: wp.array(dtype=wp.vec3d), radius: wp.array(dtype=wp.float64),
            result: wp.array(dtype=wp.vec3d), converged: wp.array(dtype=int)):
    i = wp.tid()
    value, success = corotated_return(trial[i], radius[i])
    result[i] = value
    converged[i] = success


def gpu_return(stretches, radius):
    values = np.asarray(stretches, dtype=float).reshape(-1, 3)
    radii = np.broadcast_to(radius, (len(values),)).copy()
    result = wp.empty(len(values), dtype=wp.vec3d, device='cuda:0')
    converged = wp.empty(len(values), dtype=int, device='cuda:0')
    wp.launch(project, len(values),
              inputs=[wp.array(values, dtype=wp.vec3d, device='cuda:0'),
                      wp.array(radii, dtype=wp.float64, device='cuda:0')],
              outputs=[result, converged], device='cuda:0')
    return result.numpy(), converged.numpy()


def reference_return(stretches, radius):
    basis = np.array([[1, 1], [-1, 1], [0, -2]]) / np.array([np.sqrt(2), np.sqrt(6)])
    log_trial = np.log(stretches)
    trial_stress = stretches * (stretches - 1)
    trial_gradient = basis.T @ trial_stress
    if np.linalg.norm(trial_gradient) <= radius:
        return stretches.copy()

    def value(z):
        stretch = np.exp(log_trial + basis @ z)
        return .5 * np.sum((stretch - 1) ** 2) + radius * np.linalg.norm(z)

    def gradient(z):
        stretch = np.exp(log_trial + basis @ z)
        return basis.T @ (stretch * (stretch - 1)) + radius * z / np.linalg.norm(z)

    start = -.001 * trial_gradient / np.linalg.norm(trial_gradient)
    result = minimize(value, start, jac=gradient, method='BFGS', options=dict(gtol=1e-11, maxiter=1000))
    assert np.linalg.norm(gradient(result.x)) < 1e-8
    stationary = root(gradient, result.x, tol=1e-12)
    assert np.linalg.norm(gradient(stationary.x)) < 2e-13
    return np.exp(log_trial + basis @ stationary.x)


@pytest.mark.parametrize('stretches', [[1.1, .95, .95], [.2, .1, .05], [.55, .4, .3],
                                     [1.3, 1.3, .8], [1.30001, 1.3, .8]])
def test_return_matches_independent_minimization(stretches):
    trial = np.array(stretches)
    radius = 200 / (2 * (20000 / (2 * 1.35))) * np.sqrt(2 / 3)
    actual, converged = gpu_return(trial, radius)
    assert converged[0] == 1
    expected = reference_return(trial, radius)
    np.testing.assert_allclose(actual[0], expected, rtol=1e-7, atol=1e-9)
    assert_consistency(trial[None], actual, radius)


def assert_consistency(trial, actual, radius):
    np.testing.assert_allclose(np.prod(actual, axis=1), np.prod(trial, axis=1), rtol=2e-13, atol=1e-15)
    delta = np.log(actual / trial)
    distance = np.linalg.norm(delta, axis=1)
    energy_drop = .5 * np.sum((trial - 1) ** 2 - (actual - 1) ** 2, axis=1)
    assert np.min(energy_drop - radius * distance) >= -2e-13
    stress = actual * (actual - 1)
    dev = stress - stress.mean(axis=1)[:, None]
    active = distance > 1e-12
    np.testing.assert_allclose(np.linalg.norm(dev[active], axis=1), radius, rtol=3e-9, atol=2e-12)
    # Multiplication by the plastic distance avoids amplifying reconstruction
    # roundoff in log(actual/trial) as the increment approaches zero.
    residual = distance[active, None] * dev[active] + radius * delta[active]
    tolerance = 3e-12 * distance[active, None] + 8 * np.finfo(float).eps * radius
    assert np.all(np.abs(residual) <= tolerance), residual


@pytest.mark.parametrize('mean_log', [-2., -.7, 0., .7])
def test_random_loading_energy_and_yield(mean_log):
    rng = np.random.default_rng(42)
    dev = rng.normal(scale=.25, size=(100, 3))
    dev -= dev.mean(axis=1)[:, None]
    trial = np.exp(mean_log + dev)
    radius = .03
    actual, converged = gpu_return(trial, radius)
    assert np.all(converged), np.flatnonzero(converged == 0)
    assert_consistency(trial, actual, radius)


@pytest.mark.parametrize('stretch', [.1, .5, 1., 2.])
def test_hydrostatic_states_remain_elastic(stretch):
    actual, converged = gpu_return([[stretch] * 3], .03)
    np.testing.assert_array_equal(actual[0], [stretch] * 3)
    assert converged[0] == 1


@pytest.mark.parametrize('stretch,radius', [([-1, 1, 1], .03), ([0, 1, 1], .03),
                                          ([np.nan, 1, 1], .03), ([1, 1, 1], -.01)])
def test_invalid_inputs_report_failure(stretch, radius):
    _, converged = gpu_return(stretch, radius)
    assert converged[0] == 0


@pytest.mark.parametrize('stretches', [[1.1, .95, .95], [1.05, 1.01, .99], [1.3, 1.3, .8]])
def test_yield_boundary_uses_elastic_and_plastic_sides(stretches):
    trial = np.array([stretches])
    stress = trial * (trial - 1)
    radius = float(np.linalg.norm(stress - stress.mean()))
    at_boundary, success = gpu_return(trial, radius)
    assert success[0] == 1
    np.testing.assert_allclose(at_boundary, trial, atol=1e-14)
    for epsilon in (1e-6, 1e-7, 1e-8, 1e-10, 1e-12):
        elastic, valid_elastic = gpu_return(trial, radius + epsilon)
        plastic, valid_plastic = gpu_return(trial, radius - epsilon)
        assert valid_elastic[0] == valid_plastic[0] == 1
        np.testing.assert_array_equal(elastic, trial)
        assert 0 < np.linalg.norm(plastic - trial) < 2 * epsilon
        assert_consistency(trial, plastic, radius - epsilon)


@wp.kernel
def matrix_project(trial: wp.array(dtype=wp.mat33d), radius: wp.array(dtype=wp.float64),
                   result: wp.array(dtype=wp.mat33d), converged: wp.array(dtype=int)):
    i = wp.tid()
    value, success = corotated_projection(trial[i], radius[i])
    result[i] = value
    converged[i] = success


@pytest.mark.parametrize('stretches', [[1., 1., 1.], [1.1, .95, .95], [.2, .1, .05], [.55, .4, .3],
                                     [1.3, 1.3, .8], [1.300000001, 1.3, .8], [1.30015, 1.3, .8]])
def test_implicit_matrix_vjp_matches_natural_differences(stretches, monkeypatch):
    monkeypatch.setattr(wp.config, 'verify_autograd_array_access', True)
    rng = np.random.default_rng(48)
    left, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    right, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    trial = left @ np.diag(stretches) @ right.T
    radius = .03
    x = wp.array([trial], dtype=wp.mat33d, device='cuda:0', requires_grad=True)
    r = wp.array([radius], dtype=wp.float64, device='cuda:0', requires_grad=True)
    seed = rng.normal(size=(3, 3))
    direction = rng.normal(size=(3, 3))

    def run():
        result = wp.empty(1, dtype=wp.mat33d, device='cuda:0', requires_grad=True)
        converged = wp.empty(1, dtype=int, device='cuda:0')
        wp.launch(matrix_project, 1, inputs=[x, r], outputs=[result, converged], device='cuda:0')
        assert converged.numpy()[0] == 1
        return result

    with wp.Tape() as tape:
        value = run()
    tape.backward(grads={value: wp.array([seed], dtype=wp.mat33d, device='cuda:0')})
    ad = np.sum(x.grad.numpy()[0] * direction)
    ad_radius = r.grad.numpy()[0]
    for epsilon in (1e-6, 3e-7):
        losses = []
        for sign in (-1, 1):
            x.assign([trial + sign * epsilon * direction])
            losses.append(np.sum(run().numpy()[0] * seed))
        np.testing.assert_allclose((losses[1] - losses[0]) / (2 * epsilon), ad, rtol=2e-6, atol=2e-7)
        x.assign([trial])
        losses = []
        for sign in (-1, 1):
            r.assign([radius + sign * epsilon])
            losses.append(np.sum(run().numpy()[0] * seed))
        np.testing.assert_allclose((losses[1] - losses[0]) / (2 * epsilon), ad_radius, rtol=2e-6, atol=2e-7)
        r.assign([radius])
