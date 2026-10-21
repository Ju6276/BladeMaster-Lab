"""Rotated plastic states checked against an independent NumPy constitutive law."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_material import MaterialTrajectory, constitutive_response
from blademaster_warp.solver import Config


@wp.kernel
def response(F: wp.array(dtype=wp.mat33), Fp: wp.array(dtype=wp.mat33),
             mu: wp.array(dtype=float), lam: wp.array(dtype=float), limit: wp.array(dtype=float),
             next_Fp: wp.array(dtype=wp.mat33), stress: wp.array(dtype=wp.mat33)):
    p = wp.tid()
    plastic, force = constitutive_response(F[p], Fp[p], mu[p], lam[p], limit[p], -1)
    next_Fp[p] = plastic
    stress[p] = force


def reference(F, Fp, mu, lam, limit):
    u, stretch, vt = np.linalg.svd(F @ np.linalg.inv(Fp))
    strain = np.log(stretch)
    deviation = strain - strain.mean()
    radius = np.linalg.norm(deviation)
    plastic = Fp
    if radius > limit:
        stretch = np.exp(strain - (1 - limit / radius) * deviation)
        elastic = (u * stretch) @ vt
        plastic = np.linalg.solve(elastic, F)
    else:
        elastic = F @ np.linalg.inv(Fp)
    J = np.linalg.det(elastic)
    stress = (2 * mu * (elastic - u @ vt)
              + lam * (J - 1) * J * np.linalg.inv(elastic).T) @ np.linalg.inv(plastic).T
    return plastic, stress


def samples():
    rng = np.random.default_rng(471)
    values = []
    for stretch in ([1., 1., 1.], [1.04, 1.04, 1.04], [1.3, 1.3, .8],
                    [1.30001, 1.3, .8], [1.30015, 1.3, .8], [1.2, .9, 1.02]):
        for _ in range(4):
            left, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            right, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            plastic = np.eye(3) + rng.normal(scale=.03, size=(3, 3))
            elastic = left @ np.diag(stretch) @ right.T
            values.append((elastic @ plastic, plastic, 2000., 3000., .08))
    return [np.asarray(x) for x in zip(*values)]


@pytest.mark.parametrize('parameter', range(5), ids=['F', 'Fp', 'mu', 'lambda', 'yield_radius'])
def test_constitutive_vjp_against_numpy(parameter, monkeypatch):
    monkeypatch.setattr(wp.config, 'verify_autograd_array_access', True)
    model = MaterialTrajectory(Config(size=(.012, .012, .012), grid_spacing=.006), precision='float64')
    base = samples()
    count = len(base[0])
    arrays = [model.array(value, wp.mat33 if i < 2 else float) for i, value in enumerate(base)]
    plastic, stress = model.zeros(count, wp.mat33), model.zeros(count, wp.mat33)
    rng = np.random.default_rng(478)
    seeds = [rng.normal(size=(count, 3, 3)), rng.normal(size=(count, 3, 3)) / 3000]
    with wp.Tape() as tape:
        model.launch(response, count, inputs=arrays, outputs=[plastic, stress], device=model.device)
    tape.backward(grads={out: model.array(seed, wp.mat33, False)
                         for out, seed in zip((plastic, stress), seeds)})
    expected = [reference(*args) for args in zip(*base)]
    np.testing.assert_allclose(plastic.numpy(), [x[0] for x in expected], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(stress.numpy(), [x[1] for x in expected], rtol=1e-9, atol=1e-8)
    direction = rng.normal(size=base[parameter].shape)
    axes = tuple(range(1, direction.ndim))
    actual = np.sum(arrays[parameter].grad.numpy() * direction, axis=axes)
    for epsilon in ([.01, .003] if parameter in (2, 3) else [1e-6, 3e-7]):
        losses = []
        for sign in (-1, 1):
            shifted = list(base)
            shifted[parameter] = base[parameter] + sign * epsilon * direction
            result = [reference(*args) for args in zip(*shifted)]
            losses.append(sum(np.sum(np.asarray([row[i] for row in result]) * seeds[i], axis=(1, 2))
                              for i in range(2)))
        np.testing.assert_allclose(actual, (losses[1] - losses[0]) / (2 * epsilon), rtol=2e-6, atol=1e-7)
