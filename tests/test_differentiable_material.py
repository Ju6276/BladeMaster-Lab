"""Plastic return-map derivatives and fixed-field TLMPM forward equivalence."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_material import MaterialTrajectory, plastic_project
from blademaster_warp.differentiable_contact import PrescribedContact
from blademaster_warp.solver import Config, Solver
from test_plasticity import reference_return


@wp.kernel
def projection(f: wp.array(dtype=wp.mat33), limit: wp.array(dtype=float), out: wp.array(dtype=wp.mat33), success: wp.array(dtype=int)):
    value, converged = plastic_project(f[0], limit[0])
    out[0] = value
    success[0] = converged


@wp.kernel
def material_loss(v: wp.array(dtype=wp.vec3), Fp: wp.array(dtype=wp.mat33), loss: wp.array(dtype=float)):
    p = wp.tid()
    wp.atomic_add(loss, 0, (wp.length_sq(v[p]) + 0.1 * wp.ddot(Fp[p], Fp[p])) / float(v.shape[0]))


def reference_projection(f, limit):
    u, stretch, vt = np.linalg.svd(f)
    return (u * reference_return(stretch, limit)) @ vt


@pytest.mark.parametrize("kind", ["identity", "isotropic", "rotation", "repeated", "near", "shear"])
def test_plastic_projection_derivatives(kind):
    rotation = np.array([[0.8, -0.6, 0], [0.6, 0.8, 0], [0, 0, 1]])
    matrices = {
        "identity": np.eye(3), "isotropic": np.eye(3) * 1.08,
        "rotation": rotation, "repeated": rotation @ np.diag([1.25, 1.25, 0.8]),
        "near": rotation @ np.diag([1.25001, 1.25, 0.8]),
        "shear": np.array([[1.2, 0.2, 0], [0, 0.9, 0.1], [0, 0, 1.05]]),
    }
    f, limit = matrices[kind], 0.08
    rng = np.random.default_rng(84)
    seed = rng.normal(size=(3, 3)).astype(np.float32)
    direction = rng.normal(size=(3, 3))
    direction /= np.linalg.norm(direction)
    a = wp.array([f], dtype=wp.mat33, device="cuda:0", requires_grad=True)
    b = wp.array([limit], dtype=float, device="cuda:0", requires_grad=True)
    out = wp.zeros(1, dtype=wp.mat33, device="cuda:0", requires_grad=True)
    success = wp.zeros(1, dtype=int, device="cuda:0")
    with wp.Tape() as tape:
        wp.launch(projection, 1, inputs=[a, b], outputs=[out, success], device="cuda:0")
    assert success.numpy()[0] == 1
    tape.backward(grads={out: wp.array([seed], dtype=wp.mat33, device="cuda:0")})
    np.testing.assert_allclose(out.numpy()[0], reference_projection(f, limit), atol=3e-6)
    ad_f = float(np.sum(a.grad.numpy()[0] * direction))
    ad_limit = float(b.grad.numpy()[0])
    for eps in (1e-4, 1e-5):
        fd_f = np.sum((reference_projection(f + eps * direction, limit)
                       - reference_projection(f - eps * direction, limit)) * seed) / (2 * eps)
        fd_limit = np.sum((reference_projection(f, limit + eps)
                          - reference_projection(f, limit - eps)) * seed) / (2 * eps)
        np.testing.assert_allclose(ad_f, fd_f, rtol=3e-4, atol=2e-6)
        np.testing.assert_allclose(ad_limit, fd_limit, rtol=3e-4, atol=2e-6)


def config(spacing=0.006):
    return Config(size=(0.024, 0.018, 0.018), grid_spacing=spacing, dt=0.0002,
                  yield_stress=180.0, blade_grid_constraint=False)


def initial_values(model):
    deformation = np.array([[1.18, 0.07, 0], [0, 0.93, 0.03], [0, 0, 1.01]])
    plastic = np.array([[1.02, 0.015, 0], [0, 0.99, 0], [0, 0, 1 / (1.02 * 0.99)]])
    return (model.reference @ deformation.T + [0, 0, 0.06],
            model.reference @ np.diag([1.0, -0.7, 0.2]) + [0.02, -0.01, 0],
            np.tile(deformation, (model.n, 1, 1)), np.tile(plastic, (model.n, 1, 1)))


def cut_codes(model):
    codes = np.where(model.reference[:, 0] < 0, 1, 2)
    codes[model.reference[:, 2] < 0.007] = 0
    return codes


def test_elastic_checkpoint_rejects_plastic_state():
    model = MaterialTrajectory(replace(config(), external_tool=True))
    contact = PrescribedContact(model)
    with pytest.raises(ValueError, match="elastic x/v/F"):
        contact.checkpoint_vjp(model.state(), contact.history(), 1, 1, None, None)


@pytest.mark.parametrize("spacing", [0.006, 0.004])
@pytest.mark.parametrize("steps", [1, 12])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_multifield_plastic_forward(spacing, steps, precision):
    cfg = config(spacing)
    model = MaterialTrajectory(cfg, precision=precision)
    values = initial_values(model)
    state = model.state(*values)
    grid = model.topology(cut_codes(model))
    original = Solver(cfg, precision=precision)
    for name, value in zip(("x", "v", "F", "Fp"), values):
        getattr(original, name).assign(value)
    original.codes.assign(cut_codes(model))
    original.refresh_stress()
    for _ in range(steps):
        state = model.step(state, gravity=(0, 0, 0), topology=grid)
        original.impulses.zero_()
        original.material_step()
    for name, tolerance in (("x", 1e-6), ("v", 1e-4), ("F", 1e-5), ("Fp", 1e-5)):
        if precision == "float64":
            tolerance = 1e-10
        error = np.max(np.abs(getattr(state, name).numpy() - getattr(original, name).numpy()))
        assert error < tolerance, (name, error)
    assert not original.errors.numpy().any()
    assert np.max(np.abs(state.Fp.numpy() - values[3])) > 0.01


@pytest.mark.parametrize("parameter", ["young_modulus", "yield_stress", "F", "Fp", "v"])
def test_multifield_plastic_trajectory_gradient(parameter, monkeypatch):
    monkeypatch.setattr(wp.config, "verify_autograd_array_access", True)
    cfg = config()
    model = MaterialTrajectory(cfg)
    values = initial_values(model)
    grid = model.topology(cut_codes(model))
    start = model.state(*values)
    loss = model.zeros(1, float)
    with wp.Tape() as tape:
        state = start
        for _ in range(7):
            state = model.step(state, gravity=(0, 0, 0), topology=grid)
        wp.launch(material_loss, model.n, inputs=[state.v, state.Fp], outputs=[loss], device=model.device)
    tape.backward(loss)
    rng = np.random.default_rng(79)
    if parameter in ("F", "Fp", "v"):
        index = {"v": 1, "F": 2, "Fp": 3}[parameter]
        direction = rng.normal(size=values[index].shape)
        direction /= np.linalg.norm(direction) / np.sqrt(model.n)
        ad = np.sum(getattr(start, parameter).grad.numpy().astype(float) * direction)
        epsilons = (0.002, 0.001)
    else:
        array = model.young if parameter == "young_modulus" else model.yield_stress
        ad = float(array.grad.numpy()[0])
        epsilons = (getattr(cfg, parameter) * 0.01, getattr(cfg, parameter) * 0.005)
    assert np.isfinite(ad) and abs(ad) > 1e-9
    for epsilon in epsilons:
        losses = []
        for sign in (-1, 1):
            perturbed = list(values)
            other_cfg = cfg
            if parameter in ("F", "Fp", "v"):
                perturbed[index] = values[index] + sign * epsilon * direction
            else:
                other_cfg = replace(cfg, **{parameter: getattr(cfg, parameter) + sign * epsilon})
            other = MaterialTrajectory(other_cfg, requires_grad=False)
            state = other.state(*perturbed)
            for _ in range(7):
                state = other.step(state, gravity=(0, 0, 0), topology=grid)
            losses.append(float(np.mean(np.sum(state.v.numpy().astype(float) ** 2, axis=1)
                                       + 0.1 * np.sum(state.Fp.numpy().astype(float) ** 2, axis=(1, 2)))))
        fd = (losses[1] - losses[0]) / (2 * epsilon)
        np.testing.assert_allclose(ad, fd, rtol=0.015, atol=2e-7)
