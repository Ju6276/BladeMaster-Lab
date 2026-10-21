"""Forward equivalence and directional derivatives of elastic TLMPM on CUDA."""

from dataclasses import replace
import numpy as np
import pytest
import warp as wp

from blademaster_warp import material as m
from blademaster_warp.differentiable import ElasticTrajectory, State, elastic_stress
from blademaster_warp.solver import Config, Solver
from blademaster_warp.differentiable_contact import PrescribedContact
from blademaster_warp.coupling import ExternalToolPort, ToolState


@wp.kernel
def stress_probe(F: wp.array(dtype=wp.mat33), output: wp.array(dtype=wp.mat33)):
    output[wp.tid()] = elastic_stress(F[wp.tid()], 2000.0, 3000.0)


@wp.kernel
def velocity_loss(v: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):
    wp.atomic_add(loss, 0, wp.length_sq(v[wp.tid()]) / float(v.shape[0]))


@wp.kernel
def force_loss(force: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):
    wp.atomic_add(loss, 0, wp.dot(force[0], wp.vec3(0.3, 0.7, 0.2)))


def config(spacing=0.006):
    return Config(
        size=(0.024, 0.018, 0.018), grid_spacing=spacing,
        dt=0.0002, blade_grid_constraint=False,
    )


def initial_values(model):
    deformation = np.array([[1.04, 0.025, 0.0], [0.0, 0.98, 0.01], [0.0, 0.0, 1.01]])
    x = model.reference @ deformation.T + [0.0, 0.0, 0.06]
    v = model.reference @ np.diag([1.0, -0.7, 0.2]) + [0.02, -0.01, 0.0]
    return x, v, np.tile(deformation, (model.n, 1, 1))


@pytest.mark.parametrize("kind", ["identity", "isotropic", "repeated", "shear", "rotation"])
def test_stress_derivative_at_repeated_stretches(kind):
    matrices = {
        "identity": np.eye(3),
        "isotropic": np.eye(3) * 1.08,
        "repeated": np.diag([1.1, 1.1, 0.9]),
        "shear": np.array([[1.1, 0.2, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 1.05]]),
        "rotation": np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]]),
    }
    F = matrices[kind]
    rng = np.random.default_rng(12)
    direction = rng.normal(size=(3, 3))
    direction /= np.linalg.norm(direction)
    adjoint = rng.normal(size=(3, 3)).astype(np.float32)
    inputs = wp.array([F], dtype=wp.mat33, device="cuda:0", requires_grad=True)
    out = wp.zeros(1, dtype=wp.mat33, device="cuda:0", requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(stress_probe, 1, inputs=[inputs], outputs=[out], device="cuda:0")
    tape.backward(grads={out: wp.array([adjoint], dtype=wp.mat33, device="cuda:0")})
    ad = float(np.sum(inputs.grad.numpy()[0] * direction))
    assert np.isfinite(ad)
    for epsilon in (3e-3, 1e-3):
        values = []
        for sign in (-1, 1):
            perturbed = wp.array([F + sign * epsilon * direction], dtype=wp.mat33, device="cuda:0")
            wp.launch(stress_probe, 1, inputs=[perturbed], outputs=[out], device="cuda:0")
            values.append(float(np.sum(out.numpy()[0].astype(float) * adjoint)))
        fd = (values[1] - values[0]) / (2 * epsilon)
        np.testing.assert_allclose(ad, fd, rtol=3e-3, atol=0.5)
    if kind == "identity":
        linear = 2000 * (direction + direction.T) + 3000 * np.trace(direction) * np.eye(3)
        np.testing.assert_allclose(ad, np.sum(linear * adjoint), rtol=3e-6, atol=0.01)


@pytest.mark.parametrize("spacing", [0.006, 0.004])
@pytest.mark.parametrize("steps", [1, 12])
def test_forward_matches_original_solver(spacing, steps):
    cfg = config(spacing)
    model = ElasticTrajectory(cfg)
    x, v, F = initial_values(model)
    state = model.state(x, v, F)
    original = Solver(cfg)
    original.x.assign(x)
    original.v.assign(v)
    original.F.assign(F)
    original.tool_pos.assign([[1.0, 1.0, 1.0]])
    original.launch(m.compute_stress, original.n, [
        original.F, original.Fp, original.mu, original.lam, original.stress, original.errors,
    ])
    for _ in range(steps):
        state = model.step(state)
        original.step(target_edge=(1.0, 1.0, 1.0), cutting=False)
    np.testing.assert_allclose(state.x.numpy(), original.x.numpy(), atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(state.v.numpy(), original.v.numpy(), atol=2e-5, rtol=1e-4)
    np.testing.assert_allclose(state.F.numpy(), original.F.numpy(), atol=2e-6, rtol=1e-6)
    assert not original.errors.numpy().any()


@pytest.mark.parametrize("steps", [1, 12, 32])
def test_trajectory_gradient_matches_finite_difference(steps, monkeypatch):
    monkeypatch.setattr(wp.config, "verify_autograd_array_access", True)
    model = ElasticTrajectory(config())
    x, v, F = initial_values(model)
    start = model.state(x, v, F)
    loss = model.zeros(1, float)
    with wp.Tape() as tape:
        state = start
        for _ in range(steps):
            state = model.step(state)
        wp.launch(velocity_loss, model.n, inputs=[state.v], outputs=[loss], device=model.device)
    tape.backward(loss)
    gradient = float(model.young.grad.numpy()[0])
    assert np.isfinite(gradient) and abs(gradient) > 1e-9
    for relative_step in (1e-2, 3e-3):
        epsilon = model.config.young_modulus * relative_step
        values = []
        for sign in (-1, 1):
            other = ElasticTrajectory(
                replace(model.config, young_modulus=model.config.young_modulus + sign * epsilon),
                requires_grad=False,
            )
            current = other.state(x, v, F)
            for _ in range(steps):
                current = other.step(current)
            values.append(float(np.mean(np.sum(current.v.numpy().astype(float) ** 2, axis=1))))
        fd = (values[1] - values[0]) / (2 * epsilon)
        np.testing.assert_allclose(gradient, fd, rtol=0.01, atol=1e-9)
    np.testing.assert_array_equal(start.x.numpy(), x.astype(np.float32))
    np.testing.assert_array_equal(start.F.numpy(), F.astype(np.float32))


def test_free_translation_gradient_is_one():
    model = ElasticTrajectory(config())
    start = model.state(v=np.tile([0.1, -0.2, 0.3], (model.n, 1)))
    with wp.Tape() as tape:
        state = start
        for _ in range(8):
            state = model.step(state, gravity=(0, 0, 0))
    seed = wp.array(np.ones((model.n, 3)) / model.n, dtype=wp.vec3, device=model.device)
    tape.backward(grads={state.v: seed})
    np.testing.assert_allclose(start.v.grad.numpy().sum(axis=0), 1.0, rtol=2e-5)


@pytest.mark.parametrize("variable", ["v", "F"])
def test_initial_state_directional_derivative(variable):
    model = ElasticTrajectory(config())
    values = list(initial_values(model))
    start = model.state(*values)
    loss = model.zeros(1, float)
    direction = (
        np.tile([0.4, -0.2, 0.1], (model.n, 1)) if variable == "v"
        else np.tile(np.diag([0.4, -0.2, 0.1]), (model.n, 1, 1))
    )
    with wp.Tape() as tape:
        state = start
        for _ in range(12):
            state = model.step(state)
        wp.launch(velocity_loss, model.n, inputs=[state.v], outputs=[loss], device=model.device)
    tape.backward(loss)
    ad = float(np.sum(getattr(start, variable).grad.numpy() * direction))
    assert abs(ad) > 1e-5
    for epsilon in (1e-3, 3e-4):
        losses = []
        for sign in (-1, 1):
            perturbed = [value.copy() for value in values]
            index = 1 if variable == "v" else 2
            perturbed[index] += sign * epsilon * direction
            other = ElasticTrajectory(config(), requires_grad=False)
            current = other.state(*perturbed)
            for _ in range(12):
                current = other.step(current)
            losses.append(float(np.mean(np.sum(current.v.numpy().astype(float) ** 2, axis=1))))
        fd = (losses[1] - losses[0]) / (2 * epsilon)
        np.testing.assert_allclose(ad, fd, rtol=0.01, atol=1e-6)


def test_reject_inelastic_config():
    with pytest.raises(ValueError, match="plasticity"):
        ElasticTrajectory(replace(config(), yield_stress=100))


@pytest.mark.parametrize("segment_steps", [1, 7, 32])
def test_checkpoint_vjp_matches_full_tape(segment_steps):
    model = ElasticTrajectory(config())
    start = model.state(*initial_values(model))
    rng = np.random.default_rng(48)
    seed = State(
        model.array(rng.normal(size=(model.n, 3)) / model.n, wp.vec3, requires_grad=False),
        model.array(rng.normal(size=(model.n, 3)) / model.n, wp.vec3, requires_grad=False),
        model.array(rng.normal(size=(model.n, 3, 3)) / model.n, wp.mat33, requires_grad=False),
    )
    with wp.Tape() as tape:
        state = start
        for _ in range(32):
            state = model.step(state)
    tape.backward(grads={state.x: seed.x, state.v: seed.v, state.F: seed.F})
    expected = [a.grad.numpy().copy() for a in (start.x, start.v, start.F, model.young)]
    tape.zero()
    actual, young = model.checkpoint_vjp(start, 32, segment_steps, seed)
    for found, wanted in zip((actual.x, actual.v, actual.F), expected[:3]):
        np.testing.assert_allclose(found.numpy(), wanted, atol=2e-7, rtol=5e-4)
    np.testing.assert_allclose(young, expected[3][0], atol=1e-9, rtol=2e-4)


def contact_scene(speed=0.01, spacing=0.006, requires_grad=True):
    cfg = replace(config(spacing), external_tool=True)
    model = ElasticTrajectory(cfg, requires_grad=requires_grad)
    contact = PrescribedContact(model)
    position = [
        model.reference[:, 0].min() - cfg.blade_half_size[0] - spacing / 4 + 0.0003,
        0.0, cfg.size[2] / 2,
    ]
    velocity = [0.02, speed, 0.0]
    return model, contact, np.array(position), np.array(velocity)


@pytest.mark.parametrize("spacing", [0.006, 0.004])
@pytest.mark.parametrize("speed", [0.01, 2.0])
def test_contact_forward_matches_original_solver(spacing, speed):
    model, contact, position, velocity = contact_scene(speed, spacing)
    original = Solver(model.config)
    port = ExternalToolPort(original)
    state, history = model.state(), contact.history()
    for _ in range(12):
        tool = contact.tool(position, velocity)
        state, history, force = contact.step(state, history, tool)
        expected = port.advance(
            ToolState(position, [0, 0, 0, 1], velocity, [0, 0, 0]),
            model.config.dt, cutting=False, graph=False,
        )
        np.testing.assert_allclose(force.numpy()[0], expected.force_n, atol=3e-5, rtol=3e-4)
        position += model.config.dt * velocity
    np.testing.assert_allclose(state.x.numpy(), original.x.numpy(), atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(state.v.numpy(), original.v.numpy(), atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(state.F.numpy(), original.F.numpy(), atol=3e-6, rtol=1e-5)
    np.testing.assert_allclose(history.numpy(), original.histories.numpy(), atol=1e-8, rtol=1e-4)
    assert not original.errors.numpy().any()


@pytest.mark.parametrize("speed", [0.01, 2.0])
@pytest.mark.parametrize("steps", [1, 8])
def test_contact_gradient_in_stick_and_slip(speed, steps, monkeypatch):
    monkeypatch.setattr(wp.config, "verify_autograd_array_access", True)
    model, contact, position, velocity = contact_scene(speed)
    state, history = model.state(), contact.history()
    tool = contact.tool(position, velocity)
    loss = model.zeros(1, float)
    with wp.Tape() as tape:
        for _ in range(steps):
            state, history, force = contact.step(state, history, tool)
            wp.launch(force_loss, 1, inputs=[force], outputs=[loss], device=model.device)
    tape.backward(loss)
    ad = float(tool.position.grad.numpy()[0, 0])
    assert np.isfinite(ad) and abs(ad) > 1e-3
    for epsilon in (1e-5, 3e-6):
        values = []
        for sign in (-1, 1):
            other, other_contact, p, v = contact_scene(speed, requires_grad=False)
            p[0] += sign * epsilon
            other_tool = other_contact.tool(p, v)
            current, h = other.state(), other_contact.history()
            total = 0.0
            for _ in range(steps):
                current, h, reaction = other_contact.step(current, h, other_tool)
                total += float(reaction.numpy()[0] @ np.array([0.3, 0.7, 0.2]))
            values.append(total)
        fd = (values[1] - values[0]) / (2 * epsilon)
        np.testing.assert_allclose(ad, fd, rtol=0.015, atol=0.01)
