"""Integrated interface forces, empty supports and implicit derivatives."""

from dataclasses import fields

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_cohesive import solve, solve_force
from blademaster_warp.cohesive_geometry import InterfaceGeometry, solve_geometry
from test_differentiable_cohesive import arrays, fixture, closed_pair


def solve_force_compensated(*sources, compliance_scale=1.0):
    geometry = InterfaceGeometry(sources[0], sources[1])
    return solve_geometry(geometry, *sources[2:], compliance_scale=compliance_scale, integrated_force=True)[0]


@pytest.fixture(params=[solve_force, solve_force_compensated])
def force_solver(request):
    return request.param


def outputs(result):
    return [getattr(result, f.name) for f in fields(result) if f.name != "info"]


@pytest.mark.parametrize("scenario", ["closed", "opening", "mixed", "opened"])
def test_integrated_force_matches_pressure_state_and_all_input_pullbacks(scenario, force_solver):
    values = fixture(scenario)
    source = arrays(values)
    with wp.Tape() as tape:
        result = force_solver(*source)
    pressure = solve(*arrays(values))
    for name in ["gap", "history", "storage", "fracture", "frame", "normal_gap", "displacement"]:
        np.testing.assert_allclose(
            getattr(result, name).numpy(), getattr(pressure, name).numpy(), rtol=1e-10, atol=1e-13
        )
    np.testing.assert_allclose(
        result.force.numpy(), values[7][:, None] * pressure.traction.numpy(), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        result.surface_moment.numpy(),
        values[7][:, None, None] * pressure.surface_stress.numpy(),
        rtol=1e-10,
        atol=1e-13,
    )
    balance = values[5].T @ result.force.numpy() + np.einsum(
        "qab,qpb->pa", result.surface_moment.numpy(), values[6]
    )
    np.testing.assert_allclose(
        result.displacement.numpy() + values[9][0] ** 2 / values[8][0] * balance, 0.0, atol=1e-17
    )
    rng = np.random.default_rng(723)
    target = outputs(result)
    scales = [
        0.0028,
        max(0.2, abs(result.force.numpy()).max()),
        0.0028,
        70.0,
        70.0,
        1.0,
        max(0.00028, abs(result.surface_moment.numpy()).max()),
        0.0028,
        0.0028,
    ]
    seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(target, scales)]
    tape.backward(grads={a: wp.array(s, dtype=a.dtype) for a, s in zip(target, seeds)})
    for i, a in enumerate(source):
        base = a.numpy().copy()
        direction = rng.normal(size=base.shape) * np.maximum(abs(base), 0.1 if i in [1, 2] else 1e-12)
        if i == 3:
            direction *= base > 0
        ad = float(np.sum(a.grad.numpy() * direction))
        errors = []
        for epsilon in [1e-4, 1e-5, 1e-6, 1e-7]:
            trials, frames = [], []
            for sign in [-1, 1]:
                a.assign(base + sign * epsilon * direction)
                trial = force_solver(*source)
                trials.append([b.numpy() for b in outputs(trial)])
                frames.append(
                    (
                        source[1].numpy(),
                        np.einsum("pa,qpb->qab", trial.displacement.numpy(), source[6].numpy()),
                    )
                )
            differences = [b - c for c, b in zip(*trials)]
            # Retain the small frame increment separately from its unit base.
            differences[5] = frames[1][0] - frames[0][0] + frames[1][1] - frames[0][1]
            fd = sum(np.sum(delta * seed) for delta, seed in zip(differences, seeds)) / (2 * epsilon)
            errors.append(abs(ad - fd) / max(abs(ad), abs(fd), 1e-12))
        a.assign(base)
        assert sum(e < 2e-5 for e in errors) >= 2, (scenario, i, ad, errors)


@pytest.mark.parametrize("history", [0.0, 0.0002, 0.003])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_empty_support_preserves_omitted_interface_state_and_motion_adjoint(history, sign, force_solver):
    values = fixture("opened")
    values[3][:] = history
    values[0][:, 0] = sign * (abs(values[0][:, 0]) + 1e-4)
    values[7][0] = 0.0
    source = arrays(values)
    with wp.Tape() as tape:
        result = force_solver(*source)
    positive = [v[1:] for v in values[:8]] + values[8:]
    expected = solve(*arrays(positive))
    np.testing.assert_allclose(result.displacement.numpy(), expected.displacement.numpy(), atol=1e-17)
    np.testing.assert_allclose(result.force.numpy()[0], 0.0, atol=1e-20)
    np.testing.assert_allclose(
        result.gap.numpy()[0], (values[0] + values[5] @ result.displacement.numpy())[0], atol=1e-18
    )
    rng = np.random.default_rng(501)
    seed = rng.normal(size=result.displacement.numpy().shape)
    tape.backward(grads={result.displacement: wp.array(seed, dtype=wp.vec3d)})
    for i in [0, 1, 2, 3, 4, 5, 6, 8, 9]:
        base = source[i].numpy().copy()
        direction = rng.normal(size=base.shape) * np.maximum(abs(base), 0.1 if i in [1, 2] else 1e-12)
        if i == 3:
            direction *= base > 0
        ad = float(np.sum(source[i].grad.numpy() * direction))
        for epsilon in [1e-5, 1e-6]:
            trials = []
            for sign_fd in [-1.0, 1.0]:
                source[i].assign(base + sign_fd * epsilon * direction)
                trials.append(force_solver(*source).displacement.numpy())
            fd = float(np.sum((trials[1] - trials[0]) * seed) / (2 * epsilon))
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-12, err_msg=str(i))
        source[i].assign(base)
    if sign > 0:
        ad = float(source[7].grad.numpy()[0] * 4e-6)
        for epsilon in [1e-4, 1e-5]:
            area = values[7].copy()
            area[0] = epsilon * 4e-6
            source[7].assign(area)
            trial = force_solver(*source).displacement.numpy()
            fd = float(np.sum(seed * (trial - result.displacement.numpy())) / epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-12)


@pytest.mark.parametrize("history", [0.0, 0.0002, 0.003])
def test_all_zero_supports_leave_free_kinematics_and_no_interface_force(history, force_solver):
    values = fixture("opened")
    values[3][:] = history
    values[7][:] = 0.0
    source = arrays(values)
    with wp.Tape() as tape:
        result = force_solver(*source)
    np.testing.assert_allclose(result.displacement.numpy(), 0.0, atol=1e-20)
    np.testing.assert_allclose(result.force.numpy(), 0.0, atol=1e-20)
    np.testing.assert_allclose(result.gap.numpy(), values[0], atol=1e-20)
    np.testing.assert_allclose(result.frame.numpy(), values[1], atol=1e-20)
    seed = np.random.default_rng(721).normal(size=values[0].shape)
    tape.backward(grads={result.gap: wp.array(seed, dtype=wp.vec3d)})
    np.testing.assert_allclose(source[0].grad.numpy(), seed, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("fraction", [0.6, 0.65, 0.9999])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_closed_integrated_force_preserves_unequal_partition_history_derivative(fraction, sign, force_solver):
    values = closed_pair(4e-6, [sign * 1e-6, 0.0, 0.0])
    values = [np.repeat(a, 2, axis=0) if i < 8 else a for i, a in enumerate(values)]
    values[7][:] = 4e-6 * np.array([fraction, 1 - fraction])
    source = arrays(values)
    with wp.Tape() as tape:
        result = force_solver(*source)
    np.testing.assert_allclose(result.force.numpy()[:, 0], sign * 31250.0 * values[7], rtol=1e-13)
    seed = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    tape.backward(grads={result.displacement: wp.array(seed, dtype=wp.vec3d)})
    np.testing.assert_allclose(
        source[3].grad.numpy().sum(), 0.625 if sign > 0 else 0.0, rtol=2e-13, atol=1e-14
    )
    np.testing.assert_array_equal(source[7].grad.numpy(), 0.0)


@pytest.mark.parametrize("scenario", ["opening", "mixed", "opened"])
@pytest.mark.parametrize("integrated", [False, True])
def test_redundant_selected_reaction_area_and_geometry_pullbacks(scenario, integrated, force_solver):
    solver = force_solver if integrated else solve
    base = fixture(scenario)
    values = [np.repeat(a, 2, axis=0) if i < 8 else a.copy() for i, a in enumerate(base)]
    values[7] *= np.tile([0.6, 0.4], 3)
    source = arrays(values)
    with wp.Tape() as tape:
        result = solver(*source)
    force = result.force if integrated else result.traction
    rng = np.random.default_rng(873)
    seed = rng.normal(size=force.numpy().shape) * (1.0 if integrated else 4e-6)
    tape.backward(grads={force: wp.array(seed, dtype=wp.vec3d)})
    for i in [0, 1, 2, 4, 5, 6, 7, 8, 9]:
        direction = rng.normal(size=base[i].shape) * np.maximum(abs(base[i]), 0.1 if i in [1, 2] else 1e-12)
        if i < 8:
            direction = np.repeat(direction, 2, axis=0)
        if i == 7:
            direction = values[7] * rng.normal(size=values[7].shape)
        ad = float(np.sum(source[i].grad.numpy() * direction))
        for epsilon in [1e-4, 1e-5]:
            trials = []
            for sign in [-1, 1]:
                source[i].assign(values[i] + sign * epsilon * direction)
                trial = solver(*source)
                trials.append((trial.force if integrated else trial.traction).numpy())
            fd = float(np.sum((trials[1] - trials[0]) * seed) / (2 * epsilon))
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-10, err_msg=str(i))
        source[i].assign(values[i])


@pytest.mark.parametrize("force_scale", [1e-9, 1.0, 1e3])
def test_nonpositive_canonical_contact_reaction_with_binding_constraint(force_solver, force_scale):
    B = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, -1.0]])
    force = force_scale * np.array([-0.001, -0.1, -0.001])
    coefficient = 1e-4 / (2e-5) ** 2
    values = [
        np.column_stack([B @ B.T @ force / coefficient, np.zeros((3, 2))]),
        np.tile(np.eye(3), (3, 1, 1)),
        np.tile([1.0, 0.0, 0.0], (3, 1)),
        np.full(3, 0.0002),
        np.tile([5e4, 70.0, 0.0], (3, 1)),
        B,
        np.zeros((3, 3, 3)),
        np.full(3, 4e-6),
        np.array([1e-4]),
        np.array([2e-5]),
    ]
    source = arrays(values)
    with wp.Tape() as tape:
        result = force_solver(*source)
    np.testing.assert_allclose(
        result.force.numpy()[:, 0], force_scale * np.array([-0.002, -0.099, 0.0]), atol=1e-12 * force_scale
    )
    seed = np.random.default_rng(43).normal(size=(3, 3)) / force_scale
    tape.backward(grads={result.force: wp.array(seed, dtype=wp.vec3d)})
    direction = values[7] * np.array([0.1, 0.3, 0.5])
    ad = float(np.sum(source[7].grad.numpy() * direction))
    for epsilon in [1e-4, 1e-5]:
        trials = []
        for sign in [-1, 1]:
            source[7].assign(values[7] + sign * epsilon * direction)
            trials.append(force_solver(*source).force.numpy())
        fd = float(np.sum(seed * (trials[1] - trials[0])) / (2 * epsilon))
        np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-10)

    source[7].assign(values[7])
    force_direction = force_scale * np.array([-0.03, -0.2, 0.0])
    direction = np.zeros_like(values[0])
    direction[:, 0] = B @ B.T @ force_direction / coefficient
    ad = float(np.sum(source[0].grad.numpy() * direction))
    expected = float(seed[:, 0] @ force_direction)
    np.testing.assert_allclose(ad, expected, rtol=2e-8, atol=1e-10)
    for epsilon in [1e-4, 1e-5]:
        trials = []
        for sign in [-1, 1]:
            source[0].assign(values[0] + sign * epsilon * direction)
            trials.append(force_solver(*source).force.numpy())
        fd = float(np.sum(seed * (trials[1] - trials[0])) / (2 * epsilon))
        np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-10)


def test_zero_area_birth_alongside_redundant_closed_constraints(force_solver):
    values = closed_pair(4e-6, [1e-6, 0.0, 0.0])
    values = [np.repeat(a, 3, axis=0) if i < 8 else a for i, a in enumerate(values)]
    values[7][:] = [2.4e-6, 1.6e-6, 0.0]
    values[0][2, 0] = 1e-4
    source = arrays(values)
    with wp.Tape() as tape:
        result = force_solver(*source)
    seed = np.array([[0.3, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.7, 0.0, 0.0]])
    tape.backward(grads={result.force: wp.array(seed, dtype=wp.vec3d)})
    for i, direction in [
        (7, np.array([1e-6, -1e-6, 0.0])),
        (7, np.array([0.0, 0.0, 4e-6])),
        (3, np.array([1.0, 1.0, 0.0])),
    ]:
        ad = float(np.sum(source[i].grad.numpy() * direction))
        epsilons = [1e-11, 1e-12] if i == 3 else [1e-5, 1e-6]
        for epsilon in epsilons:
            source[i].assign(values[i] + epsilon * direction)
            trial = force_solver(*source).force.numpy()
            fd = float(np.sum(seed * (trial - result.force.numpy())) / epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-8)
        source[i].assign(values[i])


def test_canonical_reaction_strength_bound_and_parameter_derivative(force_solver):
    values = closed_pair(4e-6, [1e-6, 0.0, 0.0])
    values = [np.repeat(a, 2, axis=0) if i < 8 else a for i, a in enumerate(values)]
    values[7][:] = [1e-6, 3e-6]
    values[4][:, 0] = [1e3, 5e4]
    source = arrays(values)
    with wp.Tape() as tape:
        result = force_solver(*source)
    np.testing.assert_allclose(result.gap.numpy(), 0.0, atol=1e-16)
    np.testing.assert_allclose(result.force.numpy()[:, 0], [0.001, 0.124], rtol=1e-10, atol=1e-12)
    tape.backward(grads={result.force: wp.array([[1.0, 0.0, 0.0], [-0.3, 0.0, 0.0]], dtype=wp.vec3d)})
    for i, direction in [(7, np.array([1e-6, 0.0])), (4, np.array([[1e3, 0.0, 0.0], [0.0, 0.0, 0.0]]))]:
        ad = float(np.sum(source[i].grad.numpy() * direction))
        np.testing.assert_allclose(ad, 0.0013, rtol=2e-5, atol=1e-10)
        for epsilon in [1e-4, 1e-5]:
            trials = []
            for sign in [-1, 1]:
                source[i].assign(values[i] + sign * epsilon * direction)
                trials.append(force_solver(*source).force.numpy())
            fd = float(
                ((trials[1][0, 0] - 0.3 * trials[1][1, 0]) - (trials[0][0, 0] - 0.3 * trials[0][1, 0]))
                / (2 * epsilon)
            )
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-10)
        source[i].assign(values[i])
