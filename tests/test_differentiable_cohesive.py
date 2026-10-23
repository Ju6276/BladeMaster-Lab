"""Interface momentum, area partition, rotation and implicit input derivatives."""

from dataclasses import fields

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_cohesive import solve


DTYPES = [
    wp.vec3d,
    wp.mat33d,
    wp.vec3d,
    wp.float64,
    wp.vec3d,
    wp.float64,
    wp.vec3d,
    wp.float64,
    wp.float64,
    wp.float64,
]


def arrays(values):
    return [wp.array(value, dtype=dtype, requires_grad=True) for value, dtype in zip(values, DTYPES)]


def fixture(scenario, seed=99731):
    rng = np.random.default_rng(seed)
    coordinates = [(0.2, 0.25), (0.7, 0.35), (0.4, 0.8)]
    B, G = np.zeros((3, 8)), np.zeros((3, 8, 3))
    for q, (y, z) in enumerate(coordinates):
        for side in range(2):
            for j in range(2):
                for k in range(2):
                    p = 4 * side + 2 * j + k
                    sy, sz = (y if j else 1 - y), (z if k else 1 - z)
                    B[q, p] = (2 * side - 1) * sy * sz
                    G[q, p] = [0.0, 0.5 * (2 * j - 1) * sz / 0.012, 0.5 * (2 * k - 1) * sy / 0.012]
    scale = 1e-8 if scenario in ["closed", "mixed"] else 2e-5
    gap = B @ rng.normal(scale=scale, size=(8, 3))
    frame = np.tile(np.eye(3), (3, 1, 1)) + rng.normal(scale=0.03, size=(3, 3, 3))
    history = np.zeros(3)
    if scenario in ["mixed", "opened"]:
        history[:] = rng.uniform(0.0001, 0.001, 3)
    if scenario == "mixed":
        history[::2] = 0.0
    return [
        gap,
        frame,
        np.tile([1.0, 0.0, 0.0], (3, 1)),
        history,
        np.tile([5e4, 70.0, 0.0], (3, 1)),
        B,
        G,
        np.full(3, 4e-6),
        np.array([1e-4]),
        np.array([2e-5]),
    ]


def outputs(result):
    return [getattr(result, field.name) for field in fields(result) if field.name != "info"]


@pytest.mark.parametrize("scenario", ["closed", "opening", "mixed", "opened"])
def test_moving_interface_momentum_and_all_input_derivatives(scenario):
    rng = np.random.default_rng(6193)
    values = fixture(scenario)
    sources = arrays(values)
    with wp.Tape() as tape:
        result = solve(*sources)
    if scenario == "closed":
        np.testing.assert_allclose(result.gap.numpy(), 0.0, atol=1e-17)
        np.testing.assert_array_equal(result.history.numpy(), 0.0)
    if scenario == "mixed":
        np.testing.assert_array_equal(result.history.numpy()[::2], 0.0)
    q, F, _, history, _, B, G, area, mass, dt = values
    u = result.displacement.numpy()
    np.testing.assert_allclose(result.gap.numpy(), q + B @ u, atol=1e-17, rtol=1e-12)
    np.testing.assert_allclose(result.frame.numpy(), F + np.einsum("pa,qpb->qab", u, G), atol=5e-16)
    force = B.T @ (area[:, None] * result.traction.numpy())
    force += np.einsum("q,qab,qpb->pa", area, result.surface_stress.numpy(), G)
    np.testing.assert_allclose(u + dt[0] ** 2 / mass[0] * force, 0.0, atol=1e-17)
    np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1e-12)
    assert result.normal_gap.numpy().min() >= -1e-17
    assert np.all(result.history.numpy() >= history)
    target = outputs(result)
    scales = [
        0.0028,
        max(5e4, np.abs(result.traction.numpy()).max()),
        0.0028,
        70.0,
        70.0,
        1.0,
        max(70.0, np.abs(result.surface_stress.numpy()).max()),
        0.0028,
        0.0028,
    ]
    seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(target, scales)]
    tape.backward(grads={a: wp.array(seed, dtype=a.dtype, device=a.device) for a, seed in zip(target, seeds)})
    for index, source in enumerate(sources):
        base = source.numpy()
        direction = rng.normal(size=base.shape) * np.maximum(np.abs(base), 0.1 if index in [1, 2] else 1e-12)
        if index == 3:
            direction *= base > 0
        ad = float(np.sum(source.grad.numpy() * direction))
        errors = []
        for epsilon in [1e-4, 1e-5, 1e-6, 1e-7]:
            trials, frames = [], []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                trial = solve(*sources)
                trials.append([a.numpy() for a in outputs(trial)])
                frames.append(
                    (
                        sources[1].numpy(),
                        np.einsum("pa,qpb->qab", trial.displacement.numpy(), sources[6].numpy()),
                    )
                )
            differences = [b - a for a, b in zip(*trials)]
            # Separate the unit-scale frame from its small displacement increment.
            differences[5] = frames[1][0] - frames[0][0] + frames[1][1] - frames[0][1]
            fd = sum(np.sum(delta * seed) for delta, seed in zip(differences, seeds)) / (2 * epsilon)
            errors.append(abs(ad - fd) / max(abs(ad), abs(fd), 1e-12))
        source.assign(base)
        assert sum(error <= 2e-5 for error in errors) >= 2, (scenario, index, ad, errors)


@pytest.mark.parametrize("scenario", ["closed", "opening", "mixed", "opened"])
def test_area_partition_compliance_and_rigid_rotation(scenario):
    values = fixture(scenario)
    reference = solve(*arrays(values))
    rotation, _ = np.linalg.qr(np.random.default_rng(45).normal(size=(3, 3)))
    transformed = [a.copy() for a in values]
    transformed[0] = values[0] @ rotation.T
    transformed[1] = rotation @ values[1]
    rotated = solve(*arrays(transformed))
    for name in ["gap", "traction", "displacement"]:
        np.testing.assert_allclose(
            getattr(rotated, name).numpy(),
            getattr(reference, name).numpy() @ rotation.T,
            rtol=1e-10,
            atol=1e-10,
        )
    for name in ["frame", "surface_stress"]:
        np.testing.assert_allclose(
            getattr(rotated, name).numpy(),
            rotation @ getattr(reference, name).numpy(),
            rtol=1e-10,
            atol=1e-11,
        )
    for scale in [0.5, 1.0, 2.0]:
        duplicated = [np.repeat(a, 2, axis=0) if i < 8 else a for i, a in enumerate(values)]
        duplicated[7] = duplicated[7] / 2
        result = solve(*arrays(duplicated), compliance_scale=scale)
        np.testing.assert_allclose(result.displacement.numpy(), reference.displacement.numpy(), atol=1e-15)
        for name in ["gap", "history", "storage", "fracture", "frame"]:
            actual = getattr(result, name).numpy().reshape(3, 2, *getattr(reference, name).numpy().shape[1:])
            np.testing.assert_allclose(actual[:, 0], getattr(reference, name).numpy(), rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(actual[:, 1], actual[:, 0], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("scenario", ["closed", "opened"])
def test_redundant_quadrature_preserves_physical_adjoint(scenario):
    values = fixture(scenario)
    rng = np.random.default_rng(771)
    scales = [0.0028, 5e4, 0.0028, 70.0, 70.0, 1.0, 70.0, 0.0028, 0.0028]
    seeds = None
    gradients = []
    for copies in [1, 2]:
        duplicated = [np.repeat(a, copies, axis=0) if i < 8 else a for i, a in enumerate(values)]
        duplicated[7] = duplicated[7] / copies
        sources = arrays(duplicated)
        with wp.Tape() as tape:
            result = solve(*sources)
        target = outputs(result)
        if seeds is None:
            seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(target, scales)]
        seed_values = [np.repeat(s, copies, axis=0) / copies if i < 8 else s for i, s in enumerate(seeds)]
        tape.backward(
            grads={a: wp.array(seed, dtype=a.dtype, device=a.device) for a, seed in zip(target, seed_values)}
        )
        pulled = []
        for index, (source, value) in enumerate(zip(sources, values)):
            gradient = source.grad.numpy()
            if index < 8:
                gradient = gradient.reshape(len(value), copies, *value.shape[1:]).sum(axis=1)
            if index == 7:
                gradient /= copies
            pulled.append(gradient)
        gradients.append(pulled)
    for index, (base, duplicate) in enumerate(zip(*gradients)):
        scale = max(np.max(np.abs(base)), np.max(np.abs(duplicate)), 1.0)
        np.testing.assert_allclose(base, duplicate, rtol=1e-8, atol=1e-10 * scale, err_msg=str(index))
