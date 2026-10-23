"""Interface momentum, area partition, rotation and implicit input derivatives."""

from dataclasses import fields

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_cohesive import solve
from blademaster_warp.cohesive_geometry import InterfaceGeometry, solve_geometry


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


def solve_compensated(*sources, compliance_scale=1.0):
    geometry = InterfaceGeometry(sources[0], sources[1])
    return solve_geometry(geometry, *sources[2:], compliance_scale=compliance_scale)[0]


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
    return [value for field in fields(result) if isinstance(value := getattr(result, field.name), wp.array)]


@pytest.mark.parametrize("interface_solver", [solve, solve_compensated])
@pytest.mark.parametrize("scenario", ["closed", "opening", "mixed", "opened"])
def test_moving_interface_momentum_and_all_input_derivatives(scenario, interface_solver):
    rng = np.random.default_rng(6193)
    values = fixture(scenario)
    sources = arrays(values)
    with wp.Tape() as tape:
        result = interface_solver(*sources)
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
                trial = interface_solver(*sources)
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


@pytest.mark.parametrize("interface_solver", [solve, solve_compensated])
@pytest.mark.parametrize("scenario", ["closed", "opening", "mixed", "opened"])
def test_area_partition_compliance_and_rigid_rotation(scenario, interface_solver):
    values = fixture(scenario)
    reference = interface_solver(*arrays(values))
    rotation, _ = np.linalg.qr(np.random.default_rng(45).normal(size=(3, 3)))
    transformed = [a.copy() for a in values]
    transformed[0] = values[0] @ rotation.T
    transformed[1] = rotation @ values[1]
    rotated = interface_solver(*arrays(transformed))
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
        result = interface_solver(*arrays(duplicated), compliance_scale=scale)
        np.testing.assert_allclose(result.displacement.numpy(), reference.displacement.numpy(), atol=1e-15)
        for name in ["gap", "history", "storage", "fracture", "frame"]:
            actual = getattr(result, name).numpy().reshape(3, 2, *getattr(reference, name).numpy().shape[1:])
            np.testing.assert_allclose(actual[:, 0], getattr(reference, name).numpy(), rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(actual[:, 1], actual[:, 0], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("interface_solver", [solve, solve_compensated])
@pytest.mark.parametrize("scenario", ["closed", "opened"])
def test_redundant_quadrature_preserves_physical_adjoint(scenario, interface_solver):
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
            result = interface_solver(*sources)
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


def closed_pair(area, free_gap):
    return [
        np.array([free_gap]),
        np.eye(3)[None],
        np.array([[1.0, 0.0, 0.0]]),
        np.zeros(1),
        np.array([[5e4, 70.0, 0.0]]),
        np.array([[-1.0, 1.0]]),
        np.zeros((1, 2, 3)),
        np.array([area]),
        np.array([1e-4]),
        np.array([2e-5]),
    ]


@pytest.mark.parametrize("interface_solver", [solve, solve_compensated])
@pytest.mark.parametrize("fraction", [0.6, 0.65, 0.9999])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_unequal_area_partition_preserves_pressure_and_history_derivative(interface_solver, fraction, sign):
    values = closed_pair(4e-6, [sign * 1e-6, 0.0, 0.0])
    values = [np.repeat(a, 2, axis=0) if i < 8 else a for i, a in enumerate(values)]
    values[7][:] = 4e-6 * np.array([fraction, 1 - fraction])
    source = arrays(values)
    with wp.Tape() as tape:
        result = interface_solver(*source)
    np.testing.assert_allclose(
        result.traction.numpy(), [[sign * 31250.0, 0.0, 0.0]] * 2, rtol=2e-13, atol=1e-12
    )
    seed_u = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    tape.backward(grads={result.displacement: wp.array(seed_u, dtype=wp.vec3d)})
    expected = 0.625 if sign > 0 else 0.0
    np.testing.assert_allclose(source[3].grad.numpy().sum(), expected, rtol=2e-13, atol=1e-14)
    np.testing.assert_array_equal(source[7].grad.numpy(), 0.0)
    tape.zero()
    seed_t = np.array([[1.0 / 5e4, 0.0, 0.0], [0.0, 0.0, 0.0]])
    tape.backward(grads={result.traction: wp.array(seed_t, dtype=wp.vec3d)})
    np.testing.assert_allclose(source[7].grad.numpy(), -sign * 156250.0, rtol=2e-10, atol=1e-5)
    np.testing.assert_allclose(source[3].grad.numpy().sum(), -expected * 625000.0, rtol=2e-10, atol=1e-8)


@pytest.mark.parametrize("interface_solver", [solve, solve_compensated])
def test_unequal_quadrature_partition_preserves_all_physical_pullbacks(interface_solver):
    values = fixture("closed")
    rng = np.random.default_rng(911)
    reference, gradients, seeds = None, None, None
    for fractions in [[1.0], [0.6, 0.4], [0.65, 0.35], [0.9999, 0.0001]]:
        copies = len(fractions)
        duplicated = [np.repeat(a, copies, axis=0) if i < 8 else a.copy() for i, a in enumerate(values)]
        weights = np.tile(fractions, len(values[0]))
        duplicated[7] *= weights
        source = arrays(duplicated)
        with wp.Tape() as tape:
            result = interface_solver(*source)
        target = outputs(result)
        if seeds is None:
            scales = [0.0028, 5e4, 0.0028, 70.0, 70.0, 1.0, 70.0, 0.0028, 0.0028]
            seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(target, scales)]
            reference = [a.numpy() for a in target]
        actual_seeds = [np.repeat(s, copies, axis=0) if i < 8 else s.copy() for i, s in enumerate(seeds)]
        for i in range(8):
            actual_seeds[i] *= weights.reshape((-1,) + (1,) * (actual_seeds[i].ndim - 1))
            np.testing.assert_allclose(
                target[i].numpy(), np.repeat(reference[i], copies, axis=0), rtol=2e-12, atol=1e-11
            )
        np.testing.assert_allclose(result.displacement.numpy(), reference[-1], rtol=2e-12, atol=1e-15)
        tape.backward(grads={a: wp.array(s, dtype=a.dtype) for a, s in zip(target, actual_seeds)})
        pulled = []
        for i, (a, base) in enumerate(zip(source, values)):
            gradient = a.grad.numpy()
            if i == 7:
                gradient *= weights
            if i < 8:
                gradient = gradient.reshape(len(base), copies, *base.shape[1:]).sum(axis=1)
            pulled.append(gradient)
        if gradients is None:
            gradients = pulled
        else:
            for i, (actual, expected) in enumerate(zip(pulled, gradients)):
                scale = max(np.max(np.abs(expected)), 1.0)
                np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-10 * scale, err_msg=str(i))


@pytest.mark.parametrize("interface_solver", [solve, solve_compensated])
@pytest.mark.parametrize("moving_frame", [False, True])
@pytest.mark.parametrize("history", [[0.0, 0.0, 0.0], [1e-4, 2e-4, 4e-3], [0.0, 2e-4, 4e-3]])
def test_zero_opening_map_preserves_history_energy_and_adjoint(interface_solver, moving_frame, history):
    values = fixture("closed")
    values[0][:] = 0.0
    values[3][:] = history
    values[5][:] = 0.0
    if not moving_frame:
        values[6][:] = 0.0
    sources = arrays(values)
    with np.errstate(divide="raise", invalid="raise"):
        with wp.Tape() as tape:
            result = interface_solver(*sources)
    for name in ["gap", "traction", "storage", "surface_stress", "normal_gap", "displacement"]:
        np.testing.assert_array_equal(getattr(result, name).numpy(), 0.0)
    np.testing.assert_array_equal(result.frame.numpy(), values[1])
    np.testing.assert_array_equal(result.history.numpy(), values[3])
    failure = 2 * values[4][:, 1] / values[4][:, 0]
    np.testing.assert_allclose(result.fracture.numpy(), 0.5 * values[4][:, 0] * np.minimum(history, failure))
    rng = np.random.default_rng(459)
    target = outputs(result)
    scales = [0.0028, 5e4, 0.0028, 70.0, 70.0, 1.0, 70.0, 0.0028, 0.0028]
    seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(target, scales)]
    tape.backward(grads={a: wp.array(seed, dtype=a.dtype) for a, seed in zip(target, seeds)})
    for source in sources:
        assert np.isfinite(source.grad.numpy()).all()
    # Perturbations retain the shared-field identity B=0, q=0.
    for index in [1, 2, 3, 4, 6, 7, 8, 9]:
        source = sources[index]
        base = source.numpy().copy()
        direction = rng.normal(size=base.shape) * np.maximum(np.abs(base), 1e-12)
        if index == 3:
            direction[base == 0.0] = 0.0
        ad = float(np.sum(source.grad.numpy() * direction))
        for epsilon in [1e-4, 1e-5]:
            trials = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                trials.append([a.numpy() for a in outputs(interface_solver(*sources))])
            fd = sum(np.sum(seed * (b - a)) for a, b, seed in zip(*trials, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-8)
        source.assign(base)
    for scale in [0.5, 1.0, 1.5]:
        scaled = interface_solver(*sources, compliance_scale=scale)
        for a, b in zip(target, outputs(scaled)):
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=2e-14, atol=1e-15)


@pytest.mark.parametrize("interface_solver", [solve, solve_compensated])
def test_shared_intact_point_preserves_loaded_interface_and_its_adjoint(interface_solver):
    values = fixture("opened")
    extra = [a[:1].copy() for a in values[:8]]
    for index in [0, 3, 5]:
        extra[index][:] = 0.0
    extended = [np.concatenate([a, b]) for a, b in zip(values[:8], extra)] + values[8:]
    rng = np.random.default_rng(881)
    seeds = None
    results, gradients = [], []
    for data in [values, extended]:
        sources = arrays(data)
        with wp.Tape() as tape:
            result = interface_solver(*sources)
        target = outputs(result)
        if seeds is None:
            scales = [0.0028, 5e4, 0.0028, 70.0, 70.0, 1.0, 70.0, 0.0028, 0.0028]
            seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(target, scales)]
            actual_seeds = seeds
        else:
            actual_seeds = [
                np.concatenate([s, np.zeros_like(s[:1])]) if i < 8 else s for i, s in enumerate(seeds)
            ]
            actual_seeds[1][-1] = 1.0 / 5e4
            for name in ["gap", "traction", "history", "storage", "fracture", "surface_stress"]:
                np.testing.assert_array_equal(getattr(result, name).numpy()[-1], 0.0)
        tape.backward(grads={a: wp.array(s, dtype=a.dtype) for a, s in zip(target, actual_seeds)})
        results.append([a.numpy()[:3] if i < 8 else a.numpy() for i, a in enumerate(target)])
        gradients.append([a.grad.numpy()[:3] if i < 8 else a.grad.numpy() for i, a in enumerate(sources)])
    for before, after in zip(*results):
        np.testing.assert_allclose(after, before, rtol=2e-12, atol=1e-10)
    for before, after in zip(*gradients):
        scale = max(np.max(np.abs(before)), 1.0)
        np.testing.assert_allclose(after, before, rtol=2e-8, atol=1e-9 * scale)


@pytest.mark.parametrize("interface_solver", [solve, solve_compensated])
def test_zero_mobility_retains_prescribed_opening_response_and_map_derivative(interface_solver):
    values = closed_pair(4e-6, [3e-5, 4e-5, -2e-5])
    values[3][:] = 2e-4
    values[5][:] = 0.0
    sources = arrays(values)
    with wp.Tape() as tape:
        result = interface_solver(*sources)
    failure = 2 * values[4][0, 1] / values[4][0, 0]
    stiffness = values[4][0, 0] * (1 - values[3][0] / failure) / values[3][0]
    np.testing.assert_allclose(result.gap.numpy(), values[0], rtol=2e-14, atol=1e-18)
    np.testing.assert_allclose(result.traction.numpy(), stiffness * values[0], rtol=2e-14, atol=1e-10)
    np.testing.assert_array_equal(result.displacement.numpy(), 0.0)
    rng = np.random.default_rng(435)
    target = outputs(result)
    scales = [0.0028, 5e4, 0.0028, 70.0, 70.0, 1.0, 70.0, 0.0028, 0.0028]
    seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(target, scales)]
    tape.backward(grads={a: wp.array(s, dtype=a.dtype) for a, s in zip(target, seeds)})
    for index, source in enumerate(sources):
        base = source.numpy().copy()
        direction = rng.normal(size=base.shape) * np.maximum(
            np.abs(base), 0.1 if index in [1, 2, 5, 6] else 1e-12
        )
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            trials = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                trials.append([a.numpy() for a in outputs(interface_solver(*sources))])
            fd = sum(np.sum((b - a) * s) for a, b, s in zip(*trials, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-8, err_msg=str(index))
        source.assign(base)


@pytest.mark.parametrize("interface_solver", [solve, solve_compensated])
@pytest.mark.parametrize("ratio", [1.0, 1e-4, 1e-8])
@pytest.mark.parametrize("mode", ["compression", "tension"])
def test_closed_area_limit_retains_displacement_and_pressure_derivatives(interface_solver, ratio, mode):
    area = 4e-6 * ratio
    free = -2e-5 if mode == "compression" else 0.4 * (2 * 2e-5**2 / 1e-4) * area * 5e4
    values = closed_pair(area, [free, 0.0, 0.0])
    sources = arrays(values)
    with wp.Tape() as tape:
        result = interface_solver(*sources)
    coefficient = values[8][0] / values[9][0] ** 2
    pressure = coefficient * free / (2 * area)
    np.testing.assert_array_equal(result.gap.numpy(), 0.0)
    np.testing.assert_allclose(result.displacement.numpy()[:, 0], [free / 2, -free / 2], rtol=1e-14)
    np.testing.assert_allclose(result.traction.numpy()[0, 0], pressure, rtol=1e-14)
    tape.backward(grads={result.displacement: wp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=wp.vec3d)})
    np.testing.assert_array_equal(sources[7].grad.numpy(), 0.0)
    np.testing.assert_array_equal(sources[8].grad.numpy(), 0.0)
    np.testing.assert_array_equal(sources[9].grad.numpy(), 0.0)
    np.testing.assert_allclose(sources[0].grad.numpy()[0, 0], 0.5, rtol=1e-14)
    tape.zero()
    tape.backward(grads={result.traction: wp.array([[area, 0.0, 0.0]], dtype=wp.vec3d)})
    np.testing.assert_allclose(sources[7].grad.numpy()[0], -pressure, rtol=1e-14)
    np.testing.assert_allclose(sources[8].grad.numpy()[0], free / (2 * values[9][0] ** 2), rtol=1e-14)
    np.testing.assert_allclose(sources[9].grad.numpy()[0], -coefficient * free / values[9][0], rtol=1e-14)


@pytest.mark.parametrize("interface_solver", [solve, solve_compensated])
def test_closed_branch_keeps_one_sided_damage_memory_derivative(interface_solver):
    values = closed_pair(4e-6, [-2e-5, 6.4e-7, 0.0])
    sources = arrays(values)
    with wp.Tape() as tape:
        result = interface_solver(*sources)
    tape.backward(grads={result.displacement: wp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=wp.vec3d)})
    derivative = sources[3].grad.numpy()[0]
    np.testing.assert_allclose(derivative, -0.2, rtol=1e-12)
    for epsilon in [1e-11, 1e-12]:
        sources[3].assign(np.array([epsilon]))
        trial = interface_solver(*sources)
        difference = (trial.displacement.numpy()[0, 1] - result.displacement.numpy()[0, 1]) / epsilon
        np.testing.assert_allclose(derivative, difference, rtol=2e-5, atol=1e-9)


def test_closed_moving_interface_history_derivative():
    rng = np.random.default_rng(411)
    values = fixture("closed")
    sources = arrays(values)
    with wp.Tape() as tape:
        result = solve_compensated(*sources)
    target = outputs(result)
    scales = [0.0028, 5e4, 0.0028, 70.0, 70.0, 1.0, 70.0, 0.0028, 0.0028]
    seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(target, scales)]
    tape.backward(grads={a: wp.array(seed, dtype=a.dtype) for a, seed in zip(target, seeds)})
    direction = rng.uniform(0.5, 1.0, 3) * 1e-6
    derivative = np.dot(sources[3].grad.numpy(), direction)
    errors = []
    for epsilon in [1e-5, 1e-6, 1e-7]:
        sources[3].assign(epsilon * direction)
        trial = solve_compensated(*sources)
        difference = (
            sum(np.sum(seed * (b.numpy() - a.numpy())) for a, b, seed in zip(target, outputs(trial), seeds))
            / epsilon
        )
        errors.append(abs(derivative - difference) / max(abs(derivative), abs(difference), 1e-12))
    assert sum(error <= 2e-5 for error in errors) >= 2, (derivative, errors)


def test_compensated_geometry_advection_and_snapshot():
    rng = np.random.default_rng(872)
    values = fixture("opened")
    sources = arrays(values)
    velocity = wp.array(rng.normal(scale=0.03, size=(8, 3)), dtype=wp.vec3d, requires_grad=True)
    geometry = InterfaceGeometry(sources[0], sources[1])
    B, G, duration = sources[5], sources[6], sources[9]
    with wp.Tape() as tape:
        result = geometry.advance(B, G, velocity, duration)
    low = np.longdouble
    expected_gap = values[0].astype(low) + low(values[9][0]) * (
        values[5].astype(low) @ velocity.numpy().astype(low)
    )
    expected_frame = values[1].astype(low) + low(values[9][0]) * np.einsum(
        "pa,qpb->qab", velocity.numpy().astype(low), values[6].astype(low)
    )
    np.testing.assert_array_equal(result.gap.numpy().astype(low) + result.gap_remainder, expected_gap)
    np.testing.assert_array_equal(result.frame.numpy().astype(low) + result.frame_remainder, expected_frame)
    assert np.any(result.gap_remainder != 0) and np.any(result.frame_remainder != 0)
    restored = InterfaceGeometry.restore(result.snapshot(), device=geometry.gap.device)
    for key, value in result.snapshot().items():
        np.testing.assert_array_equal(restored.snapshot()[key], value)
    seed_gap = rng.normal(size=(3, 3)) / 0.0028
    seed_frame = rng.normal(size=(3, 3, 3))
    tape.backward(
        grads={
            result.gap: wp.array(seed_gap, dtype=wp.vec3d),
            result.frame: wp.array(seed_frame, dtype=wp.mat33d),
        }
    )
    for source in [geometry.gap, geometry.frame, B, G, velocity, duration]:
        base = source.numpy()
        direction = rng.normal(size=base.shape) * np.maximum(np.abs(base), 1e-5)
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            trials = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                trial = geometry.advance(B, G, velocity, duration)
                trials.append(
                    (
                        trial.gap.numpy().astype(low) + trial.gap_remainder,
                        trial.frame.numpy().astype(low) + trial.frame_remainder,
                    )
                )
            fd = (
                np.sum((trials[1][0] - trials[0][0]) * seed_gap)
                + np.sum((trials[1][1] - trials[0][1]) * seed_frame)
            ) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-10)
        source.assign(base)


@pytest.mark.parametrize("interface_solver", [solve, solve_compensated])
@pytest.mark.parametrize("fraction", [0.6, 0.9999])
@pytest.mark.parametrize("history", [1e-10, 1e-12])
def test_small_unloading_history_matches_series_spring_solution(interface_solver, fraction, history):
    values = closed_pair(4e-6, [1e-6, 0.0, 0.0])
    values[3][:] = history
    split = [np.repeat(value, 2, axis=0) for value in values[:8]] + values[8:]
    split[7] *= np.array([fraction, 1 - fraction])
    sources = arrays(split)
    with wp.Tape() as tape:
        result = interface_solver(*sources)
    strength, gc = values[4][0, :2]
    failure = 2 * gc / strength
    stiffness = strength * (1 / history - 1 / failure)
    mobility = 2 * values[7][0] * values[9][0] ** 2 / values[8][0]
    denominator = 1 + mobility * stiffness
    opening = values[0][0, 0] / denominator
    traction = stiffness * opening
    np.testing.assert_allclose(result.gap.numpy()[:, 0], opening, rtol=2e-8, atol=1e-20)
    np.testing.assert_allclose(result.traction.numpy()[:, 0], traction, rtol=2e-12)
    np.testing.assert_allclose(
        result.displacement.numpy()[:, 0],
        (values[0][0, 0] - opening) * np.array([0.5, -0.5]),
        rtol=2e-14,
    )
    tape.backward(grads={result.displacement: wp.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3d)})
    expected = values[0][0, 0] * mobility * strength / (history * denominator) ** 2
    np.testing.assert_allclose(sources[3].grad.numpy().sum(), expected, rtol=2e-9)


@wp.kernel
def interface_velocity(
    velocity: wp.array(dtype=wp.vec3d),
    displacement: wp.array(dtype=wp.vec3d),
    duration: wp.array(dtype=wp.float64),
    result: wp.array(dtype=wp.vec3d),
):
    p = wp.tid()
    result[p] = velocity[p] + displacement[p] / duration[0]


@pytest.mark.parametrize("segment_steps", [3, 4])
def test_compensated_checkpoint_matches_complete_tape(segment_steps):
    rng = np.random.default_rng(23)
    source = arrays(fixture("opened"))
    initial_velocity = wp.array(rng.normal(scale=0.02, size=(8, 3)), dtype=wp.vec3d, requires_grad=True)
    initial = (InterfaceGeometry(source[0], source[1]), source[3], initial_velocity)
    parameters = [source[2], *source[4:]]

    def advance(state):
        geometry, history, velocity = state
        free = geometry.advance(source[5], source[6], velocity, source[9])
        result, updated = solve_geometry(free, source[2], history, *source[4:])
        next_velocity = wp.zeros_like(velocity, requires_grad=True)
        wp.launch(
            interface_velocity,
            len(velocity),
            inputs=[velocity, result.displacement, source[9]],
            outputs=[next_velocity],
        )
        return updated, result.history, next_velocity

    def state_arrays(state):
        return [state[0].gap, state[0].frame, state[1], state[2]]

    def snapshot(state):
        return state[0].snapshot(), state[1].numpy().copy(), state[2].numpy().copy()

    def restore(saved):
        return (
            InterfaceGeometry.restore(saved[0], device=source[0].device),
            wp.array(saved[1], dtype=wp.float64, requires_grad=True),
            wp.array(saved[2], dtype=wp.vec3d, requires_grad=True),
        )

    checkpoints = [(0, snapshot(initial))]
    with wp.Tape() as tape:
        state = initial
        for step in range(11):
            state = advance(state)
            if (step + 1) % segment_steps == 0 or step == 10:
                checkpoints.append((step + 1, snapshot(state)))
    target = state_arrays(state)
    seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(target, [0.0028, 1.0, 0.0028, 2.0])]
    tape.backward(grads={a: wp.array(seed, dtype=a.dtype) for a, seed in zip(target, seeds)})
    expected_initial = [a.grad.numpy().copy() for a in state_arrays(initial)]
    expected_parameters = [a.grad.numpy().copy() for a in parameters]
    tape.zero()
    parameter_gradients = [np.zeros_like(value) for value in expected_parameters]
    for (first, saved), (last, expected) in reversed(list(zip(checkpoints[:-1], checkpoints[1:]))):
        start = restore(saved)
        for parameter in parameters:
            parameter.grad.zero_()
        with wp.Tape() as local:
            state = start
            for _ in range(first, last):
                state = advance(state)
        actual = snapshot(state)
        for key in expected[0]:
            np.testing.assert_array_equal(actual[0][key], expected[0][key])
        for a, b in zip(actual[1:], expected[1:]):
            np.testing.assert_array_equal(a, b)
        local.backward(
            grads={a: wp.array(seed, dtype=a.dtype) for a, seed in zip(state_arrays(state), seeds)}
        )
        seeds = [a.grad.numpy().copy() for a in state_arrays(start)]
        for accumulated, parameter in zip(parameter_gradients, parameters):
            accumulated += parameter.grad.numpy()
        local.zero()
    for actual, expected in zip(seeds + parameter_gradients, expected_initial + expected_parameters):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
