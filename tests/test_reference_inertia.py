"""Reference inertia action and cohesive equilibria with an independent dense metric."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from blademaster_warp.reference_inertia import ReferenceInertia
from blademaster_warp.differentiable_cohesive import solve, solve_force
from blademaster_warp.cohesive_geometry import InterfaceGeometry, solve_geometry
from test_differentiable_cohesive import arrays, fixture, outputs


def transfer(size, device):
    kinds = {float: wp.float64, wp.vec3: wp.vec3d}
    model = SimpleNamespace(
        mass=1e-4,
        device=device,
        array=lambda data, kind, requires_grad=True: wp.array(
            data, dtype=kinds.get(kind, kind), device=device, requires_grad=requires_grad
        ),
        zeros=lambda count, kind: wp.zeros(
            count, dtype=kinds.get(kind, kind), device=device, requires_grad=True
        ),
    )
    rng = np.random.default_rng(170)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(size, size)))
    mapping = np.linspace(0.2, 0.9, size)[:, None] * orthogonal.T

    def check(a, kind, shape):
        assert a.dtype == kind and a.shape == shape and a.device == device

    return SimpleNamespace(
        model=model,
        grid_map=model.array(mapping, float, False),
        grid_mass=np.full(size, model.mass),
        _check=check,
    )


@pytest.mark.parametrize("theta", [0.0, 0.1, 20.0])
def test_action_inverse_and_parameter_derivatives(theta):
    device = wp.get_device()
    tr = transfer(8, device)
    m = tr.model
    t, eta = m.array([1e-4], float), m.array([theta / 1e-4], float)
    inertia = ReferenceInertia(tr, t, eta)
    P = tr.grid_map.numpy()
    L = np.eye(8) - P.T @ P
    D = np.eye(8) + theta * L
    rng = np.random.default_rng(71)
    value = m.array(rng.normal(size=(8, 3)), wp.vec3)
    original = m.array(rng.normal(size=(8, 3)), wp.vec3)
    with wp.Tape() as tape:
        result = inertia.add_momentum(original, value)
    np.testing.assert_allclose(
        result.numpy(), original.numpy() + theta * L @ value.numpy(), rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        inertia.solve(value.numpy()), np.linalg.solve(D, value.numpy()), rtol=2e-13, atol=2e-13
    )
    seed = rng.normal(size=(8, 3))
    tape.backward(grads={result: m.array(seed, wp.vec3, False)})
    np.testing.assert_allclose(value.grad.numpy(), theta * L @ seed, rtol=2e-13, atol=2e-13)
    power = np.sum(seed * (L @ value.numpy()))
    np.testing.assert_allclose(t.grad.numpy()[0], eta.numpy()[0] * power, rtol=2e-13, atol=1e-13)
    np.testing.assert_allclose(eta.grad.numpy()[0], t.numpy()[0] * power, rtol=2e-13, atol=1e-13)


@pytest.mark.parametrize("duration,rate", [(0.0, 2000.0), (1e-4, 0.0), (1e-4, 2000.0), (1e-4, 2e5)])
def test_inverse_action_pullback_including_zero_duration_and_rate(duration, rate):
    tr = transfer(8, wp.get_device())
    m = tr.model
    t, eta = m.array([duration], float), m.array([rate], float)
    inertia = ReferenceInertia(tr, t, eta)
    rng = np.random.default_rng(771)
    value = m.array(rng.normal(size=(8, 3)), wp.vec3)
    with wp.Tape() as tape:
        result = inertia.solve_array(value)
    P = tr.grid_map.numpy()
    L = np.eye(8) - P.T @ P
    D = np.eye(8) + duration * rate * L
    expected = np.linalg.solve(D, value.numpy())
    np.testing.assert_allclose(result.numpy(), expected, rtol=2e-13, atol=2e-13)
    seed = rng.normal(size=(8, 3))
    tape.backward(grads={result: m.array(seed, wp.vec3, False)})
    adjoint = np.linalg.solve(D, seed)
    power = np.sum(adjoint * (L @ expected))
    np.testing.assert_allclose(value.grad.numpy(), adjoint, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(t.grad.numpy()[0], -rate * power, rtol=2e-13, atol=1e-10)
    np.testing.assert_allclose(eta.grad.numpy()[0], -duration * power, rtol=2e-13, atol=1e-13)


@pytest.mark.parametrize("scenario", ["closed", "opening", "mixed", "opened"])
@pytest.mark.parametrize("integrated", [False, True])
@pytest.mark.parametrize("compensated", [False, True])
def test_interface_metric_equilibrium_and_all_input_gradients(scenario, integrated, compensated):
    values = fixture(scenario)
    sources = arrays(values)
    tr = transfer(values[5].shape[1], sources[0].device)
    rate = tr.model.array([2000.0], float)
    inertia = ReferenceInertia(tr, sources[9], rate)
    interface = solve_force if integrated else solve

    def run():
        if compensated:
            return solve_geometry(
                InterfaceGeometry(*sources[:2]), *sources[2:], integrated_force=integrated, inertia=inertia
            )[0]
        return interface(*sources, inertia=inertia)

    with wp.Tape() as tape:
        result = run()
    P = tr.grid_map.numpy()
    L = np.eye(P.shape[1]) - P.T @ P
    D = np.eye(len(L)) + values[9][0] * rate.numpy()[0] * L
    u = result.displacement.numpy()
    measure = np.ones_like(values[7]) if integrated else values[7]
    force = values[5].T @ (measure[:, None] * outputs(result)[1].numpy())
    force += np.einsum("q,qab,qpb->pa", measure, outputs(result)[6].numpy(), values[6])
    np.testing.assert_allclose(D @ u + values[9][0] ** 2 / values[8][0] * force, 0.0, atol=2e-17)
    eigen, vectors = np.linalg.eigh(D)
    S = (vectors / np.sqrt(eigen)) @ vectors.T
    transformed = [v.copy() for v in values]
    transformed[5] = values[5] @ S
    transformed[6] = np.einsum("qic,ij->qjc", values[6], S)
    other = interface(*arrays(transformed))
    np.testing.assert_allclose(u, S @ other.displacement.numpy(), rtol=2e-9, atol=2e-17)
    np.testing.assert_allclose(outputs(result)[1].numpy(), outputs(other)[1].numpy(), rtol=2e-9, atol=1e-10)
    rng = np.random.default_rng(819)
    target = outputs(result)
    scales = [max(np.max(np.abs(a.numpy())), 1e-5 if a.dtype == wp.vec3d else 1.0) for a in target]
    seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(target, scales)]
    tape.backward(grads={a: tr.model.array(seed, a.dtype, False) for a, seed in zip(target, seeds)})
    gradients = [a.grad.numpy().copy() for a in [*sources, rate]]
    for index, (source, gradient) in enumerate(zip([*sources, rate], gradients)):
        base = source.numpy().copy()
        direction = rng.normal(size=base.shape) * np.maximum(np.abs(base), 0.1 if index in [1, 2] else 1e-12)
        if index == 3:
            direction *= base > 0
        if not np.any(direction):
            continue
        ad = float(np.sum(gradient * direction))
        error = []
        for epsilon in [1e-4, 5e-5]:
            sampled = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                sampled.append([a.numpy() for a in outputs(run())])
            source.assign(base)
            fd = sum(np.sum((b - a) * s) for a, b, s in zip(*sampled, seeds)) / (2 * epsilon)
            error.append(abs(ad - fd))
        assert min(error) <= 2e-5 * abs(ad) + 2e-7, (scenario, integrated, index, ad, error)


def test_compensated_geometry_inertia_matches_direct_force_entry():
    values = fixture("mixed")
    source = arrays(values)
    tr = transfer(values[5].shape[1], source[0].device)
    rate = tr.model.array([2000.0], float)
    inertia = ReferenceInertia(tr, source[9], rate)
    direct = solve_force(*source, inertia=inertia)
    result, _ = solve_geometry(
        InterfaceGeometry(*source[:2]), *source[2:], integrated_force=True, inertia=inertia
    )
    for a, b in zip(outputs(direct), outputs(result)):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("zero_area", [False, True])
def test_repeated_closed_constraints_keep_partition_and_metric_derivatives(zero_area):
    values = fixture("closed")
    repeated = [v.copy() for v in values]
    for i in range(8):
        repeated[i] = np.repeat(repeated[i], 2, axis=0)
    repeated[7] *= 0.5
    if zero_area:
        repeated[7][::2] = 0.0
        repeated[7][1::2] *= 2
    sources = arrays(repeated)
    tr = transfer(values[5].shape[1], sources[0].device)
    rate = tr.model.array([2000.0], float)
    inertia = ReferenceInertia(tr, sources[9], rate)
    with wp.Tape() as tape:
        response = solve_force(*sources, inertia=inertia)
    baseline = solve_force(*arrays(values), inertia=inertia)
    np.testing.assert_allclose(
        response.displacement.numpy(), baseline.displacement.numpy(), rtol=1e-11, atol=1e-18
    )
    np.testing.assert_allclose(
        response.force.numpy().reshape(-1, 2, 3).sum(axis=1), baseline.force.numpy(), rtol=1e-11, atol=1e-12
    )
    if zero_area:
        np.testing.assert_array_equal(response.force.numpy()[::2], 0.0)
    rng = np.random.default_rng(19)
    seed = rng.normal(size=response.force.numpy().shape)
    tape.backward(grads={response.force: tr.model.array(seed, wp.vec3, False)})
    gradient = rate.grad.numpy().copy()
    original = rate.numpy().copy()
    direction = original * 0.1
    samples = []
    for sign in [-1, 1]:
        rate.assign(original + sign * 1e-4 * direction)
        samples.append(solve_force(*sources, inertia=inertia).force.numpy())
    rate.assign(original)
    fd = np.sum((samples[1] - samples[0]) * seed) / 2e-4
    np.testing.assert_allclose(gradient @ direction, fd, rtol=2e-6, atol=1e-10)


@pytest.mark.parametrize("theta", [0.0, 0.04, 20.0])
def test_closed_history_derivative_matches_compliance_limit(theta):
    values = fixture("closed")
    values[:8] = [a[:1].copy() for a in values[:8]]
    values[0] = np.array([[1e-9, 0.0, 0.0]])
    values[1] = np.eye(3)[None]
    values[5] = np.array([[-1.0, 1.0]])
    values[6] = np.zeros((1, 2, 3))
    source = arrays(values)
    tr = transfer(2, source[0].device)
    rate = tr.model.array([theta / values[9][0]], float)
    inertia = ReferenceInertia(tr, source[9], rate)
    with wp.Tape() as tape:
        response = solve_force(*source, inertia=inertia)
    tape.backward(grads={response.force: tr.model.array([[1.0, 0.0, 0.0]], wp.vec3, False)})
    P = tr.grid_map.numpy()
    D = np.eye(2) + theta * (np.eye(2) - P.T @ P)
    mobility = values[9][0] ** 2 / values[8][0] * (values[5] @ np.linalg.solve(D, values[5].T))[0, 0]
    expected = -values[0][0, 0] / (mobility**2 * values[7][0] * values[4][0, 0])
    np.testing.assert_allclose(source[3].grad.numpy()[0], expected, rtol=2e-10, atol=1e-10)
