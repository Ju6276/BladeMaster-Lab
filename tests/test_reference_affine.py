"""Affine completeness, transfer conservation, virtual power and moving-cut gradients."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.geometry import grid_cells
from blademaster_warp.reference_affine import build_transfer
from blademaster_warp.reference_cut import quadrature
from blademaster_warp.reference_grid import build_trace
from blademaster_warp.solver import Config


def setup(spacing=0.006, shape="block", seed=0):
    cfg = Config(size=(0.012, 0.012, 0.012), grid_spacing=spacing, shape=shape, sampling_seed=seed)
    model = MaterialTrajectory(cfg, precision="float64", ordered_transfers=True)
    codes = 1 + (model.reference[:, 0] > 0.0007) + 3 * (1 + (model.reference[:, 1] > 0.0003))
    codes[model.reference[:, 2] < 0.003] = 0
    grid = model.topology(codes)
    transfer = build_transfer(model, grid)
    cells = grid_cells(cfg.size, spacing)
    indices = np.array(np.unravel_index(grid.keys.numpy() % model.node_count, tuple(cells + 1))).T
    coordinates = indices * spacing - [0.006, 0.006, 0.0]
    return model, grid, transfer, coordinates


@pytest.mark.parametrize("spacing", [0.006, 0.004])
@pytest.mark.parametrize("shape,seed", [("block", 0), ("block", 1), ("cylinder", 0)])
def test_affine_fields_and_transfer_conservation(spacing, shape, seed):
    m, grid, transfer, coordinates = setup(spacing, shape, seed)
    A = np.array([[0.3, -50.0, -20.0], [50.0, -0.2, -10.0], [20.0, 10.0, 0.1]])
    v = m.reference @ A.T + [0.1, 0.2, 0.3]
    moment = transfer.pack(m.array(v, wp.vec3), m.array(np.tile(A, (m.n, 1, 1)), wp.mat33))
    nodal = transfer.grid_map.numpy() @ moment.numpy()
    np.testing.assert_allclose(nodal, coordinates @ A.T + [0.1, 0.2, 0.3], rtol=2e-14, atol=1e-14)
    flow, rate = transfer.kinematics(moment)
    np.testing.assert_allclose(flow.numpy(), v, rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(rate.numpy(), np.tile(A, (m.n, 1, 1)), rtol=1e-11, atol=2e-12)

    rng = np.random.default_rng(178)
    v = rng.normal(size=(m.n, 3))
    A = rng.normal(size=(m.n, 3, 3)) / spacing
    moment = transfer.pack(m.array(v, wp.vec3), m.array(A, wp.mat33))
    nodal = transfer.grid_map.numpy() @ moment.numpy()
    masses = transfer.grid_mass[:, None]
    np.testing.assert_allclose(np.sum(masses * nodal, axis=0), m.mass * v.sum(axis=0), atol=1e-17)
    offset = coordinates[grid.slots.numpy()] - m.reference[:, None]
    local_velocity = np.einsum("pab,pib->pia", A, offset)
    affine_angular = np.sum(m.weights.numpy()[:, :, None] * np.cross(offset, local_velocity), axis=(0, 1))
    expected = m.mass * (np.cross(m.reference, v).sum(axis=0) + affine_angular)
    np.testing.assert_allclose(np.sum(masses * np.cross(coordinates, nodal), axis=0), expected, atol=1e-19)
    before = np.sum(transfer.kinetic_energy(moment).numpy())
    after = 0.5 * np.sum(masses * nodal**2)
    assert after <= before * (1 + 1e-14)


def test_pack_inverse_and_kinetic_energy_gradient():
    m, _, transfer, _ = setup(seed=1)
    rng = np.random.default_rng(145)
    v = m.array(rng.normal(size=(m.n, 3)), wp.vec3)
    A = m.array(rng.normal(size=(m.n, 3, 3)) * 100, wp.mat33)
    with wp.Tape() as tape:
        moments = transfer.pack(v, A)
        restored_v, restored_A = transfer.unpack(moments)
        energy = transfer.kinetic_energy(moments)
    np.testing.assert_array_equal(restored_v.numpy(), v.numpy())
    np.testing.assert_allclose(restored_A.numpy(), A.numpy(), rtol=2e-14, atol=3e-14)
    tape.backward(grads={energy: m.array(np.ones(m.n), float, False)})
    root = transfer.root.numpy()
    covariance = root @ root.transpose(0, 2, 1)
    np.testing.assert_allclose(v.grad.numpy(), m.mass * v.numpy(), rtol=2e-14, atol=1e-18)
    np.testing.assert_allclose(A.grad.numpy(), m.mass * A.numpy() @ covariance, rtol=2e-13, atol=1e-20)
    tape.zero()
    seed_v = rng.normal(size=(m.n, 3))
    seed_A = rng.normal(size=(m.n, 3, 3))
    tape.backward(
        grads={restored_v: m.array(seed_v, wp.vec3, False), restored_A: m.array(seed_A, wp.mat33, False)}
    )
    np.testing.assert_allclose(v.grad.numpy(), seed_v, rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(A.grad.numpy(), seed_A, rtol=2e-14, atol=1e-14)


def test_force_and_kinematics_share_virtual_power_and_adjoint():
    m, _, transfer, _ = setup(seed=1)
    rng = np.random.default_rng(991)
    z = m.array(rng.normal(size=(4 * m.n, 3)), wp.vec3)
    force = m.array(rng.normal(size=(m.n, 3)), wp.vec3)
    moment = m.array(rng.normal(size=(m.n, 3, 3)) * 0.006, wp.mat33)
    with wp.Tape() as tape:
        v, rate = transfer.kinematics(z)
        conjugate = transfer.pullback_force(force, moment)
    work = np.sum(force.numpy() * v.numpy()) + np.sum(moment.numpy() * rate.numpy())
    np.testing.assert_allclose(np.sum(conjugate.numpy() * z.numpy()), work, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(conjugate.numpy()[::4].sum(axis=0), force.numpy().sum(axis=0), atol=1e-13)
    tape.backward(grads={v: force, rate: moment})
    np.testing.assert_allclose(z.grad.numpy(), conjugate.numpy(), rtol=2e-13, atol=1e-13)
    tape.zero()
    tape.backward(grads={conjugate: z})
    np.testing.assert_allclose(force.grad.numpy(), v.numpy(), rtol=2e-13, atol=1e-13)
    np.testing.assert_allclose(moment.grad.numpy(), rate.numpy(), rtol=2e-13, atol=1e-12)


@pytest.mark.parametrize("spacing", [0.006, 0.004])
def test_cut_geometry_and_affine_velocity_gradients(spacing):
    m, _, _, _ = setup(spacing)
    rng = np.random.default_rng(119)
    state = m.state()
    reference = m.array(m.reference, wp.vec3)
    half = m.array(np.full((m.n, 3), spacing / 4), wp.vec3)
    point = m.array([[0.0007, 0.0, 0.006]], wp.vec3)
    normal = m.array([[1.0, 0.2, 0.17]], wp.vec3)
    velocity = m.array(rng.normal(size=(m.n, 3)), wp.vec3)
    gradient = m.array(rng.normal(size=(m.n, 3, 3)) * 10, wp.mat33)

    def run():
        surface = quadrature(state.x, state.F, reference, half, point, normal)
        codes = np.where((m.reference - point.numpy()[0]) @ normal.numpy()[0] < 0, 1, 2)
        grid = m.topology(codes)
        transfer = build_transfer(m, grid)
        trace = build_trace(m, grid, surface.points, surface.normals, np.tile([1, 2], (len(surface.area), 1)))
        B, G = transfer.compose_trace(trace)
        z = transfer.pack(velocity, gradient)
        return surface.area, B, G, z

    def objective(outputs):
        area, B, G, z = [a.numpy() for a in outputs]
        gap = B @ z
        rate = np.einsum("pa,qpb->qab", z, G)
        density = 0.5 * np.sum(gap**2, axis=1) + 0.5 * spacing**2 * np.sum(rate**2, axis=(1, 2))
        return area @ density / 0.012**2, (area, gap, rate, density)

    with wp.Tape() as tape:
        outputs = run()
    _, (area, gap, rate, density) = objective(outputs)
    area = area / 0.012**2
    z = outputs[3].numpy()
    B, G = outputs[1].numpy(), outputs[2].numpy()
    seeds = [
        density / 0.012**2,
        area[:, None] * (gap @ z.T),
        area[:, None, None] * spacing**2 * np.einsum("pa,qab->qpb", z, rate),
        B.T @ (area[:, None] * gap) + spacing**2 * np.einsum("q,qab,qpb->pa", area, rate, G),
    ]
    tape.backward(grads={a: m.array(s, a.dtype, False) for a, s in zip(outputs, seeds)})
    for source, scale in [(point, 0.001), (normal, 0.1), (velocity, 0.1), (gradient, 10.0)]:
        base = source.numpy().copy()
        direction = rng.normal(size=base.shape) * scale
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            losses = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                losses.append(objective(run())[0])
            np.testing.assert_allclose(ad, (losses[1] - losses[0]) / (2 * epsilon), rtol=2e-6, atol=1e-8)
        source.assign(base)


def test_empty_trace_and_shape_validation():
    m, grid, transfer, _ = setup()
    trace = SimpleNamespace(
        separation_map=m.zeros((0, grid.count), float), frame_map=m.zeros((0, grid.count), wp.vec3)
    )
    B, G = transfer.compose_trace(trace)
    assert B.shape == G.shape == (0, 4 * m.n)
    with pytest.raises(ValueError, match="matching float64"):
        transfer.pack(m.zeros(4 * m.n, wp.vec3), m.zeros(m.n, wp.mat33))


@pytest.mark.parametrize("spacing", [0.006, 0.004])
def test_shared_fields_have_exactly_zero_separation_and_cut_location_derivative(spacing):
    m, _, _, _ = setup(spacing, seed=1)
    codes = np.zeros(m.n, dtype=int)
    codes[:3] = 1
    grid = m.topology(codes)
    transfer = build_transfer(m, grid)
    points = m.array([[0.0007, -0.002, 0.003], [0.0007, 0.002, 0.009]], wp.vec3)
    normal = m.array([[1.0, 0.2, 0.17]] * 2, wp.vec3)
    with wp.Tape() as tape:
        trace = build_trace(
            m,
            grid,
            points,
            normal,
            np.tile([1, 2], (2, 1)),
            equivalent_slots=transfer.equivalent_slots,
        )
        separation, _ = transfer.compose_trace(trace)
    np.testing.assert_array_equal(separation.numpy(), 0.0)
    seed = m.array(np.random.default_rng(832).normal(size=separation.shape), float, False)
    tape.backward(grads={separation: seed})
    np.testing.assert_array_equal(points.grad.numpy(), 0.0)
    np.testing.assert_array_equal(normal.grad.numpy(), 0.0)

    rng = np.random.default_rng(283)
    velocity = rng.normal(size=(m.n, 3))
    moment = transfer.pack(m.array(velocity, wp.vec3), m.zeros(m.n, wp.mat33))
    momentum = np.zeros((grid.count, 3))
    for p in range(m.n):
        for slot, weight in zip(grid.slots.numpy()[p], m.weights.numpy()[p]):
            momentum[slot] += m.mass * weight * velocity[p]
    expected = m.couple(grid, m.array(momentum, wp.vec3)).numpy() / transfer.grid_mass[:, None]
    np.testing.assert_allclose(transfer.grid_map.numpy() @ moment.numpy(), expected, rtol=2e-13, atol=1e-14)
