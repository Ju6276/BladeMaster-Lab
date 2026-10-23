"""Polynomial completeness and derivatives of moving, off-grid surface traces."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.cohesive_geometry import InterfaceGeometry
from blademaster_warp.cohesive_grid_update import advance_material
from blademaster_warp.reference_cut import quadrature
from blademaster_warp.reference_grid import build_trace
from blademaster_warp.solver import Config


def setup(offset, normal):
    cfg = Config(
        size=(0.012, 0.012, 0.012),
        grid_spacing=0.006,
        dt=2e-6,
        young_modulus=20000.0,
        yield_stress=0.0,
        external_tool=True,
    )
    model = MaterialTrajectory(cfg, precision="float64", ordered_transfers=True)
    state = model.state()
    reference = model.array(model.reference, wp.vec3)
    half = model.array(np.full((model.n, 3), cfg.grid_spacing / 4), wp.vec3)
    point = model.array([[offset, 0.0, 0.006]], wp.vec3)
    normal = model.array([normal], wp.vec3)

    def evaluate():
        surface = quadrature(state.x, state.F, reference, half, point, normal)
        codes = np.where((model.reference - point.numpy()[0]) @ normal.numpy()[0] < 0.0, 1, 2)
        topology = model.topology(codes)
        result = build_trace(
            model, topology, surface.points, surface.normals, np.tile([1, 2], (len(surface.area), 1))
        )
        return surface, result, topology

    return model, point, normal, evaluate


def nodal_coordinates(model, topology):
    cells = np.ceil(np.asarray(model.config.size) / model.config.grid_spacing).astype(int)
    index = np.array(np.unravel_index(topology.keys.numpy() % model.node_count, tuple(cells + 1))).T
    origin = -np.array([model.config.size[0] / 2, model.config.size[1] / 2, 0.0])
    return origin + index * model.config.grid_spacing


def polynomial(coordinates, spacing):
    x, y, z = (coordinates / spacing).T
    value = 1 + 0.2 * x - 0.3 * y + 0.4 * z + 0.3 * x * y - 0.2 * y * z + 0.1 * x * y * z
    gradient = (
        np.stack(
            [
                0.2 + 0.3 * y + 0.1 * y * z,
                -0.3 + 0.3 * x - 0.2 * z + 0.1 * x * z,
                0.4 - 0.2 * y + 0.1 * x * y,
            ],
            axis=1,
        )
        / spacing
    )
    return value, gradient


@pytest.mark.parametrize(
    "offset,normal",
    [
        (0.0, [1.0, 0.0, 0.0]),
        (0.001, [1.0, 0.0, 0.0]),
        (0.0007, [1.0, 0.2, 0.17]),
        (-0.0007, [-1.0, 0.3, -0.2]),
    ],
)
def test_two_sided_traces_reproduce_trilinear_polynomials(offset, normal):
    model, _, _, evaluate = setup(offset, normal)
    surface, trace, topology = evaluate()
    nodal, _ = polynomial(nodal_coordinates(model, topology), model.config.grid_spacing)
    expected, gradient = polynomial(surface.points.numpy(), model.config.grid_spacing)
    weights, derivatives = trace.weights.numpy(), trace.gradients.numpy()
    np.testing.assert_allclose(weights.sum(axis=2), 1.0, atol=2e-15)
    np.testing.assert_allclose(derivatives.sum(axis=2), 0.0, atol=2e-13)
    for side in [0, 1]:
        np.testing.assert_allclose(weights[:, side] @ nodal, expected, rtol=2e-14, atol=2e-15)
        np.testing.assert_allclose(
            np.einsum("qna,n->qa", derivatives[:, side], nodal), gradient, rtol=2e-14, atol=2e-13
        )
    np.testing.assert_allclose(trace.separation_map.numpy() @ nodal, 0.0, atol=2e-15)
    normal = surface.normals.numpy()
    tangential = gradient - np.sum(gradient * normal, axis=1)[:, None] * normal
    np.testing.assert_allclose(
        np.einsum("qna,n->qa", trace.frame_map.numpy(), nodal), tangential, rtol=2e-14, atol=2e-13
    )


def test_cut_location_and_normal_derivatives_through_extended_traces():
    model, point, normal, evaluate = setup(0.0007, [1.0, 0.2, 0.17])
    _, _, topology = evaluate()
    velocity = np.random.default_rng(871).normal(size=(topology.count, 3))
    spacing = model.config.grid_spacing
    area_scale = 0.012**2

    def measure(surface, trace):
        gap = trace.separation_map.numpy() @ velocity
        rate = np.einsum("na,qnb->qab", velocity, trace.frame_map.numpy())
        density = 0.5 * np.sum(gap**2, axis=1) + 0.5 * spacing**2 * np.sum(rate**2, axis=(1, 2))
        return density, gap, rate

    with wp.Tape() as tape:
        surface, trace, _ = evaluate()
    density, gap, rate = measure(surface, trace)
    area = surface.area.numpy() / area_scale
    tape.backward(
        grads={
            surface.area: model.array(density / area_scale, float, False),
            trace.separation_map: model.array(area[:, None] * (gap @ velocity.T), float, False),
            trace.frame_map: model.array(
                area[:, None, None] * spacing**2 * np.einsum("na,qab->qnb", velocity, rate), wp.vec3, False
            ),
        }
    )
    for source, direction in [(point, [[0.001, -0.0002, 0.0003]]), (normal, [[0.12, -0.18, 0.15]])]:
        base = source.numpy().copy()
        direction = np.asarray(direction)
        ad = float(np.sum(source.grad.numpy() * direction))
        for epsilon in [1e-4, 1e-5]:
            losses = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                perturbed, mapped, trial_topology = evaluate()
                np.testing.assert_array_equal(trial_topology.keys.numpy(), topology.keys.numpy())
                losses.append(perturbed.area.numpy() @ measure(perturbed, mapped)[0] / area_scale)
            np.testing.assert_allclose(ad, (losses[1] - losses[0]) / (2 * epsilon), rtol=2e-6, atol=1e-8)
        source.assign(base)


def test_moving_plane_derivatives_through_coupled_material_steps():
    model, point, normal, evaluate = setup(0.0007, [1.0, 0.2, 0.17])
    surface, _, topology = evaluate()
    count = len(surface.area)
    codes = topology.keys.numpy()[topology.slots.numpy()[:, 0]] // model.node_count
    side = np.where(codes == 1, -1.0, 1.0)[:, None]
    gap = np.array([0.00015, 0.0002, 0.0001])
    initial = model.state(x=model.reference + side * gap / 2, v=side * np.array([0.3, 0.2, -0.1]))
    geometry = InterfaceGeometry(
        model.array(np.tile(gap, (count, 1)), wp.vec3),
        model.array(np.tile(np.eye(3), (count, 1, 1)), wp.mat33),
    )
    history = model.array(np.full(count, 1.2 * np.linalg.norm(gap)), float)
    material = model.array(np.tile([5e4, 70.0, 0.0], (count, 1)), wp.vec3)

    def run():
        cut, trace, grid = evaluate()
        assert len(cut.area) == count
        state, interface, maximum = initial, geometry, history
        for _ in range(4):
            result = advance_material(
                model,
                state,
                interface,
                maximum,
                cut.normals,
                material,
                trace.separation_map,
                trace.frame_map,
                cut.area,
                topology=grid,
                gravity=(0.0, 0.0, 0.0),
            )
            state, interface, maximum = result.state, result.geometry, result.interface.history
        return state.v, interface.gap, interface.frame

    rng = np.random.default_rng(948)
    with wp.Tape() as tape:
        outputs = run()
    seeds = [rng.normal(size=a.numpy().shape) / len(a) for a in outputs]
    seeds[1] /= model.config.grid_spacing
    tape.backward(
        grads={a: wp.array(seed, dtype=a.dtype, device=a.device) for a, seed in zip(outputs, seeds)}
    )
    for source, direction in [(point, [[0.001, -0.0002, 0.0003]]), (normal, [[0.12, -0.18, 0.15]])]:
        base = source.numpy().copy()
        direction = np.asarray(direction)
        ad = float(np.sum(source.grad.numpy() * direction))
        for epsilon in [1e-4, 1e-5]:
            losses = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                losses.append(sum(float(np.sum(a.numpy() * seed)) for a, seed in zip(run(), seeds)))
            np.testing.assert_allclose(ad, (losses[1] - losses[0]) / (2 * epsilon), rtol=2e-5, atol=1e-9)
        source.assign(base)
