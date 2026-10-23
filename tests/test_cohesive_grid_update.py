"""Material-field continuity, fracture history and the coupled trajectory VJP."""

from itertools import product

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_geometry import InterfaceGeometry
from blademaster_warp.cohesive_grid_update import advance_material
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.geometry import points, stencil
from blademaster_warp.solver import Config


def patch(gc=70.0):
    cfg = Config(
        size=(0.012, 0.012, 0.012),
        grid_spacing=0.006,
        dt=2e-6,
        young_modulus=20000.0,
        yield_stress=0.0,
        flip_fraction=1.0,
        external_tool=True,
        blade_grid_constraint=False,
    )
    m = MaterialTrajectory(cfg, precision="float64", ordered_transfers=True)
    codes = np.where(m.reference[:, 0] < 0.0, 1, 2)
    grid = m.topology(codes)
    abscissa = np.polynomial.legendre.leggauss(2)[0]
    coordinate = np.concatenate([base + 0.003 * (1 + abscissa) for base in [0.0, 0.006]])
    reference = np.array([[0.0, y - 0.006, z] for y, z in product(coordinate, coordinate)])
    _, origin, cells = points(cfg.size, cfg.grid_spacing, cfg.shape)
    nodes, weights, gradients = stencil(reference, origin, cfg.grid_spacing, cells)
    gradients[:, :, 0] = 0.0
    lookup = {int(key): index for index, key in enumerate(grid.keys.numpy())}
    Q = np.zeros((len(reference), grid.count))
    G = np.zeros((*Q.shape, 3))
    for q in range(len(reference)):
        for code, sign in [(1, -1), (2, 1)]:
            for node, weight, gradient in zip(nodes[q], weights[q], gradients[q]):
                key = int(code * m.node_count + node)
                if key in lookup:
                    Q[q, lookup[key]] += sign * weight
                    G[q, lookup[key]] += 0.5 * gradient
    count = len(reference)
    data = dict(
        topology=grid,
        reference_normal=m.array(np.tile([1.0, 0.0, 0.0], (count, 1)), wp.vec3),
        material=m.array(np.tile([5e4, gc, 0.0], (count, 1)), wp.vec3),
        separation_map=m.array(Q, float),
        frame_map=m.array(G, wp.vec3),
        area=m.array(np.full(count, cfg.size[1] * cfg.size[2] / count), float),
        duration=m.array([cfg.dt], float),
    )
    geometry = InterfaceGeometry(
        m.zeros(count, wp.vec3), m.array(np.tile(np.eye(3), (count, 1, 1)), wp.mat33)
    )
    return m, codes, geometry, m.zeros(count, float), data


@pytest.mark.parametrize("motion", ["translation", "rotation", "extension", "shear"])
def test_intact_interface_matches_unsplit_material(motion):
    m, _, geometry, history, data = patch()
    F = np.eye(3)
    velocity = np.tile([0.1, 0.2, 0.3], (m.n, 1))
    if motion == "rotation":
        velocity = np.cross([30.0, -20.0, 200.0], m.reference - [0.0, 0.0, 0.006])
    elif motion == "extension":
        F[0, 0] = 1.05
    elif motion == "shear":
        F[0, 1] = 0.05
    state = m.state(x=m.reference @ F.T, v=velocity, F=np.tile(F, (m.n, 1, 1)))
    reference = state
    geometry.frame.assign(np.tile(F, (len(history), 1, 1)))
    for _ in range(8):
        result = advance_material(m, state, geometry, history, **data, gravity=(0.0, 0.0, 0.0))
        reference = m.step(reference, gravity=(0.0, 0.0, 0.0), duration=data["duration"])
        state, geometry, history = result.state, result.geometry, result.interface.history
        np.testing.assert_array_equal(history.numpy(), 0.0)
        np.testing.assert_allclose(geometry.gap.numpy(), 0.0, atol=1e-19)
        for name in ["x", "v", "F", "Fp", "stress"]:
            np.testing.assert_allclose(
                getattr(state, name).numpy(), getattr(reference, name).numpy(), rtol=1e-11, atol=1e-10
            )


def test_fracture_energy_and_history_survive_unloading_and_restart():
    m, codes, geometry, history, data = patch(gc=7.0)
    state = m.state()
    force = np.zeros((m.n, 3))
    force[:, 0] = np.where(codes == 1, -1.0, 1.0) * 2e5 * m.mass
    impulse = m.array(force * m.config.dt, wp.vec3)
    for _ in range(32):
        result = advance_material(m, state, geometry, history, **data, impulse=impulse)
        assert np.all(result.interface.history.numpy() >= history.numpy())
        state, geometry, history = result.state, result.geometry, result.interface.history
    np.testing.assert_allclose(
        result.interface.fracture.numpy() @ data["area"].numpy(), 7.0 * 0.012**2, rtol=1e-13
    )
    saved_history = history.numpy().copy()
    saved_geometry = geometry.snapshot()
    impulse.assign(-impulse.numpy())
    direct = advance_material(m, state, geometry, history, **data, impulse=impulse)
    restored = advance_material(
        m,
        state,
        InterfaceGeometry.restore(saved_geometry, device=m.device),
        m.array(saved_history, float),
        **data,
        impulse=impulse
    )
    for a, b in [
        (direct.geometry.gap, restored.geometry.gap),
        (direct.geometry.frame, restored.geometry.frame),
        (direct.interface.history, restored.interface.history),
    ]:
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    for name in ["x", "v", "F", "Fp", "stress"]:
        np.testing.assert_array_equal(
            getattr(direct.state, name).numpy(), getattr(restored.state, name).numpy()
        )
    assert np.all(direct.interface.history.numpy() >= saved_history)


def test_coupled_trajectory_input_derivatives():
    m, codes, geometry, history, data = patch()
    side = np.where(codes == 1, -1.0, 1.0)[:, None]
    gap = np.array([0.0001, 0.0002, 0.0001])
    state = m.state(x=m.reference + side * gap / 2, v=side * np.array([0.3, 0.2, -0.1]))
    geometry.gap.assign(np.tile(gap, (len(history), 1)))
    history.assign(np.full(len(history), 1.2 * np.linalg.norm(gap)))

    def run():
        current, surface, maximum = state, geometry, history
        for _ in range(8):
            result = advance_material(m, current, surface, maximum, **data, gravity=(0.0, 0.0, 0.0))
            current, surface, maximum = result.state, result.geometry, result.interface.history
        return current.x, current.v, current.F, surface.gap, surface.frame, maximum

    rng = np.random.default_rng(418)
    with wp.Tape() as tape:
        output = run()
    seeds = [
        rng.normal(size=a.numpy().shape) / scale
        for a, scale in zip(output, [0.01, 1.0, 1.0, 0.0028, 1.0, 0.0028])
    ]
    tape.backward(grads={a: m.array(s, a.dtype, False) for a, s in zip(output, seeds)})
    for source in [state.v, data["material"], m.young, data["duration"]]:
        base = source.numpy().copy()
        direction = rng.normal(size=base.shape) * np.maximum(np.abs(base), 1e-12)
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            trials = []
            for sign in [-1, 1]:
                source.assign(base + sign * epsilon * direction)
                trials.append([a.numpy() for a in run()])
            fd = sum(np.sum((b - a) * s) for a, b, s in zip(*trials, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-8)
        source.assign(base)
