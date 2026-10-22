"""Grid tractions paired with the actual material velocity field."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.solver import Config


def reference_operators(model, codes):
    keys = np.unique(codes[:, None] * model.node_count + model.nodes)
    slots = np.searchsorted(keys, codes[:, None] * model.node_count + model.nodes)
    weights = np.zeros((len(keys), model.n))
    for p in range(model.n):
        weights[slots[p], p] += model.weights.numpy()[p]
    mass = model.mass * weights.sum(axis=1)
    coupling = np.eye(len(keys))
    for node in np.unique(keys % model.node_count):
        group = np.flatnonzero(keys % model.node_count == node)
        total = mass[group].sum()
        for i in group:
            for j in group:
                left, right = int(keys[i] // model.node_count), int(keys[j] // model.node_count)
                compatible = True
                while left or right:
                    a, b = left % 3, right % 3
                    compatible &= a == 0 or b == 0 or a == b
                    left, right = left // 3, right // 3
                if i != j and compatible:
                    coupling[i, j] += mass[i] / total
                    coupling[i, i] -= mass[j] / total
    return weights, mass, coupling


@pytest.mark.parametrize('ordered', [False, True])
@pytest.mark.parametrize('case', ['uncut', 'partial', 'crossing'])
def test_grid_traction_virtual_work_and_adjoint(case, ordered):
    cfg = Config(size=(.018, .012, .012), grid_spacing=.006, dt=1e-6,
                 external_tool=True, young_modulus=20000, yield_stress=0,
                 blade_grid_constraint=False)
    model = MaterialTrajectory(cfg, precision='float64', ordered_transfers=ordered)
    codes = np.zeros(model.n, dtype=np.int64)
    if case != 'uncut':
        codes = np.where(model.reference[:, 0] < 0, 1, 2)
        codes[model.reference[:, 2] < .006] = 0
    if case == 'crossing':
        codes += 3 * np.where(model.reference[:, 1] < 0, 1, 2)
        codes[model.reference[:, 0] > .006] %= 3
    grid = model.topology(codes)
    weights, mass, coupling = reference_operators(model, codes)
    rng = np.random.default_rng(317)
    velocity = rng.normal(scale=.01, size=(model.n, 3))
    grid_force = rng.normal(size=(grid.count, 3))
    load = model.array(cfg.dt * grid_force, wp.vec3)
    state = model.state(v=velocity)
    zero = model.zeros(model.n, wp.vec3, False)
    trace = {}
    with wp.Tape() as tape:
        loaded = model.step(state, topology=grid, impulse=zero,
                            stage_base=state, grid_load=load, trace=trace)
    unloaded = model.step(state, topology=grid, impulse=zero, stage_base=state)
    acceleration = (loaded.v.numpy() - unloaded.v.numpy()) / cfg.dt
    expected = weights.T @ ((coupling @ grid_force) / mass[:, None])
    np.testing.assert_allclose(acceleration, expected, rtol=1e-12, atol=2e-11)

    grid_velocity = (coupling @ (weights @ (model.mass * velocity))) / mass[:, None]
    np.testing.assert_allclose(trace['old_v'].numpy(), grid_velocity, rtol=1e-12, atol=2e-17)
    kinetic_power = model.mass * np.sum(velocity * acceleration)
    interface_power = np.sum(grid_velocity * grid_force)
    np.testing.assert_allclose(kinetic_power, interface_power, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(model.mass * acceleration.sum(axis=0), grid_force.sum(axis=0), atol=2e-13)

    seed = rng.normal(size=(model.n, 3))
    tape.backward(grads={loaded.v: model.array(seed, wp.vec3, False)})
    expected_adjoint = coupling.T @ ((weights @ seed) / mass[:, None])
    np.testing.assert_allclose(load.grad.numpy(), expected_adjoint, rtol=2e-12, atol=2e-10)


def test_zero_grid_load_preserves_material_state():
    cfg = Config(size=(.012, .012, .012), grid_spacing=.006, dt=1e-6,
                 young_modulus=20000, yield_stress=0.)
    model = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    state = model.state(v=np.random.default_rng(83).normal(scale=.01, size=(model.n, 3)))
    first = model.step(state)
    second = model.step(state, grid_load=model.zeros(model.uncut.count, wp.vec3))
    for name in ('x', 'v', 'F', 'Fp', 'stress'):
        np.testing.assert_array_equal(getattr(first, name).numpy(), getattr(second, name).numpy())
