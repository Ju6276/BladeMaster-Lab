"""Grid-constraint forward equivalence and pose/velocity directional derivatives."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_constraint import solve
from blademaster_warp.differentiable_cutting import FixedCutContact
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.solver import Config, Solver


def setup(spacing):
    cfg = Config(size=(0.012, 0.012, 0.012), grid_spacing=spacing, dt=0.001,
                 external_tool=True, blade_grid_constraint=True, yield_stress=0)
    model = MaterialTrajectory(cfg)
    codes = np.where(model.reference[:, 0] < 0, 1, 2)
    codes[:3] = 0
    grid = model.topology(codes)
    mass = np.bincount(grid.slots.numpy().ravel(), weights=model.weights.numpy().ravel() * model.mass)
    x = model.reference + [1, 1, 1]
    x[:4] = [[cfg.blade_half_size[0] + 1e-5, 0.001, 0.002]] * 4
    velocity = np.tile([-0.12, 0.03, 0.02], (grid.count, 1))
    return cfg, model, grid, codes, mass, x, velocity


@pytest.mark.parametrize('spacing', [0.006, 0.004])
def test_grid_constraint_forward(spacing):
    cfg, model, grid, codes, mass, x, velocity = setup(spacing)
    contact = FixedCutContact(model)
    result = solve(model, grid, model.array(mass, float, False), model.array(velocity, wp.vec3),
                   model.array(x, wp.vec3), contact.tool([0, 0, 0]))
    original = Solver(cfg)
    original.codes.assign(codes)
    original._register_topology()
    keys = original.grid_keys.numpy()
    mapping = {int(key): index for index, key in enumerate(keys) if key >= 0}
    slots = np.array([mapping[int(key)] for key in grid.keys.numpy()])
    old_mass = np.zeros(original.grid_capacity)
    old_velocity = np.zeros((original.grid_capacity, 3))
    old_mass[slots] = mass
    old_velocity[slots] = velocity
    original.grid_mass.assign(old_mass)
    original.new_v.assign(old_velocity)
    original.x.assign(x)
    original.tool_pos.zero_()
    original.blade_constraint.solve()
    assert result.batches > 0
    assert result.residual <= 1e-6
    np.testing.assert_allclose(result.velocity.numpy(), original.new_v.numpy()[slots], atol=2e-7)
    np.testing.assert_allclose(result.linear_impulse.numpy(), -original.reaction.numpy(), atol=2e-10)
    np.testing.assert_array_equal(result.side.numpy(), original.blade_constraint.side.numpy())


@pytest.mark.parametrize('parameter', ['x', 'position', 'velocity'])
def test_grid_constraint_derivative(parameter, monkeypatch):
    monkeypatch.setattr(wp.config, 'verify_autograd_array_access', True)
    cfg, model, grid, codes, mass, x, velocity = setup(.006)
    contact = FixedCutContact(model)
    a, v = model.array(x, wp.vec3), model.array(velocity, wp.vec3)
    tool = contact.tool([0, 0, 0])
    seed = np.random.default_rng(52).normal(size=velocity.shape)
    force_seed = np.array([[300., 0, 0]])
    with wp.Tape() as tape:
        result = solve(model, grid, model.array(mass, float, False), v, a, tool)
    tape.backward(grads={result.velocity: model.array(seed, wp.vec3, False),
                         result.linear_impulse: model.array(force_seed, wp.vec3, False)})
    direction = np.zeros_like(x if parameter == 'x' else velocity if parameter == 'velocity' else np.zeros((1, 3)))
    direction[:4, 0] = 1.0
    source = {'x': a, 'position': tool.position, 'velocity': v}[parameter]
    ad = float(np.sum(source.grad.numpy() * direction))
    assert np.isfinite(ad) and abs(ad) > 1e-5
    for eps in ((2e-7, 1e-7) if parameter != 'velocity' else (2e-4, 1e-4)):
        losses = []
        for sign in (-1, 1):
            pos = sign * eps * direction[0] if parameter == 'position' else np.zeros(3)
            xx = x + sign * eps * direction if parameter == 'x' else x
            vv = velocity + sign * eps * direction if parameter == 'velocity' else velocity
            value = solve(model, grid, model.array(mass, float, False), model.array(vv, wp.vec3),
                          model.array(xx, wp.vec3), contact.tool(pos), fixed_batches=result.batches)
            losses.append(float(np.sum(value.velocity.numpy().astype(float) * seed)
                                + np.sum(value.linear_impulse.numpy().astype(float) * force_seed)))
        np.testing.assert_allclose(ad, (losses[1] - losses[0]) / (2 * eps), rtol=.003, atol=1e-5)
