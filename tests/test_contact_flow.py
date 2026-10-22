"""Contact virtual work under partial and intersecting cut-field transfers."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_geometry import GeometryEventContact
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.differentiable_natural import CutInstruction
from blademaster_warp.energy_ledger import EnergyLedger
from blademaster_warp.solver import Config


def reference_flow(model, codes, velocity):
    keys, slots = np.unique(codes[:, None] * model.node_count + model.nodes, return_inverse=True)
    slots = slots.reshape(model.n, 8)
    weights = model.weights.numpy()
    mass = np.zeros(len(keys))
    momentum = np.zeros((len(keys), 3))
    for a in range(8):
        np.add.at(mass, slots[:, a], model.mass * weights[:, a])
        np.add.at(momentum, slots[:, a], model.mass * weights[:, a, None] * velocity)
    before = momentum / mass[:, None]
    after = before.copy()
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
                if compatible:
                    after[i] += mass[j] / total * (before[j] - before[i])
    return np.sum(weights[:, :, None] * after[slots], axis=1)


@pytest.mark.parametrize('topology', ['uncut', 'partial', 'crossing'])
def test_contact_flow_matches_virtual_work_and_its_transpose(topology):
    cfg = Config(size=(.018, .012, .012), grid_spacing=.006, dt=1e-6,
                 external_tool=True, young_modulus=20000, yield_stress=0,
                 blade_grid_constraint=False)
    m = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    codes = np.zeros(m.n, dtype=np.int64)
    if topology != 'uncut':
        codes = np.where(m.reference[:, 0] < 0, 1, 2)
        codes[m.reference[:, 2] < .006] = 0
    if topology == 'crossing':
        codes += 3 * np.where(m.reference[:, 1] < 0, 1, 2)
        codes[m.reference[:, 0] > .006] %= 3
    grid = m.topology(codes)
    rng = np.random.default_rng(84)
    velocity = rng.normal(scale=.01, size=(m.n, 3))
    force = rng.normal(size=(m.n, 3))
    state = m.state(v=velocity)
    with wp.Tape() as tape:
        flow = m.flow_velocity(state.v, grid)
    expected = reference_flow(m, codes, velocity)
    np.testing.assert_allclose(flow.numpy(), expected, atol=1e-17)
    tape.backward(grads={flow: m.array(force, wp.vec3, False)})
    np.testing.assert_allclose(state.v.grad.numpy(), reference_flow(m, codes, force), atol=2e-15)
    result = m.step(state, impulse=m.array(force * cfg.dt, wp.vec3, False),
                    topology=grid, stage_base=state)
    rate = (1 - cfg.flip_fraction) / (cfg.flip_reference_dt or cfg.dt)
    contact_dv = result.v.numpy() - velocity - cfg.dt * rate * (expected - velocity)
    kinetic_power = m.mass * np.sum(velocity * contact_dv) / cfg.dt
    contact_power = np.sum(force * expected)
    np.testing.assert_allclose(kinetic_power, contact_power, atol=1e-15)
    assert abs(contact_power - np.sum(force * velocity)) > 1e-4


def test_nonuniform_contact_energy_error_decreases_with_time_step():
    residuals = []
    for steps in (32, 64, 128):
        cfg = Config(size=(.012, .012, .012), grid_spacing=.006, dt=.0004 / steps,
                     young_modulus=20000, yield_stress=0., external_tool=True,
                     blade_grid_constraint=False, contact_reference_dt=.0002,
                     flip_reference_dt=.0002, friction_static=10., friction_dynamic=10.)
        m = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
        c = GeometryEventContact(m)
        ledger = EnergyLedger(c)
        velocity = np.zeros_like(m.reference)
        velocity[:, 0] = .005 * (1 + np.sin(m.reference[:, 0] / cfg.grid_spacing))
        state = m.state(x=m.reference - [0., 0., .0001], v=velocity)
        history = c.history()
        for _ in range(steps):
            state, history, _ = c.step(state, history, c.tool([1, 1, 1]), CutInstruction(cutting=False))
        totals = ledger.report()['totals']
        residuals.append(abs(totals['global_balance_residual_j']))
        assert abs(totals['friction_transfer_work_j']) < 1e-19
        assert totals['friction_viscous_j'] > 1e-10
        assert totals['pic_j'] > 1e-10
    assert residuals[0] / residuals[1] > 3.5
    assert residuals[1] / residuals[2] > 3.5
