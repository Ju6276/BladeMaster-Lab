"""Repeatability and forward equivalence of ordered grid reductions."""

import numpy as np
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.solver import Config


def test_ordered_transfers_recompute_exactly():
    cfg = Config(size=(.024, .018, .018), grid_spacing=.006, dt=.0002, yield_stress=200)
    model = MaterialTrajectory(cfg, ordered_transfers=True)
    grid = model.topology(np.where(model.reference[:, 0] < 0, 1, 2))
    initial = model.state(F=np.tile(np.diag([1.1, .95, 1]), (model.n, 1, 1)))
    states = []
    for _ in range(2):
        state = initial
        for step in range(32):
            state = model.step(state, topology=grid)
        states.append(state)
    for name in ('x', 'v', 'F', 'Fp', 'stress'):
        np.testing.assert_array_equal(getattr(states[0], name).numpy(), getattr(states[1], name).numpy())


def test_ordered_and_atomic_transfers_agree():
    cfg = Config(size=(.024, .018, .018), grid_spacing=.006, dt=.0002, yield_stress=200)
    models = [MaterialTrajectory(cfg, ordered_transfers=ordered) for ordered in (False, True)]
    states = []
    for model in models:
        grid = model.topology(np.where(model.reference[:, 0] < 0, 1, 2))
        state = model.state(F=np.tile(np.diag([1.1, .95, 1]), (model.n, 1, 1)))
        for step in range(12):
            state = model.step(state, topology=grid)
        states.append(state)
    for name, tolerance in (('x', 1e-6), ('v', 1e-4), ('F', 1e-5), ('Fp', 1e-5)):
        np.testing.assert_allclose(getattr(states[0], name).numpy(), getattr(states[1], name).numpy(),
                                   rtol=0, atol=tolerance)
