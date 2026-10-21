"""Independent float64 references and reproducible fixed-field time integration."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_material import MaterialTrajectory, hencky_project
from blademaster_warp.solver import Config, Solver
from blademaster_warp.coupling import ExternalToolPort, ToolState


@wp.kernel
def project64_input(f: wp.array(dtype=wp.mat33), limit: wp.array(dtype=float), out: wp.array(dtype=wp.mat33)):
    out[0] = hencky_project(f[0], limit[0])


def numpy_projection(f, limit):
    u, s, vt = np.linalg.svd(f)
    e = np.log(s)
    dev = e - e.mean()
    radius = np.linalg.norm(dev)
    return f if radius <= limit else (u * np.exp(e - (1 - limit / radius) * dev)) @ vt


@pytest.mark.parametrize('f', [np.eye(3), np.diag([1.25, 1.25, .8]),
                               np.array([[1.2, .1, 0], [0, .9, .1], [0, 0, 1.05]])])
def test_float64_projection_matches_numpy(f):
    model = MaterialTrajectory(Config(), precision='float64', ordered_transfers=True)
    a, limit = model.array([f], wp.mat33), model.array([.08], float)
    output = model.zeros(1, wp.mat33)
    assert a.dtype == wp.mat33d and limit.dtype == wp.float64
    seed = np.random.default_rng(8).normal(size=(3, 3))
    direction = np.random.default_rng(9).normal(size=(3, 3))
    with wp.Tape() as tape:
        model.launch(project64_input, 1, inputs=[a, limit], outputs=[output], device=model.device)
    tape.backward(grads={output: model.array([seed], wp.mat33, False)})
    np.testing.assert_allclose(output.numpy()[0], numpy_projection(f, .08), atol=5e-12)
    ad = np.sum(a.grad.numpy()[0] * direction)
    eps = 1e-6
    fd = np.sum((numpy_projection(f + eps * direction, .08)
                 - numpy_projection(f - eps * direction, .08)) * seed) / (2 * eps)
    np.testing.assert_allclose(ad, fd, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_ordered_transfers_recompute_exactly(precision):
    cfg = Config(size=(.024, .018, .018), grid_spacing=.006, dt=.0002, yield_stress=200)
    model = MaterialTrajectory(cfg, ordered_transfers=True, precision=precision)
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


def test_double_reference_coupling_graph_matches_eager():
    cfg = Config(size=(.012, .012, .012), grid_spacing=.006, dt=1.5625e-5,
                 external_tool=True, blade_grid_constraint=True, contact_reference_dt=2.5e-5,
                 blade_half_size=(.0005, .01, .01))
    tool = ToolState([.0063, 0, .006], [0, 0, 0, 1], [-.05, 0, 0], [0, 0, 0])
    values = []
    for graph in (False, True):
        solver = Solver(cfg, precision='float64')
        port = ExternalToolPort(solver, record_trajectory=True)
        assert solver.x.dtype == wp.vec3d and port.packed.dtype == wp.float64
        forces = []
        for _ in range(4):
            result = port.advance(tool, 3 * cfg.dt, cutting=False, graph=graph)
            forces.append(result.force_n)
        values.append(dict(x=solver.x.numpy(), v=solver.v.numpy(), F=solver.F.numpy(),
                           force=np.asarray(forces), tool=port.trajectory))
    assert np.max(np.abs(values[0]['force'])) > 1e-4
    for name in values[0]:
        np.testing.assert_allclose(values[0][name], values[1][name], rtol=1e-10, atol=1e-12)
