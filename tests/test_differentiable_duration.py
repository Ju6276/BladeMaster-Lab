"""Time derivatives and fixed-rate material/contact coefficients in split steps."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_constraint import solve
from blademaster_warp.differentiable_cutting import EventCutContact, FixedCutContact
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.solver import Config


def config(**kwargs):
    cfg = Config(size=(.012, .012, .012), grid_spacing=.006, dt=.0002,
                 external_tool=True, blade_grid_constraint=False, young_modulus=20000,
                 yield_stress=200, contact_reference_dt=.0002, flip_reference_dt=.0002)
    return replace(cfg, **kwargs)


def check_duration_vjp(model, duration, evaluate, monkeypatch):
    monkeypatch.setattr(wp.config, 'verify_autograd_array_access', True)
    with wp.Tape() as tape:
        outputs = evaluate(duration)
    rng = np.random.default_rng(41)
    seeds = [rng.normal(size=a.numpy().shape) / max(1., np.linalg.norm(a.numpy())) for a in outputs]
    tape.backward(grads={a: model.array(seed, a.dtype, False) for a, seed in zip(outputs, seeds)})
    ad = float(duration.grad.numpy()[0])
    assert np.isfinite(ad) and abs(ad) > 1e-3
    base = float(duration.numpy()[0])
    for epsilon in (1e-8, 3e-9):
        values = []
        for sign in (-1, 1):
            changed = evaluate(model.array([base + sign * epsilon], float, False))
            values.append(sum(np.sum(a.numpy() * seed) for a, seed in zip(changed, seeds)))
        np.testing.assert_allclose(ad, (values[1] - values[0]) / (2 * epsilon), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('ordered', [False, True])
def test_material_duration_gradient_and_reference_dt(ordered, monkeypatch):
    cfg = config()
    model = MaterialTrajectory(cfg, precision='float64', ordered_transfers=ordered)
    velocity = np.random.default_rng(5).normal(scale=.07, size=(model.n, 3))
    deformation = np.tile(np.diag([1.03, .99, 1.01]), (model.n, 1, 1))
    initial = model.state(v=velocity, F=deformation)
    duration = model.array([.00013], float)

    def evaluate(dt):
        state = model.step(initial, duration=dt)
        return [state.x, state.v, state.F, state.Fp, state.stress]

    actual = evaluate(duration)
    shorter = MaterialTrajectory(replace(cfg, dt=.00013), precision='float64', ordered_transfers=ordered)
    expected = shorter.step(shorter.state(v=velocity, F=deformation))
    for a, name in zip(actual, ('x', 'v', 'F', 'Fp', 'stress')):
        np.testing.assert_allclose(a.numpy(), getattr(expected, name).numpy(), rtol=1e-10, atol=1e-9)
    assert np.any(expected.yielded.numpy())
    check_duration_vjp(model, duration, evaluate, monkeypatch)


@pytest.mark.parametrize('contact_type', [FixedCutContact, EventCutContact])
def test_cut_duration_gradient_and_fixed_stiffness(contact_type, monkeypatch):
    cfg = config(cutting_energy=1.)

    def setup(cfg):
        model = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
        contact = contact_type(model)
        positions = model.reference + [.0016, 0, .009]
        positions[0], positions[1] = [-.001, 0, .1], [.001, 0, .1]
        positions[2] = [.02, 0, .0014]
        velocity = np.zeros_like(positions)
        velocity[0, 1], velocity[1, 1] = .2, -.2
        velocity[2] = [.02, .03, -.1]
        state = model.state(x=positions, v=velocity)
        history = contact.history(pairs=[[0, 1e-6, 0]], damage=np.full(model.n, .2))
        branch = contact.freeze(np.zeros(model.n, dtype=np.int64), [1], [1], np.zeros(model.n))
        return model, contact, state, history, branch

    model, contact, state, history, branch = setup(cfg)
    tool = contact.tool([0, 0, .04], [0, 0, -.12])
    duration = model.array([.00013], float)

    def evaluate(dt):
        s, h, f = contact.step(state, history, tool, branch, duration=dt)
        return [s.x, s.v, s.F, s.Fp, s.stress, h.rigid, h.pairs, h.damage, h.damage_work, f]

    shorter, other, s, h, b = setup(replace(cfg, dt=.00013))
    s, h, f = other.step(s, h, other.tool([0, 0, .04], [0, 0, -.12]), b)
    expected = [s.x, s.v, s.F, s.Fp, s.stress, h.rigid, h.pairs, h.damage, h.damage_work, f]
    for a, b in zip(evaluate(duration), expected):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-10, atol=1e-9)
    assert np.linalg.norm(h.pairs.numpy()) > 0
    assert np.linalg.norm(h.rigid.numpy()[0, 2]) > 0
    assert np.sum(h.damage_work.numpy()) > 0
    check_duration_vjp(model, duration, evaluate, monkeypatch)


def test_active_constraint_duration_gradient(monkeypatch):
    cfg = config(dt=.001, contact_reference_dt=.001, flip_reference_dt=.001,
                 blade_grid_constraint=True, yield_stress=0)
    model = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    contact = FixedCutContact(model)
    codes = np.where(model.reference[:, 0] < 0, 1, 2)
    codes[:3] = 0
    grid = model.topology(codes)
    masses = np.bincount(grid.slots.numpy().ravel(), weights=model.weights.numpy().ravel() * model.mass)
    mass = model.array(masses, float, False)
    positions = model.reference + [1, 1, 1]
    positions[:4] = [[cfg.blade_half_size[0] + 1e-5, .001, .002]] * 4
    x = model.array(positions, wp.vec3)
    v = model.array(np.tile([-.12, .03, .02], (grid.count, 1)), wp.vec3)
    tool = contact.tool([0, 0, 0])
    duration = model.array([.0007], float)
    baseline = solve(model, grid, mass, v, x, tool, duration=duration)
    assert baseline.batches > 0

    def evaluate(dt):
        result = solve(model, grid, mass, v, x, tool, fixed_batches=baseline.batches, duration=dt)
        return [result.velocity, result.linear_impulse]

    check_duration_vjp(model, duration, evaluate, monkeypatch)
