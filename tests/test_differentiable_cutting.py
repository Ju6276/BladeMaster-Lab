"""Cut-pair history, release branches and complete fixed-field contact gradients."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from blademaster_warp.coupling import ExternalToolPort
from blademaster_warp.differentiable_cutting import FixedCutContact, pair_snapshot, pair_step
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.solver import Config, Solver
from scripts.diff_cut_validate import evaluate, numpy_tool


def split_case(cut_hold=False):
    cfg = Config(size=(0.024, 0.018, 0.018), grid_spacing=0.006, dt=0.0002,
                 yield_stress=180, external_tool=True, blade_grid_constraint=False,
                 friction_static=0.05, friction_dynamic=0.03, cut_hold=cut_hold)
    original = Solver(cfg)
    x = original.reference.copy()
    x[:, 0] *= 0.85
    original.x.assign(x)
    F = np.tile(np.diag([0.85, 1.03, 1.0]), (original.n, 1, 1))
    original.F.assign(F)
    v = np.zeros_like(x)
    v[:, 1] = np.where(x[:, 0] < 0, 0.25, -0.25)
    original.v.assign(v)
    codes = np.where(x[:, 0] < 0, 1, 2)
    original.codes.assign(codes)
    original.refresh_stress()
    values = {name: getattr(original, name).numpy() for name in ("x", "v", "F", "Fp")}
    history = dict(rigid=original.histories.numpy(), pairs=np.zeros((0, 3)), damage=original.damage.numpy())
    origin = np.array([x[:, 0].min() - 0.0016, 0, 0.009])
    controls = np.array([0, 0.2, 0.4])
    port = ExternalToolPort(original)
    keys = np.empty(0, dtype=np.int64)
    rows = []
    for i in range(5):
        previous = keys
        reaction = port.advance(numpy_tool(origin, i * cfg.dt, controls), cfg.dt, cutting=False, graph=False)
        keys, pairs = pair_snapshot(original)
        rows.append(dict(codes=codes.copy(), released=np.zeros(original.n, dtype=int),
                         previous=previous, keys=keys, pairs=pairs, force=reaction.force_n.copy(),
                         rigid=original.histories.numpy(), damage=original.damage.numpy(), cutting=False,
                         state={name: getattr(original, name).numpy() for name in values}))
    assert not original.errors.numpy().any()
    assert min(len(row["keys"]) for row in rows) > 0
    assert np.max(np.abs(rows[-1]["pairs"])) > 0
    return cfg, origin, values, history, rows, controls


@pytest.fixture(scope="module")
def split():
    case = split_case()
    cfg, origin, values, history, rows, controls = case
    recorded = evaluate(cfg, origin, 0, values, history, rows, controls)
    for row, (rigid_modes, pair_modes, yielded) in zip(rows, recorded["contact_modes"]):
        row.update(rigid_modes=rigid_modes, pair_modes=pair_modes, yielded=yielded)
    return case


def test_recorded_contact_modes_preserve_forward(split):
    cfg, origin, values, history, rows, controls = split
    fixed = evaluate(cfg, origin, 0, values, history, rows, controls, backward=True)
    natural_rows = [{k: v for k, v in row.items() if k not in ("rigid_modes", "pair_modes", "yielded")} for row in rows]
    natural = evaluate(cfg, origin, 0, values, history, natural_rows, controls, backward=True)
    np.testing.assert_allclose(fixed["loss"], natural["loss"], atol=1e-6, rtol=1e-5)
    for parameter in fixed["gradients"]:
        np.testing.assert_allclose(fixed["gradients"][parameter], natural["gradients"][parameter],
                                   atol=1e-6, rtol=5e-4)


def test_split_contact_forward(split):
    cfg, origin, values, history, rows, controls = split
    result = evaluate(cfg, origin, 0, values, history, rows, controls)
    for name, tolerance in (("x", 1e-6), ("v", 1e-4), ("F", 1e-5), ("Fp", 1e-5),
                            ("force", 1e-4), ("pair_history", 1e-7), ("rigid_history", 1e-7)):
        assert result["forward_max_abs"][name] < tolerance, (name, result["forward_max_abs"])


@pytest.mark.parametrize("parameter", ["young_modulus", "yield_stress", "friction", "controls"])
def test_split_contact_gradient(split, parameter, monkeypatch):
    monkeypatch.setattr(wp.config, "verify_autograd_array_access", True)
    cfg, origin, values, history, rows, controls = split
    base = evaluate(cfg, origin, 0, values, history, rows, controls, backward=True)
    direction = np.array([0.7, 0.3, 0.4])
    friction = np.array([cfg.friction_static, cfg.friction_dynamic])
    gradient = base["gradients"][parameter]
    if parameter == "controls":
        ad = float(np.dot(gradient, direction))
    elif parameter == "friction":
        ad = float(np.sum(gradient))
    else:
        ad = gradient * getattr(cfg, parameter)
    assert np.isfinite(ad) and abs(ad) > 1e-6
    for epsilon in (0.001, 0.0005):
        losses = []
        for sign in (-1, 1):
            other_cfg, other_q, other_friction = cfg, controls, friction
            if parameter in ("young_modulus", "yield_stress"):
                other_cfg = replace(cfg, **{parameter: getattr(cfg, parameter) * (1 + sign * epsilon)})
            elif parameter == "controls":
                other_q = controls + sign * epsilon * direction
            else:
                other_friction = friction + sign * epsilon
            result = evaluate(other_cfg, origin, 0, values, history, rows, other_q, friction=other_friction)
            losses.append(result["loss"])
        fd = (losses[1] - losses[0]) / (2 * epsilon)
        np.testing.assert_allclose(ad, fd, rtol=0.02, atol=2e-5)


@pytest.mark.parametrize("history_scale", [1e-5, 0.001])
def test_pair_history_vjp_and_momentum(history_scale):
    model = MaterialTrajectory(Config(blade_grid_constraint=False))
    x = model.array([[0, 0, 0.01], [0.0025, 0, 0.01]], wp.vec3)
    v = model.array([[0, 0.03, 0], [0, -0.02, 0]], wp.vec3)
    pairs = model.array([[0, 1]], int, False)
    previous = model.array([0], int, False)
    seed = np.array([[0.3, 0.7, 0.2], [-0.2, 0.1, 0.4]])
    history = np.array([[0, history_scale, 0.4 * history_scale]])
    h = model.array(history, wp.vec3)
    friction = model.array([0.3, 0.2], float)
    updated = model.zeros(1, wp.vec3)
    impulse = model.zeros(2, wp.vec3)
    modes = model.zeros(1, int, False)
    fixed_modes = model.array([-1], int, False)
    inputs = [x, v, pairs, previous, fixed_modes, h, friction, 0.003, 3e-5,
              model.array([0.0002], float, False), 0.0002, 0.1, 0.05, 0.1, 0]
    with wp.Tape() as tape:
        wp.launch(pair_step, 1, inputs=inputs, outputs=[updated, impulse, modes], device=model.device)
    tape.backward(grads={impulse: model.array(seed, wp.vec3, False),
                         updated: model.array([[0.4, -0.3, 0.7]], wp.vec3, False)})
    np.testing.assert_allclose(impulse.numpy().sum(axis=0), 0, atol=1e-12)
    direction = np.array([[0.3, -0.7, 0.2]])
    ad = float(np.sum(h.grad.numpy() * direction))
    assert np.isfinite(ad) and abs(ad) > 1e-6
    values = []
    eps = history_scale * 0.01
    for sign in (-1, 1):
        altered = list(inputs)
        altered[5] = model.array(history + sign * eps * direction, wp.vec3, False)
        output = model.zeros(2, wp.vec3, False)
        wp.launch(pair_step, 1, inputs=altered, outputs=[updated, output, modes], device=model.device)
        values.append(float(np.sum(output.numpy().astype(float) * seed)
                            + updated.numpy()[0].astype(float) @ [0.4, -0.3, 0.7]))
    np.testing.assert_allclose(ad, (values[1] - values[0]) / (2 * eps), rtol=0.003, atol=2e-6)


@pytest.mark.parametrize("hold", [False, True])
@pytest.mark.parametrize("initial_damage", [0.2, 1.0])
def test_cutting_work_and_release_match_original(hold, initial_damage):
    cfg = Config(size=(0.024, 0.018, 0.018), grid_spacing=0.006, dt=0.0002,
                 external_tool=True, blade_grid_constraint=False, cut_hold=hold,
                 blade_bevel_height=0.005)
    original = Solver(cfg)
    x = original.reference + [0.2, 0.2, 0.2]
    x[0] = [0.0001, 0, 0.0195]
    damage = np.zeros(original.n)
    damage[0] = initial_damage
    original.x.assign(x)
    original.damage.assign(damage)
    original.tool_pos.assign([[0, 0, 0.04]])
    original.tool_vel.assign([[0, 0, -0.12]])
    original.rigid_contacts(cutting=True)
    model = MaterialTrajectory(cfg)
    contact = FixedCutContact(model)
    state = model.state(x=x)
    history = contact.history(damage=damage)
    branch = contact.freeze(np.zeros(model.n, dtype=int), [], [], damage >= 1)
    tool = contact.tool([0, 0, 0.04], [0, 0, -0.12])
    _, updated, reaction = contact.step(state, history, tool, branch)
    np.testing.assert_allclose(reaction.numpy()[0], original.reaction.numpy()[0] / cfg.dt, atol=1e-6)
    np.testing.assert_allclose(updated.damage.numpy(), original.damage.numpy(), atol=1e-7)
    np.testing.assert_allclose(updated.rigid.numpy(), original.histories.numpy(), atol=1e-8)
    if initial_damage < 1:
        assert updated.damage.numpy()[0] > initial_damage


@pytest.mark.parametrize('segment_steps', [1, 3, 8])
def test_cut_checkpoint_preserves_plastic_and_contact_history(split, segment_steps):
    from scripts.diff_cut_validate import differentiable_tool, reaction_loss, terminal_loss

    cfg, origin, values, history, rows, controls = split
    model = MaterialTrajectory(cfg, ordered_transfers=True, precision="float64")
    contact = FixedCutContact(model)
    frozen = [contact.freeze(row['codes'], row['keys'], row['previous'], row['released'],
                             cutting=False, rigid_modes=row['rigid_modes'], pair_modes=row['pair_modes'],
                             yielded=row['yielded']) for row in rows]
    q = model.array(controls, float)
    initial = model.state(**values)
    initial_history = contact.history(**history)
    # Nonzero stress is a checkpoint state; the first step may compute it from F/Fp.
    def tool(step):
        return differentiable_tool(model, q, origin, step * cfg.dt)

    def objective(step, state, h, force, loss):
        model.launch(reaction_loss, 1, inputs=[force, 1 / len(rows)], outputs=[loss], device=model.device)
        if step == len(rows) - 1:
            model.launch(terminal_loss, model.n, inputs=[state.v, state.Fp], outputs=[loss], device=model.device)

    loss = model.zeros(1, float)
    with wp.Tape() as tape:
        state, h = initial, initial_history
        for step, branch in enumerate(frozen):
            state, h, force = contact.step(state, h, tool(step), branch)
            objective(step, state, h, force, loss)
    tape.backward(loss)
    expected_parameters = {a: a.grad.numpy().copy() for a in (q, model.young, model.yield_stress, contact.friction)}
    expected_state = {name: getattr(initial, name).grad.numpy().copy() for name in ('x', 'v', 'F', 'Fp')}
    expected_history = {name: getattr(initial_history, name).grad.numpy().copy() for name in ('rigid', 'pairs', 'damage')}
    expected_loss = loss.numpy().copy()
    tape.zero()
    timings = {}
    result = contact.checkpoint_vjp(initial, initial_history, frozen, segment_steps, tool, objective,
                                    parameters=(q,), timings=timings)
    np.testing.assert_allclose(result.loss.numpy(), expected_loss, atol=2e-7)
    for a, expected in expected_parameters.items():
        np.testing.assert_allclose(result.parameter_gradients[a], expected, rtol=2e-4, atol=2e-6)
    for name, expected in expected_state.items():
        np.testing.assert_allclose(getattr(result.initial_gradient, name).numpy(), expected, rtol=3e-4, atol=1e-5)
    for name, expected in expected_history.items():
        np.testing.assert_allclose(getattr(result.history_gradient, name).numpy(), expected, rtol=3e-4, atol=1e-5)
    assert timings['recompute_max_abs']['Fp'] < 1e-5
    assert timings['recompute_max_abs']['stress'] < 0.03
