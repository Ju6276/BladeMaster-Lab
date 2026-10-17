"""Maxwell (generalized standard linear solid) branch of the material model."""

from dataclasses import replace
import numpy as np
import pytest
from blademaster_warp import material as m
from blademaster_warp.coupling import ExternalToolPort
from blademaster_warp.scenarios import command_at, keyframes, scene_config
from blademaster_warp.solver import Config, Solver
from scripts.indentation import fit_relaxation, indent


def run_short(cfg, steps=100):
    s = Solver(cfg)
    s.tool_pos.assign([[0.0, 0.0, 0.047]])
    for _ in range(steps):
        s.step((0.0, 0.0, 0.02), cutting=True)
    return s, s.state()


@pytest.mark.parametrize("tau, fraction", [(0.0, 0.5), (0.1, 0.0)])
def test_disabled_branch_runs_the_elastic_kernels(tau, fraction):
    # Float atomics make two identical runs differ in the last bits; compare at that level.
    reference = run_short(Config())[1]
    repeat = run_short(Config())[1]
    s, out = run_short(replace(Config(), maxwell_tau=tau, maxwell_fraction=fraction))
    assert not s.maxwell
    assert out.keys() == reference.keys()
    for key in ("position", "velocity", "deformation", "blade_position"):
        noise = np.abs(repeat[key] - reference[key]).max()
        assert np.abs(out[key] - reference[key]).max() <= max(4 * noise, 1e-7), key
    np.testing.assert_array_equal(out["sides"], reference["sides"])


def held(cfg, F):
    """Solver with every particle at deformation F and no grid motion."""
    s = Solver(cfg)
    s.F.assign(np.tile(F, (s.n, 1, 1)).astype(np.float32))
    s.refresh_stress()
    s.grid_mass.zero_()
    return s


def hold_step(s):
    s.launch(
        m.update_F_maxwell,
        s.n,
        [
            s.F,
            s.Fp,
            s.viscous,
            s.slots,
            s.gradients,
            s.grid_mass,
            s.coupled_remapped,
            s.config.dt,
            s.mu,
            s.lam,
            s.mu_maxwell,
            s.maxwell_relax,
            s.config.yield_stress,
            s.stress,
            s.errors,
        ],
    )


def reference_stress(F, mu, lam, mu_maxwell, factor):
    """FP64 first Piola stress with the Maxwell deviatoric Hencky strain scaled by factor."""
    u, sigma, vt = np.linalg.svd(F)
    R, J = u @ vt, np.prod(sigma)
    mu_eq, lam_eq = mu - mu_maxwell, lam + 2 / 3 * mu_maxwell
    inv_t = np.linalg.inv(F).T
    equilibrium = 2 * mu_eq * (F - R) + lam_eq * (J - 1) * J * inv_t
    e = np.log(sigma)
    dev = e - e.mean()
    kirchhoff = u @ np.diag(2 * mu_maxwell * factor * dev) @ u.T
    return equilibrium + kirchhoff @ inv_t


def rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


# Finite stretch with shear and 1% volume change, rotated off the axes.
HELD_F = rotation([1, 2, 3], 0.4) @ np.diag([1.08, 0.95, 0.985]) @ rotation([0, 1, 1], -0.3)


@pytest.mark.parametrize("tau_steps", [0.01, 1.0, 40.0])
def test_held_deformation_relaxes_exactly(tau_steps):
    cfg = replace(Config(), maxwell_tau=tau_steps * Config().dt, maxwell_fraction=0.6)
    s = held(cfg, HELD_F)
    decay = np.exp(-1 / tau_steps)
    for n in range(0, 121):
        if n:
            hold_step(s)
        if n in (0, 1, 5, 40, 120):
            expected = reference_stress(HELD_F, s.mu, s.lam, s.mu_maxwell, decay**n)
            scale = np.abs(expected).max()
            np.testing.assert_allclose(s.stress.numpy()[0], expected, atol=2e-4 * scale, err_msg=f"step {n}")
    assert not s.errors.numpy().any()
    np.testing.assert_array_equal(s.F.numpy(), np.tile(HELD_F, (s.n, 1, 1)).astype(np.float32))
    np.testing.assert_allclose(np.linalg.det(np.eye(3) + s.viscous.numpy()), 1, atol=2e-6)


def test_slow_relaxation_is_resolved():
    """tau = 3.5e5 steps (22 s at the banana's 62.5 us); float32 Fv gave a 39% error here."""
    tau_steps, steps = 3.5e5, 20000
    cfg = replace(Config(), maxwell_tau=tau_steps * Config().dt, maxwell_fraction=0.6)
    s = held(cfg, HELD_F)
    start = s.stress.numpy()[0].copy()
    for _ in range(steps):
        hold_step(s)
    expected = reference_stress(HELD_F, s.mu, s.lam, s.mu_maxwell, np.exp(-steps / tau_steps))
    error = np.abs(s.stress.numpy()[0] - expected).max()
    assert error < 0.01 * np.abs(start - expected).max()


def test_instantaneous_and_volumetric_response_match_elastic_model():
    base = Config()
    cfg = replace(base, maxwell_tau=10 * base.dt, maxwell_fraction=0.7)
    small = np.eye(3) + 1e-3 * np.array([[0.5, 0.8, 0.0], [0.1, -0.3, 0.2], [0.0, 0.4, 0.6]])
    elastic = held(base, small).stress.numpy()[0]
    viscoelastic = held(cfg, small).stress.numpy()[0]
    np.testing.assert_allclose(viscoelastic, elastic, atol=3e-3 * np.abs(elastic).max())
    # A pure volume change loads only the elastic bulk response and does not relax.
    s = held(cfg, 0.98 * np.eye(3))
    first = s.stress.numpy()[0].copy()
    for _ in range(50):
        hold_step(s)
    scale = np.abs(first).max()
    np.testing.assert_allclose(s.stress.numpy()[0], first, atol=1e-5 * scale)
    np.testing.assert_allclose(first, np.diag(np.diag(first)), atol=1e-5 * scale)


def test_held_indentation_force_relaxes_with_tau():
    """Paper banana (with plasticity), knife indentation on the 6 mm grid held at 6 mm depth."""
    tau, fraction, hold = 0.05, 0.5, 0.35
    off = indent(0.006, 1, hold=hold)[3]
    on = indent(0.006, 1, hold=hold, maxwell_tau=tau, maxwell_fraction=fraction)[3]
    assert abs(off[-50:].mean() / off[20:40].mean() - 1) < 0.1
    fit = fit_relaxation(on)
    # Non-relaxing compliance in series (bulk modulus, contact penalties) lengthens the
    # structural time constant; measured 1.2 tau on this grid.
    assert 0.8 * tau < fit["tau_s"] < 1.5 * tau
    assert fit["f_inf"] / fit["f_0"] < 0.9
    assert 0 < fit["f_inf"] < 0.75 * off[-50:].mean()


@pytest.mark.parametrize("tau", [1.0e-6, 100.0])
def test_indentation_stable_for_extreme_tau(tau):
    forces, _, _, held_force = indent(0.006, 1, hold=0.05, maxwell_tau=tau, maxwell_fraction=0.5)
    reference = indent(0.006, 1, hold=0.05)
    assert np.isfinite(forces).all() and np.isfinite(held_force).all()
    if tau > 1:
        np.testing.assert_allclose(forces, reference[0], rtol=0.1, atol=0.05)
    else:
        assert held_force.mean() < reference[3].mean()


@pytest.mark.parametrize("scene", ["fig4", "fig8"])
@pytest.mark.parametrize("tau", [1.0e-4, 0.05, 10.0])
def test_paper_step_scenes_stay_valid(scene, tau):
    cfg = replace(scene_config(scene), maxwell_tau=tau, maxwell_fraction=0.5)
    keys = keyframes(scene)
    interval = max(0.002, cfg.dt)
    s = Solver(cfg)
    port = ExternalToolPort(s)
    for i in range(round(2.0 / interval)):
        tool, active, index = command_at(i * interval, keys)
        port.advance(tool, interval, cutting=active, cut_index=index)
    state = s.state()
    assert np.abs(state["velocity"]).max() < 2.0
