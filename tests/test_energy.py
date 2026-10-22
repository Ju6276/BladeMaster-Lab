"""Independent energy measurements, analytic states and stress-work consistency."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_material import MaterialTrajectory, compute_stress, project_plastic
from blademaster_warp.differentiable_geometry import GeometryEventContact
from blademaster_warp.differentiable_natural import CutInstruction
from blademaster_warp.energy import elastic_density, energy_snapshot, material_energy
from blademaster_warp.solver import Config


@pytest.mark.parametrize('stretch', [1., .94, 1.08])
def test_uniform_dilatation_and_rotation(stretch):
    young, poisson = 20000., .35
    rotation = np.array([[.8, -.6, 0], [.6, .8, 0], [0, 0, 1]])
    plastic = np.array([[1.1, .1, 0], [0, .9, 0], [0, 0, 1 / .99]])
    F = stretch * rotation @ plastic
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    expected = 3 * mu * (stretch - 1) ** 2 + .5 * lam * (stretch ** 3 - 1) ** 2
    np.testing.assert_allclose(elastic_density(F, plastic, young, poisson), expected,
                               rtol=1e-12, atol=1e-23)


def test_rigid_motion_energy_and_reference_volume():
    count, mass, volume = 5, .02, 3e-6
    x = np.tile([.3, -.2, .4], (count, 1))
    velocity = np.tile([.2, -.1, .3], (count, 1))
    F = np.tile(np.eye(3), (count, 1, 1))
    gravity = np.array([1., -2., -9.])
    value = material_energy(x, velocity, F, F, mass, volume, 20000, .35, gravity)
    assert value['elastic_j'] == 0
    np.testing.assert_allclose(value['kinetic_j'], .5 * count * mass * .14)
    np.testing.assert_allclose(value['gravity_j'], count * mass * 2.9)
    expanded = material_energy(x, velocity, F * 1.03, F, mass, volume, 20000, .35)
    twice_volume = material_energy(x, velocity, F * 1.03, F, mass, 2 * volume, 20000, .35)
    assert twice_volume['kinetic_j'] == expanded['kinetic_j']
    assert twice_volume['elastic_j'] == 2 * expanded['elastic_j']


@pytest.mark.parametrize('kind', ['identity', 'rotation', 'repeated', 'shear'])
def test_energy_directional_derivative_matches_solver_stress(kind):
    model = MaterialTrajectory(Config(size=(.012, .012, .012), grid_spacing=.006,
                                      young_modulus=20000, poisson_ratio=.35),
                               precision='float64', requires_grad=False)
    rotation = np.array([[.8, -.6, 0], [.6, .8, 0], [0, 0, 1]])
    elastic = dict(identity=np.eye(3), rotation=rotation,
                   repeated=rotation @ np.diag([1.1, 1.1, .92]),
                   shear=np.array([[1.1, .2, 0], [0, .94, .07], [0, 0, 1.02]]))[kind]
    plastic = np.array([[1.1, .1, 0], [0, .9, 0], [0, 0, 1 / .99]])
    F = elastic @ plastic
    tensors = lambda value: np.tile(value, (model.n, 1, 1))
    state = model.state(F=tensors(F), Fp=tensors(plastic))
    stress = model.zeros(model.n, wp.mat33, False)
    model.launch(compute_stress, model.n,
                 inputs=[state.F, state.Fp, model.young, model.config.poisson_ratio],
                 outputs=[stress], device=model.device)
    direction = np.random.default_rng(19).normal(size=(3, 3))
    contraction = float(np.sum(stress.numpy()[0] * direction))
    for epsilon in (1e-5, 1e-6):
        difference = (elastic_density(F + epsilon * direction, plastic, 20000, .35)
                      - elastic_density(F - epsilon * direction, plastic, 20000, .35)) / (2 * epsilon)
        np.testing.assert_allclose(difference, contraction, rtol=2e-8, atol=2e-5)
    snapshot = energy_snapshot(model, state)
    np.testing.assert_allclose(snapshot['elastic_j'],
                               model.n * model.volume * elastic_density(F, plastic, 20000, .35))


@pytest.mark.parametrize('integration', ['symplectic', 'midpoint'])
@pytest.mark.parametrize('divisor', [1, 2, 4])
def test_freefall_energy_matches_discrete_integrator(integration, divisor):
    dt = .0002 / divisor
    duration = .002
    cfg = Config(size=(.012, .012, .012), grid_spacing=.006, dt=dt,
                 external_tool=True, young_modulus=20000, yield_stress=0,
                 contact_reference_dt=.0002, flip_reference_dt=.0002)
    model = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True, requires_grad=False)
    contact = GeometryEventContact(model, friction_update='midpoint', integration=integration)
    state = model.state(x=model.reference + [0, 0, .1])
    history = contact.history()
    initial = energy_snapshot(model, state)
    tool = contact.tool([1., 0., 1.])
    for _ in range(round(duration / dt)):
        state, history, force = contact.step(state, history, tool, CutInstruction(cutting=False))
        np.testing.assert_array_equal(force.numpy(), 0.)
    final = energy_snapshot(model, state)
    error = sum(final.values()) - sum(initial.values())
    expected = -.5 * model.n * model.mass * 9.81 ** 2 * duration * dt if integration == 'symplectic' else 0.
    np.testing.assert_allclose(error, expected, rtol=1e-9, atol=1e-16)
    np.testing.assert_allclose(final['kinetic_j'], .5 * model.n * model.mass * (9.81 * duration) ** 2,
                               rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize('field', ['x', 'v', 'F', 'Fp', 'mass', 'volume', 'young', 'poisson', 'gravity'])
def test_energy_rejects_nonfinite_inputs(field):
    values = dict(x=np.zeros((1, 3)), v=np.zeros((1, 3)),
                  F=np.eye(3)[None], Fp=np.eye(3)[None],
                  mass=.01, volume=1e-6, young=20000., poisson=.35, gravity=np.zeros(3))
    values[field] = np.full_like(values[field], np.nan) if isinstance(values[field], np.ndarray) else np.nan
    with pytest.raises(ValueError):
        material_energy(**values)


@pytest.mark.parametrize('F,Fp', [(np.diag([-1., 1, 1]), np.eye(3)),
                                 (np.eye(3), np.diag([1., 0, 1]))])
def test_energy_rejects_inverted_or_singular_deformation(F, Fp):
    with pytest.raises(ValueError, match='determinants'):
        elastic_density(F, Fp, 20000., .35)


@pytest.mark.parametrize('stretches', [[1.1, .95, .95], [.55, .4, .3], [.2, .1, .05]])
def test_material_plastic_update_decreases_energy(stretches):
    model = MaterialTrajectory(Config(size=(.012, .012, .012), grid_spacing=.006,
                                      young_modulus=20000, poisson_ratio=.35, yield_stress=200),
                               precision='float64', requires_grad=False)
    F = np.tile(np.diag(stretches), (model.n, 1, 1))
    initial = model.state(F=F)
    updated = model.zeros(model.n, wp.mat33, False)
    yielded = model.zeros(model.n, int, False)
    stress = model.zeros(model.n, wp.mat33, False)
    errors = model.zeros(1, int, False)
    branch = model.array(np.full(model.n, -1), int, False)
    model.launch(project_plastic, model.n,
                 inputs=[initial.F, initial.Fp, model.young, model.yield_stress, .35, branch],
                 outputs=[updated, yielded, stress, errors], device=model.device)
    assert errors.numpy()[0] == 0
    before = elastic_density(F, initial.Fp.numpy(), 20000, .35)
    after = elastic_density(F, updated.numpy(), 20000, .35)
    assert np.all(before > after)
    elastic = F @ np.linalg.inv(updated.numpy())
    plastic_increment = np.log(np.linalg.svd(elastic, compute_uv=False) / np.sort(stretches)[::-1])
    dissipation_distance = 200 * np.sqrt(2 / 3) * np.linalg.norm(plastic_increment, axis=1)
    assert np.min(before - after - dissipation_distance) >= -1e-9
    np.testing.assert_allclose(np.linalg.det(updated.numpy()), 1., rtol=1e-12, atol=1e-12)
