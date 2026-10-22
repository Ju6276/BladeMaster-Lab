import numpy as np
import pytest
import warp as wp

from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.solver import Config
from test_grid_load import reference_operators


@wp.func
def transferred_velocity(
    transfer: wp.array2d(dtype=float), load: wp.array(dtype=wp.vec3),
    particle: int,
):
    velocity = wp.vec3(0.0)
    for node in range(load.shape[0]):
        velocity += transfer[particle, node] * load[node]
    return velocity


@wp.kernel
def grid_displacement(
    transfer: wp.array2d(dtype=float), load: wp.array(dtype=wp.vec3),
    duration: wp.array(dtype=float), displacement: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    displacement[p] = duration[0] * transferred_velocity(transfer, load, p)


@pytest.mark.parametrize('case', ['uncut', 'partial', 'crossing'])
@pytest.mark.parametrize('flip', [0.95, 1.0])
@pytest.mark.parametrize('yield_stress', [0., 200.])
def test_primal_correction_matches_grid_load_state_and_adjoint(case, flip, yield_stress):
    cfg = Config(size=(.018, .012, .012), grid_spacing=.006, dt=2e-5,
                 young_modulus=20000., yield_stress=yield_stress, flip_fraction=flip,
                 external_tool=True, blade_grid_constraint=False)
    model = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    codes = np.zeros(model.n, dtype=np.int64)
    if case != 'uncut':
        codes = np.where(model.reference[:, 0] < 0, 1, 2)
        codes[model.reference[:, 2] < .006] = 0
    if case == 'crossing':
        codes += 3 * np.where(model.reference[:, 1] < 0, 1, 2)
    grid = model.topology(codes)
    weights, mass, coupling = reference_operators(model, codes)
    transfer = model.array(weights.T @ (coupling / mass[:, None]), float, False)
    rng = np.random.default_rng(916)
    velocity = rng.normal(scale=.01, size=(model.n, 3))
    deformation = np.tile(np.diag([1.03, .97, 1.]), (model.n, 1, 1))
    deformation[:, 0, 1] = .2
    state = model.state(v=velocity, F=deformation)
    load = model.array(rng.normal(scale=1e-6, size=(grid.count, 3)), wp.vec3)
    duration = model.array([cfg.dt], float)
    zero = model.zeros(model.n, wp.vec3, False)
    names = ('x', 'v', 'F', 'Fp', 'stress')
    seeds = [model.array(rng.normal(size=getattr(state, key).numpy().shape) / scale,
                         getattr(state, key).dtype, False)
             for key, scale in zip(names[:-1], [.02, 2., 1., 1.])]
    seeds.append(model.array(rng.normal(size=deformation.shape) / cfg.young_modulus,
                             wp.mat33, False))
    sources = [state.v, state.F, state.Fp, load, duration, model.young, model.yield_stress]
    values, gradients = [], []
    for corrected in [False, True]:
        with wp.Tape() as tape:
            if corrected:
                free = model.step(state, topology=grid, impulse=zero, duration=duration)
                displacement = model.zeros(model.n, wp.vec3)
                model.launch(grid_displacement, model.n, inputs=[transfer, load, duration],
                             outputs=[displacement], device=model.device)
                end = model.correct_displacement(state, free, displacement, grid, duration)
            else:
                end = model.step(state, topology=grid, impulse=zero, duration=duration,
                                 grid_load=load)
        values.append([getattr(end, key).numpy().copy() for key in names])
        tape.backward(grads={getattr(end, key): seed for key, seed in zip(names, seeds)})
        gradients.append([a.grad.numpy().copy() for a in sources])
        tape.zero()
    for name, expected, actual in zip(names, *values):
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-10,
                                   err_msg=f'{case} {name}')
    source_names = ('v', 'F', 'Fp', 'load', 'duration', 'young', 'yield_stress')
    for name, expected, actual in zip(source_names, *gradients):
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-8,
                                   err_msg=f'{case} gradient {name}')
    if yield_stress:
        assert np.max(np.abs(values[0][3] - np.eye(3))) > .01


def test_zero_primal_correction_preserves_the_completed_step():
    cfg = Config(size=(.012, .012, .012), grid_spacing=.006, dt=1e-6,
                 young_modulus=20000., yield_stress=200.)
    model = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    state = model.state(F=np.tile(np.diag([1.2, .9, 1.]), (model.n, 1, 1)))
    free = model.step(state)
    end = model.correct_displacement(state, free, model.zeros(model.n, wp.vec3))
    for name in ('x', 'v', 'F', 'Fp', 'stress', 'yielded'):
        np.testing.assert_array_equal(getattr(end, name).numpy(), getattr(free, name).numpy())
