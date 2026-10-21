"""Full and checkpoint-only experiment drivers must return the same derivatives."""

import numpy as np
import pytest

from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.differentiable_natural import NaturalCutContact
from blademaster_warp.solver import Config
from scripts.diff_natural_validate import WindowState, evaluate


@pytest.mark.parametrize('phase', [0., .25, .5])
def test_checkpoint_only_validation_matches_full_tape(phase):
    cfg = Config(size=(.012, .012, .012), grid_spacing=.006, dt=.0002,
                 external_tool=True, blade_bevel_height=.005,
                 young_modulus=20000, yield_stress=200,
                 contact_reference_dt=.0002, flip_reference_dt=.0002)
    model = MaterialTrajectory(cfg, precision='float64', ordered_transfers=True)
    contact = NaturalCutContact(model)
    state = model.state(x=model.reference + [0, 0, .03])
    values = {name: getattr(state, name).numpy() for name in ('x', 'v', 'F', 'Fp')}
    history = contact.history()
    saved = {name: getattr(history, name).numpy() for name in ('rigid', 'pairs', 'damage', 'side')}
    origin = np.array([.0004, 0, values['x'][:, 2].max() + cfg.blade_half_size[2] + .00005])
    window = WindowState(origin, values, saved, np.zeros(model.n, np.int64), np.empty(0, np.int64),
                         0., cfg.dt, [])
    results = [evaluate(cfg, 6, 'integrated', [1e-7], [2, 3], 'event', 'projected',
                        initial_window=window, adjoint=mode, phase=phase,
                        controls_to_check=('knife_x_mm', 'knife_depth_scale', 'knife_lateral_scale'))
               for mode in ('full', 'checkpoint')]
    first, second = results
    assert second['checkpoint'][0]['controls_max_abs'] is None
    assert second['checkpoint'][1]['gradient_reference'] == 'checkpoint:2'
    assert second['checkpoint'][1]['controls_max_abs'] < 1e-10
    assert first['newly_labelled_particles'] > 0
    np.testing.assert_allclose(first['gradient_controls'], second['gradient_controls'], rtol=1e-9, atol=1e-10)
    for name in first['gradient_materials']:
        np.testing.assert_allclose(first['gradient_materials'][name], second['gradient_materials'][name],
                                   rtol=1e-9, atol=1e-10)
    for result in results:
        assert max(row['relative_error'] for row in result['finite_difference']) < .02
