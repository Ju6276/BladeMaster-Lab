import numpy as np
from blademaster_warp.scenarios import keyframes, command_at


def test_scenario_velocities_match_poses_and_cut_history_is_monotone():
    for name in ("parallel", "cross-shallow", "slice-lift", "fig4", "fig7-shallow", "fig7-deep", "fig8"):
        keys = keyframes(name)
        previous = -1
        for a, b in zip(keys[:-1], keys[1:]):
            t = (a[0] + b[0]) / 2
            dt = 1.0e-5
            state, active, index = command_at(t, keys)
            before = command_at(t - dt, keys)[0]
            after = command_at(t + dt, keys)[0]
            np.testing.assert_allclose(
                (after.position - before.position) / (2 * dt), state.linear_velocity, atol=1.0e-10
            )
            assert index >= previous
            previous = index
            assert active == (a[3] >= 0)


def test_explicit_normal_scale_is_the_per_step_factor_at_the_actual_dt():
    from blademaster_warp.scenarios import scene_config

    default = scene_config("fig4")
    assert default.dt == 0.002 and default.contact_scales()[0] == 1.0
    # An overriding dt keeps the default step's stiffness: s_n scales with dt^2.
    finer = scene_config("fig4", 0.0005)
    np.testing.assert_allclose(finer.contact_scales()[0], 1.0 / 16)
    for dt in (None, 0.002, 0.0005, 0.00025):
        cfg = scene_config("fig4", dt, normal_scale=0.1)
        np.testing.assert_allclose(cfg.contact_scales()[0], 0.1)
        assert cfg.contact_scales()[1] == scene_config("fig4", dt).contact_scales()[1]
