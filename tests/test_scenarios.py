import numpy as np
from blademaster_warp.scenarios import keyframes, command_at


def test_scenario_velocities_match_poses_and_cut_history_is_monotone():
    for name in ("parallel", "cross-shallow", "slice-lift"):
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
