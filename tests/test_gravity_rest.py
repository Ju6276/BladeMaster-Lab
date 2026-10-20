import numpy as np

from blademaster_warp.franka_run import block_has_settled


def test_rest_waits_until_motion_stays_inside_a_particle_fraction():
    radius, frame_s, window_s = 0.0025, 0.04, 0.30
    falling = [(0.02 * i, 0.05, 0.067) for i in range(1, 40)]
    assert not block_has_settled(falling, radius, frame_s, window_s)

    # One frame of the 95th-percentile speed is under a tenth of a particle,
    # and the top does not drift by more than that over the window.
    still = [(0.02 * i, 0.001, 0.064) for i in range(1, 40)]
    assert block_has_settled(still, radius, frame_s, window_s)

    drifting = [(0.02 * i, 0.001, 0.064 + 0.0004 * i) for i in range(1, 40)]
    assert not block_has_settled(drifting, radius, frame_s, window_s)

    short = [(0.02 * i, 0.001, 0.064) for i in range(1, 5)]
    assert not block_has_settled(short, radius, frame_s, window_s)
    assert not block_has_settled(still, 0.0, frame_s, window_s)
