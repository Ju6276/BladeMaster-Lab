import numpy as np

from blademaster_warp.spacemouse import MouseState


def report(axis_values):
    return bytes([1]) + np.asarray(axis_values, dtype="<i2").tobytes()


def test_six_axis_report_is_clipped_to_full_scale():
    state = MouseState()
    state.feed(report([350, -700, 0, 175, 0, 0]), now=1.0)
    snap = state.sample()
    np.testing.assert_allclose(snap.axes, [1.0, -1.0, 0.0, 0.5, 0.0, 0.0])
    np.testing.assert_array_equal(snap.received, [1.0, 1.0])


def test_buttons_left_right_and_both():
    state = MouseState()
    state.feed([3, 1], now=1.0)
    state.feed([3, 3], now=2.0)
    state.feed([3, 2], now=3.0)
    snap = state.sample()
    assert snap.presses == ("close", "hold")
    assert snap.buttons == 2
    assert state.sample().presses == ()
