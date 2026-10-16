import numpy as np
import pytest
from blademaster_warp.cut_cycle import CutCycle, CycleSettings, evaluate_cut


def edges(position):
    return np.array(position) + [[0, -0.04, -0.02], [0, 0.04, -0.02]]


def test_resisted_cut_does_not_withdraw_on_clock():
    c = CutCycle([0, 0, 0.06], 0, 0.02)
    c.update(0, c.initial, edges(c.initial))
    c.update(0.3, c.initial, edges(c.initial))
    # The old timed script would already be withdrawing at t=2.6.
    for t in [2.3, 2.6, 4.6, 20.0]:
        target = c.update(t, [0, 0, 0.032], edges([0, 0, 0.032]))
        assert c.phase == "descend"
        np.testing.assert_allclose(target, c.bottom)


def test_tilted_edge_and_interrupted_contact_do_not_pass():
    c = CutCycle([0, 0, 0.06], 0, 0.02)
    c.update(0, c.initial, edges(c.initial))
    c.update(0.3, c.initial, edges(c.initial))
    tilted = edges(c.bottom)
    tilted[1, 2] = 0.002
    c.update(3, c.bottom, tilted)
    c.update(3.2, c.bottom, tilted)
    assert c.phase == "descend"
    c.update(3.3, c.bottom, edges(c.bottom))
    c.update(3.35, c.bottom, tilted)
    c.update(3.4, c.bottom, edges(c.bottom))
    c.update(3.49, c.bottom, edges(c.bottom))
    assert c.phase == "descend"
    c.update(3.51, c.bottom, edges(c.bottom))
    assert c.phase == "withdraw"


def test_retraction_and_mechanical_success_are_separate():
    c = CutCycle([0, 0, 0.06], 0, 0.02)
    for t, p in [
        (0, c.initial),
        (0.3, c.initial),
        (3, c.bottom),
        (3.11, c.bottom),
        (5, c.initial),
        (5.01, c.initial),
        (5.52, c.initial),
    ]:
        c.update(t, p, edges(p))
    assert c.done

    def row(parts, clear=True):
        return dict(phase="verify", component_sizes=parts, knife_clear_of_material=clear)

    assert evaluate_cut([row([10, 10])], True)["success"]
    assert not evaluate_cut([row([20])], True)["success"]
    assert not evaluate_cut([row([10, 10], False)], True)["success"]
    assert not evaluate_cut([row([10, 10]), row([20])], True)["success"]
    assert not evaluate_cut([row([10, 10])], False)["success"]
    assert not evaluate_cut([], True)["success"]


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        CycleSettings(position_tolerance_m=0)
    with pytest.raises(ValueError):
        CycleSettings(descent_speed_m_s=float("nan"))
