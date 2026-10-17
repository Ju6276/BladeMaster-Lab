import numpy as np
from blademaster_warp.cut_cycle import CutSequence, SequenceSettings, evaluate_cut


def test_sequence_cuts_each_offset_to_the_board_then_pushes():
    settings = SequenceSettings(cut_offsets_m=(0.03, 0.0))
    sequence = CutSequence([0.0, 0.0, 0.06], 0.0, 0.02, settings)
    bottoms = {}
    actual = np.array([0.0, 0.0, 0.06])
    for t in np.arange(0.0, sequence.duration_s + 1.0, 0.001):
        target = sequence.update(t, actual, np.zeros((2, 3)))
        actual = target
        if sequence.cutting:
            bottoms[sequence.cut_index] = target
    assert sequence.done
    assert sorted(bottoms) == [0, 1]
    for k, dx in enumerate(settings.cut_offsets_m):
        np.testing.assert_allclose(bottoms[k], [dx, 0.0, 0.02], atol=1e-4)
    phases = [e["phase"] for e in sequence.events]
    assert phases.count("push") == 2 and phases.count("descend") == 2
    assert sequence.expected_pieces == 3


def test_evaluation_counts_expected_pieces():
    observations = [dict(phase="verify", component_sizes=[1, 2, 3], knife_clear_of_material=True)]
    assert evaluate_cut(observations, True, 3)["success"]
    assert not evaluate_cut(observations, True, 2)["success"]


def test_push_waits_for_the_knife_to_reach_the_board():
    sequence = CutSequence([0.0, 0.0, 0.06], 0.0, 0.02, SequenceSettings(cut_offsets_m=(0.0,)))
    stuck = np.array([0.0, 0.0, 0.025])
    for t in np.arange(0.0, 3 * sequence.duration_s, 0.001):
        sequence.update(t, stuck, np.zeros((2, 3)))
    assert sequence.phase == "descend"


def test_tilted_knife_with_one_low_corner_does_not_end_the_descent():
    sequence = CutSequence([0.0, 0.0, 0.06], 0.0, 0.02, SequenceSettings(cut_offsets_m=(0.0,)))
    bottom = np.array([0.0, 0.0, 0.02])
    tilted = np.array([[-0.04, 0.0, 0.0], [0.04, 0.0, 0.004]])
    for t in np.arange(0.0, 3 * sequence.duration_s, 0.001):
        sequence.update(t, bottom, tilted)
    assert sequence.phase == "descend"
    level = np.array([[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]])
    t0 = 3 * sequence.duration_s
    for t in np.arange(t0, t0 + 0.5, 0.001):
        sequence.update(t, bottom, level)
    assert sequence.phase != "descend"
