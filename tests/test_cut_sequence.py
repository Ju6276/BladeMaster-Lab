import numpy as np
from blademaster_warp.cut_cycle import CutSequence, SequenceSettings, evaluate_cut


def test_sequence_cuts_each_offset_to_the_board_then_pushes():
    settings = SequenceSettings(cut_offsets_m=(0.03, 0.0))
    sequence = CutSequence([0.0, 0.0, 0.06], 0.0, 0.02, settings)
    bottoms = {}
    for t in np.arange(0.0, sequence.duration_s + 0.1, 0.001):
        target = sequence.update(t, [0, 0, 0], np.zeros((2, 3)))
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
