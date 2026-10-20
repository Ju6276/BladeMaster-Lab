import numpy as np

from blademaster_warp.franka_task import PaperTask, PaperTaskMetrics


def follow(name, until):
    controller = PaperTask(name, [0.0, 0.0, 0.12], np.eye(3), np.zeros(3))
    position = controller.start_position.copy()
    rotation = controller.start_rotation.copy()
    cuts = set()
    lowest = np.inf
    for t in np.arange(0.0, until, 0.01):
        position, rotation = controller.update(t, position, rotation, np.zeros((2, 3)))
        lowest = min(lowest, position[2])
        if controller.cutting:
            cuts.add(controller.cut_index)
        if controller.done or controller.failed:
            break
    return controller, cuts, lowest


def test_fig7_path_makes_both_scores_and_finishes():
    controller, cuts, _ = follow("fig7-shallow", 90)
    assert controller.done and not controller.failed, controller.failure_reason
    assert cuts == {0, 1}
    # The rolled knife's origin stays high; the leading tip is what enters the block.
    plunged = [segment.position[2] for segment in controller.segments if segment.phase == "cut"]
    assert min(plunged) < max(plunged) - 0.01
    assert controller.events[-1]["phase"] == "done"


def test_fig8_path_cuts_four_slices():
    controller, cuts, lowest = follow("fig8", 180)
    assert controller.done and not controller.failed, controller.failure_reason
    assert cuts == {0, 1, 2, 3}
    assert lowest < 0.05


def _hold(metrics, positions, labels, codes, shifts=None):
    metrics.observe(0.0, "settle", positions, labels, np.zeros_like(positions), codes)
    for index, t in enumerate((0.2, 0.5, 0.8, 1.1, 1.4)):
        x = positions if shifts is None else positions + shifts
        phase = "done" if index == 4 else "verify"
        metrics.observe(t, phase, x, labels, np.zeros_like(positions), codes)
    return metrics.evaluate(True)


def test_fig7_accepts_one_piece_with_both_cuts_and_rejects_a_split():
    reference = np.array([[0.0, 0.0, 0.06], [0.02, 0.0, 0.05], [0.0, 0.02, 0.05], [-0.02, 0.0, 0.04]])
    codes = np.array([1, 3, 0, 0])
    metrics = PaperTaskMetrics("fig7-shallow", reference)
    assert _hold(metrics, reference, np.zeros(4, int), codes)["success"]
    metrics = PaperTaskMetrics("fig7-deep", reference)
    labels = np.array([0, 0, 1, 1])
    assert not _hold(metrics, reference, labels, codes)["pieces_match"]


def test_fig8_accepts_four_shifted_slices():
    reference = np.array([[0.044, 0, 0.02], [0.032, 0, 0.02], [0.020, 0, 0.02], [0.008, 0, 0.02], [-0.02, 0, 0.02]])
    codes = np.array([1, 3, 9, 27, 0])
    labels = np.arange(5)
    metrics = PaperTaskMetrics("fig8", reference)
    result = _hold(metrics, reference, labels, codes, shifts=np.array([0.01, 0.0, 0.0]))
    assert result["success"], result
    metrics = PaperTaskMetrics("fig8", reference)
    assert not _hold(metrics, reference, np.zeros(5, int), codes)["pieces_match"]
