from dataclasses import replace
import numpy as np
from blademaster_warp.solver import Config, Solver


def test_cached_map_rebuilds_when_labels_change_and_matches_uncached():
    plain = Solver()
    cached = Solver(replace(Config(), cache_topology=True))
    for s in (plain, cached):
        s.tool_pos.assign([[0.0, 0.0, 0.047]])
    for count, cut, on in [(8, 0, True), (8, 0, True), (8, 0, False), (7, 1, True), (8, 1, True)]:
        targets = np.tile([0.0, 0.0, 0.019], (count, 1))
        plain.advance(targets, cutting=on, cut_index=cut)
        cached.advance(targets, cutting=on, cut_index=cut)
        a, b = plain.state(), cached.state()
        np.testing.assert_array_equal(a["sides"], b["sides"])
        np.testing.assert_allclose(a["position"], b["position"], atol=2.0e-7)
        np.testing.assert_allclose(a["velocity"], b["velocity"], atol=2.0e-5)
        np.testing.assert_allclose(a["deformation"], b["deformation"], atol=2.0e-5)
    assert np.count_nonzero(cached.codes.numpy()) > 0


def test_unmodified_material_retains_cached_slots():
    s = Solver(replace(Config(), cache_topology=True))
    s.advance(np.tile([0.0, 0.0, 1.0], (2, 1)), cutting=False)
    slots = s.slots.numpy().copy()
    keys = s.keys.numpy().copy()
    s.advance(np.tile([0.0, 0.0, 1.0], (2, 1)), cutting=False)
    np.testing.assert_array_equal(s.slots.numpy(), slots)
    np.testing.assert_array_equal(s.keys.numpy(), keys)
    assert s.topology_dirty.numpy()[0] == 0
