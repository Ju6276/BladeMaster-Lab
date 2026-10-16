"""Storage optimization must preserve contact, topology, and material updates."""
from dataclasses import replace
import numpy as np
import pytest
from blademaster_warp.solver import Config, Solver


@pytest.mark.parametrize('viscosity', [0., 2.])
def test_compact_transfer_matches_hash_with_incompatible_codes(viscosity):
    cfg = Config(dynamic_viscosity=viscosity)
    a = Solver(cfg)
    b = Solver(replace(cfg, grid_storage='compact', cache_topology=True))
    rng = np.random.default_rng(23)
    velocity = rng.normal(0, .01, (a.n, 3))
    codes = rng.integers(0, 27, a.n, dtype=np.int32)
    impulse = rng.normal(0, 1.e-7, (a.n, 3))
    for s in (a, b):
        s.v.assign(velocity); s.codes.assign(codes); s.impulses.assign(impulse)
        s.material_step()
    for name, atol in [('x', 1.e-8), ('v', 2.e-6), ('F', 2.e-6)]:
        np.testing.assert_allclose(getattr(a, name).numpy(), getattr(b, name).numpy(), atol=atol)
    packed = b.grid_keys.numpy()[:int(b.active_count.numpy()[0])]
    np.testing.assert_array_equal(np.sort(packed), np.sort(a.keys.numpy()[a.keys.numpy() >= 0]))
    np.testing.assert_allclose(a.mass*a.v.numpy().sum(0), b.mass*b.v.numpy().sum(0), atol=2.e-9)


def test_compact_graph_contacts_new_cuts_and_persistent_history():
    a = Solver(Config())
    b = Solver(Config(grid_storage='compact', cache_topology=True))
    for s in (a, b):
        s.bx.assign([[0., 0., .047]])
    for count, index, cutting in [(20, 0, True), (40, 0, True), (13, 0, False),
                                  (25, 1, True), (8, 1, False)]:
        for s in (a, b):
            s.advance(np.tile([0., 0., .019], (count, 1)), cutting=cutting, cut_index=index)
        np.testing.assert_array_equal(a.codes.numpy(), b.codes.numpy())
        np.testing.assert_allclose(a.x.numpy(), b.x.numpy(), atol=2.e-7)
        np.testing.assert_allclose(a.v.numpy(), b.v.numpy(), atol=2.e-5)
        np.testing.assert_allclose(a.F.numpy(), b.F.numpy(), atol=2.e-5)
        np.testing.assert_allclose(a.reaction.numpy(), b.reaction.numpy(), atol=2.e-8)
        assert not b.errors.numpy().any()
    assert np.count_nonzero(b.codes.numpy()) > 0
