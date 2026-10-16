import numpy as np
import warp as wp


@wp.kernel
def counts(grid: wp.uint64, x: wp.array(dtype=wp.vec3), r: float, out: wp.array(dtype=int)):
    i = wp.tid()
    found = int(0)
    for j in wp.hash_grid_query(grid, x[i], r):
        if wp.length(x[i] - x[j]) < r:
            found += 1
    out[i] = found


def test_compact_hash_grid_keeps_neighbors_outside_reference_box():
    rng = np.random.default_rng(45)
    # Far outside the banana reference box; bucket wrap must not clip space.
    x = np.vstack([rng.normal(c, 0.003, (60, 3)) for c in [(2, 3, 4), (-2, 0, -3), (0.1, 0, 0.2)]]).astype(
        np.float32
    )
    r = 0.006
    expected = (np.linalg.norm(x[:, None] - x[None], axis=2) < r).sum(axis=1)
    xd = wp.array(x, dtype=wp.vec3, device="cuda:0")
    for shape in [(64, 64, 64), (64, 32, 16), (8, 8, 8)]:
        grid = wp.HashGrid(*shape, device="cuda:0")
        grid.build(xd, 0.003)
        out = wp.zeros(len(x), dtype=int, device="cuda:0")
        wp.launch(counts, len(x), [grid.id, xd, r, out], device="cuda:0")
        np.testing.assert_array_equal(out.numpy(), expected)
