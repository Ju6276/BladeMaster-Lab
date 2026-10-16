import numpy as np
import pytest
from blademaster_warp.geometry import points, levelset


def test_banana_must_not_silently_clip_shape():
    with pytest.raises(ValueError, match="clipped"):
        points((0.072, 0.024, 0.03), 0.006, "banana")
    a, _, _ = points((0.072, 0.06, 0.03), 0.006, "banana")
    b, _, _ = points((0.072, 0.072, 0.03), 0.006, "banana")
    np.testing.assert_allclose(a, b)
    assert len(a) == 892
    assert (levelset(a, (0.072, 0.06, 0.03), "banana") <= 0).all()


@pytest.mark.parametrize("spacing", [0.004, 0.007])
def test_spacing_that_does_not_divide_the_size_still_covers_the_banana(spacing):
    from blademaster_warp.geometry import BANANA_DEMO_SIZE_M

    size = np.array(BANANA_DEMO_SIZE_M)
    fine, _, _ = points(size, 0.003, "banana")
    x, origin, cells = points(size, spacing, "banana")
    assert np.all(cells * spacing >= size - 1.0e-12)
    assert np.all((x >= origin) & (x <= origin + cells * spacing))
    # Same tips as the 3 mm (dividing) sampling, within one particle spacing.
    assert abs(np.ptp(x[:, 0]) - np.ptp(fine[:, 0])) < spacing / 2
    block, origin, _ = points((0.03, 0.03, 0.03), spacing, "block")
    assert np.all(block - origin <= 0.03)
