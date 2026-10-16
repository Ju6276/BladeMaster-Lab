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
