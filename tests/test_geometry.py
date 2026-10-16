import numpy as np
import pytest
from blademaster_warp.geometry import points,levelset


def test_banana_must_not_silently_clip_shape():
    with pytest.raises(ValueError,match='clipped'):
        points((.072,.024,.03),.006,'banana')
    a,_,_=points((.072,.06,.03),.006,'banana')
    b,_,_=points((.072,.072,.03),.006,'banana')
    np.testing.assert_allclose(a,b)
    assert len(a)==892
    assert (levelset(a,(.072,.06,.03),'banana')<=0).all()
