import numpy as np
import pytest
from blademaster_warp.geometry import points, reference_cloud, stencil

def test_reference_cloud_preserves_uniform_volume_and_stencils():
    p,o,c=points((.02,.02,.02),.01,"block")
    got,origin,cells=reference_cloud(p,(.02,.02,.02),.01)
    np.testing.assert_array_equal(got,p)
    np.testing.assert_array_equal(origin,o)
    np.testing.assert_array_equal(cells,c)
    for a,b in zip(stencil(p,o,.01,c),stencil(got,origin,.01,cells)):
        np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match="lattice"):
        reference_cloud(p+.001,(.02,.02,.02),.01)
    with pytest.raises(ValueError,match="duplicate"):
        reference_cloud(np.concatenate((p,p[:1])),(.02,.02,.02),.01)
