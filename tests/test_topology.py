import numpy as np
from blademaster_warp.topology import components,compatible


def test_uncut_bridges_and_persistent_conflicts():
    nodes=np.zeros((3,1),dtype=int);weights=np.ones((3,1))
    assert components(nodes,weights,[1,0,2])[1].tolist()==[3]
    assert components(nodes,weights,[1,1,2])[1].tolist()==[2,1]
    assert not compatible(1,2)
    assert not compatible(4,7)
    assert compatible(1,4)


def test_zero_weight_and_spatially_disjoint_pieces():
    nodes=np.array([[0,1],[1,2],[3,4]])
    weights=np.array([[1.,0.],[1.,0.],[1.,0.]])
    assert components(nodes,weights,[0,0,0])[1].tolist()==[1,1,1]
