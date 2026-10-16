import numpy as np
from blademaster_warp.surface import LinkedSurface,MaterialEmbedding
from blademaster_warp.geometry import points


def test_original_banana_tips_remain_connected_on_refined_surface():
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from blademaster_warp.geometry import BANANA_DEMO_SIZE_M
    s=LinkedSurface(BANANA_DEMO_SIZE_M,.003,'banana')
    mesh=s.mesh();tri=mesh.triangles
    assert (edge_incidence(mesh)==2).all()
    edges=np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]])
    graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(mesh.reference),len(mesh.reference)))
    assert connected_components(graph,directed=False,return_labels=False)==1


def edge_incidence(mesh):
    triangles=mesh.triangles
    edges=np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1)
    return np.unique(edges,axis=0,return_counts=True)[1]


def test_original_banana_surface_has_consistent_outward_winding():
    # QEF vertices on curved/flat boundaries can give the wrong face normal
    # despite a closed mesh. Primal cell ordering must determine its winding.
    from blademaster_warp.benchmark_surface import mesh_violations
    s=LinkedSurface((.12,.06,.03),.003,'banana')
    for cut in [False,True]:
        if cut:
            s.sweep([[0,-.04,.04],[0,.04,.04]],[[0,-.04,-.01],[0,.04,-.01]],normal_world=[1,0,0])
        assert not any(mesh_violations(s.mesh().triangles).values())


def test_reference_mesh_cache_invalidates_on_cuts_without_freezing_motion():
    s=LinkedSurface((.06,.03,.03),.006)
    first=s.mesh()
    assert s.mesh() is first
    s.sweep([[0,-.04,.04],[0,.04,.04]],[[0,-.04,-.01],[0,.04,-.01]])
    cut=s.mesh()
    assert cut is not first
    assert s.mesh() is cut
    recomputed=s.mesh(use_cache=False)
    for name in ['reference','triangles','codes','cells','weights']:
        np.testing.assert_array_equal(getattr(cut,name),getattr(recomputed,name))
    ref,origin,cells=points(s.size,.006,'block')
    emb=MaterialEmbedding(s.reference,ref,.006,origin,cells)
    codes=np.where(ref[:,0]<0,1,2);F=np.tile(np.eye(3),(len(ref),1,1))
    before=emb.vertices(s.mesh(),ref,F,codes)
    after=emb.vertices(s.mesh(),ref+[.01,.02,.03],F,codes)
    np.testing.assert_allclose(after-before,np.tile([.01,.02,.03],(len(before),1)),atol=1.e-12)


def test_original_surface_and_finite_sweep_are_closed_and_persistent():
    s=LinkedSurface((.06,.03,.03),.006)
    initial=s.mesh()
    assert len(initial.triangles)>0
    assert (edge_incidence(initial)==2).all()
    assert s.sweep([[.2,-.04,.04],[.2,.04,.04]],[[.2,-.04,-.01],[.2,.04,-.01]])==0
    assert not s.broken.any()
    assert s.sweep([[0,-.04,.04],[0,.04,.04]],[[0,-.04,-.01],[0,.04,-.01]])>0
    cut=s.mesh()
    assert len(cut.reference)>len(initial.reference)
    assert (edge_incidence(cut)==2).all()
    old=s.broken.copy()
    s.sweep([[0,-.04,-.01],[0,.04,-.01]],[[0,-.04,.04],[0,.04,.04]])
    np.testing.assert_array_equal(s.broken,old)
    assert len(s.mesh().reference)==len(cut.reference)


def test_affine_exact_embedding_and_cut_faces_separate():
    s=LinkedSurface((.06,.03,.03),.006)
    ref,_,_=points(s.size,.006,'block')
    emb=MaterialEmbedding(s.reference,ref,.006)
    mesh=s.mesh()
    A=np.array([[1,.1,0],[0,1.2,0],[0,0,.9]])
    t=np.array([.1,.2,.3]);F=np.tile(A,(len(ref),1,1))
    x=ref@A.T+t
    np.testing.assert_allclose(emb.vertices(mesh,x,F,np.zeros(len(ref),int)),mesh.reference@A.T+t,atol=1.e-10)
    s.sweep([[0,-.04,-.01],[0,.04,-.01]],[[0,-.04,.04],[0,.04,.04]])
    mesh=s.mesh();codes=np.where(ref[:,0]<0,1,2)
    displacement=np.where(codes==1,-.003,.003)
    x=ref.copy();x[:,0]+=displacement
    moved=emb.vertices(mesh,x,np.tile(np.eye(3),(len(ref),1,1)),codes)
    for code,dx in [(1,-.003),(2,.003)]:
        selected=mesh.codes==code
        assert selected.any()
        np.testing.assert_allclose(moved[selected,0],mesh.reference[selected,0]+dx,atol=1.e-10)


def test_embedding_uses_material_grid_not_padded_surface_grid():
    size=(.06,.03,.03);h=.006
    s=LinkedSurface(size,h)
    ref,origin,cells=points(size,h,'block')
    emb=MaterialEmbedding(s.reference,ref,h,origin,cells)
    s.sweep([[0,-.04,.04],[0,.04,.04]],[[0,-.04,-.01],[0,.04,-.01]],normal_world=[1,0,0])
    mesh=s.mesh();codes=np.where(ref[:,0]<0,1,2);x=ref.copy()
    x[:,0]+=np.where(codes==1,-.01,.01)
    result=emb.vertices(mesh,x,np.tile(np.eye(3),(len(ref),1,1)),codes)
    assert np.isfinite(result).all()
    for code,delta in [(1,-.01),(2,.01)]:
        selected=mesh.codes==code
        np.testing.assert_allclose(result[selected,0],mesh.reference[selected,0]+delta,atol=1.e-10)


def test_partial_cut_front_uses_connected_region_hermite():
    s=LinkedSurface((.06,.03,.03),.006)
    s.sweep([[0,-.04,.04],[0,.04,.04]],[[0,-.04,.015],[0,.04,.015]],normal_world=[1,0,0])
    mesh=s.mesh()
    assert (edge_incidence(mesh)==2).all()
    # Shallow intersecting cut; the uncut lower volume remains connected.
    s.sweep([[-.04,0,.04],[.04,0,.04]],[[-.04,0,.015],[.04,0,.015]],cut_index=1,normal_world=[0,1,0])
    assert (edge_incidence(s.mesh())==2).all()


def test_moving_material_link_crosses_stationary_finite_blade():
    from blademaster_warp.surface import moving_segment_hits
    import numpy as np
    a=np.array([[-1.,0.,1.],[-1.,3.,1.]])
    b=np.array([[1.,0.,1.],[1.,3.,1.]])
    knife=np.array([[0.,-1.,0.],[0.,1.,0.]])
    hit,alpha,t=moving_segment_hits(a,b,a-[0,0,2],b-[0,0,2],knife,knife)
    np.testing.assert_array_equal(hit,[True,False])
    assert abs(alpha[0]-.5)<1.e-12 and abs(t[0]-.5)<1.e-12


def test_relative_sweep_separates_surface_without_mechanical_labels():
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    s=LinkedSurface((.06,.03,.03),.006)
    knife=np.array([[0,-.04,0],[0,.04,0]])
    s.sweep(knife,knife,s.reference-[0,0,.04],normal_world=[1,0,0],
            previous_nodes=s.reference+[0,0,.01])
    mesh=s.mesh();tri=mesh.triangles
    assert (edge_incidence(mesh)==2).all()
    edges=np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]])
    adjacency=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),
                         shape=(len(mesh.reference),len(mesh.reference)))
    count,_=connected_components(adjacency,directed=False)
    assert count==2

