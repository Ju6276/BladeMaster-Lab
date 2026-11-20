import hashlib,json
from pathlib import Path
import numpy as np
import pytest
import warp as wp
from blademaster_warp.mesh_geometry import MeshGeometry,MeshTool,mesh_query,edge_at,side_surface
from blademaster_warp.mesh_runtime import mesh_kernel
from blademaster_warp import contact,blade_constraint

@wp.kernel
def queries(g:MeshTool,points:wp.array(dtype=wp.vec3),out:wp.array2d(dtype=float)):
 i=wp.tid();p=points[i]
 c,n,d=mesh_query(p,g);e,inside=edge_at(p[1],g)
 s=1.0
 if p[0]<0.0:s=-1.0
 near,nn,found=side_surface(p,s,g)
 out[i,0]=d;out[i,1]=n[0];out[i,2]=e[2];out[i,3]=float(inside)
 out[i,4]=near[0];out[i,5]=float(found)

@pytest.fixture
def geometry(tmp_path):
 # Closed convex blade, sloping cutting edge: z=-20mm+0.2*y.
 v=np.array([[-.002,-.04,-.028],[.002,-.04,-.028],[.002,.04,-.012],[-.002,.04,-.012],
             [-.002,-.04,.02],[.002,-.04,.02],[.002,.04,.02],[-.002,.04,.02]])
 quads=[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
 f=np.array([(a,b,c) for a,b,c,d in quads]+[(a,c,d) for a,b,c,d in quads])
 path=tmp_path/"tool.npz";np.savez(path,vertices_blademaster=v,faces=f,edge_blademaster=[[0,-.04,-.028],[0,.04,-.012]])
 manifest=tmp_path/"tool.json";manifest.write_text(json.dumps(dict(geometry=str(path),geometry_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
 return manifest

def test_mesh_distance_normals_edge_and_original_side(geometry):
 g=MeshGeometry(geometry,"cuda:0")
 points=np.array([[.003,0,0],[.001,0,0],[-.003,.02,0],[0,.05,0]],np.float32)
 out=wp.zeros((4,6),device="cuda:0");wp.launch(queries,4,[g.data,wp.array(points,dtype=wp.vec3,device="cuda:0"),out],device="cuda:0")
 a=out.numpy()
 np.testing.assert_allclose(a[:3,0],[.001,-.001,.001],atol=2e-8)
 np.testing.assert_allclose(a[:3,1],[1,1,-1],atol=1e-5)
 np.testing.assert_allclose(a[:3,2],[-.02,-.02,-.016],atol=2e-8)
 np.testing.assert_equal(a[:,3],[1,1,1,0])
 np.testing.assert_allclose(a[:3,4],[.002,.002,-.002],atol=2e-8)
 np.testing.assert_equal(a[:,5],[1,1,1,0])

def test_every_geometry_dependent_kernel_dispatches(geometry):
 g=MeshGeometry(geometry,"cuda:0")
 for kernel in (contact.rigid_and_speed,contact.extra_blade,contact.assign,contact.count_embedded,blade_constraint.prepare):
  for batch in (False,True):
   target,args=mesh_kernel(kernel,[],g,batch)
   assert target is not None and args==[g.data]
   assert target.module.name.endswith(("batch_mesh_" if batch else "mesh_")+kernel.module.name.rsplit(".",1)[-1])
 target,args=mesh_kernel(contact.clear_step,[],g)
 assert target is None and args==[]

def test_asset_checksum_is_required(geometry):
 meta=json.loads(geometry.read_text());meta["geometry_sha256"]="0"*64;geometry.write_text(json.dumps(meta))
 with pytest.raises(ValueError,match="hash"):MeshGeometry(geometry,"cuda:0")

def test_actual_ikea_exterior_below_edge_is_outside():
    path=Path('/dataHDD/dong/Assets/Cutting_DEV/robots/ur10e_ati_ikea_estimated_v1/cutting_geometry.json')
    if not path.exists():pytest.skip('user CAD asset not installed')
    g=MeshGeometry(path,'cuda:0')
    # A normal-based signed mesh query in the real prism run classified
    # this exterior location as 40 mm INSIDE the blade.
    center=np.array([-.033566,.011533,-.031440],np.float32)
    offsets=np.stack(np.meshgrid(*([np.linspace(-2e-6,2e-6,9)]*3)),axis=-1).reshape(-1,3)
    points=(center+offsets).astype(np.float32)
    out=wp.zeros((len(points),6),device='cuda:0')
    wp.launch(queries,len(points),[g.data,wp.array(points,dtype=wp.vec3,device='cuda:0'),out],device='cuda:0')
    assert (out.numpy()[:,0]>.03).all()

def test_mesh_contact_uses_each_world_material_parameters(geometry):
    from dataclasses import replace
    from blademaster_warp.batched import BatchedSolver
    from blademaster_warp.solver import Solver
    from blademaster_warp.presets import banana_config
    from blademaster_warp.coupling import ExternalToolPort,ToolState
    cfg=replace(banana_config(),grid_storage='compact',external_tool=True,tool_substep='hold',
        size=(.03,.03,.03),shape='block',dt=.002/94,blade_geometry_asset=str(geometry))
    batch=BatchedSolver(cfg,2)
    scales=np.array([.6,1.4])
    batch.set_material_scales([1,1],youngs_modulus=scales,density=scales,cutting_energy=scales)
    scalar=[Solver(replace(cfg,young_modulus=cfg.young_modulus*s,density=cfg.density*s,
                           cutting_energy=cfg.cutting_energy*s)) for s in scales]
    ports=[ExternalToolPort(s) for s in scalar]
    packed=np.zeros((2,49),np.float32);packed[:,6]=1;packed[:,9]=-.1
    peaks=np.zeros(2)
    for step in range(1):
        packed[:,2]=.049-.1*(step+1)*.002
        out=batch.advance(packed,[1,1]).numpy()
        assert not out[:,6:9].any()
        for i,port in enumerate(ports):
            ref=port.advance(ToolState(packed[i,:3],packed[i,3:7],packed[i,7:10],packed[i,10:13]),.002).about(np.zeros(3))
            np.testing.assert_allclose(out[i,:6]/.002,np.r_[ref.force_n,ref.torque_nm],atol=.02,rtol=.02)
            np.testing.assert_allclose(batch.x.numpy()[i],scalar[i].x.numpy(),atol=2e-5,rtol=0)
            peaks[i]=max(peaks[i],np.linalg.norm(out[i,:3]/.002))
    assert (peaks>.01).all()
    assert not np.isclose(peaks[0],peaks[1],rtol=.01)
