from dataclasses import replace
import numpy as np
import warp as wp
from blademaster_warp.solver import Solver,Config
from blademaster_warp import tool


def run_tool(free=False):
    cfg=replace(Config(),dt=.00025,pose_servo=not free,free_blade=free,blade_bevel_height=.005)
    s=Solver(cfg)
    s.bx.assign([[.12,0.,.05]])
    angle=.2
    q=np.array([0,np.sin(angle/2),0,np.cos(angle/2)])
    s.bq.assign([q])
    targets=wp.array([[.12,0.,-.01]],dtype=wp.vec3,device=s.device)
    rotations=wp.array([q],dtype=wp.quat,device=s.device)
    states=[]
    for i in range(3000):
        s.launch(tool.step,1,[s.bx,s.bv,s.bq,s.bw,targets,rotations,0,
            s.reaction,s.torque,s.tool_table_impulse,cfg.dt,cfg.blade_mass,s.half,
            cfg.blade_bevel_height,cfg.drive_stiffness,cfg.drive_damping,cfg.drive_force_limit,
            cfg.rotational_stiffness,cfg.rotational_damping,s.inertia_inv,cfg.friction_dynamic,int(free)])
        if i%20==0:
            pos=s.bx.numpy()[0];rot=s.bq.numpy()[0]
            from scipy.spatial.transform import Rotation
            h=np.array(cfg.blade_half_size)
            vertices=np.array([[0,-h[1],-h[2]],[0,h[1],-h[2]],
                              [h[0],-h[1],-h[2]+.005],[-h[0],h[1],-h[2]+.005]])
            minimum=(Rotation.from_quat(rot).apply(vertices)+pos)[:,2].min()
            states.append(minimum)
    assert np.isfinite(s.bv.numpy()).all()
    assert min(states)>-2e-5
    assert max(states)-min(states)>.01
    return s


def test_six_dof_servo_contacts_plane_without_target_clip():
    s=run_tool()
    assert s.bx.numpy()[0,2]>.018
    assert s.tool_table_impulse.numpy()[0,2]>0


def test_free_tool_falls_and_contacts_plane():
    run_tool(free=True)


def test_servo_graph_accepts_rotations_and_material_reaction():
    cfg=replace(Config(),pose_servo=True,dt=.00025)
    eager,graph=Solver(cfg),Solver(cfg)
    for s in (eager,graph):s.bx.assign([[0.,0.,.045]])
    targets=np.tile([0.,0.,.02],(40,1))
    q=np.tile([0.,np.sin(.05),0.,np.cos(.05)],(40,1))
    eager.advance(targets,orientations=q,graph=False)
    graph.advance(targets,orientations=q,graph=True)
    a,b=eager.state(),graph.state()
    np.testing.assert_allclose(a['blade_orientation'],b['blade_orientation'],atol=3e-6)
    np.testing.assert_allclose(a['position'],b['position'],atol=3e-7)
    assert abs(b['blade_orientation'][0,1])>0.001
