"""Dynamics checks independent of a successful-looking cutting video."""
import numpy as np
import pytest
import mujoco as mj
from blademaster_warp.franka import Franka


def test_banana_material_and_blade_match_standalone_driver():
    from dataclasses import asdict
    from types import SimpleNamespace
    from blademaster_warp.run import preset
    from blademaster_warp.franka_run import material_config
    from blademaster_warp.franka import HALF,BEVEL
    standalone=asdict(preset('paper-material'))
    robot=asdict(material_config(SimpleNamespace(material='paper-banana',dt=None,spacing=None,cutting_energy=None)))
    differences={k for k in robot if robot[k]!=standalone[k]}
    assert differences=={'external_tool'}
    assert robot['external_tool'] and not standalone['external_tool']
    assert HALF==robot['blade_half_size'] and BEVEL==robot['blade_bevel_height']==.005


def test_explicit_material_overrides_do_not_change_other_parameters():
    from dataclasses import asdict
    from types import SimpleNamespace
    from blademaster_warp.franka_run import material_config
    base=asdict(material_config(SimpleNamespace(material='paper-banana',dt=None,spacing=None,cutting_energy=None)))
    changed=asdict(material_config(SimpleNamespace(material='paper-banana',dt=.0000125,spacing=.003,cutting_energy=750.)))
    assert {k for k in base if base[k]!=changed[k]}=={'dt','grid_spacing','cutting_energy'}


def test_jacobian_finite_difference():
    r=Franka()
    analytic=r.jacobian()[:3,:7]
    q=r.data.qpos.copy()
    numerical=np.zeros((3,7))
    for j in range(7):
        r.data.qpos[:]=q; r.data.qpos[j]+=1.e-6
        mj.mj_forward(r.model,r.data); plus=r.position
        r.data.qpos[:]=q; r.data.qpos[j]-=1.e-6
        mj.mj_forward(r.model,r.data)
        numerical[:,j]=(plus-r.position)/2.e-6
    np.testing.assert_allclose(analytic,numerical,atol=1.e-8)


def test_wrench_virtual_work_and_effective_mass():
    r=Franka()
    wrench=np.array([2.,-3.,1.,.05,-.1,.02])
    j=r.jacobian()
    generalized=np.zeros(r.model.nv)
    mj.mj_applyFT(r.model,r.data,wrench[:3],wrench[3:],r.position,r.body,generalized)
    np.testing.assert_allclose(generalized,j.T@wrench,atol=1.e-12)
    velocity=np.arange(r.model.nv)*.03
    np.testing.assert_allclose(velocity@generalized,(j@velocity)@wrench,atol=1.e-12)
    mobility=j@np.linalg.solve(r.mass(),j.T)
    assert np.linalg.eigvalsh(mobility).min()>0
    np.testing.assert_allclose(mobility@wrench,j@np.linalg.solve(r.mass(),generalized),atol=1.e-12)


@pytest.mark.parametrize('controller',['libfranka','serl'])
def test_bias_hold_and_external_force_changes_motion(controller):
    free=Franka(controller=controller)
    held=Franka(controller=controller)
    target=free.position
    for _ in range(500):
        free.step(target)
        held.step(target,np.array([0,0,2.,0,0,0]))
    assert np.linalg.norm(free.position-target)<1.e-4
    # Reaction is not discarded and the end effector is not teleported.
    assert held.position[2]-free.position[2]>.002
    assert np.linalg.norm(held.data.qpos[:7]-free.data.qpos[:7])>.005
    assert np.max(np.abs(held.data.ctrl[:7]))<87


def test_frictional_grasp_lifts_and_releases_free_knife():
    r=Franka(attachment='grasp')
    assert r.model.nq==16 and r.model.nv==15
    assert r.model.jnt_type[r.model.joint('knife_free').id]==mj.mjtJoint.mjJNT_FREE
    # Only the upstream finger synchronizer remains; no tool weld.
    assert r.model.neq==1
    for _ in range(500):r.step(r.initial_position)
    settled=r.position.copy();relative=r.position-r.control_position
    for i in range(1000):
        r.step(r.initial_position+[0,0,.03*min(i/500,1)])
    assert r.position[2]-settled[2]>.02
    assert np.linalg.norm(r.position-r.control_position-relative)<.003
    height=r.position[2]
    r.release()
    for _ in range(200):r.step(r.initial_position+[0,0,.03])
    assert r.position[2]<height-.015
