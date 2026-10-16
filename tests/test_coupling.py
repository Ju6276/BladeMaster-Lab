"""External-tool tests use physical invariants, independent of DexLab."""
from dataclasses import replace
import numpy as np
import pytest
from blademaster_warp.solver import Config, Solver
from blademaster_warp.coupling import ExternalToolPort, ToolState, Reaction


def setup():
    s = Solver(replace(Config(), external_tool=True))
    # Move the specimen off the floor: only tool contact and gravity remain.
    x = s.x.numpy(); x[:, 2] += .1; s.x.assign(x)
    angle = .15
    state = ToolState([.001, 0., .132], [0., np.sin(angle/2), 0., np.cos(angle/2)],
                      [.01, 0., -.02], [0., .1, 0.])
    return s, ExternalToolPort(s), state


def test_tool_is_frozen_and_impulse_is_equal_opposite():
    s, port, state = setup()
    duration = 8*s.config.dt
    before = s.v.numpy().astype(float).sum(0)*s.mass
    reaction = port.advance(state, duration, graph=False, cutting=False)
    after = s.v.numpy().astype(float).sum(0)*s.mass
    gravity = np.array([0., 0., -9.81])*s.mass*s.n*duration
    np.testing.assert_allclose(after-before+reaction.impulse_ns, gravity, atol=3.e-9)
    for array, value in [(s.bx,state.position),(s.bq,state.orientation_xyzw),
                          (s.bv,state.linear_velocity),(s.bw,state.angular_velocity)]:
        np.testing.assert_allclose(array.numpy()[0],value,atol=2.e-8)
    assert np.linalg.norm(reaction.force_n) > .01
    np.testing.assert_allclose(reaction.force_n*duration,reaction.impulse_ns)


def test_interval_accumulation_matches_single_substeps():
    a, pa, state = setup(); b, pb, _ = setup()
    result = pa.advance(state, 9*a.config.dt, graph=True)
    pieces = [pb.advance(state,b.config.dt,graph=False) for _ in range(9)]
    np.testing.assert_allclose(result.impulse_ns, sum(r.impulse_ns for r in pieces), atol=2.e-9)
    np.testing.assert_allclose(result.angular_impulse_nms,
                               sum(r.angular_impulse_nms for r in pieces), atol=2.e-10)
    np.testing.assert_allclose(a.x.numpy(),b.x.numpy(),atol=1.e-7)
    np.testing.assert_array_equal(a.codes.numpy(),b.codes.numpy())
    # Cache hit plus history parity change, without retaining old impulses.
    ra=pa.advance(state,9*a.config.dt,graph=True)
    rb=pb.advance(state,9*b.config.dt,graph=False)
    np.testing.assert_allclose(ra.impulse_ns,rb.impulse_ns,atol=2.e-9)


def test_moment_shift_and_power_invariance():
    r = Reaction(.02,np.array([.1,.2,.3]),np.array([.01,-.02,.03]),np.array([1.,2.,3.]))
    origin=np.array([-.5,.3,2.]); shifted=r.about(origin)
    velocity=np.array([.2,-.1,.7]); omega=np.array([.3,.5,-.2])
    v_new=velocity+np.cross(omega,origin-r.origin_world_m)
    assert np.dot(r.force_n,velocity)+np.dot(r.torque_nm,omega) == pytest.approx(
        np.dot(shifted.force_n,v_new)+np.dot(shifted.torque_nm,omega))


def test_external_api_rejects_wrong_ownership_and_time():
    s,port,state=setup()
    with pytest.raises(ValueError): s.step()
    with pytest.raises(ValueError): s.advance([[0.,0.,0.]])
    for duration in [0.,-1.,np.nan,1.5*s.config.dt]:
        with pytest.raises(ValueError): port.advance(state,duration)
    assert s.step_count==0
    with pytest.raises(ValueError):
        port.advance(ToolState([0,0,0],[0,0,0,2],[0,0,0],[0,0,0]),s.config.dt)
    with pytest.raises(ValueError): replace(Config(),external_tool=True,pose_servo=True)


def test_invalid_material_interval_never_returns_a_wrench():
    s,port,state=setup()
    f=s.F.numpy();f[0,0,0]=-1.;s.F.assign(f)
    with pytest.raises(RuntimeError,match='Invalid material coupling interval'):
        port.advance(state,s.config.dt,graph=False)


def test_articulated_effective_mass_in_single_contact():
    from blademaster_warp import contact as c
    s=Solver(replace(Config(),external_tool=True))
    cfg=s.config
    s.bx.assign([[0,0,.15]])
    x=s.x.numpy()
    x[0]=[cfg.blade_half_size[0]+s.radius/2,.01,.15]
    s.x.assign(x)
    mobility=np.diag([1000,2000,3000,4000,5000,6000.])
    def impulse(a):
        s.tool_mobility.assign(a.astype(np.float32))
        s.reaction.zero_();s.torque.zero_();s.histories.zero_()
        s.launch(c.rigid,1,[s.x,s.v,s.bx,s.bv,s.bq,s.bw,s.half,cfg.blade_bevel_height,
            s.radius,cfg.grid_spacing,s.mass,cfg.blade_mass,s.inertia_inv,s.tool_mobility,-1,
            cfg.dt,cfg.normal_scale,cfg.friction_static,cfg.friction_dynamic,
            cfg.tangent_stiffness,cfg.tangent_damping,s.histories,s.damage,s.threshold,0,
            s.impulses,s.reaction,s.torque,s.board,s.work])
        return s.reaction.numpy()[0,0]
    prescribed=impulse(np.zeros((6,6)))
    articulated=impulse(mobility)
    direction=np.array([1,0,0,0,0,-.01])
    expected_ratio=1/(1+s.mass*(direction@mobility@direction))
    assert prescribed<0
    assert articulated/prescribed == pytest.approx(expected_ratio,rel=2.e-6)


def test_mobility_rejects_invalid_mass_response():
    for invalid in [np.eye(5),-np.eye(6),np.full((6,6),np.nan),np.triu(np.ones((6,6)))]:
        with pytest.raises(ValueError,match='Mobility'):
            ToolState([0,0,0],[0,0,0,1],[0,0,0],[0,0,0],invalid).checked()
