from dataclasses import replace
import numpy as np
import pytest
from blademaster_warp.batched import BatchedSolver
from blademaster_warp.solver import Solver
from blademaster_warp.presets import banana_config
from blademaster_warp.coupling import ExternalToolPort, ToolState


def configuration():
    return replace(banana_config(), grid_storage="compact", external_tool=True,
                   tool_substep="hold", size=(.03,.03,.03), shape="block",
                   dt=.002/94, blade_half_size=(.00175,.075,.02952),
                   blade_bevel_height=.00004, blade_edge_half_width=.00004)


def test_independent_materials_match_scalar_contact_and_live_graph_updates():
    cfg=configuration()
    batch=BatchedSolver(cfg,2)
    e=np.array([.5,2.]);rho=np.array([1.25,.8]);energy=np.array([.5,2.])
    batch.set_material_scales([1,1],youngs_modulus=e,density=rho,cutting_energy=energy)
    scalar=[Solver(replace(cfg,young_modulus=cfg.young_modulus*e[i],
                           density=cfg.density*rho[i],cutting_energy=cfg.cutting_energy*energy[i]))
            for i in range(2)]
    ports=[ExternalToolPort(s) for s in scalar]
    packed=np.zeros((2,49),np.float32);packed[:,6]=1
    errors=np.zeros(2);peaks=np.zeros(2)
    for step in range(32):
        packed[:,2]=.059-.1*(step+1)*.002
        packed[:,9]=-.1
        out=batch.advance(packed,[1,1]).numpy()
        assert not out[:,6:9].any()
        for i,port in enumerate(ports):
            ref=port.advance(ToolState(packed[i,:3],packed[i,3:7],packed[i,7:10],packed[i,10:13]),.002).about(np.zeros(3))
            errors[i]=max(errors[i],np.max(np.abs(out[i,:6]/.002-np.r_[ref.force_n,ref.torque_nm])))
            peaks[i]=max(peaks[i],np.linalg.norm(out[i,:3]/.002))
            np.testing.assert_allclose(batch.x.numpy()[i],scalar[i].x.numpy(),atol=2e-5,rtol=0)
    print("per-world reaction errors",errors,"peaks",peaks,flush=True)
    assert (peaks>.01).all()
    assert (errors<.02).all()
    assert not np.isclose(peaks[0],peaks[1],rtol=.01)

    names=("x","v","F","Fp","codes","damage","histories","impulses")
    untouched={n:getattr(batch,n).numpy()[1].copy() for n in names}
    graph_ids={k:id(v) for k,v in batch._graphs.items()}
    batch.reset([1,0],raw=True)
    batch.set_material_scales([1,0],youngs_modulus=[1.3,999.],density=[1.1,999.],
                              cutting_energy=[.9,999.])
    for name,value in untouched.items():
        np.testing.assert_array_equal(getattr(batch,name).numpy()[1],value)
    np.testing.assert_allclose(batch.material_parameters["mass"].numpy(),
                               cfg.density*batch.volume*np.array([1.1,.8]),rtol=1e-6)
    np.testing.assert_allclose(batch.material_parameters["threshold"].numpy(),
                               batch.threshold*np.array([.9,2]),rtol=1e-6)
    new=Solver(replace(cfg,young_modulus=cfg.young_modulus*1.3,
                       density=cfg.density*1.1,cutting_energy=cfg.cutting_energy*.9))
    port=ExternalToolPort(new)
    packed[:,2]=.065;packed[:,9]=0
    for _ in range(3):
        batch.advance(packed,[1,0],cutting=True).numpy()
        port.advance(ToolState(packed[0,:3],packed[0,3:7],packed[0,7:10],packed[0,10:13]),.002,cutting=True)
    np.testing.assert_allclose(batch.x.numpy()[0],new.x.numpy(),atol=2e-5,rtol=0)
    for name,value in untouched.items():
        np.testing.assert_array_equal(getattr(batch,name).numpy()[1],value)
    batch.save_reset(mask=[1,0]) # the other world already has cut codes
    saved=batch.x.numpy()[0].copy()
    batch.advance(packed,[1,0],cutting=True).numpy()
    batch.reset([1,0])
    np.testing.assert_array_equal(batch.x.numpy()[0],saved)
    assert all(id(batch._graphs[k])==v for k,v in graph_ids.items())


def test_material_input_validation_is_atomic():
    batch=BatchedSolver(configuration(),2)
    old={k:v.copy() for k,v in batch.material_scales.items()}
    with pytest.raises(ValueError):
        batch.set_material_scales([1,0],youngs_modulus=[2,1],density=[0,1],cutting_energy=[1,1])
    for key in old:np.testing.assert_array_equal(batch.material_scales[key],old[key])
