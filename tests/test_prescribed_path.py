import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation

from blademaster_warp.prescribed_path import PrescribedPath
from blademaster_warp.differentiable_contact import Tool
from blademaster_warp.differentiable_geometry import GeometryEvents
from blademaster_warp.differentiable_material import MaterialTrajectory
from blademaster_warp.solver import Config


@pytest.mark.parametrize("maximum_step", [0.0004, 0.0002, 0.0001])
def test_continuous_multiphase_pose_and_controls_against_independent_rotation(maximum_step):
    model = MaterialTrajectory(Config(size=(0.012, 0.012, 0.012), grid_spacing=0.006), precision="float64")
    driver = GeometryEvents()
    driver.model = model
    durations = np.array([0.0011, 0.0007, 0.0013, 0.0009])
    path = PrescribedPath(durations, maximum_step)
    rng = np.random.default_rng(720)
    values = rng.normal(size=(4, 6)) * [1.0, 1.0, 1.0, 20.0, 20.0, 20.0]
    controls = model.array(values, float)
    origin = np.array([0.001, -0.002, 0.018])
    rotation = Rotation.from_rotvec([0.2, -0.1, 0.3])
    tool = Tool(
        model.array([origin], wp.vec3),
        model.zeros(1, wp.vec3),
        model.array([rotation.as_quat()], wp.quat),
        model.zeros(1, wp.vec3),
    )
    initial_position = tool.position
    with wp.Tape() as tape:
        for i in range(len(path.steps)):
            controlled, duration = path.inputs(i, tool, controls)
            assert controlled.position is tool.position
            assert controlled.rotation is tool.rotation
            tool = driver.pose(controlled, duration)
    position_seed, rotation_seed = rng.normal(size=3), rng.normal(size=4)
    tape.backward(
        grads={
            tool.position: model.array([position_seed], wp.vec3, False),
            tool.rotation: model.array([rotation_seed], wp.quat, False),
        }
    )

    def independent(v):
        orientation = rotation
        for dt, row in zip(durations, v):
            orientation = Rotation.from_rotvec(dt * row[3:]) * orientation
        position = origin + durations @ v[:, :3]
        return position, orientation.as_quat()

    expected_position, expected_rotation = independent(values)
    np.testing.assert_allclose(tool.position.numpy()[0], expected_position, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(tool.rotation.numpy()[0], expected_rotation, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(controls.grad.numpy()[:, :3], durations[:, None] * position_seed, rtol=1e-13)
    np.testing.assert_allclose(initial_position.grad.numpy()[0], position_seed, rtol=1e-13)
    for phase in range(4):
        for axis in range(3, 6):
            epsilon = 1e-5
            plus, minus = values.copy(), values.copy()
            plus[phase, axis] += epsilon
            minus[phase, axis] -= epsilon
            fd = (independent(plus)[1] - independent(minus)[1]) @ rotation_seed / (2 * epsilon)
            np.testing.assert_allclose(controls.grad.numpy()[phase, axis], fd, rtol=2e-7, atol=2e-11)
