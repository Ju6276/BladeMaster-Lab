import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from blademaster_warp.franka import Franka, HOME, quat_wxyz

VENDOR = Path(__file__).resolve().parents[1] / "vendor/serl_controller"


def upstream_impedance_control():
    pytest.importorskip("scipy")
    manifest = json.loads((VENDOR / "manifest.json").read_text())
    path = VENDOR / "impedance.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    spec = importlib.util.spec_from_file_location("upstream_serl_impedance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.impedance_control


def test_impedance_matches_upstream_serl():
    import mujoco as mj

    reference = upstream_impedance_control()
    robot = Franka(controller="serl")
    rng = np.random.default_rng(0)
    for _ in range(20):
        robot.data.qpos[:7] = HOME[:7] + rng.normal(scale=0.2, size=7)
        robot.data.qvel[:7] = rng.normal(scale=0.5, size=7)
        mj.mj_forward(robot.model, robot.data)
        target = robot.control_position + rng.normal(scale=0.02, size=3)
        axis = rng.normal(size=3)
        rotation = np.empty(9)
        quat = np.r_[np.cos(0.2), np.sin(0.2) * axis / np.linalg.norm(axis)]
        mj.mju_quat2Mat(rotation, quat)
        quat_des = quat_wxyz(robot.control_rotation @ rotation.reshape(3, 3))
        expected = reference(
            robot.model,
            robot.data,
            robot.control_site,
            np.arange(7),
            pos=target,
            ori=quat_des,
            joint=HOME[:7],
        )
        actual = robot.impedance(robot.data, target, quat_des, HOME[:7])
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
