"""Operational-space impedance controller of SERL++ (vendor/serl_controller/impedance.py).

Same equations and default gains as the upstream impedance_control, with the
quaternion algebra done by MuJoCo instead of scipy and the gains computed once.
tests/test_impedance.py checks equality against the pinned upstream file.
"""

import mujoco as mj
import numpy as np


def _gains(stiffness, damping_ratio):
    stiffness = np.asarray(stiffness, dtype=float)
    return stiffness, damping_ratio * 2.0 * np.sqrt(stiffness)


class ImpedanceController:
    def __init__(
        self,
        model,
        site_id,
        dof_ids,
        pos_gains=(100.0, 100.0, 100.0),
        ori_gains=(2000.0, 2000.0, 200.0),
        damping_ratio=1.0,
        nullspace_stiffness=0.5,
    ):
        self.model = model
        self.site_id = site_id
        self.dof_ids = np.asarray(dof_ids)
        self.kp_pos, self.kd_pos = _gains(pos_gains, damping_ratio)
        self.kp_ori, self.kd_ori = _gains(ori_gains, damping_ratio)
        self.kp_joint, self.kd_joint = _gains(np.full(len(self.dof_ids), nullspace_stiffness), damping_ratio)
        self._jacobian = np.zeros((6, model.nv))
        self._mass = np.zeros((model.nv, model.nv))

    def orientation_error(self, data, quat_des):
        """Rotation vector of q_current * q_des^-1 (shortest rotation)."""
        quat = np.empty(4)
        mj.mju_mat2Quat(quat, data.site_xmat[self.site_id])
        inverse_des = np.empty(4)
        mj.mju_negQuat(inverse_des, quat_des)
        difference = np.empty(4)
        mj.mju_mulQuat(difference, quat, inverse_des)
        rotation_vector = np.empty(3)
        mj.mju_quat2Vel(rotation_vector, difference, 1.0)
        return rotation_vector

    def __call__(self, data, pos, quat_des, joint):
        """Joint torques for target position pos, orientation quat_des (w, x, y, z) and nullspace posture joint."""
        ids = self.dof_ids
        q = data.qpos[ids]
        dq = data.qvel[ids]
        mj.mj_jacSite(self.model, data, self._jacobian[:3], self._jacobian[3:], self.site_id)
        J = self._jacobian[:, ids]

        twist = J @ dq
        ddx = -self.kp_pos * (data.site_xpos[self.site_id] - pos) - self.kd_pos * twist[:3]
        dw = -self.kp_ori * self.orientation_error(data, quat_des) - self.kd_ori * twist[3:]

        mj.mj_fullM(self.model, self._mass, data.qM)
        M_inv = np.linalg.inv(self._mass[np.ix_(ids, ids)])
        Mx_inv = J @ M_inv @ J.T
        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            Mx = np.linalg.inv(Mx_inv)
        else:
            Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

        tau = J.T @ Mx @ np.r_[ddx, dw]
        ddq = -self.kp_joint * (q - joint) - self.kd_joint * dq
        Jnull = M_inv @ J.T @ Mx
        tau += (np.eye(len(q)) - J.T @ Jnull.T) @ ddq
        return tau + data.qfrc_bias[ids]
