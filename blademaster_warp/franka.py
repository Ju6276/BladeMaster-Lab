"""Torque-driven MuJoCo Panda with a fixed-mount or grasped knife."""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
from .geometry import DEMO_KNIFE_HALF_SIZE_M, DEMO_KNIFE_BEVEL_HEIGHT_M
from .impedance import ImpedanceController

SOURCE = Path(__file__).resolve().parents[1] / "vendor/mujoco_menagerie/franka_emika_panda/panda.xml"
HOME = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7853981634, 0.035, 0.035])
HALF = DEMO_KNIFE_HALF_SIZE_M
BEVEL = DEMO_KNIFE_BEVEL_HEIGHT_M
# Cutting board half extents (x, y), centred under the material; shared with the MPM board.
BOARD_HALF_EXTENT = (0.17, 0.16)
# Fixed mount: a cylinder between the fingers; the fingers close onto it.
MOUNT_RADIUS = 0.008
# Upstream actuator8: ctrl 0..255 maps to 0..0.04 m opening per finger.
FINGER_CTRL_PER_M = 255 / 0.04


def quat_wxyz(matrix):
    """Unit quaternion (w, x, y, z) of a 3x3 rotation matrix."""
    result = np.empty(4)
    mj.mju_mat2Quat(result, np.ascontiguousarray(matrix, dtype=np.float64).ravel())
    return result


def quat_xyzw(matrix):
    w, x, y, z = quat_wxyz(matrix)
    return np.array([x, y, z, w])


def numbers(x):
    return " ".join(format(float(v), ".14g") for v in np.asarray(x).ravel())


def make_model(interval=0.001, blade_mass=0.1, attachment="fixed"):
    if attachment not in ("fixed", "grasp"):
        raise ValueError("Unknown attachment")
    root = ET.parse(SOURCE).getroot()
    root.find("compiler").set("meshdir", str(SOURCE.parent / "assets"))
    # FK only to set the constant mounting rotation and board position.
    original = mj.MjModel.from_xml_path(str(SOURCE))
    initial = mj.MjData(original)
    initial.qpos[:] = HOME
    mj.mj_forward(original, initial)
    hand = original.body("hand").id
    hand_r = initial.xmat[hand].reshape(3, 3)
    tool_origin = initial.xpos[hand] + hand_r @ np.array([0.0, 0.0, 0.16])
    material_origin = tool_origin - np.array([0.0, 0.0, HALF[2] + 0.04])
    mounting_q = quat_wxyz(hand_r.T)
    root.find("option").set("timestep", str(interval))
    root.remove(root.find("keyframe"))  # Its ctrl is a position target, not torque.
    actuator = root.find("actuator")
    for index, node in enumerate(list(actuator)[:7]):
        limit = 87 if index < 4 else 12
        actuator.remove(node)
        actuator.insert(
            index,
            ET.Element(
                "motor",
                name=f"actuator{index+1}",
                joint=f"joint{index+1}",
                ctrlrange=f"-{limit} {limit}",
                forcerange=f"-{limit} {limit}",
            ),
        )
    hx, hy, hz = HALF
    section = [(0.0, -hz), (hx, -hz + BEVEL), (hx, hz), (-hx, hz), (-hx, -hz + BEVEL)]
    vertices = [(x, y, z) for y in (-hy, hy) for x, z in section]
    ET.SubElement(root.find("asset"), "mesh", name="knife_prism", vertex=numbers(vertices))
    hand_node = root.find(".//body[@name='hand']")
    # The controller always follows a hand-fixed TCP. A grasped knife has its
    # own 6 DoF; its motion must never be substituted for the controlled site.
    ET.SubElement(
        hand_node,
        "site",
        name="control_tcp",
        pos="0 0 .16",
        quat=numbers(mounting_q),
        size=".001",
        rgba="0 0 0 0",
    )
    if attachment == "fixed":
        ET.SubElement(
            hand_node,
            "geom",
            name="tool_mount",
            type="cylinder",
            pos="0 0 .1",
            size=f"{MOUNT_RADIUS} .04",
            mass=".03",
            rgba=".16 .18 .2 1",
            group="2",
        )
        knife = ET.SubElement(hand_node, "body", name="knife", pos="0 0 .16", quat=numbers(mounting_q))
    else:
        # Free knife with a handle. No weld/equality involving
        # this free body and no artificial attraction to the fingers.
        knife = ET.SubElement(root.find("worldbody"), "body", name="knife", pos=numbers(tool_origin))
        ET.SubElement(knife, "freejoint", name="knife_free")
        ET.SubElement(
            knife,
            "geom",
            name="knife_handle",
            type="box",
            pos="0 0 .06",
            size=".008 .008 .04",
            mass=".03",
            friction=".2 .005 .0001",
            rgba=".16 .18 .2 1",
        )
        actuator.remove(list(actuator)[7])
        ET.SubElement(
            actuator, "motor", name="actuator8", tendon="split", ctrlrange="-100 100", forcerange="-100 100"
        )
    ET.SubElement(
        knife,
        "geom",
        name="knife",
        type="mesh",
        mesh="knife_prism",
        mass=str(blade_mass),
        rgba=".65 .69 .75 1",
        friction=".2 .005 .0001",
        group="2",
    )
    ET.SubElement(knife, "site", name="knife_edge", pos=f"0 0 {-hz}", size=".001", rgba="1 .2 0 1")
    ET.SubElement(knife, "site", name="knife_origin", size=".001", rgba="0 0 0 0")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 .1", rgba=".25 .29 .32 1")
    ET.SubElement(
        world,
        "geom",
        name="board",
        type="box",
        pos=numbers(material_origin - [0, 0, 0.025]),
        size=numbers([*BOARD_HALF_EXTENT, 0.025]),
        rgba=".64 .46 .27 1",
        friction=".2 .005 .0001",
    )
    ET.SubElement(world, "light", pos="1 -1 2", dir="-.4 .3 -1", diffuse=".7 .7 .7")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    xml = ET.tostring(root, encoding="unicode")
    return mj.MjModel.from_xml_string(xml), material_origin, xml


class Franka:
    def __init__(
        self,
        interval=0.001,
        translation_stiffness=None,
        rotation_stiffness=None,
        controller="libfranka",
        attachment="fixed",
        grip_force_per_finger=10.0,
    ):
        """translation/rotation_stiffness: libfranka Cartesian gains (default 150 N/m, 10 Nm/rad)."""
        if controller == "serl" and (translation_stiffness is not None or rotation_stiffness is not None):
            raise ValueError("SERL mode uses its upstream gains, not libfranka stiffness arguments")
        translation_stiffness = 150.0 if translation_stiffness is None else translation_stiffness
        rotation_stiffness = 10.0 if rotation_stiffness is None else rotation_stiffness
        if interval <= 0 or min(translation_stiffness, rotation_stiffness) <= 0:
            raise ValueError("Positive integration interval and stiffness required")
        self.model, self.material_origin, self.xml = make_model(interval, attachment=attachment)
        self.attachment = attachment
        self.grip_force_per_finger = float(grip_force_per_finger)
        if not 0 < self.grip_force_per_finger <= 50:
            raise ValueError("Grip force must fit the upstream tendon actuator range (0, 50] N per finger")
        if controller not in ("libfranka", "serl"):
            raise ValueError("Unknown torque controller")
        self.controller = controller
        self.impedance = None
        self.data = mj.MjData(self.model)
        self.data.qpos[:9] = HOME  # Initialization only; never written during control.
        if attachment == "fixed":
            self.data.qpos[7:9] = MOUNT_RADIUS
            self.data.ctrl[7] = MOUNT_RADIUS * FINGER_CTRL_PER_M
        else:
            self.data.qpos[7:9] = 0.0075
            self.data.ctrl[7] = -2 * self.grip_force_per_finger
        self.control_site = self.model.site("control_tcp").id
        if controller == "serl":
            self.impedance = ImpedanceController(self.model, self.control_site, np.arange(7))
        self.body = self.model.body("knife").id
        self.k = np.array([translation_stiffness] * 3 + [rotation_stiffness] * 3)
        self.damping = 2 * np.sqrt(self.k)
        mj.mj_forward(self.model, self.data)
        self.initial_position = self.control_position.copy()
        self.initial_rotation = self.control_rotation.copy()
        self.initial_quat_wxyz = quat_wxyz(self.initial_rotation)

    @property
    def position(self):
        return self.data.xpos[self.body].copy()

    @property
    def rotation(self):
        return self.data.xmat[self.body].reshape(3, 3).copy()

    @property
    def control_position(self):
        return self.data.site_xpos[self.control_site].copy()

    @property
    def control_rotation(self):
        return self.data.site_xmat[self.control_site].reshape(3, 3).copy()

    @property
    def edge_points(self):
        local = np.array([[0.0, -HALF[1], -HALF[2]], [0.0, HALF[1], -HALF[2]]])
        return local @ self.rotation.T + self.position

    def board_normal_force(self):
        """Measured MuJoCo knife/board contact, separate from MPM reaction."""
        blade, board = self.model.geom("knife").id, self.model.geom("board").id
        total = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if {contact.geom1, contact.geom2} == {blade, board}:
                force = np.zeros(6)
                mj.mj_contactForce(self.model, self.data, i, force)
                total += max(0.0, float(force[0]))
        return total

    def control_jacobian(self):
        j = np.zeros((6, self.model.nv))
        mj.mj_jacSite(self.model, self.data, j[:3], j[3:], self.control_site)
        return j

    def release(self):
        if self.attachment != "grasp":
            raise ValueError("Fixed mount cannot be released")
        self.data.ctrl[7] = 2 * self.grip_force_per_finger

    def jacobian(self):
        j = np.zeros((6, self.model.nv))
        mj.mj_jacBody(self.model, self.data, j[:3], j[3:], self.body)
        return j

    def mass(self):
        result = np.zeros((self.model.nv, self.model.nv))
        mj.mj_fullM(self.model, result, self.data.qM)
        return result

    def state(self):
        from .coupling import ToolState

        j = self.jacobian()
        twist = j @ self.data.qvel
        # J M^-1 J^T with the mass matrix factorization from the last forward pass.
        inverse_mass_jt = np.empty_like(j)
        mj.mj_solveM(self.model, self.data, inverse_mass_jt, j)
        mobility = j @ inverse_mass_jt.T
        return ToolState(
            self.position - self.material_origin,
            quat_xyzw(self.rotation),
            twist[:3],
            twist[3:],
            mobility,
        )

    def step(self, target, wrench=None, target_rotation=None):
        """Hold interval-averaged material wrench, integrate the full robot once.

        Controller equation/gains follow libfranka's cartesian_impedance_control
        example. MuJoCo needs gravity AND Coriolis compensation (qfrc_bias).
        Torque saturation is the upstream actuator limit; it is logged.
        """
        target = np.asarray(target, dtype=float)
        wrench = np.zeros(6) if wrench is None else np.asarray(wrench, dtype=float)
        if target.shape != (3,) or wrench.shape != (6,) or not np.isfinite([*target, *wrench]).all():
            raise ValueError("Invalid robot command/wrench")
        rotation = self.initial_rotation if target_rotation is None else np.asarray(target_rotation)
        if self.controller == "libfranka":
            qr = quat_xyzw(self.control_rotation.T @ rotation)
            if qr[3] < 0:
                qr = -qr
            error = np.r_[self.control_position - target, -self.control_rotation @ qr[:3]]
            j = self.control_jacobian()
            requested = (j.T @ (-self.k * error - self.damping * (j @ self.data.qvel)) + self.data.qfrc_bias)[
                :7
            ]
        else:
            quat = self.initial_quat_wxyz if target_rotation is None else quat_wxyz(rotation)
            requested = self.impedance(self.data, target, quat, HOME[:7])
        limits = self.model.actuator_ctrlrange[:7]
        self.data.ctrl[:7] = np.clip(requested, limits[:, 0], limits[:, 1])
        self.data.qfrc_applied[:] = 0
        mj.mj_applyFT(
            self.model, self.data, wrench[:3], wrench[3:], self.position, self.body, self.data.qfrc_applied
        )
        applied = self.data.qfrc_applied.copy()
        mj.mj_step(self.model, self.data)
        # Derived body states must correspond to the newly integrated qpos.
        mj.mj_forward(self.model, self.data)
        if not np.isfinite(np.r_[self.data.qpos, self.data.qvel]).all() or self.data.warning.number.any():
            raise RuntimeError(f"MuJoCo invalid state/warnings: {self.data.warning.number}")
        return requested, applied
