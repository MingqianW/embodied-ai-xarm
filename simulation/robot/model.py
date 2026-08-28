"""Named access to the canonical xArm6 MuJoCo model."""

from __future__ import annotations

import mujoco
import numpy as np


ARM_DOF = 6
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, ARM_DOF + 1))
ARM_ACTUATOR_NAMES = tuple(f"joint{i}_actuator" for i in range(1, ARM_DOF + 1))
GRIPPER_ACTUATOR_NAME = "gripper_actuator"
GRIPPER_DRIVER_JOINT_NAMES = ("left_driver_joint", "right_driver_joint")
XARM_FOUR_BAR_JOINT_NAMES = (
    "left_driver_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_driver_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)
LEFT_GRIPPER_DRIVER_JOINT_NAME = GRIPPER_DRIVER_JOINT_NAMES[0]
RIGHT_GRIPPER_DRIVER_JOINT_NAME = GRIPPER_DRIVER_JOINT_NAMES[1]
BASE_CAMERA_NAME = "base_camera"
WRIST_CAMERA_NAME = "wrist_camera"
OVERVIEW_CAMERA_NAME = "overview_camera"


def _required_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return object_id


def joint_id(model: mujoco.MjModel, name: str) -> int:
    return _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def actuator_id(model: mujoco.MjModel, name: str) -> int:
    return _required_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def body_id(model: mujoco.MjModel, name: str) -> int:
    return _required_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def camera_id(model: mujoco.MjModel, name: str) -> int:
    return _required_id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)


def joint_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
) -> float:
    identifier = joint_id(model, name)
    return float(data.qpos[int(model.jnt_qposadr[identifier])])


def arm_joint_limits(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(
        [model.jnt_range[joint_id(model, name)] for name in ARM_JOINT_NAMES],
        dtype=np.float32,
    )


def arm_actuator_ctrl_limits(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(
        [model.actuator_ctrlrange[actuator_id(model, name)] for name in ARM_ACTUATOR_NAMES],
        dtype=np.float32,
    )


def gripper_actuator_ctrl_limits(model: mujoco.MjModel) -> tuple[float, float]:
    low, high = model.actuator_ctrlrange[actuator_id(model, GRIPPER_ACTUATOR_NAME)]
    return float(low), float(high)
