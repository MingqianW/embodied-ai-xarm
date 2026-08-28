"""Extract the canonical seven-dimensional xArm policy state from MuJoCo."""

from __future__ import annotations

from typing import Any, Mapping

import mujoco
import numpy as np

from simulation.robot.gripper import read_raw_gripper_position
from simulation.robot.joint_mapping import mujoco_qpos_to_raw_arm_state
from simulation.robot.model import ARM_JOINT_NAMES
from simulation.robot.model import joint_position


def get_robot_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config_or_mapping: Mapping[str, Any],
) -> np.ndarray:
    """Return ``[joint1..joint6 rad, gripper hardware units]`` as float32."""

    mujoco_arm = np.asarray(
        [joint_position(model, data, name) for name in ARM_JOINT_NAMES],
        dtype=np.float64,
    )
    arm_state = mujoco_qpos_to_raw_arm_state(mujoco_arm).astype(np.float32)
    gripper_raw = read_raw_gripper_position(model, data, config_or_mapping)
    return np.concatenate(
        [arm_state, np.asarray([gripper_raw], dtype=np.float32)]
    ).astype(np.float32)
