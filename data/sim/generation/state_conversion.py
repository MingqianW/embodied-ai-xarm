"""Conversions between MuJoCo internals and the real xArm policy contract.

The stored real dataset convention is:

    [joint1_rad, ..., joint6_rad, gripper_raw]

Arm coordinates are an explicitly audited identity mapping. The gripper is
not identity: the canonical LOCAL MuJoCo model uses a 0.005-open/0.85-closed
driver-angle target and linkage state while the policy uses the xArm
controller's raw convention.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from simulation.robot.gripper import read_raw_gripper_position
from simulation.robot.gripper_mapping import actuator_ctrl_rad_to_raw_hardware
from simulation.robot.gripper_mapping import raw_hardware_to_actuator_ctrl_rad
from simulation.robot.legacy_gripper import raw_hardware_to_legacy_slide_m
from simulation.robot.legacy_gripper import legacy_slide_m_to_raw_hardware
from simulation.robot.joint_mapping import (
    mujoco_qpos_to_raw_arm_state,
    raw_arm_state_to_mujoco_qpos,
)
from simulation.robot.model import ARM_JOINT_NAMES
from simulation.robot.model import joint_position
from simulation.resources import DEFAULT_CAMERA_CONFIG_PATH
from simulation.configuration import load_simulation_config


POLICY_DOF = 7
ARM_DOF = 6


@lru_cache(maxsize=4)
def _mapping_config(path: str) -> dict[str, Any]:
    return load_simulation_config(Path(path))


def _config(camera_config_path: Path = DEFAULT_CAMERA_CONFIG_PATH) -> dict[str, Any]:
    return _mapping_config(str(camera_config_path.resolve()))


def _vector7(values: np.ndarray, *, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (POLICY_DOF,):
        raise ValueError(f"{label} must have shape ({POLICY_DOF},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} contains NaN or Inf")
    return vector


def gripper_hardware_raw_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    camera_config_path: Path = DEFAULT_CAMERA_CONFIG_PATH,
) -> float:
    """Return the simulated linkage state in the real policy convention."""

    return float(
        read_raw_gripper_position(
            model,
            data,
            _config(camera_config_path),
        )
    )


def mujoco_gripper_actuator_ctrl_from_hardware_raw(
    gripper_hardware_raw: float,
    *,
    camera_config_path: Path = DEFAULT_CAMERA_CONFIG_PATH,
) -> float:
    """Convert an absolute xArm raw gripper target to configured actuator ctrl."""

    value = float(gripper_hardware_raw)
    if not np.isfinite(value):
        raise ValueError("raw gripper target contains NaN or Inf")
    config = _config(camera_config_path)
    if "sim_slide_min_m" in config.get("gripper_mapping", {}):
        return float(raw_hardware_to_legacy_slide_m(value, config))
    return float(raw_hardware_to_actuator_ctrl_rad(value, config))


def policy_state_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    camera_config_path: Path = DEFAULT_CAMERA_CONFIG_PATH,
) -> np.ndarray:
    """Build the exact 7D policy state using named MuJoCo joints."""

    arm_qpos = np.asarray(
        [joint_position(model, data, name) for name in ARM_JOINT_NAMES],
        dtype=np.float64,
    )
    arm_raw = mujoco_qpos_to_raw_arm_state(arm_qpos)
    gripper_hardware_raw = gripper_hardware_raw_from_mujoco(
        model,
        data,
        camera_config_path=camera_config_path,
    )
    return np.concatenate((arm_raw, [gripper_hardware_raw])).astype(np.float32)


def mujoco_joint_target_from_policy_action(
    action: np.ndarray,
    *,
    camera_config_path: Path = DEFAULT_CAMERA_CONFIG_PATH,
) -> np.ndarray:
    """Convert a 7D policy action to six arm targets plus gripper ctrl."""

    policy_action = _vector7(action, label="action")
    arm_target = raw_arm_state_to_mujoco_qpos(policy_action[:ARM_DOF])
    gripper_actuator_ctrl = mujoco_gripper_actuator_ctrl_from_hardware_raw(
        float(policy_action[ARM_DOF]),
        camera_config_path=camera_config_path,
    )
    return np.concatenate((arm_target, [gripper_actuator_ctrl])).astype(np.float64)


def policy_action_from_mujoco_target(
    target: np.ndarray,
    *,
    camera_config_path: Path = DEFAULT_CAMERA_CONFIG_PATH,
) -> np.ndarray:
    """Convert six arm targets plus gripper ctrl to a 7D policy action."""

    internal_target = _vector7(target, label="target")
    arm_raw = mujoco_qpos_to_raw_arm_state(internal_target[:ARM_DOF])
    config = _config(camera_config_path)
    if "sim_slide_min_m" in config.get("gripper_mapping", {}):
        gripper_raw = legacy_slide_m_to_raw_hardware(float(internal_target[ARM_DOF]), config)
    else:
        gripper_raw = actuator_ctrl_rad_to_raw_hardware(
            float(internal_target[ARM_DOF]), config
        )
    return np.concatenate((arm_raw, [gripper_raw])).astype(np.float32)
