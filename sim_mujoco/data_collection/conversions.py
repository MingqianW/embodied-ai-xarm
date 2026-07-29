"""Conversions between MuJoCo internals and the real xArm policy contract.

The stored real dataset convention is:

    [joint1_rad, ..., joint6_rad, gripper_raw]

Arm coordinates are an explicitly audited identity mapping. The gripper is
not identity: MuJoCo stores a finger-slide distance in metres while the policy
uses the xArm controller's raw position convention.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from sim_mujoco.gripper_mapping import (
    raw_gripper_to_sim_slide,
    sim_slide_to_raw_gripper,
)
from sim_mujoco.joint_mapping import (
    mujoco_qpos_to_raw_arm_state,
    raw_arm_state_to_mujoco_qpos,
)
from sim_mujoco.remote_policy_observation import (
    ARM_JOINT_NAMES,
    DEFAULT_CAMERA_CONFIG_PATH,
    GRIPPER_LEFT_JOINT,
    joint_qpos,
    load_camera_config,
)


POLICY_DOF = 7
ARM_DOF = 6


@lru_cache(maxsize=4)
def _mapping_config(path: str) -> dict[str, Any]:
    return load_camera_config(Path(path))


def _config(camera_config_path: Path = DEFAULT_CAMERA_CONFIG_PATH) -> dict[str, Any]:
    return _mapping_config(str(camera_config_path.resolve()))


def _vector7(values: np.ndarray, *, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (POLICY_DOF,):
        raise ValueError(f"{label} must have shape ({POLICY_DOF},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} contains NaN or Inf")
    return vector


def gripper_raw_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    camera_config_path: Path = DEFAULT_CAMERA_CONFIG_PATH,
) -> float:
    """Return the left finger slide expressed in the real policy convention."""

    slide = joint_qpos(model, data, GRIPPER_LEFT_JOINT)
    return float(sim_slide_to_raw_gripper(slide, _config(camera_config_path)))


def mujoco_gripper_target_from_raw(
    raw: float,
    *,
    camera_config_path: Path = DEFAULT_CAMERA_CONFIG_PATH,
) -> float:
    """Convert an absolute xArm raw gripper target to MuJoCo slide metres."""

    value = float(raw)
    if not np.isfinite(value):
        raise ValueError("raw gripper target contains NaN or Inf")
    return float(raw_gripper_to_sim_slide(value, _config(camera_config_path)))


def policy_state_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    camera_config_path: Path = DEFAULT_CAMERA_CONFIG_PATH,
) -> np.ndarray:
    """Build the exact 7D policy state using named MuJoCo joints."""

    arm_qpos = np.asarray(
        [joint_qpos(model, data, name) for name in ARM_JOINT_NAMES],
        dtype=np.float64,
    )
    arm_raw = mujoco_qpos_to_raw_arm_state(arm_qpos)
    gripper_raw = gripper_raw_from_mujoco(
        model,
        data,
        camera_config_path=camera_config_path,
    )
    return np.concatenate((arm_raw, [gripper_raw])).astype(np.float32)


def mujoco_joint_target_from_policy_action(
    action: np.ndarray,
    *,
    camera_config_path: Path = DEFAULT_CAMERA_CONFIG_PATH,
) -> np.ndarray:
    """Convert a 7D absolute policy action to six qpos plus slide metres."""

    policy_action = _vector7(action, label="action")
    arm_target = raw_arm_state_to_mujoco_qpos(policy_action[:ARM_DOF])
    gripper_target = mujoco_gripper_target_from_raw(
        float(policy_action[ARM_DOF]),
        camera_config_path=camera_config_path,
    )
    return np.concatenate((arm_target, [gripper_target])).astype(np.float64)


def policy_action_from_mujoco_target(
    target: np.ndarray,
    *,
    camera_config_path: Path = DEFAULT_CAMERA_CONFIG_PATH,
) -> np.ndarray:
    """Convert six qpos plus a slide-metre target to a 7D policy action."""

    internal_target = _vector7(target, label="target")
    arm_raw = mujoco_qpos_to_raw_arm_state(internal_target[:ARM_DOF])
    gripper_raw = sim_slide_to_raw_gripper(
        float(internal_target[ARM_DOF]),
        _config(camera_config_path),
    )
    return np.concatenate((arm_raw, [gripper_raw])).astype(np.float32)
