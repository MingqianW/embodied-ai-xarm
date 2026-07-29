from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from sim_mujoco.joint_mapping import raw_arm_state_to_mujoco_qpos
from policy_runtime.action_decoder import validate_policy_actions as _validate_policy_actions
from policy_runtime.safety import clamp_absolute_joint_target, clamp_scalar
from sim_mujoco.remote_policy_observation import (
    arm_actuator_ctrl_limits,
    arm_joint_limits,
    get_robot_state,
    gripper_actuator_ctrl_limits,
    gripper_raw_to_sim,
)


ACTION_SHAPE = (10, 7)
GRIPPER_RAW_MIN = 50.0
GRIPPER_RAW_MAX = 845.0
DEFAULT_MAX_JOINT_STEP = 0.05


@dataclass
class SafeControlTarget:
    raw_action: np.ndarray
    current_state: np.ndarray
    arm_target_raw: np.ndarray
    arm_target_clamped: np.ndarray
    arm_target_mujoco: np.ndarray
    gripper_raw: float
    gripper_raw_clamped: float
    gripper_sim_target: float
    ctrl_target: np.ndarray
    joint_delta_raw: np.ndarray
    joint_delta_clamped: np.ndarray
    clipped: bool
    clip_messages: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "raw_action": self.raw_action.tolist(),
            "current_state": self.current_state.tolist(),
            "arm_target_raw": self.arm_target_raw.tolist(),
            "arm_target_clamped": self.arm_target_clamped.tolist(),
            "arm_target_mujoco": self.arm_target_mujoco.tolist(),
            "gripper_raw": self.gripper_raw,
            "gripper_raw_clamped": self.gripper_raw_clamped,
            "gripper_sim_target": self.gripper_sim_target,
            "ctrl_target": self.ctrl_target.tolist(),
            "joint_delta_raw": self.joint_delta_raw.tolist(),
            "joint_delta_clamped": self.joint_delta_clamped.tolist(),
            "clipped": self.clipped,
            "clip_messages": self.clip_messages,
        }


def validate_policy_actions(actions: np.ndarray) -> np.ndarray:
    return _validate_policy_actions(actions)


def extract_first_action(actions: np.ndarray) -> np.ndarray:
    value = validate_policy_actions(actions)
    return np.asarray(value[0], dtype=np.float32).copy()


def clamp_joint_target(
    raw_target: np.ndarray,
    current_joints: np.ndarray,
    joint_limits: np.ndarray,
    actuator_limits: np.ndarray,
    *,
    max_joint_step: float = DEFAULT_MAX_JOINT_STEP,
) -> tuple[np.ndarray, list[str]]:
    return clamp_absolute_joint_target(
        raw_target,
        current_joints,
        joint_limits,
        max_joint_delta_rad=max_joint_step,
        additional_limits=actuator_limits,
    )


def clamp_gripper_raw(raw_value: float) -> tuple[float, list[str]]:
    return clamp_scalar(
        raw_value,
        GRIPPER_RAW_MIN,
        GRIPPER_RAW_MAX,
        label="gripper_raw",
    )


def convert_gripper_raw_to_sim_target(raw_value: float, config: dict[str, Any]) -> float:
    return float(gripper_raw_to_sim(float(raw_value), config))


def compute_safe_control_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: dict[str, Any],
    first_action: np.ndarray,
    *,
    max_joint_step: float = DEFAULT_MAX_JOINT_STEP,
) -> SafeControlTarget:
    action = np.asarray(first_action, dtype=np.float32).reshape(7)
    if not np.isfinite(action).all():
        raise ValueError("First action contains NaN or Inf")

    current_state = get_robot_state(model, data, config)
    arm_raw = np.asarray(action[:6], dtype=np.float32)
    arm_clamped, arm_messages = clamp_joint_target(
        arm_raw,
        current_state[:6],
        arm_joint_limits(model),
        arm_actuator_ctrl_limits(model),
        max_joint_step=max_joint_step,
    )
    gripper_clamped, gripper_messages = clamp_gripper_raw(float(action[6]))
    gripper_sim = convert_gripper_raw_to_sim_target(gripper_clamped, config)
    low, high = gripper_actuator_ctrl_limits(model)
    gripper_sim_limited = float(np.clip(gripper_sim, low, high))
    arm_mujoco = raw_arm_state_to_mujoco_qpos(arm_clamped).astype(np.float32)
    limit_messages = []
    if not np.isclose(gripper_sim, gripper_sim_limited, rtol=0.0, atol=1e-8):
        limit_messages.append(
            f"gripper sim target actuator-clamped from {gripper_sim:.6f} to {gripper_sim_limited:.6f}"
        )
    ctrl_target = np.concatenate(
        [arm_mujoco, np.asarray([gripper_sim_limited], dtype=np.float32)]
    ).astype(np.float32)
    messages = [*arm_messages, *gripper_messages, *limit_messages]
    return SafeControlTarget(
        raw_action=action,
        current_state=current_state,
        arm_target_raw=arm_raw,
        arm_target_clamped=arm_clamped,
        arm_target_mujoco=arm_mujoco,
        gripper_raw=float(action[6]),
        gripper_raw_clamped=gripper_clamped,
        gripper_sim_target=gripper_sim_limited,
        ctrl_target=ctrl_target,
        joint_delta_raw=(arm_raw - current_state[:6]).astype(np.float32),
        joint_delta_clamped=(arm_clamped - current_state[:6]).astype(np.float32),
        clipped=bool(messages),
        clip_messages=messages,
    )


def apply_safe_control_target(data: mujoco.MjData, target: SafeControlTarget) -> None:
    if target.ctrl_target.shape != (7,):
        raise ValueError(f"Control target must have shape (7,), got {target.ctrl_target.shape}")
    if not np.isfinite(target.ctrl_target).all():
        raise ValueError("Control target contains NaN or Inf")
    data.ctrl[:7] = target.ctrl_target
