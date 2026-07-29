from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from policy_runtime.schemas import SafetyResult


ActionMode = Literal["absolute_joint_position", "joint_position_delta"]


@dataclass(frozen=True)
class SafetyConfig:
    action_mode: ActionMode = "absolute_joint_position"
    max_joint_delta_rad: float = 0.05
    gripper_min: float = 50.0
    gripper_max: float = 845.0
    reject_on_clip: bool = False
    reject_if_clip_exceeds_rad: float | None = None


def clamp_absolute_joint_target(
    target: np.ndarray,
    current_joints: np.ndarray,
    joint_limits: np.ndarray,
    *,
    max_joint_delta_rad: float,
    additional_limits: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Clamp one canonical absolute arm target with detailed messages."""

    raw = np.asarray(target, dtype=np.float32).reshape(6)
    current = np.asarray(current_joints, dtype=np.float32).reshape(6)
    limits = _validate_limits(joint_limits)
    lower = limits[:, 0].copy()
    upper = limits[:, 1].copy()
    if additional_limits is not None:
        extra = _validate_limits(additional_limits)
        lower = np.maximum(lower, extra[:, 0])
        upper = np.minimum(upper, extra[:, 1])
    if np.any(lower > upper):
        raise ValueError("Joint and additional limits do not overlap")
    if max_joint_delta_rad <= 0:
        raise ValueError("max_joint_delta_rad must be positive")

    messages: list[str] = []
    step_clamped = current + np.clip(
        raw - current,
        -float(max_joint_delta_rad),
        float(max_joint_delta_rad),
    )
    for index, (before, after) in enumerate(zip(raw, step_clamped), start=1):
        if not np.isclose(float(before), float(after), rtol=0.0, atol=1e-6):
            messages.append(
                f"joint{index} step-clamped from {float(before):.6f} to {float(after):.6f}"
            )
    bounded = np.clip(step_clamped, lower, upper).astype(np.float32)
    for index, (before, after) in enumerate(zip(step_clamped, bounded), start=1):
        if not np.isclose(float(before), float(after), rtol=0.0, atol=1e-6):
            messages.append(
                f"joint{index} limit-clamped from {float(before):.6f} to {float(after):.6f}"
            )
    return bounded, messages


def clamp_scalar(
    value: float,
    minimum: float,
    maximum: float,
    *,
    label: str,
) -> tuple[float, list[str]]:
    if minimum >= maximum:
        raise ValueError(f"{label} minimum must be below maximum")
    raw = float(value)
    bounded = float(np.clip(raw, minimum, maximum))
    if np.isclose(raw, bounded, rtol=0.0, atol=1e-6):
        return bounded, []
    return bounded, [f"{label} clamped from {raw:.6f} to {bounded:.6f}"]


def _validate_limits(joint_limits: np.ndarray) -> np.ndarray:
    limits = np.asarray(joint_limits, dtype=np.float32)
    if limits.shape != (6, 2):
        raise ValueError(f"Joint limits must have shape (6, 2), got {limits.shape}")
    if not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("Joint limits are invalid")
    return limits


def validate_action_chunk(
    actions: np.ndarray,
    current_state: np.ndarray,
    joint_limits: np.ndarray,
    config: SafetyConfig = SafetyConfig(),
) -> SafetyResult:
    value = np.asarray(actions, dtype=np.float32)
    state = np.asarray(current_state, dtype=np.float32)
    limits = _validate_limits(joint_limits)
    if value.ndim != 2 or value.shape[1] != 7 or value.shape[0] < 1:
        return SafetyResult(False, False, f"actions must have shape (H, 7), got {value.shape}", value.copy())
    if state.shape != (7,):
        return SafetyResult(False, False, f"current_state must have shape (7,), got {state.shape}", value.copy())
    if not np.isfinite(value).all():
        return SafetyResult(False, False, "actions contain NaN or Inf", value.copy())
    if not np.isfinite(state).all():
        return SafetyResult(False, False, "current_state contains NaN or Inf", value.copy())
    if config.max_joint_delta_rad <= 0:
        raise ValueError("max_joint_delta_rad must be positive")
    if config.gripper_min >= config.gripper_max:
        raise ValueError("gripper_min must be lower than gripper_max")
    if config.action_mode not in ("absolute_joint_position", "joint_position_delta"):
        raise ValueError(f"Unsupported action mode: {config.action_mode}")

    safe = value.copy()
    reference = state[:6].copy()
    messages: list[str] = []
    clipped_magnitudes: list[float] = []
    clipped_indices: set[int] = set()

    for row_index in range(value.shape[0]):
        raw_arm = value[row_index, :6]
        absolute = raw_arm if config.action_mode == "absolute_joint_position" else reference + raw_arm
        delta_limited = reference + np.clip(
            absolute - reference,
            -config.max_joint_delta_rad,
            config.max_joint_delta_rad,
        )
        bounded = np.clip(delta_limited, limits[:, 0], limits[:, 1]).astype(np.float32)
        clip_magnitude = float(np.max(np.abs(absolute - bounded)))
        if clip_magnitude > 1e-6:
            clipped_indices.add(row_index)
            clipped_magnitudes.append(clip_magnitude)
            messages.append(
                f"action[{row_index}] arm target clipped by up to {clip_magnitude:.6f} rad"
            )
        safe[row_index, :6] = bounded

        raw_gripper = float(value[row_index, 6])
        bounded_gripper = float(np.clip(raw_gripper, config.gripper_min, config.gripper_max))
        if not np.isclose(raw_gripper, bounded_gripper, rtol=0.0, atol=1e-6):
            clipped_indices.add(row_index)
            messages.append(
                f"action[{row_index}] gripper clipped from "
                f"{raw_gripper:.6f} to {bounded_gripper:.6f}"
            )
        safe[row_index, 6] = bounded_gripper
        reference = bounded

    clipped = bool(clipped_indices)
    reason: str | None = None
    if clipped and config.reject_on_clip:
        reason = "action chunk requires clipping"
    if (
        clipped_magnitudes
        and config.reject_if_clip_exceeds_rad is not None
        and max(clipped_magnitudes) > config.reject_if_clip_exceeds_rad
    ):
        reason = (
            f"action chunk requires {max(clipped_magnitudes):.6f} rad clipping, "
            f"above rejection threshold {config.reject_if_clip_exceeds_rad:.6f}"
        )

    return SafetyResult(
        accepted=reason is None,
        clipped=clipped,
        reason=reason,
        actions=safe,
        messages=tuple(messages),
        rejected_indices=tuple(sorted(clipped_indices)) if reason else (),
    )
