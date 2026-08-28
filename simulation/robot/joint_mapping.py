from __future__ import annotations

import numpy as np


ARM_DOF = 6


def _validated_arm_values(values: np.ndarray, *, label: str) -> np.ndarray:
    arm = np.asarray(values, dtype=np.float64)
    if arm.shape != (ARM_DOF,):
        raise ValueError(f"{label} must have shape ({ARM_DOF},), got {arm.shape}")
    if not np.isfinite(arm).all():
        raise ValueError(f"{label} contains NaN or Inf")
    return arm


def raw_arm_state_to_mujoco_qpos(raw_joint_values: np.ndarray) -> np.ndarray:
    """Map xArm controller radians to named MuJoCo arm qpos.

    The kinematic audit demonstrates identical joint order, signs, and zero
    references. Keep the identity mapping explicit at the runtime boundary so
    policy-space and simulation-space values cannot be conflated later.
    """
    return _validated_arm_values(
        raw_joint_values,
        label="raw_joint_values",
    ).copy()


def mujoco_qpos_to_raw_arm_state(qpos: np.ndarray) -> np.ndarray:
    """Map named MuJoCo arm qpos to the policy's xArm controller convention."""
    return _validated_arm_values(qpos, label="qpos").copy()
