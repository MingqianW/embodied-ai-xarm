"""Offline-testable authorization gate for the existing xArm runtime.

Nothing in this module imports a robot SDK or sends a command.  The gate is a
necessary precondition for the hardware entrypoint, not a claim that software
can replace the operator, workspace assessment, or physical emergency stop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policy_runtime.safety import SafetyConfig
from policy_runtime.safety import validate_action_chunk
from policy_runtime.schemas import SafetyResult


# xArm6 position limits in radians, from the vendored official xArm ROS 2
# description at third_party/xarm_ros2/xarm_description/urdf/xarm6/xarm6.urdf.xacro.
# Site-specific reduced-mode limits may be supplied by the caller when stricter.
XARM6_JOINT_LIMITS_RAD = np.asarray(
    [
        [-2.0 * np.pi, 2.0 * np.pi],
        [-2.059, 2.0944],
        [-3.927, 0.19198],
        [-2.0 * np.pi, 2.0 * np.pi],
        [-1.69297, np.pi],
        [-2.0 * np.pi, 2.0 * np.pi],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class RealExecutionAuthorization:
    operator_present: bool
    workspace_clear: bool
    emergency_stop_accessible: bool
    robot_motion_confirmed: bool

    @property
    def authorized(self) -> bool:
        return all(
            (
                self.operator_present,
                self.workspace_clear,
                self.emergency_stop_accessible,
                self.robot_motion_confirmed,
            )
        )

    def require(self) -> None:
        if not self.authorized:
            raise PermissionError(
                "Real-robot motion requires an attending operator, a clear workspace, "
                "an accessible emergency stop, and explicit motion confirmation"
            )


def validate_real_action_chunk(
    actions: np.ndarray,
    *,
    current_state: np.ndarray,
    joint_limits: np.ndarray,
    authorization: RealExecutionAuthorization,
    config: SafetyConfig,
) -> SafetyResult:
    """Authorize and validate a proposed chunk without executing it."""

    authorization.require()
    value = np.asarray(actions, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] < 7:
        return SafetyResult(
            accepted=False,
            clipped=False,
            reason=f"actions must have shape (H, >=7), got {value.shape}",
            actions=value.copy(),
        )
    return validate_action_chunk(
        value[:, :7],
        np.asarray(current_state, dtype=np.float32),
        np.asarray(joint_limits, dtype=np.float32),
        config,
    )
