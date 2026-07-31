from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from policy_runtime.config import load_yaml


@dataclass(frozen=True)
class RobotMapping:
    canonical_arm_joint_names: tuple[str, ...]
    isaac_arm_joint_names: tuple[str, ...]
    gripper_joint_name: str
    gripper_mimic_joint_names: tuple[str, ...]
    joint_limits_rad: np.ndarray
    initial_arm_positions_rad: np.ndarray
    initial_gripper_policy: float
    gripper_policy_closed: float
    gripper_policy_open: float
    gripper_isaac_closed: float
    gripper_isaac_open: float
    action_mode: str
    base_frame: str
    end_effector_frame: str
    articulation_prim_path: str
    asset_path: str
    gravity_enabled: bool
    gripper_visual_frame: str
    gripper_color_rgb: tuple[float, float, float]
    physical_aperture_validated: bool


def load_robot_mapping(path: Path) -> RobotMapping:
    config = load_yaml(path)
    robot = config["robot"]
    canonical = config["canonical"]
    isaac = config["isaac"]
    gripper = config["gripper"]
    canonical_names = tuple(str(name) for name in canonical["arm_joint_names"])
    isaac_names = tuple(str(name) for name in isaac["arm_joint_names"])
    limits_by_name = config["joint_limits_rad"]
    limits = np.asarray([limits_by_name[name] for name in canonical_names], dtype=np.float32)
    mapping = RobotMapping(
        canonical_arm_joint_names=canonical_names,
        isaac_arm_joint_names=isaac_names,
        gripper_joint_name=str(isaac["gripper_joint_name"]),
        gripper_mimic_joint_names=tuple(
            str(name) for name in isaac.get("gripper_mimic_joint_names", [])
        ),
        joint_limits_rad=limits,
        initial_arm_positions_rad=np.asarray(
            config["initial_pose"]["arm_joint_positions_rad"],
            dtype=np.float32,
        ),
        initial_gripper_policy=float(config["initial_pose"]["gripper_mm"]),
        gripper_policy_closed=float(gripper["policy_closed"]),
        gripper_policy_open=float(gripper["policy_open"]),
        gripper_isaac_closed=float(gripper["isaac_closed"]),
        gripper_isaac_open=float(gripper["isaac_open"]),
        action_mode=str(robot["action_mode"]),
        base_frame=str(robot["base_frame"]),
        end_effector_frame=str(robot["end_effector_frame"]),
        articulation_prim_path=str(robot["articulation_prim_path"]),
        asset_path=str(robot["asset_path"]),
        gravity_enabled=bool(isaac["gravity_enabled"]),
        gripper_visual_frame=str(isaac["gripper_visual_frame"]),
        gripper_color_rgb=tuple(
            float(value) for value in isaac["gripper_color_rgb"]
        ),
        physical_aperture_validated=bool(gripper["physical_aperture_validated"]),
    )
    validate_robot_mapping(mapping)
    return mapping


def validate_robot_mapping(mapping: RobotMapping) -> None:
    if len(mapping.canonical_arm_joint_names) != 6:
        raise ValueError("Canonical arm mapping must contain exactly six joints")
    if len(mapping.isaac_arm_joint_names) != 6:
        raise ValueError("Isaac arm mapping must contain exactly six joints")
    if len(set(mapping.canonical_arm_joint_names)) != 6:
        raise ValueError("Canonical arm joint mapping is not one-to-one")
    if len(set(mapping.isaac_arm_joint_names)) != 6:
        raise ValueError("Isaac arm joint mapping is not one-to-one")
    if mapping.gripper_joint_name in mapping.isaac_arm_joint_names:
        raise ValueError("Gripper joint must not duplicate an arm joint")
    gripper_names = (
        mapping.gripper_joint_name,
        *mapping.gripper_mimic_joint_names,
    )
    if len(gripper_names) != len(set(gripper_names)):
        raise ValueError("Gripper drive and mimic joint mappings must be unique")
    if set(gripper_names).intersection(mapping.isaac_arm_joint_names):
        raise ValueError("Gripper joints must not duplicate arm joints")
    if mapping.joint_limits_rad.shape != (6, 2):
        raise ValueError("Joint limits must have shape (6, 2)")
    if not np.isfinite(mapping.joint_limits_rad).all():
        raise ValueError("Joint limits contain NaN or Inf")
    if np.any(mapping.joint_limits_rad[:, 0] >= mapping.joint_limits_rad[:, 1]):
        raise ValueError("Every lower joint limit must be below its upper limit")
    if mapping.initial_arm_positions_rad.shape != (6,):
        raise ValueError("Initial arm position must have shape (6,)")
    if not np.all(
        (mapping.initial_arm_positions_rad >= mapping.joint_limits_rad[:, 0])
        & (mapping.initial_arm_positions_rad <= mapping.joint_limits_rad[:, 1])
    ):
        raise ValueError("Initial arm position is outside configured limits")
    if mapping.gripper_policy_closed >= mapping.gripper_policy_open:
        raise ValueError("Policy gripper closed value must be below open value")
    if np.isclose(mapping.gripper_isaac_closed, mapping.gripper_isaac_open):
        raise ValueError("Isaac gripper closed and open values must differ")
    if mapping.action_mode not in ("absolute_joint_position", "joint_position_delta"):
        raise ValueError(f"Unsupported action mode: {mapping.action_mode}")
    if len(mapping.gripper_color_rgb) != 3 or not all(
        0.0 <= value <= 1.0 for value in mapping.gripper_color_rgb
    ):
        raise ValueError("Isaac gripper color must contain three values in [0, 1]")


def gripper_mm_to_isaac(value: float, mapping: RobotMapping) -> float:
    normalized = np.clip(
        (float(value) - mapping.gripper_policy_closed)
        / (mapping.gripper_policy_open - mapping.gripper_policy_closed),
        0.0,
        1.0,
    )
    return float(
        mapping.gripper_isaac_closed
        + normalized * (mapping.gripper_isaac_open - mapping.gripper_isaac_closed)
    )


def isaac_gripper_to_mm(value: float, mapping: RobotMapping) -> float:
    normalized = np.clip(
        (float(value) - mapping.gripper_isaac_closed)
        / (mapping.gripper_isaac_open - mapping.gripper_isaac_closed),
        0.0,
        1.0,
    )
    return float(
        mapping.gripper_policy_closed
        + normalized * (mapping.gripper_policy_open - mapping.gripper_policy_closed)
    )


def validate_articulation_joint_names(
    available_joint_names: list[str] | tuple[str, ...],
    mapping: RobotMapping,
) -> dict[str, int]:
    names = list(available_joint_names)
    if len(names) != len(set(names)):
        raise ValueError("Isaac articulation reports duplicate joint names")
    required = [
        *mapping.isaac_arm_joint_names,
        mapping.gripper_joint_name,
        *mapping.gripper_mimic_joint_names,
    ]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(
            f"Isaac articulation is missing required joint(s): {missing}; available={names}"
        )
    return {name: names.index(name) for name in required}


def policy_state_to_isaac(
    policy_state: np.ndarray,
    available_joint_names: list[str] | tuple[str, ...],
    mapping: RobotMapping,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(policy_state, dtype=np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("Policy state must be finite with shape (7,)")
    indices = validate_articulation_joint_names(available_joint_names, mapping)
    ordered_names = [*mapping.isaac_arm_joint_names, mapping.gripper_joint_name]
    gripper_target = gripper_mm_to_isaac(float(state[6]), mapping)
    targets = np.concatenate(
        [
            state[:6],
            np.asarray([gripper_target], dtype=np.float32),
        ]
    ).astype(np.float32)
    return targets, np.asarray([indices[name] for name in ordered_names], dtype=np.int64)


def policy_state_to_isaac_reset(
    policy_state: np.ndarray,
    available_joint_names: list[str] | tuple[str, ...],
    mapping: RobotMapping,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a canonical state into a constraint-consistent teleport state.

    Runtime commands actuate only the six arm joints and the gripper drive joint.
    A reset is different: every independent DOF in the imported closed-chain
    gripper must be teleported to the corresponding mimic coordinate. Leaving
    the passive mimic DOFs at zero while opening ``drive_joint`` injects a large
    constraint impulse through the wrist on the first physics step.
    """

    targets, actuated_indices = policy_state_to_isaac(
        policy_state,
        available_joint_names,
        mapping,
    )
    indices = validate_articulation_joint_names(available_joint_names, mapping)
    gripper_target = float(targets[-1])
    mimic_indices = np.asarray(
        [indices[name] for name in mapping.gripper_mimic_joint_names],
        dtype=np.int64,
    )
    mimic_targets = np.full(
        len(mapping.gripper_mimic_joint_names),
        gripper_target,
        dtype=np.float32,
    )
    return (
        np.concatenate([targets, mimic_targets]).astype(np.float32),
        np.concatenate([actuated_indices, mimic_indices]).astype(np.int64),
    )


def isaac_state_to_policy(
    joint_positions: np.ndarray,
    available_joint_names: list[str] | tuple[str, ...],
    mapping: RobotMapping,
) -> np.ndarray:
    positions = np.asarray(joint_positions, dtype=np.float32).reshape(-1)
    indices = validate_articulation_joint_names(available_joint_names, mapping)
    if positions.size < len(available_joint_names):
        raise ValueError(
            f"Received {positions.size} positions for {len(available_joint_names)} joint names"
        )
    arm = np.asarray(
        [positions[indices[name]] for name in mapping.isaac_arm_joint_names],
        dtype=np.float32,
    )
    gripper = isaac_gripper_to_mm(
        float(positions[indices[mapping.gripper_joint_name]]),
        mapping,
    )
    return np.concatenate([arm, np.asarray([gripper], dtype=np.float32)])


def policy_action_to_isaac(
    action: np.ndarray,
    available_joint_names: list[str] | tuple[str, ...],
    mapping: RobotMapping,
    *,
    current_policy_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("Policy action must be finite with shape (7,)")
    if mapping.action_mode == "joint_position_delta":
        if current_policy_state is None:
            raise ValueError("Delta action conversion requires current_policy_state")
        current = np.asarray(current_policy_state, dtype=np.float32)
        if current.shape != (7,):
            raise ValueError("Current policy state must have shape (7,)")
        value = value.copy()
        value[:6] += current[:6]
    return policy_state_to_isaac(value, available_joint_names, mapping)


class IsaacArticulation:
    """Thin runtime wrapper; imports Isaac APIs only when instantiated."""

    def __init__(self, mapping: RobotMapping, prim: Any | None = None) -> None:
        from sim_isaac.version_compat import create_articulation

        self.mapping = mapping
        self.prim = prim or create_articulation(mapping.articulation_prim_path)
        if hasattr(self.prim, "initialize"):
            self.prim.initialize()
        self.joint_names = tuple(self.prim.dof_names)
        validate_articulation_joint_names(self.joint_names, mapping)

    def get_policy_state(self) -> np.ndarray:
        return isaac_state_to_policy(
            np.asarray(self.prim.get_joint_positions(), dtype=np.float32),
            self.joint_names,
            self.mapping,
        )

    def apply_policy_action(self, action: np.ndarray) -> None:
        from sim_isaac.version_compat import apply_joint_position_targets

        targets, indices = policy_action_to_isaac(
            action,
            self.joint_names,
            self.mapping,
            current_policy_state=self.get_policy_state(),
        )
        apply_joint_position_targets(self.prim, targets, indices)

    def apply_canonical_target(self, target: np.ndarray) -> None:
        """Apply an absolute canonical target after shared action validation."""

        from sim_isaac.version_compat import apply_joint_position_targets

        targets, indices = policy_state_to_isaac(
            target,
            self.joint_names,
            self.mapping,
        )
        apply_joint_position_targets(self.prim, targets, indices)

    def set_home_pose(self) -> None:
        state = np.concatenate(
            [
                self.mapping.initial_arm_positions_rad,
                np.asarray([self.mapping.initial_gripper_policy], dtype=np.float32),
            ]
        )
        self.set_policy_state(state)

    def set_policy_state(self, state: np.ndarray) -> None:
        from sim_isaac.version_compat import (
            set_joint_positions,
            set_joint_velocities,
        )

        targets, indices = policy_state_to_isaac_reset(
            state,
            self.joint_names,
            self.mapping,
        )
        set_joint_positions(self.prim, targets, indices)
        set_joint_velocities(
            self.prim,
            np.zeros(len(self.joint_names), dtype=np.float32),
            np.arange(len(self.joint_names), dtype=np.int64),
        )

    def hold_position(self) -> None:
        self.apply_canonical_target(self.get_policy_state())

    def max_contact_impulse(self, physics_dt: float) -> float | None:
        """Return a conservative per-step impulse when the articulation exposes forces."""

        getter = getattr(self.prim, "get_net_contact_forces", None)
        if getter is None:
            return None
        try:
            forces = np.asarray(getter(), dtype=np.float64)
        except (RuntimeError, TypeError, ValueError):
            return None
        if forces.size == 0 or not np.isfinite(forces).all():
            return None
        vectors = forces.reshape(-1, forces.shape[-1])[:, :3]
        return float(np.linalg.norm(vectors, axis=1).max(initial=0.0) * physics_dt)
