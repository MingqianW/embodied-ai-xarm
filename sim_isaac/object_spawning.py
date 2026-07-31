from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from policy_runtime.config import load_yaml


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    size_m: np.ndarray
    mass_kg: float
    color_rgb: np.ndarray
    position_m: np.ndarray
    orientation_wxyz: np.ndarray


@dataclass(frozen=True)
class TaskConfig:
    name: str
    prompt: str
    object_name: str
    object_size_m: np.ndarray
    object_mass_kg: float
    object_position_m: np.ndarray
    object_orientation_wxyz: np.ndarray
    table_size_m: np.ndarray
    table_position_m: np.ndarray
    object_xy_range_m: float
    object_yaw_range_deg: float
    arm_joint_noise_rad: float
    success_lift_height_m: float
    partial_credit_height_m: float
    target_color_rgb: np.ndarray
    target_type: str
    distractors: tuple[ObjectSpec, ...]
    distractor_probability: float
    place_in_ring: bool
    initial_arm_positions_rad: np.ndarray | None
    initial_gripper_policy: float | None

    @property
    def table_top_z_m(self) -> float:
        return float(self.table_position_m[2] + self.table_size_m[2] / 2.0)


def load_task_config(path: Path, task_name: str = "pick_up_object") -> TaskConfig:
    root = load_yaml(path)
    try:
        task = root["tasks"][task_name]
    except KeyError as exc:
        available = sorted(root.get("tasks", {}))
        raise ValueError(f"Unknown task {task_name!r}; available={available}") from exc
    obj = task["object"]
    table = task["table"]
    randomization = task.get("randomization", {})
    success = task.get("success", {})
    config = TaskConfig(
        name=task_name,
        prompt=str(task["prompt"]),
        object_name=str(obj["name"]),
        object_size_m=np.asarray(obj["size_m"], dtype=np.float32),
        object_mass_kg=float(obj["mass_kg"]),
        object_position_m=np.asarray(obj["pose"]["position_m"], dtype=np.float32),
        object_orientation_wxyz=np.asarray(
            obj["pose"]["orientation_quaternion_wxyz"], dtype=np.float32
        ),
        table_size_m=np.asarray(table["size_m"], dtype=np.float32),
        table_position_m=np.asarray(table["position_m"], dtype=np.float32),
        object_xy_range_m=float(randomization.get("object_xy_range_m", 0.0)),
        object_yaw_range_deg=float(randomization.get("object_yaw_range_deg", 0.0)),
        arm_joint_noise_rad=float(randomization.get("arm_joint_noise_rad", 0.0)),
        success_lift_height_m=float(success.get("lift_height_m", 0.08)),
        partial_credit_height_m=float(
            success.get("partial_credit_height_m", 0.04)
        ),
        target_color_rgb=np.asarray(
            obj.get("color_rgb", [0.85, 0.12, 0.08]), dtype=np.float32
        ),
        target_type=str(task.get("target_type", "object")),
        distractors=tuple(_load_object_spec(item) for item in task.get("distractors", [])),
        distractor_probability=float(
            randomization.get("distractor_probability", 0.0)
        ),
        place_in_ring=bool(task.get("place_in_ring", task.get("place_on_paper", False))),
        initial_arm_positions_rad=(
            None
            if "initial_arm_positions_rad" not in task
            else np.asarray(task["initial_arm_positions_rad"], dtype=np.float32)
        ),
        initial_gripper_policy=(
            None
            if "initial_gripper_policy" not in task
            else float(task["initial_gripper_policy"])
        ),
    )
    validate_task_config(config)
    return config


def validate_task_config(config: TaskConfig) -> None:
    for name, value in (
        ("object_size_m", config.object_size_m),
        ("object_position_m", config.object_position_m),
        ("table_size_m", config.table_size_m),
        ("table_position_m", config.table_position_m),
    ):
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite with shape (3,)")
    if config.object_orientation_wxyz.shape != (4,):
        raise ValueError("Object orientation must have shape (4,) in WXYZ order")
    if not np.isclose(np.linalg.norm(config.object_orientation_wxyz), 1.0, atol=1e-4):
        raise ValueError("Object orientation quaternion must be normalized")
    if np.any(config.object_size_m <= 0) or np.any(config.table_size_m <= 0):
        raise ValueError("Object and table sizes must be positive")
    if config.object_mass_kg <= 0:
        raise ValueError("Object mass must be positive")
    if min(
        config.object_xy_range_m,
        config.object_yaw_range_deg,
        config.arm_joint_noise_rad,
    ) < 0:
        raise ValueError("Randomization ranges cannot be negative")
    if not 0.0 <= config.distractor_probability <= 1.0:
        raise ValueError("distractor_probability must be in [0, 1]")
    if config.target_color_rgb.shape != (3,) or not np.isfinite(config.target_color_rgb).all():
        raise ValueError("target_color_rgb must be finite with shape (3,)")
    if config.initial_arm_positions_rad is not None and (
        config.initial_arm_positions_rad.shape != (6,)
        or not np.isfinite(config.initial_arm_positions_rad).all()
    ):
        raise ValueError("initial_arm_positions_rad must be finite with shape (6,)")
    for distractor in config.distractors:
        if distractor.size_m.shape != (3,) or np.any(distractor.size_m <= 0):
            raise ValueError("distractor sizes must be positive vectors")


def _load_object_spec(item: dict[str, Any]) -> ObjectSpec:
    pose = item.get("pose", {})
    return ObjectSpec(
        name=str(item["name"]),
        size_m=np.asarray(item["size_m"], dtype=np.float32),
        mass_kg=float(item.get("mass_kg", 0.03)),
        color_rgb=np.asarray(item.get("color_rgb", [0.3, 0.3, 0.3]), dtype=np.float32),
        position_m=np.asarray(pose.get("position_m", [0.45, 0.0, 0.065]), dtype=np.float32),
        orientation_wxyz=np.asarray(
            pose.get("orientation_quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]),
            dtype=np.float32,
        ),
    )
    if not (
        0
        < config.partial_credit_height_m
        <= config.success_lift_height_m
    ):
        raise ValueError("Task lift thresholds must satisfy 0 < partial <= success")


def randomized_object_pose(
    config: TaskConfig, seed: int | None
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    position = config.object_position_m.astype(np.float32, copy=True)
    if config.object_xy_range_m:
        position[:2] += rng.uniform(
            -config.object_xy_range_m, config.object_xy_range_m, size=2
        ).astype(np.float32)
    yaw = np.deg2rad(
        rng.uniform(-config.object_yaw_range_deg, config.object_yaw_range_deg)
    )
    yaw_quaternion = np.asarray(
        [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float32
    )
    w1, x1, y1, z1 = yaw_quaternion
    w2, x2, y2, z2 = config.object_orientation_wxyz
    orientation = np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )
    return position, orientation


class ObjectSpawner:
    def __init__(
        self,
        target_object: Any,
        config: TaskConfig,
        distractor_objects: tuple[Any, ...] = (),
    ) -> None:
        self.target_object = target_object
        self.config = config
        self.distractor_objects = distractor_objects
        self.last_position_m = config.object_position_m.copy()
        self.active_distractors: list[str] = []

    def reset(self, seed: int | None = None, *, force_distractors: bool | None = None) -> None:
        position, orientation = randomized_object_pose(self.config, seed)
        self.target_object.set_world_pose(position=position, orientation=orientation)
        if hasattr(self.target_object, "set_linear_velocity"):
            self.target_object.set_linear_velocity(np.zeros(3, dtype=np.float32))
        if hasattr(self.target_object, "set_angular_velocity"):
            self.target_object.set_angular_velocity(np.zeros(3, dtype=np.float32))
        self.last_position_m = position
        rng = np.random.default_rng(seed)
        include_distractors = (
            bool(force_distractors)
            if force_distractors is not None
            else bool(rng.random() < self.config.distractor_probability)
        )
        self.active_distractors = []
        for index, (prim, spec) in enumerate(zip(self.distractor_objects, self.config.distractors)):
            active = include_distractors
            # Inactive objects are moved safely below the local ground plane.
            distractor_position = spec.position_m if active else np.asarray([0.45, 0.0, -1.0], dtype=np.float32)
            prim.set_world_pose(position=distractor_position, orientation=spec.orientation_wxyz)
            if hasattr(prim, "set_linear_velocity"):
                prim.set_linear_velocity(np.zeros(3, dtype=np.float32))
            if hasattr(prim, "set_angular_velocity"):
                prim.set_angular_velocity(np.zeros(3, dtype=np.float32))
            if active:
                self.active_distractors.append(spec.name)

    def position(self) -> np.ndarray:
        position, _ = self.target_object.get_world_pose()
        return np.asarray(position, dtype=np.float32)
