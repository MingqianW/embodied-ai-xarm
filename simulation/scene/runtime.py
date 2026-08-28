from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from simulation.robot.gripper_mapping import raw_hardware_to_actuator_ctrl_rad
from simulation.scene.objects import _body_id
from simulation.scene.objects import _freejoint_dof_address
from simulation.scene.objects import _freejoint_qpos_address
from simulation.scene.objects import _set_body_enabled
from simulation.scene.tasks import TABLE_TOP_Z


@dataclass
class TaskSceneRuntime:
    model: mujoco.MjModel
    data: mujoco.MjData
    task_name: str
    spec: dict[str, Any]
    catalog: dict[str, Any]
    released: bool = False
    success_streak: int = 0
    initial_target_z: float = 0.0
    release_simulation_time_s: float | None = None

    @property
    def prompt(self) -> str:
        return str(self.spec["prompt"])

    @property
    def target_body(self) -> str:
        return str(self.spec["target_body"])

    @property
    def active_target_body(self) -> str:
        """Return the physical object representing the task target right now."""

        if self.spec["success"]["type"] == "place_in_ring" and not self.released:
            return "held_red_pepper"
        return self.target_body

    def _target_pose(self) -> np.ndarray:
        return np.asarray(
            self.data.xpos[_body_id(self.model, self.active_target_body)],
            dtype=np.float64,
        )

    def adjust_observation(self, observation: dict[str, Any]) -> None:
        if (
            self.spec["success"]["type"] == "place_in_ring"
            and not self.released
            and "initial_gripper_raw" in self.spec
        ):
            observation["observation/state"][6] = float(
                self.spec["initial_gripper_raw"]
            )

    def physical_gripper_raw_target(self, gripper_raw_target: float) -> float:
        if self.spec["success"]["type"] != "place_in_ring":
            return float(gripper_raw_target)
        held_raw = float(self.spec["initial_gripper_raw"])
        if not self.released:
            return held_raw
        if float(gripper_raw_target) >= float(self.spec["release_gripper_raw"]):
            return max(
                float(gripper_raw_target),
                float(self.spec.get("release_min_gripper_raw", held_raw)),
            )
        return float(gripper_raw_target)

    def physical_gripper_target(
        self,
        gripper_raw_target: float,
        default_sim_target: float,
        gripper_config: dict[str, Any],
    ) -> float:
        """Legacy slide target used only by frozen runtime diagnostics."""

        if self.spec["success"]["type"] != "place_in_ring":
            return float(default_sim_target)
        mapping = gripper_config.get("gripper_mapping", {})
        canonical_four_bar = "sim_joint_min_rad" in mapping
        held = (
            raw_hardware_to_actuator_ctrl_rad(
                float(self.spec["initial_gripper_raw"]), gripper_config
            )
            if canonical_four_bar
            else float(self.spec.get("initial_gripper_sim_half_width", 0.012))
        )
        if not self.released:
            return held
        if float(gripper_raw_target) >= float(self.spec["release_gripper_raw"]):
            return max(
                float(default_sim_target),
                float(
                    self.spec.get(
                        "release_min_sim_half_width",
                        raw_hardware_to_actuator_ctrl_rad(652.0, gripper_config),
                    )
                ),
            )
        return float(default_sim_target)

    def release_if_requested(self, gripper_raw_target: float) -> bool:
        threshold = self.spec.get("release_gripper_raw")
        if (
            threshold is None
            or self.released
            or float(gripper_raw_target) < float(threshold)
        ):
            return False

        held_body_id = _body_id(self.model, "held_red_pepper")
        object_addr = _freejoint_qpos_address(self.model, "red_pepper")
        dof_addr = _freejoint_dof_address(self.model, "red_pepper")
        self.data.qpos[object_addr : object_addr + 3] = self.data.xpos[held_body_id]
        self.data.qpos[object_addr + 3 : object_addr + 7] = self.data.xquat[held_body_id]
        self.data.qvel[dof_addr : dof_addr + 6] = 0.0
        _set_body_enabled(self.model, "held_red_pepper", False)
        _set_body_enabled(self.model, "red_pepper", True)
        self.released = True
        self.release_simulation_time_s = float(self.data.time)
        mujoco.mj_forward(self.model, self.data)
        return True

    def metrics(self) -> dict[str, Any]:
        target_position = self._target_pose()
        success_config = self.spec["success"]
        success_type = str(success_config["type"])
        metrics: dict[str, Any] = {
            "task": self.task_name,
            "success_type": success_type,
            "target_body": self.target_body,
            "target_position": target_position.tolist(),
            "released": self.released,
            "release_simulation_time_s": self.release_simulation_time_s,
        }
        if success_type == "lift":
            lift_height = float(target_position[2] - self.initial_target_z)
            metrics["lift_height_m"] = lift_height
            metrics["instant_success"] = lift_height >= float(
                success_config["lift_height_m"]
            )
        elif success_type == "place_in_ring":
            ring_position = np.asarray(
                self.data.xpos[_body_id(self.model, str(success_config["ring_body"]))],
                dtype=np.float64,
            )
            ring_distance = float(
                np.linalg.norm(target_position[:2] - ring_position[:2])
            )
            metrics["ring_position"] = ring_position.tolist()
            metrics["ring_xy_distance_m"] = ring_distance
            metrics["height_above_table_m"] = float(target_position[2] - TABLE_TOP_Z)
            metrics["instant_success"] = bool(
                self.released
                and ring_distance <= float(success_config["ring_radius_m"])
                and target_position[2] - TABLE_TOP_Z
                <= float(success_config["max_height_above_table_m"])
            )
        else:
            raise ValueError(f"Unsupported success type: {success_type}")
        metrics["required_success_streak"] = int(
            success_config.get("sustained_policy_steps", 1)
        )
        metrics["success_streak"] = self.success_streak
        metrics["task_success"] = bool(
            self.success_streak >= metrics["required_success_streak"]
        )
        return metrics

    def update_success(self) -> dict[str, Any]:
        metrics = self.metrics()
        if metrics["instant_success"]:
            self.success_streak += 1
        else:
            self.success_streak = 0
        metrics["success_streak"] = self.success_streak
        metrics["task_success"] = bool(
            self.success_streak >= metrics["required_success_streak"]
        )
        return metrics
