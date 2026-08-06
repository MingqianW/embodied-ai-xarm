from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from sim_mujoco.collision import collision_diagnostics
from policy_runtime.image_preprocessing import ImagePreprocessingConfig
from policy_runtime.observation_builder import build_policy_observation
from policy_runtime.schemas import PolicyObservation
from sim_mujoco.remote_policy_observation import (
    BASE_CAMERA,
    DEFAULT_CAMERA_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    WRIST_CAMERA,
    arm_actuator_ctrl_limits,
    arm_joint_limits,
    get_robot_state,
    gripper_raw_to_sim,
    initialize_scene,
    load_simulation,
    policy_image,
    render_native_rgb,
)
from sim_mujoco.task_scenes import (
    TASK_CONFIG_PATH,
    TaskSceneRuntime,
    configure_task_scene,
    resolve_task,
)


class MuJoCoEnvironment:
    """Canonical xArm environment backed by the existing MuJoCo scene."""

    def __init__(
        self,
        *,
        model_path: Path = DEFAULT_MODEL_PATH,
        camera_config_path: Path = DEFAULT_CAMERA_CONFIG_PATH,
        task_scene_config_path: Path = TASK_CONFIG_PATH,
        task: str = "red_block",
        prompt: str | None = None,
        settle_steps: int = 500,
        object_xy_range: float = 0.0,
        object_yaw_range_deg: float = 0.0,
        joint_noise: float = 0.0,
        scene_variant: str = "clean",
    ) -> None:
        self.context = load_simulation(model_path, camera_config_path)
        self.task_scene_config_path = Path(task_scene_config_path).resolve()
        self.task, task_spec = resolve_task(task, self.task_scene_config_path)
        self.prompt = str(prompt or task_spec["prompt"])
        self.settle_steps = int(settle_steps)
        self.object_xy_range = float(object_xy_range)
        self.object_yaw_range_deg = float(object_yaw_range_deg)
        self.joint_noise = float(joint_noise)
        self.scene_variant = str(scene_variant)
        self.task_runtime: TaskSceneRuntime | None = None
        self.initial_conditions: dict[str, Any] = {}
        self._closed = False
        self._last_step_started_s = float(self.context.data.time)
        self._last_step_duration_s = 0.0

    @property
    def joint_limits(self) -> np.ndarray:
        model_limits = arm_joint_limits(self.context.model)
        actuator_limits = arm_actuator_ctrl_limits(self.context.model)
        return np.column_stack(
            (
                np.maximum(model_limits[:, 0], actuator_limits[:, 0]),
                np.minimum(model_limits[:, 1], actuator_limits[:, 1]),
            )
        ).astype(np.float32)

    def _preprocessing(self) -> ImagePreprocessingConfig:
        render = self.context.config["render"]
        return ImagePreprocessingConfig(
            width=int(render["policy_width"]),
            height=int(render["policy_height"]),
            input_color_order="RGB",
        )

    def reset(self, seed: int | None = None) -> PolicyObservation:
        initialize_scene(
            self.context.model,
            self.context.data,
            settle_steps=0,
        )
        self.task_runtime, self.initial_conditions = configure_task_scene(
            self.context.model,
            self.context.data,
            task=self.task,
            seed=seed,
            object_xy_range=self.object_xy_range,
            object_yaw_range_deg=self.object_yaw_range_deg,
            joint_noise=self.joint_noise,
            scene_variant=self.scene_variant,
            settle_steps=self.settle_steps,
            config_path=self.task_scene_config_path,
        )
        self._last_step_started_s = float(self.context.data.time)
        self._last_step_duration_s = 0.0
        return self.observe()

    def observe(self) -> PolicyObservation:
        base_native = render_native_rgb(
            self.context.renderer,
            self.context.data,
            BASE_CAMERA,
        )
        wrist_native = render_native_rgb(
            self.context.renderer,
            self.context.data,
            WRIST_CAMERA,
        )
        observation = build_policy_observation(
            base_native,
            wrist_native,
            get_robot_state(self.context.model, self.context.data, self.context.config),
            self.prompt,
            base_preprocessing=self._preprocessing(),
            wrist_preprocessing=self._preprocessing(),
            timestamp_s=float(self.context.data.time),
            frame_ids={"base": f"{self.context.data.time:.9f}", "wrist": f"{self.context.data.time:.9f}"},
            metadata={"simulator": "mujoco", "task": self.task},
        )
        return observation

    def apply_action(self, action: np.ndarray) -> None:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError(f"Canonical action must be finite with shape (7,), got {value.shape}")
        arm_limits = self.joint_limits
        self.context.data.ctrl[:6] = np.clip(value[:6], arm_limits[:, 0], arm_limits[:, 1])
        gripper_target = gripper_raw_to_sim(float(value[6]), self.context.config)
        if self.task_runtime is not None:
            self.task_runtime.release_if_requested(float(value[6]))
            gripper_target = self.task_runtime.physical_gripper_target(
                float(value[6]),
                gripper_target,
            )
        gripper_limits = self.context.model.actuator_ctrlrange[6]
        self.context.data.ctrl[6] = float(
            np.clip(gripper_target, gripper_limits[0], gripper_limits[1])
        )

    def step_physics(self, duration_s: float) -> None:
        if duration_s <= 0:
            raise ValueError("Physics duration must be positive")
        start = float(self.context.data.time)
        steps = max(1, int(round(float(duration_s) / self.context.model.opt.timestep)))
        for _ in range(steps):
            mujoco.mj_step(self.context.model, self.context.data)
        self._last_step_started_s = start
        self._last_step_duration_s = float(self.context.data.time) - start

    def hold_position(self) -> None:
        state = get_robot_state(self.context.model, self.context.data, self.context.config)
        self.context.data.ctrl[:6] = state[:6]
        self.context.data.ctrl[6] = gripper_raw_to_sim(float(state[6]), self.context.config)

    def _max_contact_force(self) -> float:
        maximum = 0.0
        force = np.zeros(6, dtype=np.float64)
        for contact_index in range(self.context.data.ncon):
            mujoco.mj_contactForce(self.context.model, self.context.data, contact_index, force)
            maximum = max(maximum, float(np.linalg.norm(force[:3])))
        return maximum

    def is_safe(self) -> bool:
        data = self.context.data
        finite = (
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.ctrl).all()
            and np.isfinite(data.time)
        )
        limits = self.joint_limits
        state = get_robot_state(self.context.model, data, self.context.config)
        joints_valid = bool(
            np.all(state[:6] >= limits[:, 0] - 1e-4)
            and np.all(state[:6] <= limits[:, 1] + 1e-4)
        )
        time_advanced = self._last_step_duration_s == 0.0 or float(data.time) > self._last_step_started_s
        collision_safe = not collision_diagnostics(self.context.model, data)["forbidden"]
        return bool(finite and joints_valid and time_advanced and collision_safe)

    def safety_diagnostics(self) -> dict[str, Any]:
        state = get_robot_state(self.context.model, self.context.data, self.context.config)
        tracking_error = np.abs(
            np.asarray(self.context.data.ctrl[:6], dtype=np.float64)
            - np.asarray(state[:6], dtype=np.float64)
        )
        diagnostics = {
            "simulator": "mujoco",
            "simulation_time_s": float(self.context.data.time),
            "last_step_duration_s": self._last_step_duration_s,
            "finite_qpos": bool(np.isfinite(self.context.data.qpos).all()),
            "finite_qvel": bool(np.isfinite(self.context.data.qvel).all()),
            "contact_count": int(self.context.data.ncon),
            "max_contact_force_n": self._max_contact_force(),
            "max_joint_tracking_error_rad": float(tracking_error.max(initial=0.0)),
        }
        diagnostics["collision"] = collision_diagnostics(
            self.context.model,
            self.context.data,
        )
        return diagnostics

    def recording_frames(self) -> dict[str, np.ndarray]:
        viewer = render_native_rgb(
            self.context.renderer,
            self.context.data,
            "overview_camera",
        )
        base = policy_image(
            render_native_rgb(self.context.renderer, self.context.data, BASE_CAMERA),
            self.context.config,
        )
        wrist = policy_image(
            render_native_rgb(self.context.renderer, self.context.data, WRIST_CAMERA),
            self.context.config,
        )
        return {"viewer": viewer, "base": base, "wrist": wrist}

    def close(self) -> None:
        if not self._closed:
            self.context.close()
            self._closed = True

    def __enter__(self) -> "MuJoCoEnvironment":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
