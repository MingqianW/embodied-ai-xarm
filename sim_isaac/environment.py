from __future__ import annotations

import time
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from policy_runtime.config import load_yaml, repository_path
from policy_runtime.observation_builder import build_policy_observation
from policy_runtime.schemas import PolicyObservation
from sim_isaac.articulation import load_robot_mapping
from sim_isaac.cameras import IsaacCameraRig, load_camera_configs
from sim_isaac.dependencies import require_isaac_runtime
from sim_isaac.object_spawning import load_task_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "config"


class IsaacEnvironment:
    """Canonical xArm policy environment backed by local Isaac Sim.

    Isaac imports happen only after the bundled SimulationApp starts. Construct
    this class using Isaac Sim's python.bat/python.sh, never ordinary CPython.
    """

    simulator_name = "isaac"

    def __init__(
        self,
        *,
        robot_config_path: Path = DEFAULT_CONFIG_DIR / "robot.yaml",
        camera_config_path: Path = DEFAULT_CONFIG_DIR / "cameras.yaml",
        control_config_path: Path = DEFAULT_CONFIG_DIR / "control.yaml",
        task_config_path: Path = DEFAULT_CONFIG_DIR / "tasks.yaml",
        task_name: str = "pick_up_object",
        prompt: str | None = None,
        headless: bool | None = None,
        seed: int | None = None,
        scene_variant: str | None = None,
    ) -> None:
        require_isaac_runtime()
        self.mapping = load_robot_mapping(Path(robot_config_path))
        self.camera_configs = load_camera_configs(Path(camera_config_path))
        self.control_config = load_yaml(Path(control_config_path))
        self.task = load_task_config(Path(task_config_path), task_name)
        simulation = self.control_config["simulation"]
        configured_action_mode = str(
            self.control_config["policy"]["action_mode"]
        )
        if configured_action_mode != self.mapping.action_mode:
            raise ValueError(
                "robot.yaml and control.yaml action_mode disagree: "
                f"{self.mapping.action_mode!r} != {configured_action_mode!r}"
            )
        safety = self.control_config["safety"]
        self.physics_hz = float(simulation["physics_hz"])
        self.rendering_hz = float(simulation["rendering_hz"])
        if self.physics_hz <= 0 or self.rendering_hz <= 0:
            raise ValueError("Physics and rendering rates must be positive")
        self.physics_dt = 1.0 / self.physics_hz
        self.render_every_steps = max(1, int(round(self.physics_hz / self.rendering_hz)))
        self.prompt = self.task.prompt if prompt is None else str(prompt)
        self.seed = int(simulation.get("seed", 0) if seed is None else seed)
        if scene_variant not in (None, "auto", "clean", "distractors"):
            raise ValueError("scene_variant must be auto, clean, or distractors")
        self.scene_variant = scene_variant or "auto"
        self._max_tracking_error = float(safety["max_joint_tracking_error_rad"])
        self._max_contact_impulse = float(safety["max_contact_impulse_ns"])
        self._require_contact_sensor = bool(
            safety.get("require_contact_impulse_sensor", False)
        )
        self._minimum_table_clearance = float(safety["minimum_table_clearance_m"])
        self._stale_camera_timeout = float(safety["stale_camera_timeout_s"])
        self._real_time_factor_warning = float(
            safety.get("real_time_factor_warning", 0.5)
        )
        self._simulation_app: Any | None = None
        self.scene: Any | None = None
        self.cameras: IsaacCameraRig | None = None
        self._closed = False
        self._last_command: np.ndarray | None = None
        self._last_step_started_s = 0.0
        self._last_step_duration_s = 0.0
        self._last_step_wall_s = 0.0
        self._last_contact_impulse_ns: float | None = None
        self._last_frames: dict[str, np.ndarray] = {}
        self._last_camera_render_sim_s: float | None = None
        self._steps = 0

        from sim_isaac.version_compat import create_simulation_app

        self._simulation_app = create_simulation_app(
            headless=bool(
                simulation["headless"] if headless is None else headless
            ),
            renderer=str(simulation.get("renderer", "RayTracedLighting")),
            anti_aliasing=int(simulation.get("anti_aliasing", 1)),
        )
        try:
            # SimulationApp must exist before importing any remaining Isaac modules.
            from sim_isaac.scene import build_scene

            configured_asset = os.environ.get(
                "XARM_ASSET_PATH", self.mapping.asset_path
            )
            asset_path = repository_path(PROJECT_ROOT, configured_asset).resolve()
            self.scene = build_scene(
                robot_asset_path=asset_path,
                mapping=self.mapping,
                task=self.task,
                physics_hz=self.physics_hz,
                rendering_hz=self.rendering_hz,
            )
            self.cameras = IsaacCameraRig(
                self.camera_configs,
                rendering_hz=self.rendering_hz,
            )
            self.reset(self.seed)
        except Exception:
            # SimulationApp.close() may terminate the bundled Python process before
            # the original exception can propagate to the CLI entry point.
            import traceback

            traceback.print_exc()
            sys.stderr.flush()
            self.close()
            raise

    @property
    def joint_limits(self) -> np.ndarray:
        return self.mapping.joint_limits_rad.copy()

    def _simulation_time(self) -> float:
        if self.scene is None:
            return 0.0
        return float(getattr(self.scene.world, "current_time", 0.0))

    def reset(self, seed: int | None = None) -> PolicyObservation:
        if self.scene is None:
            raise RuntimeError("Isaac scene is not initialized")
        reset_seed = self.seed if seed is None else int(seed)
        self.scene.world.reset()
        home_state = np.concatenate(
            [
                (
                    self.mapping.initial_arm_positions_rad
                    if self.task.initial_arm_positions_rad is None
                    else self.task.initial_arm_positions_rad
                ),
                np.asarray(
                    [
                        self.mapping.initial_gripper_policy
                        if self.task.initial_gripper_policy is None
                        else self.task.initial_gripper_policy
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        if self.task.arm_joint_noise_rad:
            rng = np.random.default_rng(reset_seed)
            home_state[:6] += rng.uniform(
                -self.task.arm_joint_noise_rad,
                self.task.arm_joint_noise_rad,
                size=6,
            ).astype(np.float32)
            home_state[:6] = np.clip(
                home_state[:6],
                self.mapping.joint_limits_rad[:, 0],
                self.mapping.joint_limits_rad[:, 1],
            )
        self.scene.robot.set_policy_state(home_state)
        self.scene.robot.apply_canonical_target(home_state)
        self.scene.objects.reset(
            reset_seed,
            force_distractors=(
                None
                if self.scene_variant == "auto"
                else self.scene_variant == "distractors"
            ),
        )
        self._last_command = home_state.copy()
        self._last_step_started_s = self._simulation_time()
        self._last_step_duration_s = 0.0
        self._steps = 0
        # Advance and render enough frames for articulation/camera buffers to be valid.
        for index in range(max(4, self.render_every_steps * 2)):
            render = index % self.render_every_steps == 0
            self.scene.world.step(render=render)
            self._steps += 1
            if render:
                self._last_camera_render_sim_s = self._simulation_time()
        # RTX annotators can lag the first rendered physics frame on a cold,
        # fully offline startup. Warm them up for a bounded number of frames.
        import omni.timeline

        # Follow Isaac Sim 6.0's official CameraSensor standalone example:
        # start the Omni timeline directly, then drive SimulationApp updates
        # until the Replicator annotator has produced its first frame.
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for attempt in range(600):
            if self._simulation_app is not None:
                self._simulation_app.update()
            try:
                return self.observe()
            except (RuntimeError, ValueError) as exc:
                camera_not_ready = (
                    "camera frame is not ready" in str(exc)
                    or "Expected HxWx3 or HxWx4 image" in str(exc)
                )
                if not camera_not_ready:
                    raise
                if attempt == 599:
                    raise RuntimeError(
                        "Isaac cameras did not produce a frame after 600 warm-up renders"
                    ) from exc
                self.scene.world.step(render=True)
                self._steps += 1
                self._last_camera_render_sim_s = self._simulation_time()
        raise AssertionError("unreachable camera warm-up state")

    def observe(self) -> PolicyObservation:
        if self.scene is None or self.cameras is None:
            raise RuntimeError("Isaac environment is not initialized")
        frames = self.cameras.read()
        self._last_frames = frames
        captured_sim_s = self._simulation_time()
        self._last_camera_render_sim_s = captured_sim_s
        state = self.scene.robot.get_policy_state()
        return build_policy_observation(
            frames["base"],
            frames["wrist"],
            state,
            self.prompt,
            base_preprocessing=self.camera_configs["base"].preprocessing,
            wrist_preprocessing=self.camera_configs["wrist"].preprocessing,
            timestamp_s=captured_sim_s,
            frame_ids=self.cameras.frame_ids,
            metadata={
                "simulator": "isaac",
                "camera_backend": self.cameras.backend,
                "physics_hz": self.physics_hz,
                "rendering_hz": self.rendering_hz,
            },
        )

    def apply_action(self, action: np.ndarray) -> None:
        if self.scene is None:
            raise RuntimeError("Isaac environment is not initialized")
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError(f"Canonical action must be finite shape (7,), got {value.shape}")
        value = value.copy()
        value[:6] = np.clip(
            value[:6],
            self.mapping.joint_limits_rad[:, 0],
            self.mapping.joint_limits_rad[:, 1],
        )
        value[6] = np.clip(
            value[6],
            self.mapping.gripper_policy_closed,
            self.mapping.gripper_policy_open,
        )
        # Shared safety converts either configured policy mode into absolute
        # canonical targets before the environment boundary.
        self.scene.robot.apply_canonical_target(value)
        self._last_command = value

    def step_physics(self, duration_s: float) -> None:
        if self.scene is None:
            raise RuntimeError("Isaac environment is not initialized")
        if duration_s <= 0:
            raise ValueError("Physics duration must be positive")
        start = self._simulation_time()
        wall_started = time.perf_counter()
        steps = max(1, int(round(float(duration_s) / self.physics_dt)))
        for _ in range(steps):
            if (
                self._simulation_app is not None
                and hasattr(self._simulation_app, "is_running")
                and not self._simulation_app.is_running()
            ):
                raise KeyboardInterrupt("Isaac Sim window was closed")
            self._steps += 1
            render = self._steps % self.render_every_steps == 0
            self.scene.world.step(render=render)
            if render:
                self._last_camera_render_sim_s = self._simulation_time()
        self._last_step_started_s = start
        self._last_step_duration_s = self._simulation_time() - start
        self._last_step_wall_s = time.perf_counter() - wall_started
        self._last_contact_impulse_ns = self.scene.robot.max_contact_impulse(
            self.physics_dt
        )

    def hold_position(self) -> None:
        if self.scene is not None:
            self.scene.robot.hold_position()

    def _end_effector_z(self) -> float | None:
        if self.scene is None:
            return None
        try:
            from pxr import UsdGeom

            stage = self.scene.world.stage
            path = f"{self.mapping.articulation_prim_path}/{self.mapping.end_effector_frame}"
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                return None
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
            return float(matrix.ExtractTranslation()[2])
        except (AttributeError, RuntimeError, TypeError):
            return None

    def is_safe(self) -> bool:
        diagnostics = self.safety_diagnostics()
        contact_ok = (
            diagnostics["contact_impulse_available"]
            or not self._require_contact_sensor
        ) and (
            diagnostics["max_contact_impulse_ns"] is None
            or diagnostics["max_contact_impulse_ns"] <= self._max_contact_impulse
        )
        clearance = diagnostics["end_effector_table_clearance_m"]
        clearance_ok = clearance is None or clearance >= self._minimum_table_clearance
        return bool(
            diagnostics["finite_state"]
            and diagnostics["transforms_finite"]
            and diagnostics["joints_within_limits"]
            and diagnostics["time_advanced"]
            and diagnostics["camera_fresh"]
            and diagnostics["max_joint_tracking_error_rad"] <= self._max_tracking_error
            and diagnostics["object_above_table"]
            and contact_ok
            and clearance_ok
        )

    def require_safe(self, phase: str) -> None:
        """Refuse to continue a phase when the current bounded state is unsafe."""

        if not self.is_safe():
            raise RuntimeError(
                f"{phase} preflight failed; refusing to continue from an unsafe "
                f"Isaac state: {self.safety_diagnostics()}"
            )

    def safety_diagnostics(self) -> dict[str, Any]:
        if self.scene is None:
            return {"simulator": "isaac", "initialized": False}
        state = self.scene.robot.get_policy_state()
        target = state if self._last_command is None else self._last_command
        tracking = np.abs(state[:6] - target[:6])
        limits = self.mapping.joint_limits_rad
        object_position = self.scene.objects.position()
        object_position_finite = bool(np.isfinite(object_position).all())
        object_bottom = float(object_position[2] - self.task.object_size_m[2] / 2.0)
        ee_z = self._end_effector_z()
        clearance = None if ee_z is None else ee_z - self.task.table_top_z_m
        camera_age = (
            float("inf")
            if self._last_camera_render_sim_s is None
            else max(0.0, self._simulation_time() - self._last_camera_render_sim_s)
        )
        sim_time = self._simulation_time()
        real_time_factor = (
            None
            if self._last_step_wall_s <= 0
            else self._last_step_duration_s / self._last_step_wall_s
        )
        return {
            "simulator": "isaac",
            "initialized": True,
            "simulation_time_s": sim_time,
            "last_step_duration_s": self._last_step_duration_s,
            "last_step_wall_s": self._last_step_wall_s,
            "real_time_factor": real_time_factor,
            "real_time_factor_degraded": bool(
                real_time_factor is not None
                and real_time_factor < self._real_time_factor_warning
            ),
            "finite_state": bool(np.isfinite(state).all()),
            "transforms_finite": bool(
                object_position_finite
                and ee_z is not None
                and np.isfinite(ee_z)
            ),
            "joints_within_limits": bool(
                np.all(state[:6] >= limits[:, 0] - 1e-4)
                and np.all(state[:6] <= limits[:, 1] + 1e-4)
            ),
            "time_advanced": bool(
                self._last_step_duration_s == 0.0
                or sim_time > self._last_step_started_s
            ),
            "camera_fresh": camera_age <= self._stale_camera_timeout,
            "camera_age_simulation_s": camera_age,
            "camera_backend": None if self.cameras is None else self.cameras.backend,
            "max_joint_tracking_error_rad": float(tracking.max(initial=0.0)),
            "contact_impulse_available": self._last_contact_impulse_ns is not None,
            "max_contact_impulse_ns": self._last_contact_impulse_ns,
            "max_contact_impulse_limit_ns": self._max_contact_impulse,
            "contact_impulse_required": self._require_contact_sensor,
            "object_position_m": object_position.tolist(),
            "object_above_table": bool(
                object_position_finite
                and object_bottom >= self.task.table_top_z_m - 0.003
            ),
            "end_effector_z_m": ee_z,
            "end_effector_table_clearance_m": clearance,
        }

    def recording_frames(self) -> dict[str, np.ndarray]:
        if not self._last_frames:
            self.observe()
        return {
            "viewer": self._last_frames["base"],
            "base": self._last_frames["base"],
            "wrist": self._last_frames["wrist"],
        }

    def close(self) -> None:
        if self._closed:
            return
        if self.cameras is not None:
            self.cameras.close()
        if self.scene is not None:
            stop = getattr(self.scene.world, "stop", None)
            if stop is not None:
                stop()
            clear = getattr(self.scene.world, "clear", None)
            if clear is not None:
                clear()
        if self._simulation_app is not None:
            self._simulation_app.close()
        self._closed = True

    def __enter__(self) -> "IsaacEnvironment":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
