"""Write MuJoCo trajectories in the existing real xArm raw episode format."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from sim_mujoco.data_collection.conversions import policy_state_from_mujoco
from sim_mujoco.environment import MuJoCoEnvironment
from sim_mujoco.remote_policy_observation import (
    BASE_CAMERA,
    WRIST_CAMERA,
    render_native_rgb,
)


ROBOT_LOG_FIELDS = (
    "ts",
    "j1_rad",
    "j2_rad",
    "j3_rad",
    "j4_rad",
    "j5_rad",
    "j6_rad",
    "tcp_x_m",
    "tcp_y_m",
    "tcp_z_m",
    "tcp_rx_rad",
    "tcp_ry_rad",
    "tcp_rz_rad",
    "gripper_mm",
    "realsense_0_file",
    "realsense_1_file",
    "realsense_2_file",
)
CAMERA_MAPPING = {
    "realsense_0": BASE_CAMERA,
    "realsense_1": WRIST_CAMERA,
    "realsense_2": "overview_camera",
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _deterministic_created_ts(task: str, episode_index: int) -> float:
    task_code = int.from_bytes(
        hashlib.sha256(task.encode("utf-8")).digest()[:3],
        byteorder="big",
    )
    return 1_800_000_000.0 + float(task_code % 500_000) + episode_index * 100.0


def _matrix_to_xyz_euler(rotation: np.ndarray) -> np.ndarray:
    """Return fixed-axis XYZ angles, matching the xArm robot-log convention."""

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    singular = sy < 1e-8
    if not singular:
        x = math.atan2(matrix[2, 1], matrix[2, 2])
        y = math.atan2(-matrix[2, 0], sy)
        z = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        x = math.atan2(-matrix[1, 2], matrix[1, 1])
        y = math.atan2(-matrix[2, 0], sy)
        z = 0.0
    return np.asarray([x, y, z], dtype=np.float64)


class RealRawEpisodeRecorder:
    """Record one episode as meta/CSV/three-camera PNG files."""

    def __init__(
        self,
        output_dir: Path,
        *,
        task: str,
        episode_index: int,
        seed: int,
        scene_variant: str,
        environment: MuJoCoEnvironment,
        save_hz: int = 10,
        task_id: str | None = None,
        task_prompt: str | None = None,
        requested_episode_index: int | None = None,
        base_seed: int | None = None,
        retry_index: int | None = None,
    ) -> None:
        if save_hz != 10:
            raise ValueError("The audited real raw format is recorded at 10 Hz")
        self.output_dir = Path(output_dir)
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty episode: {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for camera_name in CAMERA_MAPPING:
            (self.output_dir / camera_name).mkdir(parents=True, exist_ok=True)
        self.task = str(task)
        self.task_id = str(task_id or task)
        self.task_prompt = str(task_prompt or task)
        self.episode_index = int(episode_index)
        self.requested_episode_index = int(
            episode_index if requested_episode_index is None else requested_episode_index
        )
        self.base_seed = int(seed if base_seed is None else base_seed)
        self.retry_index = int(0 if retry_index is None else retry_index)
        self.seed = int(seed)
        self.scene_variant = str(scene_variant)
        self.environment = environment
        self.save_hz = int(save_hz)
        self.created_ts = _deterministic_created_ts(
            self.task,
            self.episode_index,
        )
        self.rows: list[dict[str, Any]] = []
        self.gripper_events: list[dict[str, Any]] = []
        self._last_gripper_target: float | None = None

    def _state(self) -> np.ndarray:
        state = policy_state_from_mujoco(
            self.environment.context.model,
            self.environment.context.data,
        ).astype(np.float64)
        return state

    def _tcp(self) -> tuple[np.ndarray, np.ndarray]:
        model = self.environment.context.model
        data = self.environment.context.data
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "tool_center_point",
        )
        if site_id < 0:
            raise RuntimeError("TCP site not found: tool_center_point")
        position_mm = (
            np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
            * 1000.0
        )
        rotation = np.asarray(
            data.site_xmat[site_id],
            dtype=np.float64,
        ).reshape(3, 3)
        return position_mm, _matrix_to_xyz_euler(rotation)

    def record_observation(
        self,
        *,
        gripper_target: float | None = None,
    ) -> None:
        frame_index = len(self.rows)
        timestamp = self.created_ts + frame_index / float(self.save_hz)
        timestamp_ms = int(round(timestamp * 1000.0))
        image_paths: dict[str, str] = {}
        for raw_camera, mujoco_camera in CAMERA_MAPPING.items():
            image = render_native_rgb(
                self.environment.context.renderer,
                self.environment.context.data,
                mujoco_camera,
            )
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError(
                    f"{mujoco_camera} must render RGB uint8 480x640, "
                    f"got {image.shape} {image.dtype}"
                )
            relative = (
                Path(raw_camera)
                / f"RGB_{timestamp_ms}_{raw_camera}.png"
            )
            Image.fromarray(image).save(self.output_dir / relative)
            image_paths[raw_camera] = relative.as_posix()

        state = self._state()
        tcp_position_mm, tcp_euler = self._tcp()
        row: dict[str, Any] = {
            "ts": f"{timestamp:.6f}",
            **{
                f"j{index + 1}_rad": f"{state[index]:.8f}"
                for index in range(6)
            },
            "tcp_x_m": f"{tcp_position_mm[0]:.6f}",
            "tcp_y_m": f"{tcp_position_mm[1]:.6f}",
            "tcp_z_m": f"{tcp_position_mm[2]:.6f}",
            "tcp_rx_rad": f"{tcp_euler[0]:.8f}",
            "tcp_ry_rad": f"{tcp_euler[1]:.8f}",
            "tcp_rz_rad": f"{tcp_euler[2]:.8f}",
            "gripper_mm": f"{state[6]:.3f}",
            "realsense_0_file": image_paths["realsense_0"],
            "realsense_1_file": image_paths["realsense_1"],
            "realsense_2_file": image_paths["realsense_2"],
        }
        self.rows.append(row)

        if gripper_target is not None and (
            self._last_gripper_target is None
            or abs(float(gripper_target) - self._last_gripper_target) >= 0.5
        ):
            target = float(gripper_target)
            self.gripper_events.extend(
                (
                    {
                        "ts": f"{timestamp:.6f}",
                        "key": "ORACLE",
                        "target_width_mm": f"{target:.3f}",
                        "sent": 0,
                        "ret_code": 0,
                    },
                    {
                        "ts": f"{timestamp + 0.001:.6f}",
                        "key": "SEND",
                        "target_width_mm": f"{target:.3f}",
                        "sent": 1,
                        "ret_code": 0,
                    },
                )
            )
            self._last_gripper_target = target

    def finalize(
        self,
        *,
        success: bool,
        failure_reason: str | None,
        initial_conditions: dict[str, Any],
        task_metrics: dict[str, Any],
        oracle_transitions: list[dict[str, Any]],
        oracle_plan: dict[str, Any],
        validation_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if len(self.rows) < 2:
            raise ValueError("A raw episode needs at least two robot-log rows")
        with (self.output_dir / "robot_log.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=ROBOT_LOG_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        with (self.output_dir / "gripper_events.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "ts",
                    "key",
                    "target_width_mm",
                    "sent",
                    "ret_code",
                ),
            )
            writer.writeheader()
            writer.writerows(self.gripper_events)

        meta = {
            "task": self.task_prompt,
            "task_id": self.task_id,
            "task_prompt": self.task_prompt,
            "episode_index": self.episode_index,
            "requested_episode_index": self.requested_episode_index,
            "created_ts": self.created_ts,
            "poll_hz": 30.0,
            "save_hz": float(self.save_hz),
            "async_writer": True,
            "queue_max": 8192,
            "drop_policy": "drop_newest",
            "cameras": [
                {
                    "name": raw_name,
                    "kind": "mujoco",
                    "serial": mujoco_name,
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                }
                for raw_name, mujoco_name in CAMERA_MAPPING.items()
            ],
            "simulation": {
                "schema_version": "xarm_real_raw_compatible_sim_v1",
                "seed": self.seed,
                "base_seed": self.base_seed,
                "retry_index": self.retry_index,
                "resolved_seed": self.seed,
                "scene_variant": self.scene_variant,
                "success": bool(success),
                "failure_reason": failure_reason,
                "robot_log_rows": len(self.rows),
                "training_samples_after_real_converter": len(self.rows) - 1,
                "temporal_alignment": (
                    "row t is observation_t; the existing real converter uses "
                    "row t+1 state as the absolute action target"
                ),
                "initial_conditions": initial_conditions,
                "task_metrics": task_metrics,
                "oracle_transitions": oracle_transitions,
                "oracle_plan": oracle_plan,
                "validation": validation_metadata or {},
            },
        }
        _write_json(self.output_dir / "meta.json", meta)
        return meta
