"""Raw MuJoCo episode recorder with real-dataset-aligned training fields."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from policy_runtime.recording import tile_recording_frame
from sim_mujoco.collision import collision_diagnostics
from sim_mujoco.data_collection.conversions import policy_state_from_mujoco
from sim_mujoco.environment import MuJoCoEnvironment
from sim_mujoco.remote_policy_observation import (
    BASE_CAMERA,
    DEFAULT_CAMERA_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    WRIST_CAMERA,
    policy_image,
    render_native_rgb,
)


RAW_SCHEMA_VERSION = "xarm_mujoco_raw_v1"
ORACLE_VERSION = "red_block_fsm_dls_ik_v1"
REAL_TRAINING_PROMPT = "pick up the red block"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


class _RgbVideoWriter:
    def __init__(self, path: Path, *, fps: int) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.frame_count = 0
        self._writer: Any | None = None
        self.frames_dir = path.with_suffix("")
        try:
            import cv2

            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.fps),
                (640, 480),
            )
            if writer.isOpened():
                self._writer = writer
            else:
                writer.release()
        except ModuleNotFoundError:
            self._writer = None
        if self._writer is None:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

    def write(self, rgb: np.ndarray) -> None:
        value = np.asarray(rgb, dtype=np.uint8)
        if value.shape != (480, 640, 3):
            raise ValueError(
                f"Video frame must have shape (480, 640, 3), got {value.shape}"
            )
        if self._writer is not None:
            self._writer.write(value[:, :, ::-1])
        else:
            Image.fromarray(value).save(
                self.frames_dir / f"frame_{self.frame_count:06d}.png"
            )
        self.frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def metadata(self) -> dict[str, Any]:
        return {
            "path": (
                self.path.name
                if self.path.exists()
                else self.frames_dir.name
            ),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "codec": "mp4v" if self.path.exists() else "png_sequence",
        }


@dataclass(frozen=True)
class EpisodeRecorderConfig:
    output_dir: Path
    task: str
    prompt: str
    seed: int
    fps: int = 10
    record_video: bool = False

    @property
    def action_dt_s(self) -> float:
        return 1.0 / float(self.fps)


class EpisodeRecorder:
    """Record observations immediately before applying each oracle action."""

    def __init__(
        self,
        config: EpisodeRecorderConfig,
        environment: MuJoCoEnvironment,
    ) -> None:
        if config.fps <= 0:
            raise ValueError("Recorder FPS must be positive")
        if config.prompt != REAL_TRAINING_PROMPT:
            raise ValueError(
                "red_block prompt must match the real training prompt exactly: "
                f"{REAL_TRAINING_PROMPT!r}"
            )
        self.config = config
        self.environment = environment
        self.output_dir = Path(config.output_dir)
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty episode directory: "
                f"{self.output_dir}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for directory in (
            "base_images",
            "wrist_images",
            "policy_base_images",
            "policy_wrist_images",
        ):
            (self.output_dir / directory).mkdir(parents=True, exist_ok=True)

        self.records: list[dict[str, Any]] = []
        self._overview_video = (
            _RgbVideoWriter(self.output_dir / "overview.mp4", fps=config.fps)
            if config.record_video
            else None
        )
        self._combined_video = (
            _RgbVideoWriter(self.output_dir / "combined.mp4", fps=config.fps)
            if config.record_video
            else None
        )
        runtime = environment.task_runtime
        if runtime is None:
            raise RuntimeError("Reset the environment before starting a recorder")
        self._initial_object_pose = self._body_pose(runtime.target_body)
        self._initial_robot_state = policy_state_from_mujoco(
            environment.context.model,
            environment.context.data,
        )

    def _body_pose(self, body_name: str) -> dict[str, Any]:
        model = self.environment.context.model
        data = self.environment.context.data
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        if body_id < 0:
            raise RuntimeError(f"Body not found while recording: {body_name}")
        return {
            "body": body_name,
            "position_m": np.asarray(data.xpos[body_id]).copy(),
            "quaternion_wxyz": np.asarray(data.xquat[body_id]).copy(),
        }

    def _tcp_pose(self) -> dict[str, Any]:
        model = self.environment.context.model
        data = self.environment.context.data
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "tool_center_point",
        )
        if site_id < 0:
            raise RuntimeError("TCP site not found: tool_center_point")
        rotation = np.asarray(data.site_xmat[site_id]).reshape(3, 3)
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
        return {
            "site": "tool_center_point",
            "position_m": np.asarray(data.site_xpos[site_id]).copy(),
            "quaternion_wxyz": quaternion,
        }

    def record_pre_action(
        self,
        *,
        action: np.ndarray,
        oracle_stage: str,
    ) -> None:
        """Record one aligned pair `(observation_t, action_t)`."""

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError(
                f"Recorded action must be finite with shape (7,), got "
                f"{action.shape}"
            )
        model = self.environment.context.model
        data = self.environment.context.data
        runtime = self.environment.task_runtime
        if runtime is None:
            raise RuntimeError("Task runtime is missing")

        frame_index = len(self.records)
        base_native = render_native_rgb(
            self.environment.context.renderer,
            data,
            BASE_CAMERA,
        )
        wrist_native = render_native_rgb(
            self.environment.context.renderer,
            data,
            WRIST_CAMERA,
        )
        base_policy = policy_image(
            base_native,
            self.environment.context.config,
        )
        wrist_policy = policy_image(
            wrist_native,
            self.environment.context.config,
        )
        state = policy_state_from_mujoco(model, data)
        timestamp = frame_index * self.config.action_dt_s

        image_rel = Path("base_images") / f"frame_{frame_index:06d}.png"
        wrist_rel = Path("wrist_images") / f"frame_{frame_index:06d}.png"
        policy_image_rel = (
            Path("policy_base_images") / f"frame_{frame_index:06d}.png"
        )
        policy_wrist_rel = (
            Path("policy_wrist_images") / f"frame_{frame_index:06d}.png"
        )
        Image.fromarray(base_native).save(self.output_dir / image_rel)
        Image.fromarray(wrist_native).save(self.output_dir / wrist_rel)
        Image.fromarray(base_policy).save(self.output_dir / policy_image_rel)
        Image.fromarray(wrist_policy).save(self.output_dir / policy_wrist_rel)

        target_pose = self._body_pose(runtime.target_body)
        tcp_pose = self._tcp_pose()
        contacts = collision_diagnostics(model, data)
        record = {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "simulation_time": float(data.time),
            "image": image_rel.as_posix(),
            "wrist_image": wrist_rel.as_posix(),
            "policy_image": policy_image_rel.as_posix(),
            "policy_wrist_image": policy_wrist_rel.as_posix(),
            "state": state.copy(),
            "actions": action.copy(),
            "task": self.config.prompt,
            "oracle_stage": str(oracle_stage),
            "object_position": np.asarray(target_pose["position_m"]).copy(),
            "object_quaternion": np.asarray(
                target_pose["quaternion_wxyz"]
            ).copy(),
            "tcp_position": np.asarray(tcp_pose["position_m"]).copy(),
            "tcp_quaternion": np.asarray(
                tcp_pose["quaternion_wxyz"]
            ).copy(),
            "mujoco_qpos": np.asarray(data.qpos).copy(),
            "mujoco_ctrl": np.asarray(data.ctrl).copy(),
            "contact_summary": contacts,
            "seed": self.config.seed,
        }
        self.records.append(record)

        if self._overview_video is not None:
            overview = render_native_rgb(
                self.environment.context.renderer,
                data,
                "overview_camera",
            )
            self._overview_video.write(overview)
            self._combined_video.write(
                tile_recording_frame(
                    overview,
                    base_native,
                    wrist_native,
                )
            )

    def _write_npz(self) -> None:
        records = self.records
        np.savez_compressed(
            self.output_dir / "observations.npz",
            image=np.asarray([record["image"] for record in records]),
            wrist_image=np.asarray(
                [record["wrist_image"] for record in records]
            ),
            policy_image=np.asarray(
                [record["policy_image"] for record in records]
            ),
            policy_wrist_image=np.asarray(
                [record["policy_wrist_image"] for record in records]
            ),
            state=np.asarray(
                [record["state"] for record in records],
                dtype=np.float32,
            ),
            actions=np.asarray(
                [record["actions"] for record in records],
                dtype=np.float32,
            ),
            task=np.asarray([record["task"] for record in records]),
            timestamp=np.asarray(
                [record["timestamp"] for record in records],
                dtype=np.float64,
            ),
            simulation_time=np.asarray(
                [record["simulation_time"] for record in records],
                dtype=np.float64,
            ),
            oracle_stage=np.asarray(
                [record["oracle_stage"] for record in records]
            ),
            object_position=np.asarray(
                [record["object_position"] for record in records],
                dtype=np.float64,
            ),
            object_quaternion=np.asarray(
                [record["object_quaternion"] for record in records],
                dtype=np.float64,
            ),
            tcp_position=np.asarray(
                [record["tcp_position"] for record in records],
                dtype=np.float64,
            ),
            tcp_quaternion=np.asarray(
                [record["tcp_quaternion"] for record in records],
                dtype=np.float64,
            ),
            mujoco_qpos=np.asarray(
                [record["mujoco_qpos"] for record in records],
                dtype=np.float64,
            ),
            mujoco_ctrl=np.asarray(
                [record["mujoco_ctrl"] for record in records],
                dtype=np.float64,
            ),
            seed=np.asarray(
                [record["seed"] for record in records],
                dtype=np.int64,
            ),
        )

    def _write_trajectory_csv(self) -> None:
        fields = [
            "frame_index",
            "timestamp",
            "simulation_time",
            "oracle_stage",
            *[f"state_{index}" for index in range(7)],
            *[f"action_{index}" for index in range(7)],
            "object_x",
            "object_y",
            "object_z",
            "tcp_x",
            "tcp_y",
            "tcp_z",
            "mujoco_qpos_json",
            "mujoco_ctrl_json",
            "contact_summary_json",
            "seed",
        ]
        with (self.output_dir / "trajectory.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in self.records:
                row: dict[str, Any] = {
                    "frame_index": record["frame_index"],
                    "timestamp": f"{record['timestamp']:.9f}",
                    "simulation_time": f"{record['simulation_time']:.9f}",
                    "oracle_stage": record["oracle_stage"],
                    "object_x": record["object_position"][0],
                    "object_y": record["object_position"][1],
                    "object_z": record["object_position"][2],
                    "tcp_x": record["tcp_position"][0],
                    "tcp_y": record["tcp_position"][1],
                    "tcp_z": record["tcp_position"][2],
                    "mujoco_qpos_json": json.dumps(
                        record["mujoco_qpos"].tolist(),
                        separators=(",", ":"),
                    ),
                    "mujoco_ctrl_json": json.dumps(
                        record["mujoco_ctrl"].tolist(),
                        separators=(",", ":"),
                    ),
                    "contact_summary_json": json.dumps(
                        record["contact_summary"],
                        separators=(",", ":"),
                        default=_json_default,
                    ),
                    "seed": record["seed"],
                }
                row.update(
                    {
                        f"state_{index}": float(value)
                        for index, value in enumerate(record["state"])
                    }
                )
                row.update(
                    {
                        f"action_{index}": float(value)
                        for index, value in enumerate(record["actions"])
                    }
                )
                writer.writerow(row)

    def finalize(
        self,
        *,
        success: bool,
        failure_reason: str | None,
        task_metrics: dict[str, Any],
        initial_conditions: dict[str, Any],
        randomization: dict[str, float],
        transitions: list[dict[str, Any]],
        oracle_plan: dict[str, Any],
    ) -> dict[str, Any]:
        if self._overview_video is not None:
            self._overview_video.close()
            self._combined_video.close()
        self._write_npz()
        self._write_trajectory_csv()

        runtime = self.environment.task_runtime
        if runtime is None:
            raise RuntimeError("Task runtime is missing")
        model = self.environment.context.model
        final_object_pose = self._body_pose(runtime.target_body)
        final_state = policy_state_from_mujoco(
            model,
            self.environment.context.data,
        )
        project_root = Path(__file__).resolve().parents[2]
        metadata = {
            "schema_version": RAW_SCHEMA_VERSION,
            "oracle_version": ORACLE_VERSION,
            "task": self.config.task,
            "prompt": self.config.prompt,
            "seed": self.config.seed,
            "success": bool(success),
            "failure_reason": failure_reason,
            "number_of_samples": len(self.records),
            "fps": self.config.fps,
            "action_dt_s": self.config.action_dt_s,
            "physics_timestep_s": float(model.opt.timestep),
            "physics_steps_per_action": int(
                round(self.config.action_dt_s / float(model.opt.timestep))
            ),
            "temporal_alignment": (
                "observation_t is recorded immediately before action_t; "
                "action_t is the absolute target for the next 0.1 s interval"
            ),
            "training_fields": {
                "image": {
                    "path_key": "image",
                    "shape": [480, 640, 3],
                    "dtype": "uint8",
                    "color_order": "RGB",
                },
                "wrist_image": {
                    "path_key": "wrist_image",
                    "shape": [480, 640, 3],
                    "dtype": "uint8",
                    "color_order": "RGB",
                },
                "state": {
                    "shape": [7],
                    "dtype": "float32",
                    "order": [
                        "j1_rad",
                        "j2_rad",
                        "j3_rad",
                        "j4_rad",
                        "j5_rad",
                        "j6_rad",
                        "gripper_raw",
                    ],
                },
                "actions": {
                    "shape": [7],
                    "dtype": "float32",
                    "semantics": "absolute next-interval target",
                },
                "task": self.config.prompt,
            },
            "policy_input_validation_fields": {
                "policy_image": [224, 224, 3],
                "policy_wrist_image": [224, 224, 3],
            },
            "debug_fields": [
                "simulation_time",
                "oracle_stage",
                "object_position",
                "object_quaternion",
                "tcp_position",
                "tcp_quaternion",
                "mujoco_qpos",
                "mujoco_ctrl",
                "contact_summary",
                "seed",
            ],
            "randomization": randomization,
            "initial_conditions": initial_conditions,
            "object_initial_pose": self._initial_object_pose,
            "object_final_pose": final_object_pose,
            "initial_robot_state": self._initial_robot_state,
            "final_robot_state": final_state,
            "task_metrics": task_metrics,
            "oracle_transitions": transitions,
            "oracle_plan": oracle_plan,
            "git_commit": _git_commit(project_root),
            "active_xml_path": str(DEFAULT_MODEL_PATH.resolve()),
            "camera_calibration_path": str(
                DEFAULT_CAMERA_CONFIG_PATH.resolve()
            ),
            "camera_calibration_sha256": _sha256(
                DEFAULT_CAMERA_CONFIG_PATH
            ),
        }
        if self._overview_video is not None:
            metadata["videos"] = {
                "overview": self._overview_video.metadata(),
                "combined": self._combined_video.metadata(),
            }
        _write_json(self.output_dir / "metadata.json", metadata)
        return metadata
