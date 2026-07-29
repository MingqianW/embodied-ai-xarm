from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from policy_runtime.episode_logging import (
    json_default as _shared_json_default,
    write_json as _shared_write_json,
)
from policy_runtime.evaluation import (
    LABELS,
    summarize_episode_rows as _shared_summarize_episode_rows,
    validate_label as _shared_validate_label,
)
from policy_runtime.recording import (
    pad_to_aspect as _shared_pad_to_aspect,
    tile_recording_frame,
)
from sim_mujoco.remote_policy_observation import (
    ARM_JOINT_NAMES,
    BASE_CAMERA,
    WRIST_CAMERA,
    arm_joint_limits,
    joint_qpos,
    render_native_rgb,
)


OVERVIEW_CAMERA = "overview_camera"


def json_default(value: Any) -> Any:
    try:
        return _shared_json_default(value)
    except TypeError:
        return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    _shared_write_json(path, payload)


def validate_label(label: str) -> str:
    return _shared_validate_label(label)


def summarize_episode_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _shared_summarize_episode_rows(rows)


def write_episodes_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode_index",
        "seed",
        "task",
        "prompt",
        "label",
        "valid",
        "automatic_task_success",
        "comment",
        "termination_reason",
        "policy_steps",
        "sim_time",
        "wall_time",
        "initial_object_x",
        "initial_object_y",
        "initial_object_yaw",
        "video_frames",
        "video_fps",
        "combined_video_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_episodes_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def write_summary(run_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_episode_rows(rows)
    tasks = sorted({str(row.get("task") or "") for row in rows if row.get("task")})
    summary["task_breakdown"] = {
        task: summarize_episode_rows(
            [row for row in rows if str(row.get("task") or "") == task]
        )
        for task in tasks
    }
    write_json(run_dir / "summary.json", summary)
    rate = summary["human_rated_task_success_rate"]
    e2e = summary["end_to_end_success_rate"]
    lines = [
        f"attempted episodes: {summary['attempted_episodes']}",
        f"labeled episodes: {summary['labeled_episodes']}",
        f"successes: {summary['successes']}",
        f"failures: {summary['failures']}",
        f"invalid episodes: {summary['invalid_episodes']}",
        f"human-rated task success rate: {'n/a' if rate is None else f'{rate:.3f}'}",
        f"end-to-end success rate: {'n/a' if e2e is None else f'{e2e:.3f}'}",
        f"label counts: {summary['label_counts']}",
        f"termination reason counts: {summary['termination_reason_counts']}",
        f"mean policy steps: {summary['mean_policy_steps']}",
        f"mean simulation time: {summary['mean_simulation_time']}",
        f"mean wall time: {summary['mean_wall_time']}",
        f"task breakdown: {summary['task_breakdown']}",
    ]
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_episodes_csv(run_dir / "episodes.csv", rows)
    return summary


def quaternion_from_yaw(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float64)


def yaw_from_quaternion(quat: np.ndarray) -> float:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def object_qpos_address(model: mujoco.MjModel) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
    if joint_id < 0:
        raise RuntimeError("Object freejoint not found: object_freejoint")
    return int(model.jnt_qposadr[joint_id])


def apply_initial_randomization(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    seed: int,
    object_xy_range: float,
    object_yaw_range_deg: float,
    joint_noise: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    object_addr = object_qpos_address(model)
    nominal_object_xy = np.asarray(data.qpos[object_addr : object_addr + 2], dtype=np.float64).copy()
    nominal_object_z = float(data.qpos[object_addr + 2])
    xy_delta = rng.uniform(-float(object_xy_range), float(object_xy_range), size=2)
    yaw = math.radians(float(rng.uniform(-float(object_yaw_range_deg), float(object_yaw_range_deg))))

    data.qpos[object_addr : object_addr + 2] = nominal_object_xy + xy_delta
    data.qpos[object_addr + 2] = nominal_object_z
    data.qpos[object_addr + 3 : object_addr + 7] = quaternion_from_yaw(yaw)

    limits = arm_joint_limits(model)
    joint_values = []
    for index, joint_name in enumerate(ARM_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_addr = int(model.jnt_qposadr[joint_id])
        noisy = float(data.qpos[qpos_addr] + rng.normal(0.0, float(joint_noise)))
        clamped = float(np.clip(noisy, limits[index, 0], limits[index, 1]))
        data.qpos[qpos_addr] = clamped
        if index < model.nu:
            data.ctrl[index] = clamped
        joint_values.append(clamped)

    mujoco.mj_forward(model, data)
    return {
        "seed": int(seed),
        "initial_object_x": float(data.qpos[object_addr]),
        "initial_object_y": float(data.qpos[object_addr + 1]),
        "initial_object_z": float(data.qpos[object_addr + 2]),
        "initial_object_yaw": yaw,
        "initial_joint_positions": joint_values,
        "object_xy_delta": xy_delta.tolist(),
    }


def pad_to_aspect(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return _shared_pad_to_aspect(image, width, height)


def tile_video_frame(
    overview: np.ndarray,
    base: np.ndarray,
    wrist: np.ndarray,
    *,
    tile_width: int = 320,
    tile_height: int = 240,
) -> np.ndarray:
    return tile_recording_frame(
        overview,
        base,
        wrist,
        width=tile_width * 2,
        height=tile_height * 2,
    )


def require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for MP4 video recording; install opencv-python") from exc
    return cv2


def bgr(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.uint8)[:, :, ::-1]


@dataclass
class VideoRecorder:
    output_dir: Path
    fps: int = 30
    enabled: bool = True

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            import cv2

            self.cv2 = cv2
        except ModuleNotFoundError:
            self.cv2 = None
        self.paths = {
            "overview": self.output_dir / "overview.mp4",
            "base_camera": self.output_dir / "base_camera.mp4",
            "wrist_camera": self.output_dir / "wrist_camera.mp4",
            "combined": self.output_dir / "combined.mp4",
        }
        self.writers: dict[str, Any] = {}
        self.codecs: dict[str, str] = {}
        self.frame_count = 0
        self.next_frame_time: float | None = None
        self._open_writer("overview", (640, 480))
        self._open_writer("base_camera", (224, 224))
        self._open_writer("wrist_camera", (224, 224))
        self._open_writer("combined", (640, 480))

    def _open_writer(self, key: str, size: tuple[int, int]) -> None:
        writer = None
        if self.cv2 is not None:
            fourcc = self.cv2.VideoWriter_fourcc(*"mp4v")
            candidate = self.cv2.VideoWriter(
                str(self.paths[key]), fourcc, float(self.fps), size
            )
            if candidate.isOpened():
                writer = candidate
                self.codecs[key] = "mp4v"
            else:
                candidate.release()
                avi_path = self.paths[key].with_suffix(".avi")
                fourcc = self.cv2.VideoWriter_fourcc(*"MJPG")
                candidate = self.cv2.VideoWriter(
                    str(avi_path), fourcc, float(self.fps), size
                )
                if candidate.isOpened():
                    writer = candidate
                    self.paths[key] = avi_path
                    self.codecs[key] = "MJPG"
                else:
                    candidate.release()
        if writer is None:
            frames_dir = self.paths[key].with_suffix("")
            frames_dir.mkdir(parents=True, exist_ok=True)
            self.paths[key] = frames_dir
            self.codecs[key] = "png_sequence"
        self.writers[key] = writer

    def _write(self, key: str, rgb: np.ndarray) -> None:
        writer = self.writers[key]
        if writer is not None:
            writer.write(bgr(rgb))
            return
        from PIL import Image

        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(
            self.paths[key] / f"frame_{self.frame_count:06d}.png"
        )

    def maybe_record(self, context: Any) -> None:
        sim_time = float(context.data.time)
        if self.next_frame_time is None:
            self.next_frame_time = sim_time
        if sim_time + 1e-12 < self.next_frame_time:
            return
        self.record(context)
        self.next_frame_time += 1.0 / float(self.fps)

    def record(self, context: Any) -> None:
        overview = render_native_rgb(context.renderer, context.data, OVERVIEW_CAMERA)
        base_native = render_native_rgb(context.renderer, context.data, BASE_CAMERA)
        wrist_native = render_native_rgb(context.renderer, context.data, WRIST_CAMERA)
        from sim_mujoco.remote_policy_observation import policy_image

        base = policy_image(base_native, context.config)
        wrist = policy_image(wrist_native, context.config)
        combined = tile_video_frame(overview, base, wrist)
        self._write("overview", pad_to_aspect(overview, 640, 480))
        self._write("base_camera", base)
        self._write("wrist_camera", wrist)
        self._write("combined", combined)
        self.frame_count += 1

    def close(self) -> None:
        for writer in self.writers.values():
            if writer is not None:
                writer.release()
        self.writers.clear()

    def validate_outputs(self) -> None:
        for key, path in self.paths.items():
            valid = (
                path.is_dir()
                and any(path.glob("frame_*.png"))
                or path.is_file()
                and path.stat().st_size > 0
            )
            if not valid:
                raise RuntimeError(
                    f"Recording was not written or is empty for {key}: {path}. "
                    "Install a working FFmpeg/OpenCV codec or inspect PNG fallback errors."
                )

    def metadata(self) -> dict[str, Any]:
        unique_codecs = sorted(set(self.codecs.values()))
        return {
            "video_frames": int(self.frame_count),
            "video_fps": int(self.fps),
            "video_codec": unique_codecs[0] if len(unique_codecs) == 1 else "mixed",
            "video_codecs": dict(self.codecs),
            "video_paths": {key: str(path) for key, path in self.paths.items()},
            "combined_video_path": str(self.paths["combined"]),
        }


def replay_video(path: Path) -> None:
    import subprocess
    import os

    if not path.exists():
        raise FileNotFoundError(path)
    if os.name == "nt":
        subprocess.Popen(["cmd", "/c", "start", "", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
