"""Simulation-evaluation video capture and operator replay."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from policy_runtime.recording import pad_to_aspect
from policy_runtime.recording import tile_recording_frame
from simulation.observation.cameras import render_rgb
from simulation.observation.policy import policy_image
from simulation.robot.model import BASE_CAMERA_NAME
from simulation.robot.model import WRIST_CAMERA_NAME


OVERVIEW_CAMERA = "overview_camera"


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


def _bgr(image: np.ndarray) -> np.ndarray:
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
            writer.write(_bgr(rgb))
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
        overview = render_rgb(context.renderer, context.data, OVERVIEW_CAMERA)
        base_native = render_rgb(context.renderer, context.data, BASE_CAMERA_NAME)
        wrist_native = render_rgb(context.renderer, context.data, WRIST_CAMERA_NAME)
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
    if not path.exists():
        raise FileNotFoundError(path)
    if os.name == "nt":
        subprocess.Popen(["cmd", "/c", "start", "", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
