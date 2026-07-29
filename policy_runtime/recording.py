from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def pad_to_aspect(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("Target width and height must be positive")
    value = np.asarray(image, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"Expected RGB HxWx3 frame, got {value.shape}")
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for recording") from exc
    source_h, source_w = value.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(value, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    output = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - resized_w) // 2
    y0 = (height - resized_h) // 2
    output[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return output


def tile_recording_frame(
    viewer: np.ndarray,
    base: np.ndarray,
    wrist: np.ndarray,
    *,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    half_width = width // 2
    half_height = height // 2
    top = np.hstack(
        [
            pad_to_aspect(viewer, half_width, half_height),
            pad_to_aspect(base, width - half_width, half_height),
        ]
    )
    bottom = pad_to_aspect(wrist, width, height - half_height)
    return np.vstack([top, bottom]).astype(np.uint8)


@dataclass
class VideoRecorder:
    output_dir: Path
    fps: int = 30
    max_frames: int = 18_000
    fallback_to_frames: bool = True

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0
        self.codec: str | None = None
        self.video_path = self.output_dir / "evaluation.mp4"
        self.frames_dir = self.output_dir / "frames"
        self._writer: Any | None = None
        try:
            import cv2

            writer = cv2.VideoWriter(
                str(self.video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.fps),
                (640, 480),
            )
            if writer.isOpened():
                self._writer = writer
                self.codec = "mp4v"
            else:
                writer.release()
        except ModuleNotFoundError:
            pass
        if self._writer is None:
            if not self.fallback_to_frames:
                raise RuntimeError("No working video encoder and frame fallback is disabled")
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            self.codec = "png_sequence"

    def write(self, frames: dict[str, np.ndarray]) -> None:
        if self.frame_count >= self.max_frames:
            raise RuntimeError(f"Recording reached configured max_frames={self.max_frames}")
        base = frames.get("base")
        wrist = frames.get("wrist")
        viewer = frames.get("viewer", base)
        if base is None or wrist is None or viewer is None:
            raise ValueError("Recording frames must include base and wrist images")
        combined = tile_recording_frame(viewer, base, wrist)
        if self._writer is not None:
            self._writer.write(combined[:, :, ::-1])
        else:
            from PIL import Image

            Image.fromarray(combined).save(self.frames_dir / f"frame_{self.frame_count:06d}.png")
        self.frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def metadata(self) -> dict[str, Any]:
        path = self.video_path if self.codec != "png_sequence" else self.frames_dir
        return {
            "video_frames": self.frame_count,
            "video_fps": self.fps,
            "video_codec": self.codec,
            "video_path": str(path),
        }
