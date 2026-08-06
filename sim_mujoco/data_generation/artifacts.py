"""Videos and inspectable key-frame contact sheets for accepted episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import numpy as np


CAMERA_DIRS = ("realsense_0", "realsense_1", "realsense_2")


def _pngs(episode_dir: Path, camera: str) -> list[Path]:
    return sorted((episode_dir / camera).glob("*.png"))


def _write_video(paths: list[Path], output: Path, *, fps: int = 10) -> str | None:
    if not paths:
        return None
    try:
        import cv2
    except ModuleNotFoundError:
        return None
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (640, 480)
    )
    if not writer.isOpened():
        writer.release()
        return None
    try:
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None or frame.shape != (480, 640, 3):
                raise ValueError(f"Cannot decode video frame: {path}")
            writer.write(frame)
    finally:
        writer.release()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Video writer produced no output: {output}")
    return str(output)


def write_episode_visuals(episode_dir: Path) -> dict[str, Any]:
    episode_dir = Path(episode_dir)
    meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
    transitions = meta["simulation"].get("oracle_transitions") or []
    row_count = int(meta["simulation"]["robot_log_rows"])
    named_indices: dict[str, int] = {
        "first_recorded_frame": 0,
        "grasp_or_lift_frame": 0,
        "verification_end_frame": row_count - 1,
        "final_frame": row_count - 1,
    }
    for transition in transitions:
        stage = str(transition.get("to_stage", "")).lower()
        index = min(max(int(transition.get("action_step", 0)), 0), row_count - 1)
        if stage == "lift":
            named_indices["grasp_or_lift_frame"] = index
        elif stage == "verify":
            named_indices["verification_start_frame"] = index
        elif stage == "release":
            named_indices["release_frame"] = index
        elif stage == "complete":
            named_indices["verification_end_frame"] = index
    ordered = list(named_indices.items())
    overview = _pngs(episode_dir, "realsense_2")
    cards: list[Image.Image] = []
    key_frames: dict[str, str] = {}
    key_dir = episode_dir / "key_frames"
    key_dir.mkdir(exist_ok=True)
    for label, index in ordered:
        if not overview:
            break
        source = overview[min(index, len(overview) - 1)]
        image = Image.open(source).convert("RGB")
        destination = key_dir / f"{label}.png"
        image.save(destination)
        key_frames[label] = str(destination)
        card = image.resize((320, 240))
        canvas = Image.new("RGB", (320, 266), "white")
        canvas.paste(card, (0, 26))
        ImageDraw.Draw(canvas).text((8, 6), label, fill="black")
        cards.append(canvas)
    if cards:
        sheet = Image.new("RGB", (320 * len(cards), 266), "white")
        for index, card in enumerate(cards):
            sheet.paste(card, (320 * index, 0))
        sheet_path = episode_dir / "contact_sheet.png"
        sheet.save(sheet_path)
    else:
        sheet_path = None
    videos = {}
    for camera in CAMERA_DIRS:
        videos[camera] = _write_video(
            _pngs(episode_dir, camera), episode_dir / f"{camera}.mp4"
        )
    result = {
        "episode_dir": str(episode_dir),
        "contact_sheet": str(sheet_path) if sheet_path else None,
        "key_frames": key_frames,
        "videos": videos,
    }
    (episode_dir / "visual_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def write_diagnostic_visuals(
    output_dir: Path,
    frames: dict[str, list[np.ndarray]],
) -> dict[str, Any]:
    """Persist excluded pre-recording frames only for a sampled failed attempt."""

    output_dir = Path(output_dir)
    diagnostic_root = output_dir / "diagnostic_frames"
    paths_by_camera: dict[str, list[Path]] = {}
    for camera in CAMERA_DIRS:
        camera_dir = diagnostic_root / camera
        camera_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for index, frame in enumerate(frames.get(camera) or []):
            value = np.asarray(frame)
            if value.shape != (480, 640, 3) or value.dtype != np.uint8:
                raise ValueError(f"Invalid diagnostic frame for {camera}: {value.shape}")
            path = camera_dir / f"frame_{index:03d}.png"
            Image.fromarray(value).save(path)
            paths.append(path)
        paths_by_camera[camera] = paths
    overview = paths_by_camera["realsense_2"]
    selected = [overview[index] for index in sorted({0, len(overview) // 2, len(overview) - 1})] if overview else []
    if selected:
        sheet = Image.new("RGB", (320 * len(selected), 240), "white")
        for index, path in enumerate(selected):
            sheet.paste(Image.open(path).convert("RGB").resize((320, 240)), (320 * index, 0))
        sheet_path = output_dir / "diagnostic_contact_sheet.png"
        sheet.save(sheet_path)
    else:
        sheet_path = None
    videos = {
        camera: _write_video(paths, output_dir / f"diagnostic_{camera}.mp4")
        for camera, paths in paths_by_camera.items()
    }
    result = {
        "diagnostic_contact_sheet": str(sheet_path) if sheet_path else None,
        "diagnostic_videos": videos,
        "diagnostic_frame_counts": {
            camera: len(paths) for camera, paths in paths_by_camera.items()
        },
        "excluded_from_training": True,
    }
    (output_dir / "diagnostic_visual_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
