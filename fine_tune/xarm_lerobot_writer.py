"""Shared canonical LeRobot writer for real and MuJoCo xArm episodes."""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path
from typing import Any

import numpy as np


XARM_STATE_COLUMNS = (
    "j1_rad",
    "j2_rad",
    "j3_rad",
    "j4_rad",
    "j5_rad",
    "j6_rad",
    "gripper_mm",
)
XARM_IMAGE_SHAPE = (480, 640, 3)


def load_rgb(value: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(value, np.ndarray):
        image = np.asarray(value)
        if image.shape != XARM_IMAGE_SHAPE:
            raise ValueError(
                f"xArm image must have shape {XARM_IMAGE_SHAPE}, got "
                f"{image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"xArm image must have dtype uint8, got {image.dtype}"
            )
        return np.ascontiguousarray(image)

    path = Path(value)
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except ModuleNotFoundError:
        from PIL import Image

        image = np.asarray(Image.open(path).convert("RGB"))
    if image.shape != XARM_IMAGE_SHAPE:
        raise ValueError(
            f"xArm image {path} must have shape {XARM_IMAGE_SHAPE}, got "
            f"{image.shape}"
        )
    return np.ascontiguousarray(image, dtype=np.uint8)


def _lerobot_imports():
    try:
        from lerobot.datasets.lerobot_dataset import (  # type: ignore
            HF_LEROBOT_HOME,
            LeRobotDataset,
        )
    except ModuleNotFoundError:
        from lerobot.common.constants import HF_LEROBOT_HOME  # type: ignore
        from lerobot.common.datasets.lerobot_dataset import (  # type: ignore
            LeRobotDataset,
        )
    return Path(HF_LEROBOT_HOME), LeRobotDataset


def default_lerobot_output_path(repo_id: str) -> Path:
    hf_lerobot_home, _ = _lerobot_imports()
    return hf_lerobot_home / repo_id


def _accepted_kwargs(callable_value, values: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_value)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return values
    return {
        key: value
        for key, value in values.items()
        if key in signature.parameters
    }


def _start_image_writer(
    dataset: Any,
    *,
    image_writer_threads: int,
    image_writer_processes: int,
) -> None:
    if not hasattr(dataset, "start_image_writer"):
        return
    values = {
        "num_threads": image_writer_threads,
        "num_processes": image_writer_processes,
        "image_writer_threads": image_writer_threads,
        "image_writer_processes": image_writer_processes,
    }
    dataset.start_image_writer(
        **_accepted_kwargs(dataset.start_image_writer, values)
    )


def _open_dataset(
    *,
    repo_id: str,
    output_path: Path,
    robot_type: str,
    fps: int,
    overwrite: bool,
    resume: bool,
    image_writer_threads: int,
    image_writer_processes: int,
):
    _, LeRobotDataset = _lerobot_imports()
    needs_image_writer_start = False
    if output_path.exists() and overwrite:
        if any(output_path.iterdir()) and not (
            output_path / "meta" / "info.json"
        ).is_file():
            raise ValueError(
                "Refusing --overwrite because the target is a non-empty "
                f"directory without a LeRobot meta/info.json marker: {output_path}"
            )
        shutil.rmtree(output_path)
    if output_path.exists():
        if not resume:
            raise FileExistsError(
                f"LeRobot output already exists; pass resume or overwrite: "
                f"{output_path}"
            )
        dataset = LeRobotDataset(
            **_accepted_kwargs(
                LeRobotDataset,
                {
                    "repo_id": repo_id,
                    "root": output_path,
                },
            )
        )
        needs_image_writer_start = True
    else:
        create_values = {
            "repo_id": repo_id,
            "root": output_path,
            "robot_type": robot_type,
            "fps": int(fps),
            "features": {
                "image": {
                    "dtype": "image",
                    "shape": XARM_IMAGE_SHAPE,
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": XARM_IMAGE_SHAPE,
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": (len(XARM_STATE_COLUMNS),),
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (len(XARM_STATE_COLUMNS),),
                    "names": ["actions"],
                },
            },
            "image_writer_threads": image_writer_threads,
            "image_writer_processes": image_writer_processes,
        }
        accepted_create_values = _accepted_kwargs(
            LeRobotDataset.create,
            create_values,
        )
        dataset = LeRobotDataset.create(**accepted_create_values)
        needs_image_writer_start = not any(
            key in accepted_create_values
            for key in (
                "image_writer_threads",
                "image_writer_processes",
            )
        )
    if needs_image_writer_start:
        _start_image_writer(
            dataset,
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_processes,
        )
    return dataset


def _add_frame(dataset: Any, frame: dict[str, Any], task: str) -> None:
    try:
        dataset.add_frame(frame, task=task)
    except TypeError:
        dataset.add_frame({**frame, "task": task})


def write_xarm_lerobot_dataset(
    records_by_episode: list[list[dict[str, Any]]],
    *,
    repo_id: str,
    output_path: Path | None,
    robot_type: str,
    fps: int,
    overwrite: bool,
    resume: bool,
    image_writer_threads: int = 4,
    image_writer_processes: int = 0,
    push_to_hub: bool = False,
    hub_private: bool = True,
) -> dict[str, Any]:
    """Write canonical episodes through LeRobotDataset, never a custom clone."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    if image_writer_threads < 0 or image_writer_processes < 0:
        raise ValueError("image writer worker counts cannot be negative")
    if output_path is None:
        output_path = default_lerobot_output_path(repo_id)
    output_path = Path(output_path)

    dataset = _open_dataset(
        repo_id=repo_id,
        output_path=output_path,
        robot_type=robot_type,
        fps=fps,
        overwrite=overwrite,
        resume=resume,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )
    starting_episodes = int(
        getattr(
            getattr(dataset, "meta", None),
            "total_episodes",
            getattr(dataset, "num_episodes", 0),
        )
    )
    written_frames = 0
    try:
        for episode_records in records_by_episode:
            if not episode_records:
                raise ValueError("Cannot write an empty LeRobot episode")
            for record in episode_records:
                state = np.asarray(record["state"], dtype=np.float32)
                actions = np.asarray(record["actions"], dtype=np.float32)
                if state.shape != (7,) or actions.shape != (7,):
                    raise ValueError(
                        "xArm state/actions must both have shape (7,)"
                    )
                if not np.isfinite(state).all() or not np.isfinite(
                    actions
                ).all():
                    raise ValueError("xArm state/actions contain NaN or Inf")
                task = str(record["task"])
                frame = {
                    "image": load_rgb(record["image"]),
                    "wrist_image": load_rgb(record["wrist_image"]),
                    "state": state,
                    "actions": actions,
                }
                _add_frame(dataset, frame, task)
                written_frames += 1
            dataset.save_episode()
        if hasattr(dataset, "finalize"):
            dataset.finalize()
    finally:
        if hasattr(dataset, "stop_image_writer"):
            dataset.stop_image_writer()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["xarm", "xarm6", "openpi"],
            private=hub_private,
            push_videos=True,
            license="apache-2.0",
        )

    ending_episodes = int(
        getattr(
            getattr(dataset, "meta", None),
            "total_episodes",
            getattr(dataset, "num_episodes", starting_episodes),
        )
    )
    return {
        "output_path": str(output_path),
        "repo_id": repo_id,
        "fps": int(fps),
        "starting_episodes": starting_episodes,
        "written_episodes": len(records_by_episode),
        "ending_episodes": ending_episodes,
        "written_frames": written_frames,
    }
