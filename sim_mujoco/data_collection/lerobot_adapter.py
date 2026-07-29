"""Validated adapter from raw MuJoCo oracle episodes to xArm LeRobot frames."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fine_tune.xarm_lerobot_writer import XARM_IMAGE_SHAPE
from sim_mujoco.data_collection.episode_recorder import (
    RAW_SCHEMA_VERSION,
    REAL_TRAINING_PROMPT,
)


CONVERSION_MANIFEST_VERSION = "xarm_mujoco_lerobot_conversion_v1"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RawOracleEpisode:
    episode_index: int
    relative_path: str
    directory: Path
    metadata: dict[str, Any]
    observations_sha256: str

    @property
    def source_id(self) -> str:
        return (
            f"{self.relative_path}:"
            f"{self.observations_sha256}"
        )


def _validate_image(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB":
            raise ValueError(f"Image must decode as RGB: {path} ({image.mode})")
        array = np.asarray(image)
    if array.shape != XARM_IMAGE_SHAPE or array.dtype != np.uint8:
        raise ValueError(
            f"Image must be uint8 with shape {XARM_IMAGE_SHAPE}: "
            f"{path} ({array.shape}, {array.dtype})"
        )


def discover_successful_episodes(input_dir: Path) -> list[RawOracleEpisode]:
    """Return only manifest-approved, successful, completed episodes."""

    input_dir = Path(input_dir).resolve()
    manifest_path = input_dir / "manifest.json"
    run_config_path = input_dir / "run_config.json"
    if not manifest_path.is_file() or not run_config_path.is_file():
        raise FileNotFoundError(
            f"Raw collection requires manifest.json and run_config.json: "
            f"{input_dir}"
        )
    manifest = read_json(manifest_path)
    run_config = read_json(run_config_path)
    if run_config.get("task") != "red_block":
        raise ValueError("Only the audited red_block collection is supported")
    if int(run_config.get("action_hz", -1)) != 10:
        raise ValueError("Raw collection must use the audited 10 Hz cadence")

    completed = manifest.get("completed_episodes")
    if not isinstance(completed, list):
        raise ValueError("manifest.completed_episodes must be a list")

    episodes: list[RawOracleEpisode] = []
    expected_index = 0
    seen_paths: set[str] = set()
    for entry in completed:
        if not isinstance(entry, dict):
            raise ValueError("Each completed episode manifest entry must be an object")
        episode_index = int(entry.get("episode_index", -1))
        if episode_index != expected_index:
            raise ValueError(
                "Completed episode indices must be contiguous from zero: "
                f"expected {expected_index}, got {episode_index}"
            )
        expected_index += 1
        if entry.get("success") is not True:
            raise ValueError(
                f"Completed episode {episode_index} is not marked successful"
            )
        relative_path = str(entry.get("path", ""))
        if not relative_path or relative_path in seen_paths:
            raise ValueError(
                f"Invalid or duplicate completed episode path: {relative_path!r}"
            )
        seen_paths.add(relative_path)
        directory = (input_dir / relative_path).resolve()
        try:
            directory.relative_to(input_dir)
        except ValueError as exc:
            raise ValueError(
                f"Completed episode escapes input directory: {relative_path}"
            ) from exc
        metadata_path = directory / "metadata.json"
        observations_path = directory / "observations.npz"
        if not metadata_path.is_file() or not observations_path.is_file():
            raise FileNotFoundError(
                f"Incomplete raw episode directory: {directory}"
            )
        metadata = read_json(metadata_path)
        if metadata.get("schema_version") != RAW_SCHEMA_VERSION:
            raise ValueError(
                f"Unexpected raw schema in {metadata_path}: "
                f"{metadata.get('schema_version')!r}"
            )
        if metadata.get("success") is not True:
            raise ValueError(
                f"Raw episode {episode_index} is not successful"
            )
        if metadata.get("failure_reason") is not None:
            raise ValueError(
                f"Successful raw episode has a failure reason: {metadata_path}"
            )
        if metadata.get("task") != "red_block":
            raise ValueError(f"Unexpected task in {metadata_path}")
        if metadata.get("prompt") != REAL_TRAINING_PROMPT:
            raise ValueError(f"Unexpected prompt in {metadata_path}")
        if int(metadata.get("fps", -1)) != 10:
            raise ValueError(f"Unexpected FPS in {metadata_path}")
        episodes.append(
            RawOracleEpisode(
                episode_index=episode_index,
                relative_path=relative_path,
                directory=directory,
                metadata=metadata,
                observations_sha256=sha256(observations_path),
            )
        )
    return episodes


def load_episode_records(
    episode: RawOracleEpisode,
    *,
    validate_images: bool = True,
) -> list[dict[str, Any]]:
    observations_path = episode.directory / "observations.npz"
    with np.load(observations_path, allow_pickle=False) as data:
        required = {
            "image",
            "wrist_image",
            "state",
            "actions",
            "task",
            "timestamp",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(
                f"Missing raw training arrays in {observations_path}: {missing}"
            )
        image_paths = np.asarray(data["image"])
        wrist_paths = np.asarray(data["wrist_image"])
        states = np.asarray(data["state"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        tasks = np.asarray(data["task"])
        timestamps = np.asarray(data["timestamp"], dtype=np.float64)

    sample_count = int(episode.metadata["number_of_samples"])
    if states.shape != (sample_count, 7):
        raise ValueError(
            f"State array must have shape ({sample_count}, 7): {states.shape}"
        )
    if actions.shape != (sample_count, 7):
        raise ValueError(
            f"Action array must have shape ({sample_count}, 7): {actions.shape}"
        )
    for name, values in (
        ("image", image_paths),
        ("wrist_image", wrist_paths),
        ("task", tasks),
        ("timestamp", timestamps),
    ):
        if values.shape != (sample_count,):
            raise ValueError(
                f"{name} array must have shape ({sample_count},): {values.shape}"
            )
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"NaN or Inf in {observations_path}")
    if not np.isfinite(timestamps).all():
        raise ValueError(f"NaN or Inf timestamps in {observations_path}")
    expected_timestamps = np.arange(sample_count, dtype=np.float64) / 10.0
    if not np.allclose(timestamps, expected_timestamps, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"Episode timestamps do not equal frame_index / 10: "
            f"{observations_path}"
        )
    if any(str(task) != REAL_TRAINING_PROMPT for task in tasks):
        raise ValueError(f"Prompt mismatch in {observations_path}")

    records: list[dict[str, Any]] = []
    for frame_index in range(sample_count):
        image_path = (episode.directory / str(image_paths[frame_index])).resolve()
        wrist_path = (
            episode.directory / str(wrist_paths[frame_index])
        ).resolve()
        for path in (image_path, wrist_path):
            try:
                path.relative_to(episode.directory.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"Raw image path escapes episode directory: {path}"
                ) from exc
            if validate_images:
                _validate_image(path)
            elif not path.is_file():
                raise FileNotFoundError(path)
        records.append(
            {
                "image": image_path,
                "wrist_image": wrist_path,
                "state": states[frame_index],
                "actions": actions[frame_index],
                "task": REAL_TRAINING_PROMPT,
                "source_episode_index": episode.episode_index,
                "source_frame_index": frame_index,
            }
        )
    return records


def validate_temporal_alignment(
    records: list[dict[str, Any]],
    *,
    arm_tolerance_rad: float = 0.08,
    gripper_tolerance_raw: float = 100.0,
) -> dict[str, Any]:
    """Check that action_t points toward the observed state at t+1.

    MuJoCo position servos need not exactly reach each target in one 100 ms
    interval, so alignment is measured as movement toward the target rather
    than equality with the next state.
    """

    if len(records) < 2:
        raise ValueError("Temporal alignment validation needs at least two frames")
    states = np.asarray([record["state"] for record in records], dtype=np.float64)
    actions = np.asarray(
        [record["actions"] for record in records],
        dtype=np.float64,
    )
    current_error = np.abs(actions[:-1] - states[:-1])
    next_error = np.abs(actions[:-1] - states[1:])
    moved_toward = next_error <= current_error + np.asarray(
        [arm_tolerance_rad] * 6 + [gripper_tolerance_raw],
        dtype=np.float64,
    )
    per_dimension_fraction = moved_toward.mean(axis=0)
    overall_fraction = float(moved_toward.mean())
    if overall_fraction < 0.95 or float(per_dimension_fraction.min()) < 0.90:
        raise ValueError(
            "Raw action/state alignment failed: "
            f"overall={overall_fraction:.4f}, "
            f"per_dimension={per_dimension_fraction.tolist()}"
        )
    return {
        "checked_transitions": len(records) - 1,
        "overall_moved_toward_fraction": overall_fraction,
        "per_dimension_moved_toward_fraction": per_dimension_fraction.tolist(),
    }
