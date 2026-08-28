"""Canonical raw CSV/PNG to LeRobot v2.1 conversion for all six tasks."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from data.common.lerobot_writer import write_xarm_lerobot_dataset
from data.common.records import EpisodeRecord, FrameRecord, SourceBackend
from data.common.schema import XARM_STATE_COLUMNS
from data.common.task_identity import canonical_prompt
from data.sim.generation.config import PipelineConfig, repository_root
from data.sim.generation.manifest import atomic_write_json
from data.sim.generation.safety import replace_authorized_roots
from data.sim.generation.status import git_sha


STATE_COLUMNS = XARM_STATE_COLUMNS


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def canonical_episode_from_raw_episode(
    episode_dir: Path,
    *,
    task_id: str,
    episode_index: int,
) -> EpisodeRecord:
    """Convert one accepted real-compatible sim episode without serializing it."""

    prompt = canonical_prompt(task_id)
    rows = _rows(Path(episode_dir) / "robot_log.csv")
    canonical_frames: list[FrameRecord] = []
    for frame_index, row in enumerate(rows[:-1]):
        next_row = rows[frame_index + 1]
        state = np.asarray(
            [float(row[name]) for name in STATE_COLUMNS], dtype=np.float32
        )
        action = np.asarray(
            [float(next_row[name]) for name in STATE_COLUMNS], dtype=np.float32
        )
        record = {
            "image": Path(episode_dir) / row["realsense_0_file"],
            "wrist_image": Path(episode_dir) / row["realsense_1_file"],
            "state": state,
            "actions": action,
            "task": prompt,
            "task_id": task_id,
            "source": "sim",
            "episode_index": int(episode_index),
            "source_frame_index": frame_index,
            "frame_index": frame_index,
            "timestamp": float(row["ts"]),
        }
        canonical_frames.append(FrameRecord.from_mapping(record))
    return EpisodeRecord(
        episode_index=episode_index,
        source=SourceBackend.SIM,
        frames=tuple(canonical_frames),
        metadata={"task_id": task_id},
    )


def training_records_from_raw_episode(
    episode_dir: Path,
    *,
    task_id: str,
    episode_index: int,
) -> list[dict[str, Any]]:
    """Expose canonical sim frames with provenance for audits and tooling."""

    return canonical_episode_from_raw_episode(
        episode_dir,
        task_id=task_id,
        episode_index=episode_index,
    ).as_records()


def convert_dataset(
    config: PipelineConfig, raw: Path, output: Path, *, overwrite: bool
) -> dict[str, Any]:
    raw = Path(raw).resolve()
    output = Path(output).resolve(strict=False)
    if raw != config.outputs.raw or output != config.outputs.converted:
        raise ValueError("Raw and converted paths must equal the configured dataset roots")
    raw_audit = json.loads(
        (config.outputs.log / "RAW_DATASET_AUDIT.json").read_text(encoding="utf-8")
    )
    if raw_audit.get("status") != "RAW_PASS":
        raise ValueError("Conversion requires a successful RAW_DATASET_AUDIT.json")
    summary = json.loads((raw / "collection_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((raw / "collection_manifest.json").read_text(encoding="utf-8"))
    if summary.get("complete") is not True or manifest.get("complete") is not True:
        raise ValueError("Raw collection must be complete before conversion")
    expected_counts = {task.task_id: task.episodes for task in config.tasks}
    completed = manifest.get("completed") or []
    completed_counts = Counter(str(entry.get("task_id")) for entry in completed)
    if (
        len(completed) != config.total_episodes
        or summary.get("total_accepted_episodes") != config.total_episodes
        or summary.get("total_distractor_episodes") != 0
        or dict(completed_counts) != expected_counts
        or summary.get("accepted_counts_by_task") != expected_counts
    ):
        raise ValueError("Raw collection counts do not match the configured dataset plan")
    replacement = replace_authorized_roots(
        [output], overwrite=overwrite, git_sha=git_sha(repository_root()),
        config_path=config.path,
    )
    overwrite_marker = json.loads(
        (output / "OVERWRITE_MARKER.json").read_text(encoding="utf-8")
    )
    records_by_episode: list[EpisodeRecord] = []
    episode_metadata: list[dict[str, Any]] = []
    for expected_episode_index, entry in enumerate(
        sorted(completed, key=lambda row: int(row["global_episode_index"]))
    ):
        if int(entry["global_episode_index"]) != expected_episode_index:
            raise ValueError("Raw global episode indices must be contiguous and monotonic")
        episode_dir = raw / entry["path"]
        meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
        task_id = str(meta["task_id"])
        prompt = canonical_prompt(task_id)
        if meta["task_prompt"] != prompt or meta["task"] != prompt:
            raise ValueError(f"Prompt mismatch in {episode_dir}")
        if entry["scene_variant"] != "clean":
            raise ValueError(f"Distractor episode cannot be converted: {episode_dir}")
        validation = meta["simulation"]["validation"]
        if task_id == "place_red_pepper_in_ring":
            if not validation["place_initial_grasp"]["initial_grasp_success"]:
                raise ValueError(f"Place initial grasp failed: {episode_dir}")
            if validation["place_initial_grasp"]["initialization_frames_recorded"] != 0:
                raise ValueError(f"Place initialization frames entered training: {episode_dir}")
            if not validation["stable_place"]["stable_place_success"]:
                raise ValueError(f"Place verification failed: {episode_dir}")
        elif not validation["stable_grasp"]["stable_grasp_success"]:
            raise ValueError(f"Pick stability failed: {episode_dir}")
        canonical_episode = canonical_episode_from_raw_episode(
            episode_dir,
            task_id=task_id,
            episode_index=expected_episode_index,
        )
        records_by_episode.append(canonical_episode)
        episode_metadata.append(
            {
                "episode_index": int(entry["global_episode_index"]),
                "task_id": task_id,
                "task_prompt": prompt,
                "source_path": entry["path"],
                "scene_variant": "clean",
                "frames": len(canonical_episode.frames),
            }
        )
    result = write_xarm_lerobot_dataset(
        records_by_episode,
        repo_id=f"local/{config.dataset_version}",
        output_path=output,
        robot_type="xarm6",
        fps=config.action_hz,
        overwrite=True,
        resume=False,
        image_writer_threads=8,
        image_writer_processes=0,
        push_to_hub=False,
    )
    metadata = {
        "schema_version": "xarm_mujoco_multitask_conversion_v1",
        "dataset_version": config.dataset_version,
        "raw_input": str(raw),
        "converted_output": str(output),
        "camera_config": str(config.camera_config),
        "canonical_prompts": {task.task_id: task.prompt for task in config.tasks},
        "task_index_order": [task.task_id for task in config.tasks],
        "clean_scene_only": True,
        "source_backend": "sim",
        "total_distractor_episodes": 0,
        "episodes": episode_metadata,
        "overwrite": replacement,
        "writer_result": result,
    }
    atomic_write_json(output / "OVERWRITE_MARKER.json", overwrite_marker)
    atomic_write_json(output / "meta" / "mujoco_multitask_metadata.json", metadata)
    return metadata
