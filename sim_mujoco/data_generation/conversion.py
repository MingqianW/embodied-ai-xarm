"""Canonical raw CSV/PNG to LeRobot v2.1 conversion for all six tasks."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from fine_tune.xarm_lerobot_writer import write_xarm_lerobot_dataset
from sim_mujoco.data_generation.config import PipelineConfig
from sim_mujoco.data_generation.manifest import atomic_write_json
from sim_mujoco.data_generation.registry import canonical_prompt
from sim_mujoco.data_generation.safety import replace_authorized_roots
from sim_mujoco.data_generation.status import git_sha


STATE_COLUMNS = tuple(f"j{index}_rad" for index in range(1, 7)) + ("gripper_mm",)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def convert_dataset(
    config: PipelineConfig, raw: Path, output: Path, *, overwrite: bool
) -> dict[str, Any]:
    raw = Path(raw).resolve()
    output = Path(output).resolve(strict=False)
    if raw != config.outputs.raw or output != config.outputs.converted:
        raise ValueError("Raw and converted paths must equal the configured v3 roots")
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
        len(completed) != 200
        or summary.get("total_accepted_episodes") != 200
        or summary.get("total_distractor_episodes") != 0
        or dict(completed_counts) != expected_counts
        or summary.get("accepted_counts_by_task") != expected_counts
    ):
        raise ValueError("Raw collection counts do not match the exact v3 plan")
    replacement = replace_authorized_roots(
        [output], overwrite=overwrite, git_sha=git_sha(config.path.parents[3]),
        config_path=config.path,
    )
    overwrite_marker = json.loads(
        (output / "OVERWRITE_MARKER.json").read_text(encoding="utf-8")
    )
    records_by_episode: list[list[dict[str, Any]]] = []
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
        rows = _rows(episode_dir / "robot_log.csv")
        episode_records: list[dict[str, Any]] = []
        for frame_index, row in enumerate(rows[:-1]):
            next_row = rows[frame_index + 1]
            state = np.asarray([float(row[name]) for name in STATE_COLUMNS], dtype=np.float32)
            action = np.asarray([float(next_row[name]) for name in STATE_COLUMNS], dtype=np.float32)
            if state.shape != (7,) or action.shape != (7,) or not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"Invalid state/action in {episode_dir}")
            episode_records.append(
                {
                    "image": episode_dir / row["realsense_0_file"],
                    "wrist_image": episode_dir / row["realsense_1_file"],
                    "state": state,
                    "actions": action,
                    "task": prompt,
                    "task_id": task_id,
                    "source_frame_index": frame_index,
                }
            )
        records_by_episode.append(episode_records)
        episode_metadata.append(
            {
                "episode_index": int(entry["global_episode_index"]),
                "task_id": task_id,
                "task_prompt": prompt,
                "source_path": entry["path"],
                "scene_variant": "clean",
                "frames": len(episode_records),
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
        "total_distractor_episodes": 0,
        "episodes": episode_metadata,
        "overwrite": replacement,
        "writer_result": result,
    }
    atomic_write_json(output / "OVERWRITE_MARKER.json", overwrite_marker)
    atomic_write_json(output / "meta" / "mujoco_multitask_metadata.json", metadata)
    return metadata
