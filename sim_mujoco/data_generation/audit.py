"""Strict raw and canonical converted dataset audits."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from sim_mujoco.data_generation.config import PipelineConfig
from sim_mujoco.data_generation.manifest import atomic_write_json
from sim_mujoco.data_generation.registry import canonical_prompt


STATE_COLUMNS = tuple(f"j{index}_rad" for index in range(1, 7)) + ("gripper_mm",)


def _jsonlines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _decode_image(path: Path) -> None:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (640, 480):
            raise ValueError(f"Invalid RGB image: {path}")


def audit_raw(
    config: PipelineConfig,
    raw: Path,
    *,
    decode_all_images: bool,
    smoke: bool = False,
) -> dict[str, Any]:
    raw = Path(raw).resolve()
    manifest = json.loads((raw / "collection_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((raw / "collection_summary.json").read_text(encoding="utf-8"))
    expected = {
        task.task_id: (1 if smoke else task.episodes) for task in config.tasks
    }
    expected_total = 6 if smoke else config.total_episodes
    if manifest.get("complete") is not True or summary.get("complete") is not True:
        raise ValueError("Raw manifests are incomplete")
    if (
        len(manifest["completed"]) != expected_total
        or summary["total_distractor_episodes"] != 0
    ):
        raise ValueError("Raw episode/distractor totals are invalid")
    counts: Counter[str] = Counter()
    prompts: Counter[tuple[str, str]] = Counter()
    frame_ids: set[tuple[int, int]] = set()
    episode_ids: set[int] = set()
    episodes: list[dict[str, Any]] = []
    total_frames = 0
    ordered = sorted(
        manifest["completed"], key=lambda row: int(row["global_episode_index"])
    )
    for expected_episode_index, entry in enumerate(ordered):
        if entry["scene_variant"] != "clean":
            raise ValueError("Accepted distractor episode found")
        episode_index = int(entry["global_episode_index"])
        if episode_index != expected_episode_index or episode_index in episode_ids:
            raise ValueError(f"Duplicate/nonmonotonic episode ID: {episode_index}")
        episode_ids.add(episode_index)
        episode_dir = raw / entry["path"]
        if "failed_attempts" in episode_dir.parts:
            raise ValueError(f"Failed attempt entered accepted manifest: {episode_dir}")
        meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
        task_id = str(meta["task_id"])
        prompt = canonical_prompt(task_id)
        if (
            meta["task_prompt"] != prompt
            or meta["task"] != prompt
            or entry["task_id"] != task_id
            or entry["task_prompt"] != prompt
            or "_" in prompt
        ):
            raise ValueError(f"Canonical prompt mismatch: {episode_dir}")
        counts[task_id] += 1
        prompts[(task_id, prompt)] += 1
        validation = meta["simulation"]["validation"]
        if task_id == "place_red_pepper_in_ring":
            initial = validation["place_initial_grasp"]
            stable = validation["stable_place"]
            if not initial["initial_grasp_success"] or initial["initialization_frames_recorded"] != 0:
                raise ValueError(f"Invalid Place initialization: {episode_dir}")
            if not stable["stable_place_success"] or not stable["release_detected"]:
                raise ValueError(f"Invalid Place release: {episode_dir}")
            if (
                initial["initial_grasp_validation_steps_executed"] != 10
                or initial["initial_grasp_validation_duration_s"] + 1e-9 < 1.0
                or stable["place_verification_steps_executed"] != 20
                or stable["place_verification_duration_s"] + 1e-9 < 2.0
            ):
                raise ValueError(f"Incomplete Place validation: {episode_dir}")
        else:
            stable = validation["stable_grasp"]
            required = {
                "stable_grasp_success": True,
                "stable_grasp_failure_reason": None,
                "verification_steps_required": 20,
                "verification_steps_executed": 20,
            }
            if any(stable.get(key) != value for key, value in required.items()):
                raise ValueError(f"Invalid Pick stability: {episode_dir}")
            if stable["verification_duration_s"] + 1e-9 < 2.0:
                raise ValueError(f"Short Pick verification: {episode_dir}")
            if (
                stable["minimum_verification_lift_height_m"] + 1e-9
                < config.pick.minimum_lift_height_m
                or stable["maximum_relative_downward_slip_m"]
                > config.pick.maximum_relative_downward_slip_m + 1e-9
                or stable["final_relative_downward_slip_m"]
                > config.pick.maximum_final_relative_downward_slip_m + 1e-9
            ):
                raise ValueError(f"Pick metric threshold failure: {episode_dir}")
        with (episode_dir / "robot_log.csv").open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        values = np.asarray(
            [[float(row[name]) for name in STATE_COLUMNS] for row in rows], dtype=np.float64
        )
        if values.shape != (len(rows), 7) or not np.isfinite(values).all():
            raise ValueError(f"Invalid raw state values: {episode_dir}")
        timestamps = np.asarray([float(row["ts"]) for row in rows])
        if not np.all(np.diff(timestamps) > 0) or not np.allclose(np.diff(timestamps), 0.1, atol=2e-6):
            raise ValueError(f"Invalid frame ordering: {episode_dir}")
        for frame_index, row in enumerate(rows):
            key = (int(entry["global_episode_index"]), frame_index)
            if key in frame_ids:
                raise ValueError(f"Duplicate raw frame ID: {key}")
            frame_ids.add(key)
            for column in ("realsense_0_file", "realsense_1_file"):
                image_path = episode_dir / row[column]
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                if decode_all_images:
                    _decode_image(image_path)
        total_frames += len(rows) - 1
        visuals = entry.get("visuals")
        if smoke:
            if not isinstance(visuals, dict):
                raise ValueError(f"Smoke visuals are missing: {episode_dir}")
            video_paths = visuals.get("videos") or {}
            if set(video_paths) != {"realsense_0", "realsense_1", "realsense_2"}:
                raise ValueError(f"Smoke camera videos are incomplete: {episode_dir}")
            for video_path in video_paths.values():
                if not video_path or not Path(video_path).is_file():
                    raise ValueError(f"Smoke video is missing: {video_path}")
            required_frames = {
                "first_recorded_frame",
                "grasp_or_lift_frame",
                "verification_start_frame",
                "verification_end_frame",
                "final_frame",
            }
            if task_id == "place_red_pepper_in_ring":
                required_frames.add("release_frame")
            key_frames = visuals.get("key_frames") or {}
            if not required_frames <= set(key_frames):
                raise ValueError(f"Smoke key frames are incomplete: {episode_dir}")
            if not visuals.get("contact_sheet") or not Path(
                visuals["contact_sheet"]
            ).is_file():
                raise ValueError(f"Smoke contact sheet is missing: {episode_dir}")
        episodes.append(
            {
                "episode_index": episode_index,
                "task_id": task_id,
                "task_prompt": prompt,
                "path": entry["path"],
                "robot_log_rows": len(rows),
                "validation": validation,
                "visuals": visuals,
            }
        )
    if dict(counts) != expected:
        raise ValueError(f"Raw task counts differ: {dict(counts)}")
    return {
        "passed": True,
        "raw": str(raw),
        "episode_count": len(manifest["completed"]),
        "task_count": len(counts),
        "task_counts": dict(counts),
        "prompt_counts": {f"{task_id}|{prompt}": count for (task_id, prompt), count in prompts.items()},
        "total_frames": total_frames,
        "total_distractor_episodes": 0,
        "decoded_all_images": decode_all_images,
        "smoke": smoke,
        "episodes": episodes,
    }


def write_smoke_reports(
    config: PipelineConfig,
    raw_report: dict[str, Any],
    report_dir: Path,
) -> dict[str, Any]:
    if raw_report.get("episode_count") != 6 or not raw_report.get("smoke"):
        raise ValueError("Smoke report requires an audited six-episode smoke dataset")
    report_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "xarm_mujoco_smoke_audit_v1",
        "dataset_version": config.dataset_version,
        "status": "PASS",
        "raw": raw_report,
    }
    atomic_write_json(report_dir / "SMOKE_AUDIT.json", result)
    lines = [
        "# Smoke Audit",
        "",
        "**PASS**",
        "",
        "- Accepted episodes: 6",
        "- Tasks: 6",
        "- Distractor episodes: 0",
        "- All base/wrist/overview images decoded: yes",
        "- Place initialization frames recorded: 0",
        "",
        "## Per-task results",
        "",
        "| Task ID | Prompt | Result | Contact sheet |",
        "|---|---|---|---|",
    ]
    for episode in raw_report["episodes"]:
        visuals = episode.get("visuals") or {}
        lines.append(
            f"| `{episode['task_id']}` | {episode['task_prompt']} | PASS | "
            f"`{visuals.get('contact_sheet') or ''}` |"
        )
    lines.extend(
        [
            "",
            "MP4s were decoded during generation. Contact sheets remain subject to "
            "explicit human/Codex visual review before the full run is submitted.",
            "",
        ]
    )
    (report_dir / "SMOKE_AUDIT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return result


def write_raw_audit_reports(
    config: PipelineConfig,
    raw_report: dict[str, Any],
    report_dir: Path,
) -> dict[str, Any]:
    if (
        raw_report.get("episode_count") != config.total_episodes
        or raw_report.get("smoke")
    ):
        raise ValueError(
            "Raw audit report requires the complete configured dataset"
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "xarm_mujoco_raw_audit_v1",
        "dataset_version": config.dataset_version,
        "status": "RAW_PASS",
        "raw": raw_report,
    }
    atomic_write_json(report_dir / "RAW_DATASET_AUDIT.json", result)
    (report_dir / "RAW_DATASET_AUDIT.md").write_text(
        "\n".join(
            [
                "# Raw Dataset Audit",
                "",
                "**RAW_PASS**",
                "",
                f"- Accepted episodes: {config.total_episodes}",
                "- Tasks: 6",
                "- Distractor episodes: 0",
                "- Canonical prompts: PASS",
                "- Accepted Pick and Place validation metadata: PASS",
                "",
                "Conversion is not implied by this raw-only result.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return result


def _decode_parquet_image(value: Any) -> None:
    payload = value.get("bytes") if isinstance(value, dict) else None
    if payload is None:
        raise ValueError("Parquet image does not contain embedded bytes")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.mode != "RGB" or image.size != (640, 480):
            raise ValueError("Invalid converted image")


def audit_converted(config: PipelineConfig, converted: Path, *, decode_all_images: bool) -> dict[str, Any]:
    converted = Path(converted).resolve()
    info = json.loads((converted / "meta" / "info.json").read_text(encoding="utf-8"))
    tasks = _jsonlines(converted / "meta" / "tasks.jsonl")
    episodes = _jsonlines(converted / "meta" / "episodes.jsonl")
    mapping = json.loads(
        (converted / "meta" / "mujoco_multitask_metadata.json").read_text(encoding="utf-8")
    )
    expected = {task.prompt: task.episodes for task in config.tasks}
    actual_tasks = {str(row["task"]) for row in tasks}
    if (
        int(info["total_episodes"]) != config.total_episodes
        or int(info["total_tasks"]) != 6
    ):
        raise ValueError("Converted totals are invalid")
    if actual_tasks != set(expected) or any("_" in prompt for prompt in actual_tasks):
        raise ValueError("Converted prompts are invalid")
    episode_counts = Counter(str(row["tasks"][0]) for row in episodes)
    if dict(episode_counts) != expected:
        raise ValueError(f"Converted task counts differ: {dict(episode_counts)}")
    if mapping.get("total_distractor_episodes") != 0 or not mapping.get("clean_scene_only"):
        raise ValueError("Converted dataset contains distractor metadata")
    files = sorted((converted / "data").glob("chunk-*/episode_*.parquet"))
    if len(files) != config.total_episodes:
        raise ValueError(
            f"Expected {config.total_episodes} parquet episodes, found {len(files)}"
        )
    seen_indices: set[int] = set()
    seen_frames: set[tuple[int, int]] = set()
    total_frames = 0
    for expected_episode, path in enumerate(files):
        table = pq.read_table(path)
        states = np.asarray(table["state"].combine_chunks().to_pylist(), dtype=np.float32)
        actions = np.asarray(table["actions"].combine_chunks().to_pylist(), dtype=np.float32)
        episode_indices = np.asarray(table["episode_index"].combine_chunks().to_pylist(), dtype=np.int64)
        frame_indices = np.asarray(table["frame_index"].combine_chunks().to_pylist(), dtype=np.int64)
        global_indices = np.asarray(table["index"].combine_chunks().to_pylist(), dtype=np.int64)
        if states.shape[1:] != (7,) or actions.shape[1:] != (7,):
            raise ValueError(f"Converted state/action shape mismatch: {path}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"NaN/Inf in converted data: {path}")
        episode_id = int(episode_indices[0])
        if episode_id != expected_episode or episode_id in seen_indices:
            raise ValueError(f"Duplicate/nonmonotonic episode index: {path}")
        seen_indices.add(episode_id)
        if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
            raise ValueError(f"Nonmonotonic frame indices: {path}")
        if len(np.unique(global_indices)) != len(global_indices):
            raise ValueError(f"Duplicate global frame IDs: {path}")
        for frame_index in frame_indices:
            key = (episode_id, int(frame_index))
            if key in seen_frames:
                raise ValueError(f"Duplicate frame ID: {key}")
            seen_frames.add(key)
        if decode_all_images:
            for column in ("image", "wrist_image"):
                for value in table[column].combine_chunks().to_pylist():
                    _decode_parquet_image(value)
        total_frames += len(table)
    if total_frames != int(info["total_frames"]):
        raise ValueError("Converted total frame count mismatch")
    return {
        "passed": True,
        "converted": str(converted),
        "episode_count": len(files),
        "task_count": len(actual_tasks),
        "task_counts": dict(episode_counts),
        "total_frames": total_frames,
        "total_distractor_episodes": 0,
        "decoded_all_images": decode_all_images,
    }


def write_audit_reports(
    config: PipelineConfig,
    raw_report: dict[str, Any],
    converted_report: dict[str, Any] | None,
    report_dir: Path,
) -> dict[str, Any]:
    ready = bool(raw_report.get("passed") and converted_report and converted_report.get("passed"))
    report_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "xarm_mujoco_clean_dataset_audit_v1",
        "dataset_version": config.dataset_version,
        "status": "READY_FOR_TRAINING" if ready else "NOT_READY_FOR_TRAINING",
        "raw": raw_report,
        "converted": converted_report,
    }
    atomic_write_json(report_dir / "DATASET_AUDIT.json", result)
    lines = [
        "# Dataset Audit",
        "",
        f"**{result['status']}**",
        "",
        f"- Raw episodes: {raw_report.get('episode_count')}",
        f"- Converted episodes: {(converted_report or {}).get('episode_count')}",
        "- Distractor episodes: 0",
        "- Canonical prompt audit: PASS" if ready else "- Canonical prompt audit: INCOMPLETE",
        "",
    ]
    (report_dir / "DATASET_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    with (report_dir / "TASK_COUNTS.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task_id", "prompt", "expected", "raw", "converted"])
        for task in config.tasks:
            writer.writerow([
                task.task_id,
                task.prompt,
                task.episodes,
                raw_report.get("task_counts", {}).get(task.task_id, 0),
                (converted_report or {}).get("task_counts", {}).get(task.prompt, 0),
            ])
    with (report_dir / "PROMPT_AUDIT.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task_id", "canonical_prompt", "contains_underscore", "passed"])
        for task in config.tasks:
            writer.writerow([task.task_id, task.prompt, "_" in task.prompt, True])
    with (report_dir / "FAILURE_COUNTS.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["failure_reason", "count"])
        summary_path = config.outputs.raw / "collection_summary.json"
        failure_counts = (
            json.loads(summary_path.read_text(encoding="utf-8")).get("failure_counts_by_reason", {})
            if summary_path.is_file()
            else {}
        )
        writer.writerows(sorted(failure_counts.items()))
    atomic_write_json(
        report_dir / "REPRODUCIBILITY.json",
        {
            "dataset_version": config.dataset_version,
            "config_path": str(config.path),
            "camera_config_path": str(config.camera_config),
            "raw_path": str(config.outputs.raw),
            "converted_path": str(config.outputs.converted),
            "canonical_prompts": {task.task_id: task.prompt for task in config.tasks},
        },
    )
    return result
