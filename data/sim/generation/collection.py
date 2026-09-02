"""Resumable six-task clean-scene oracle collection."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from data.sim.generation.core.generator import GeneratorContext
from data.sim.generation.core.registry import create_generator
from data.sim.generation.real_raw_compatible_recorder import (
    RealCompatibleRawEpisodeRecorder,
)
from data.sim.generation.acceptance import (
    accepted_oracle_episode,
    simulation_is_finite,
    update_task_success,
)
from data.sim.generation.artifacts import (
    write_diagnostic_visuals,
    write_episode_visuals,
)
from data.sim.generation.config import PipelineConfig, TaskPlan
from data.sim.generation.manifest import (
    atomic_write_json,
    initial_manifest,
    mark_updated,
)
from data.sim.generation.safety import replace_authorized_roots
from simulation.environment import MuJoCoEnvironment


def resolve_seed(task: TaskPlan, requested_episode_index: int, retry_index: int, stride: int) -> int:
    if requested_episode_index < 0 or retry_index < 0 or stride < 1:
        raise ValueError("Episode index, retry index, and stride must be valid")
    return task.base_seed + requested_episode_index + retry_index * stride


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _record_attempt(
    environment: MuJoCoEnvironment,
    *,
    config: PipelineConfig,
    task: TaskPlan,
    requested_episode_index: int,
    global_episode_index: int,
    retry_index: int,
    resolved_seed: int,
    staging_dir: Path,
) -> tuple[bool, dict[str, Any]]:
    environment.reset(seed=resolved_seed)
    generator = create_generator(
        GeneratorContext(
            environment=environment,
            pipeline_config=config,
            task=task,
            requested_episode_index=requested_episode_index,
            retry_index=retry_index,
            seed=resolved_seed,
        )
    )
    if not generator.initialization.success:
        staging_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "task_id": task.task_id,
            "task_prompt": task.prompt,
            "generator_id": generator.generator_id,
            "generator_version": generator.generator_version,
            "requested_episode_index": requested_episode_index,
            "base_seed": task.base_seed,
            "retry_index": retry_index,
            "resolved_seed": resolved_seed,
            "scene_variant": "clean",
            "success": False,
            "failure_reason": generator.failure_reason,
            "validation": generator.validation_metadata(),
        }
        if requested_episode_index == 0 and retry_index == 0 and generator.initialization.diagnostic_frames:
            failure["diagnostic_visuals"] = write_diagnostic_visuals(
                staging_dir, generator.initialization.diagnostic_frames
            )
        atomic_write_json(staging_dir / "failure.json", failure)
        return False, failure

    recorder = RealCompatibleRawEpisodeRecorder(
        staging_dir,
        task=task.prompt,
        task_id=task.task_id,
        task_prompt=task.prompt,
        episode_index=global_episode_index,
        requested_episode_index=requested_episode_index,
        base_seed=task.base_seed,
        retry_index=retry_index,
        seed=resolved_seed,
        scene_variant="clean",
        generator_id=generator.generator_id,
        generator_version=generator.generator_version,
        environment=environment,
        save_hz=config.action_hz,
    )
    runtime = environment.task_runtime
    assert runtime is not None
    metrics = runtime.metrics()
    last_action = None
    while not generator.terminal:
        action = generator.next_action()
        if action is None:
            break
        recorder.record_observation(gripper_target_raw=float(action[6]))
        environment.apply_action(action)
        environment.step_physics(1.0 / config.action_hz)
        metrics = update_task_success(environment)
        collision = environment.safety_diagnostics()["collision"]
        generator.notify_post_step(
            task_metrics=metrics,
            collision=collision,
            simulation_finite=simulation_is_finite(environment),
        )
        last_action = action
    recorder.record_observation(
        gripper_target_raw=(
            float(last_action[6]) if last_action is not None else None
        )
    )
    validation = generator.validation_metadata()
    success = accepted_oracle_episode(
        terminal_stage=generator.stage.value if hasattr(generator.stage, "value") else str(generator.stage),
        task_metrics=metrics,
        failure_reason=generator.failure_reason,
        validation_success=generator.accepted(),
    )
    meta = recorder.finalize(
        success=success,
        failure_reason=generator.failure_reason,
        initial_conditions=environment.initial_conditions,
        task_metrics=metrics,
        oracle_transitions=generator.transition_log(),
        oracle_plan=generator.plan_metadata(),
        validation_metadata=validation,
    )
    return success, meta


def _run_config(config: PipelineConfig, output: Path, *, smoke: bool, overwrite: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "schema_version": "xarm_mujoco_clean_collection_run_v1",
        "dataset_version": config.dataset_version,
        "generation_commit_sha": _git_sha(),
        "pipeline_config_path": str(config.path),
        "camera_config_path": str(config.camera_config),
        "task_scene_config_path": str(config.task_scene_config),
        "absolute_raw_output_path": str(output),
        "absolute_log_path": str(config.outputs.log),
        "action_hz": config.action_hz,
        "scene_variant": "clean",
        "distractor_count": 0,
        "randomization_config": {
            "object_xy_range_m": config.object_xy_range_m,
            "object_yaw_range_deg": config.object_yaw_range_deg,
            "joint_noise_rad": config.joint_noise_rad,
        },
        "retry_policy": {
            "max_attempts_per_episode": config.max_attempts_per_episode,
            "seed_retry_stride": config.seed_retry_stride,
        },
        "verification_config": {
            "pick": asdict(config.pick),
            "place_initial": asdict(config.place_initial),
            "place": asdict(config.place),
        },
        "plan": [
            {
                **asdict(task),
                "episodes": 1 if smoke else task.episodes,
                "clean_episodes": 1 if smoke else task.episodes,
                "distractor_episodes": 0,
                "generators": (
                    [{"generator_id": task.generator_for_episode(0), "episodes": 1}]
                    if smoke
                    else [asdict(generator) for generator in task.generators]
                ),
            }
            for task in config.tasks
        ],
        "total_target_episodes": len(config.tasks) if smoke else config.total_episodes,
        "smoke": smoke,
        "overwrite": overwrite,
    }


def collect(
    config: PipelineConfig,
    output: Path,
    *,
    overwrite: bool,
    resume: bool,
    smoke: bool,
) -> dict[str, Any]:
    output = Path(output).resolve(strict=False)
    expected = config.outputs.smoke if smoke else config.outputs.raw
    if output != expected:
        raise ValueError(f"Resolved output must equal configured root: {expected}")
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    overwrite_record = None
    if overwrite:
        overwrite_record = replace_authorized_roots(
            [output], overwrite=True, git_sha=_git_sha(), config_path=config.path
        )
    run_config = _run_config(config, output, smoke=smoke, overwrite=overwrite_record)
    config_path = output / "run_config.json"
    manifest_path = output / "collection_manifest.json"
    if resume:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_config["overwrite"] = existing.get("overwrite")
        if existing != run_config:
            raise ValueError("Resume configuration differs from saved run_config.json")
    else:
        if not overwrite:
            raise ValueError("A new run requires explicit --overwrite")
        atomic_write_json(config_path, run_config)
        manifest = initial_manifest(config.dataset_version, run_config)
        atomic_write_json(manifest_path, manifest)
    for directory in ("accepted", "failed_attempts", ".staging", "visuals"):
        (output / directory).mkdir(exist_ok=True)

    completed = list(manifest.get("completed") or [])
    failed = list(manifest.get("failed_attempts") or [])
    completed_keys = {
        (str(row["task_id"]), int(row["requested_episode_index"]))
        for row in completed
    }
    offset = 0
    for task in config.tasks:
        target = 1 if smoke else task.episodes
        with MuJoCoEnvironment(
            task=task.task_id,
            prompt=task.prompt,
            camera_config_path=config.camera_config,
            task_scene_config_path=config.task_scene_config,
            object_xy_range=config.object_xy_range_m,
            object_yaw_range_deg=config.object_yaw_range_deg,
            joint_noise=config.joint_noise_rad,
            scene_variant="clean",
        ) as environment:
            for requested_index in range(target):
                key = (task.task_id, requested_index)
                if key in completed_keys:
                    continue
                succeeded = False
                for retry_index in range(config.max_attempts_per_episode):
                    seed = resolve_seed(
                        task, requested_index, retry_index, config.seed_retry_stride
                    )
                    staging = output / ".staging" / task.task_id / (
                        f"episode_{requested_index:03d}_attempt_{retry_index:02d}"
                    )
                    if staging.exists():
                        raise FileExistsError(f"Stale staging directory: {staging}")
                    try:
                        success, metadata = _record_attempt(
                            environment,
                            config=config,
                            task=task,
                            requested_episode_index=requested_index,
                            global_episode_index=offset + requested_index,
                            retry_index=retry_index,
                            resolved_seed=seed,
                            staging_dir=staging,
                        )
                        failure_reason = (
                            metadata.get("simulation", {}).get("failure_reason")
                            if "simulation" in metadata
                            else metadata.get("failure_reason")
                        )
                    except Exception as exc:
                        staging.mkdir(parents=True, exist_ok=True)
                        success = False
                        failure_reason = f"exception:{type(exc).__name__}:{exc}"
                        metadata = {
                            "task_id": task.task_id,
                            "task_prompt": task.prompt,
                            "failure_reason": failure_reason,
                        }
                        atomic_write_json(staging / "failure.json", metadata)
                    record = {
                        "task_id": task.task_id,
                        "task_prompt": task.prompt,
                        "generator_id": metadata.get("simulation", {}).get(
                            "generator_id", task.generator_for_episode(requested_index)
                        ),
                        "generator_version": metadata.get("simulation", {}).get("generator_version", "v1"),
                        "requested_episode_index": requested_index,
                        "global_episode_index": offset + requested_index,
                        "base_seed": task.base_seed,
                        "retry_index": retry_index,
                        "resolved_seed": seed,
                        "scene_variant": "clean",
                        "success": bool(success),
                        "failure_reason": failure_reason,
                    }
                    if success:
                        destination = output / "accepted" / task.task_id / f"episode_{requested_index:03d}"
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if destination.exists():
                            raise FileExistsError(destination)
                        staging.rename(destination)
                        record["path"] = destination.relative_to(output).as_posix()
                        record["robot_log_rows"] = int(
                            metadata["simulation"]["robot_log_rows"]
                        )
                        record["training_frames"] = int(
                            metadata["simulation"]["training_samples_after_real_converter"]
                        )
                        if smoke or (
                            config.representative_video_every > 0
                            and requested_index % config.representative_video_every == 0
                        ):
                            record["visuals"] = write_episode_visuals(destination)
                        completed.append(record)
                        completed_keys.add(key)
                        succeeded = True
                    else:
                        destination = output / "failed_attempts" / task.task_id / (
                            f"episode_{requested_index:03d}_attempt_{retry_index:02d}"
                        )
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if destination.exists():
                            raise FileExistsError(destination)
                        staging.rename(destination)
                        record["path"] = destination.relative_to(output).as_posix()
                        if (
                            requested_index == 0
                            and retry_index == 0
                            and (destination / "meta.json").is_file()
                        ):
                            record["visuals"] = write_episode_visuals(destination)
                        failed.append(record)
                    manifest.update({"completed": completed, "failed_attempts": failed})
                    mark_updated(manifest)
                    atomic_write_json(manifest_path, manifest)
                    print(
                        f"task={task.task_id} requested={requested_index} retry={retry_index} "
                        f"seed={seed} success={success} failure={failure_reason}",
                        flush=True,
                    )
                    if success:
                        break
                if not succeeded:
                    raise RuntimeError(
                        f"Failed {task.task_id} requested episode {requested_index} "
                        f"after {config.max_attempts_per_episode} attempts"
                    )
        offset += target

    requested_counts = {
        task.task_id: (1 if smoke else task.episodes) for task in config.tasks
    }
    accepted_counts = Counter(str(row["task_id"]) for row in completed)
    accepted_counts_by_generator = Counter(
        f"{row['task_id']}:{row.get('generator_id', 'unknown')}" for row in completed
    )
    failure_counts = Counter(
        str(row.get("failure_reason") or "unknown") for row in failed
    )
    lengths = [int(row["training_frames"]) for row in completed]
    complete = (
        dict(accepted_counts) == requested_counts
        and len(completed) == sum(requested_counts.values())
        and all(row["scene_variant"] == "clean" for row in completed)
    )
    summary = {
        "schema_version": "xarm_mujoco_clean_collection_summary_v1",
        "dataset_version": config.dataset_version,
        "complete": complete,
        "generation_commit_sha": _git_sha(),
        "absolute_raw_output_path": str(output),
        "absolute_log_path": str(config.outputs.log),
        "canonical_prompts": {task.task_id: task.prompt for task in config.tasks},
        "requested_counts_by_task": requested_counts,
        "accepted_counts_by_task": dict(accepted_counts),
        "accepted_counts_by_task_generator": dict(accepted_counts_by_generator),
        "clean_counts_by_task": dict(accepted_counts),
        "distractor_counts_by_task": {task.task_id: 0 for task in config.tasks},
        "total_distractor_episodes": 0,
        "failed_attempt_counts_by_task": dict(Counter(row["task_id"] for row in failed)),
        "failure_counts_by_reason": dict(failure_counts),
        "stable_grasp_failure_counts": {
            key: value for key, value in failure_counts.items() if key.startswith("stable_grasp")
        },
        "initial_place_grasp_failure_counts": {
            key: value
            for key, value in failure_counts.items()
            if key.startswith("initial_place_grasp")
        },
        "total_accepted_episodes": len(completed),
        "total_failed_attempts": len(failed),
        "total_frames": sum(lengths),
        "minimum_episode_length": min(lengths),
        "median_episode_length": float(np.median(lengths)),
        "maximum_episode_length": max(lengths),
        "seed_ranges": {
            task.task_id: {
                "base_seed": task.base_seed,
                "accepted_requested_indices": [0, requested_counts[task.task_id] - 1],
                "retry_stride": config.seed_retry_stride,
            }
            for task in config.tasks
        },
        "randomization_config": run_config["randomization_config"],
        "verification_config": run_config["verification_config"],
        "converted": False,
    }
    if not complete:
        raise RuntimeError(f"Collection completed with invalid summary: {summary}")
    manifest["complete"] = True
    mark_updated(manifest)
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(output / "collection_summary.json", summary)
    return summary
