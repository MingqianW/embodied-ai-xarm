"""Resumable six-task clean-scene oracle collection."""

from __future__ import annotations

import json
import hashlib
import subprocess
from collections import Counter
from dataclasses import asdict, replace
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
from data.sim.generation.config import GeneratorPlan, PipelineConfig, TaskPlan
from data.sim.generation.manifest import (
    atomic_write_json,
    initial_manifest,
    mark_updated,
)
from data.sim.generation.trajectory import (
    resolve_parameters,
    scene_seed as paired_scene_seed,
    trajectory_seed as paired_trajectory_seed,
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
    scene_seed: int | None = None,
    trajectory_parameters: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    environment.reset(seed=resolved_seed if scene_seed is None else scene_seed)
    generator = create_generator(
        GeneratorContext(
            environment=environment,
            pipeline_config=config,
            task=task,
            requested_episode_index=requested_episode_index,
            retry_index=retry_index,
            seed=resolved_seed,
            scene_seed=scene_seed,
            trajectory_parameters=trajectory_parameters,
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
        provenance=provenance,
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


def _scene_fingerprint(environment: MuJoCoEnvironment) -> str:
    """Fingerprint actual reset state, not just the seed used to request it."""

    data = environment.context.data
    payload = {
        "qpos": np.asarray(data.qpos, dtype=np.float64).round(12).tolist(),
        "qvel": np.asarray(data.qvel, dtype=np.float64).round(12).tolist(),
        "act": np.asarray(data.act, dtype=np.float64).round(12).tolist(),
        "ctrl": np.asarray(data.ctrl, dtype=np.float64).round(12).tolist(),
        "time": round(float(data.time), 12),
        "initial_conditions": environment.initial_conditions,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _collection_task(
    task: TaskPlan,
    *,
    smoke: bool,
    smoke_all_generators: bool,
) -> TaskPlan:
    """Return the exact task allocation to execute for this collection mode."""

    if not smoke_all_generators:
        return task
    if not smoke:
        raise ValueError("smoke_all_generators requires smoke=True")
    generators = tuple(
        GeneratorPlan(generator.generator_id, 1) for generator in task.generators
    )
    return replace(
        task,
        episodes=len(generators),
        clean_episodes=len(generators),
        generators=generators,
    )


def _run_config(
    config: PipelineConfig,
    output: Path,
    *,
    smoke: bool,
    smoke_all_generators: bool,
    overwrite: dict[str, Any] | None,
) -> dict[str, Any]:
    run_tasks = tuple(
        _collection_task(
            task,
            smoke=smoke,
            smoke_all_generators=smoke_all_generators,
        )
        for task in config.tasks
    )
    target_count = lambda task: (
        task.episodes
        if config.generation_mode == "paired_scene_groups"
        else 1 if smoke and not smoke_all_generators else task.episodes
    )
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
            "layout_profile": config.layout_profile,
            "object_yaw_range_deg": config.object_yaw_range_deg,
            "joint_noise_rad": config.joint_noise_rad,
        },
        "retry_policy": {
            "max_attempts_per_episode": config.max_attempts_per_episode,
            "seed_retry_stride": config.seed_retry_stride,
        },
        "generation_mode": config.generation_mode,
        "scenes_per_task": config.scenes_per_task,
        "trajectory_profile": (
            None
            if config.trajectory_profile is None
            else {
                "name": config.trajectory_profile.name,
                "version": config.trajectory_profile.version,
            }
        ),
        "verification_config": {
            "pick": asdict(config.pick),
            "place_initial": asdict(config.place_initial),
            "place": asdict(config.place),
        },
        "plan": [
            {
                **asdict(task),
                "episodes": target_count(task),
                "clean_episodes": target_count(task),
                "distractor_episodes": 0,
                "generators": (
                    [asdict(generator) for generator in task.generators]
                    if smoke_all_generators or config.generation_mode == "paired_scene_groups"
                    else [{"generator_id": task.generator_for_episode(0), "episodes": 1}]
                    if smoke
                    else [asdict(generator) for generator in task.generators]
                ),
            }
            for task in run_tasks
        ],
        "total_target_episodes": sum(target_count(task) for task in run_tasks),
        "smoke": smoke,
        "smoke_all_generators": smoke_all_generators,
        "overwrite": overwrite,
    }


def _paired_member_episode_index(task: TaskPlan, scene_index: int, member_index: int) -> int:
    """Map a stable (scene, member) tuple to the existing allocation boundary."""

    return sum(member.episodes for member in task.generators[:member_index]) + scene_index


def paired_smoke_config(config: PipelineConfig) -> PipelineConfig:
    """Return a one-scene-per-task paired configuration without changing the plan file."""

    if config.generation_mode != "paired_scene_groups":
        raise ValueError("paired_smoke_config requires paired_scene_groups")
    tasks = tuple(
        replace(
            task,
            episodes=len(task.generators),
            clean_episodes=len(task.generators),
            generators=tuple(GeneratorPlan(member.generator_id, 1) for member in task.generators),
        )
        for task in config.tasks
    )
    return replace(config, scenes_per_task=1, tasks=tasks)


def _collect_paired_scene_groups(
    config: PipelineConfig,
    output: Path,
    *,
    overwrite: bool,
    resume: bool,
    smoke: bool,
) -> dict[str, Any]:
    assert config.scenes_per_task is not None and config.trajectory_profile is not None
    overwrite_record = None
    if overwrite:
        overwrite_record = replace_authorized_roots(
            [output], overwrite=True, git_sha=_git_sha(), config_path=config.path
        )
    run_config = _run_config(
        config, output, smoke=smoke, smoke_all_generators=False, overwrite=overwrite_record
    )
    config_path, manifest_path = output / "run_config.json", output / "collection_manifest.json"
    if resume:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_config["overwrite"] = existing.get("overwrite")
        if json.dumps(existing, sort_keys=True) != json.dumps(run_config, sort_keys=True):
            raise ValueError("Resume configuration differs from saved run_config.json")
    else:
        if not overwrite:
            raise ValueError("A new run requires explicit --overwrite")
        atomic_write_json(config_path, run_config)
        manifest = initial_manifest(config.dataset_version, run_config)
        manifest.update({"scene_groups": [], "incomplete_scene_groups": []})
        atomic_write_json(manifest_path, manifest)
    for directory in ("accepted", "failed_attempts", ".staging", "visuals"):
        (output / directory).mkdir(exist_ok=True)

    completed = list(manifest.get("completed") or [])
    failed = list(manifest.get("failed_attempts") or [])
    groups = {str(row["scene_group_id"]): row for row in manifest.get("scene_groups") or []}
    completed_keys = {
        (str(row["task_id"]), int(row["scene_index"]), str(row["family_member_id"]))
        for row in completed
    }
    global_indices = {
        (task.task_id, scene_index, member.generator_id): (
            sum(len(candidate.generators) for candidate in config.tasks[:task_index]) * config.scenes_per_task
            + scene_index * len(task.generators) + member_index
        )
        for task_index, task in enumerate(config.tasks)
        for scene_index in range(config.scenes_per_task)
        for member_index, member in enumerate(task.generators)
    }
    # A process may stop after an atomic directory rename but before the manifest
    # write. Reconcile only finalized accepted episodes so resume never overwrites
    # valid paired work at that narrow interruption boundary.
    accepted_root = output / "accepted"
    if accepted_root.is_dir():
        for meta_path in accepted_root.rglob("meta.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            simulation = meta.get("simulation") or {}
            provenance = simulation.get("provenance") or {}
            if provenance.get("generation_mode") != "paired_scene_groups" or not simulation.get("success"):
                continue
            task_id = str(meta["task_id"])
            scene_index = int(provenance["scene_index"])
            member_id = str(provenance["family_member_id"])
            key = (task_id, scene_index, member_id)
            if key in completed_keys:
                continue
            completed.append(
                {
                    "task_id": task_id, "task_prompt": meta["task_prompt"],
                    "generator_id": simulation["generator_id"], "generator_version": simulation["generator_version"],
                    "family_member_id": member_id, "scene_group_id": provenance["scene_group_id"],
                    "scene_index": scene_index, "initialized_scene_fingerprint": provenance["initialized_scene_fingerprint"],
                    "scene_seed": provenance["scene_seed"], "trajectory_seed": provenance["trajectory_seed"],
                    "requested_episode_index": int(meta["requested_episode_index"]),
                    "global_episode_index": global_indices[key], "base_seed": simulation["base_seed"],
                    "retry_index": provenance["retry_index"], "resolved_seed": simulation["resolved_seed"],
                    "scene_variant": simulation["scene_variant"], "success": True, "failure_reason": None,
                    "trajectory_profile": provenance["trajectory_profile"],
                    "resolved_trajectory_parameters": provenance["resolved_trajectory_parameters"],
                    "path": meta_path.parent.relative_to(output).as_posix(),
                    "robot_log_rows": simulation["robot_log_rows"],
                    "training_frames": simulation["training_samples_after_real_converter"],
                }
            )
            completed_keys.add(key)
        manifest["completed"] = completed
        mark_updated(manifest)
        atomic_write_json(manifest_path, manifest)
    for task in config.tasks:
        with MuJoCoEnvironment(
            task=task.task_id, prompt=task.prompt, camera_config_path=config.camera_config,
            task_scene_config_path=config.task_scene_config, object_xy_range=config.object_xy_range_m,
            object_yaw_range_deg=config.object_yaw_range_deg, joint_noise=config.joint_noise_rad,
            layout_profile=config.layout_profile, scene_variant="clean",
        ) as environment:
            for scene_index in range(config.scenes_per_task):
                group_id = f"{task.task_id}:scene_{scene_index:05d}"
                seed = paired_scene_seed(task.base_seed, task.task_id, scene_index)
                members: list[dict[str, Any]] = []
                group_fingerprint: str | None = None
                incomplete_reason: str | None = None
                for member_index, member in enumerate(task.generators):
                    key = (task.task_id, scene_index, member.generator_id)
                    if key in completed_keys:
                        members.append({"family_member_id": member.generator_id, "status": "accepted"})
                        continue
                    requested_index = _paired_member_episode_index(task, scene_index, member_index)
                    previous = [
                        int(row["retry_index"])
                        for row in failed
                        if (str(row.get("task_id")), int(row.get("scene_index", -1)), str(row.get("family_member_id"))) == key
                    ]
                    succeeded = False
                    for retry_index in range((max(previous) + 1) if previous else 0, config.max_attempts_per_episode):
                        traj_seed = paired_trajectory_seed(
                            task.base_seed, task.task_id, scene_index, member.generator_id, retry_index
                        )
                        parameters = resolve_parameters(
                            config.trajectory_profile, task_id=task.task_id,
                            member_id=member.generator_id, seed=traj_seed,
                        )
                        # Deterministic reconstruction is the restoration mechanism.  The
                        # reset fingerprint proves each member begins from the same state.
                        environment.reset(seed=seed, build_policy_observation=False)
                        fingerprint = _scene_fingerprint(environment)
                        if group_fingerprint is None:
                            group_fingerprint = fingerprint
                        elif group_fingerprint != fingerprint:
                            raise RuntimeError(f"scene restoration fingerprint mismatch for {group_id}")
                        staging = output / ".staging" / task.task_id / f"scene_{scene_index:05d}" / member.generator_id / f"attempt_{retry_index:02d}"
                        if staging.exists():
                            if not resume:
                                raise FileExistsError(f"Stale staging directory: {staging}")
                            finalized_meta = staging / "meta.json"
                            if finalized_meta.is_file():
                                finalized = json.loads(finalized_meta.read_text(encoding="utf-8"))
                                if (finalized.get("simulation") or {}).get("success") is True:
                                    destination = output / "accepted" / task.task_id / f"scene_{scene_index:05d}" / member.generator_id
                                    destination.parent.mkdir(parents=True, exist_ok=True)
                                    if destination.exists():
                                        raise FileExistsError(f"Duplicate finalized staging destination: {destination}")
                                    # Windows can transiently reject the rename while
                                    # the recorder exits. A later resume retries this
                                    # exact finalized move; no successful episode is
                                    # demoted to a failed attempt.
                                    staging.rename(destination)
                                    return _collect_paired_scene_groups(
                                        config, output, overwrite=False, resume=True, smoke=smoke
                                    )
                            # An interrupted process can leave partial frames before
                            # finalization. Preserve them as an explicit failed attempt
                            # and consume that retry rather than silently deleting it.
                            interrupted = {
                                "task_id": task.task_id, "task_prompt": task.prompt,
                                "generator_id": member.generator_id, "generator_version": "v1",
                                "family_member_id": member.generator_id, "scene_group_id": group_id,
                                "scene_index": scene_index, "initialized_scene_fingerprint": fingerprint,
                                "scene_seed": seed, "trajectory_seed": traj_seed,
                                "requested_episode_index": requested_index,
                                "global_episode_index": global_indices[key], "base_seed": task.base_seed,
                                "retry_index": retry_index, "resolved_seed": traj_seed,
                                "scene_variant": "clean", "success": False,
                                "failure_reason": "interrupted_staging_attempt",
                                "trajectory_profile": provenance["trajectory_profile"],
                                "resolved_trajectory_parameters": parameters,
                            }
                            destination = output / "failed_attempts" / task.task_id / f"scene_{scene_index:05d}" / member.generator_id / f"attempt_{retry_index:02d}"
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            if destination.exists():
                                raise FileExistsError(f"Duplicate interrupted attempt destination: {destination}")
                            atomic_write_json(staging / "failure.json", interrupted)
                            staging.rename(destination)
                            interrupted["path"] = destination.relative_to(output).as_posix()
                            failed.append(interrupted)
                            manifest.update({"completed": completed, "failed_attempts": failed, "scene_groups": list(groups.values())})
                            mark_updated(manifest)
                            atomic_write_json(manifest_path, manifest)
                            continue
                        provenance = {
                            "generation_mode": "paired_scene_groups",
                            "scene_group_id": group_id,
                            "scene_index": scene_index,
                            "scene_seed": seed,
                            "initialized_scene_fingerprint": fingerprint,
                            "trajectory_seed": traj_seed,
                            "retry_index": retry_index,
                            "family_member_id": member.generator_id,
                            "trajectory_profile": {"name": config.trajectory_profile.name, "version": config.trajectory_profile.version},
                            "resolved_trajectory_parameters": parameters,
                        }
                        try:
                            success, metadata = _record_attempt(
                                environment, config=config, task=task,
                                requested_episode_index=requested_index,
                                global_episode_index=global_indices[key], retry_index=retry_index,
                                resolved_seed=traj_seed, scene_seed=seed,
                                trajectory_parameters=parameters, provenance=provenance,
                                staging_dir=staging,
                            )
                            failure_reason = metadata.get("simulation", {}).get("failure_reason") if "simulation" in metadata else metadata.get("failure_reason")
                        except Exception as exc:
                            staging.mkdir(parents=True, exist_ok=True)
                            success, failure_reason = False, f"exception:{type(exc).__name__}:{exc}"
                            metadata = {"failure_reason": failure_reason}
                            atomic_write_json(staging / "failure.json", {**provenance, **metadata})
                        record = {
                            "task_id": task.task_id, "task_prompt": task.prompt,
                            "generator_id": member.generator_id, "generator_version": "v1",
                            "family_member_id": member.generator_id, "scene_group_id": group_id,
                            "scene_index": scene_index, "initialized_scene_fingerprint": fingerprint,
                            "scene_seed": seed, "trajectory_seed": traj_seed,
                            "requested_episode_index": requested_index,
                            "global_episode_index": global_indices[key], "base_seed": task.base_seed,
                            "retry_index": retry_index, "resolved_seed": traj_seed,
                            "scene_variant": "clean", "success": bool(success),
                            "failure_reason": failure_reason,
                            "trajectory_profile": provenance["trajectory_profile"],
                            "resolved_trajectory_parameters": parameters,
                        }
                        if success:
                            destination = output / "accepted" / task.task_id / f"scene_{scene_index:05d}" / member.generator_id
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            if destination.exists():
                                raise FileExistsError(destination)
                            staging.rename(destination)
                            record["path"] = destination.relative_to(output).as_posix()
                            record["robot_log_rows"] = int(metadata["simulation"]["robot_log_rows"])
                            record["training_frames"] = int(metadata["simulation"]["training_samples_after_real_converter"])
                            if smoke:
                                record["visuals"] = write_episode_visuals(destination)
                            completed.append(record)
                            completed_keys.add(key)
                            members.append({"family_member_id": member.generator_id, "status": "accepted"})
                            succeeded = True
                        else:
                            destination = output / "failed_attempts" / task.task_id / f"scene_{scene_index:05d}" / member.generator_id / f"attempt_{retry_index:02d}"
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            staging.rename(destination)
                            record["path"] = destination.relative_to(output).as_posix()
                            failed.append(record)
                        manifest.update({"completed": completed, "failed_attempts": failed, "scene_groups": list(groups.values())})
                        mark_updated(manifest)
                        atomic_write_json(manifest_path, manifest)
                        if success:
                            break
                    if not succeeded:
                        incomplete_reason = f"member_exhausted:{member.generator_id}"
                        members.append({"family_member_id": member.generator_id, "status": "incomplete", "reason": incomplete_reason})
                groups[group_id] = {
                    "scene_group_id": group_id, "task_id": task.task_id,
                    "scene_index": scene_index, "scene_seed": seed,
                    "initialized_scene_fingerprint": group_fingerprint,
                    "required_family_members": [member.generator_id for member in task.generators],
                    "members": members,
                    "complete": incomplete_reason is None and len(members) == len(task.generators),
                    "incomplete_reason": incomplete_reason,
                }
                manifest.update({"completed": completed, "failed_attempts": failed, "scene_groups": list(groups.values())})
                manifest["incomplete_scene_groups"] = [row for row in groups.values() if not row["complete"]]
                mark_updated(manifest)
                atomic_write_json(manifest_path, manifest)
    expected_total = sum(len(task.generators) * config.scenes_per_task for task in config.tasks)
    complete = len(completed) == expected_total and not manifest["incomplete_scene_groups"]
    summary = {
        "schema_version": "xarm_mujoco_paired_scene_collection_summary_v1",
        "dataset_version": config.dataset_version, "complete": complete,
        "generation_mode": config.generation_mode, "scenes_per_task": config.scenes_per_task,
        "target_accepted_episodes": expected_total, "total_accepted_episodes": len(completed),
        "total_failed_attempts": len(failed), "requested_scene_groups": len(config.tasks) * config.scenes_per_task,
        "completed_scene_groups": sum(bool(row["complete"]) for row in groups.values()),
        "incomplete_scene_groups": manifest["incomplete_scene_groups"],
        "accepted_counts_by_task": dict(Counter(str(row["task_id"]) for row in completed)),
        "accepted_counts_by_task_generator": dict(Counter(f"{row['task_id']}:{row['family_member_id']}" for row in completed)),
        "total_distractor_episodes": 0, "converted": False,
    }
    manifest["complete"] = complete
    mark_updated(manifest)
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(output / "collection_summary.json", summary)
    return summary


def collect(
    config: PipelineConfig,
    output: Path,
    *,
    overwrite: bool,
    resume: bool,
    smoke: bool,
    smoke_all_generators: bool = False,
) -> dict[str, Any]:
    output = Path(output).resolve(strict=False)
    expected = config.outputs.smoke if smoke else config.outputs.raw
    if output != expected:
        raise ValueError(f"Resolved output must equal configured root: {expected}")
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if smoke_all_generators and not smoke:
        raise ValueError("smoke_all_generators requires smoke=True")
    if config.generation_mode == "paired_scene_groups":
        if smoke_all_generators:
            raise ValueError("smoke_all_generators does not combine with paired_scene_groups")
        paired_config = paired_smoke_config(config) if smoke else config
        return _collect_paired_scene_groups(
            paired_config, output, overwrite=overwrite, resume=resume, smoke=smoke
        )
    overwrite_record = None
    if overwrite:
        overwrite_record = replace_authorized_roots(
            [output], overwrite=True, git_sha=_git_sha(), config_path=config.path
        )
    run_config = _run_config(
        config,
        output,
        smoke=smoke,
        smoke_all_generators=smoke_all_generators,
        overwrite=overwrite_record,
    )
    config_path = output / "run_config.json"
    manifest_path = output / "collection_manifest.json"
    if resume:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_config["overwrite"] = existing.get("overwrite")
        if json.dumps(existing, sort_keys=True) != json.dumps(run_config, sort_keys=True):
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
    run_tasks = tuple(
        _collection_task(
            task,
            smoke=smoke,
            smoke_all_generators=smoke_all_generators,
        )
        for task in config.tasks
    )
    offset = 0
    for task in run_tasks:
        target = 1 if smoke and not smoke_all_generators else task.episodes
        with MuJoCoEnvironment(
            task=task.task_id,
            prompt=task.prompt,
            camera_config_path=config.camera_config,
            task_scene_config_path=config.task_scene_config,
            object_xy_range=config.object_xy_range_m,
            object_yaw_range_deg=config.object_yaw_range_deg,
            joint_noise=config.joint_noise_rad,
            layout_profile=config.layout_profile,
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
        task.task_id: (1 if smoke and not smoke_all_generators else task.episodes)
        for task in run_tasks
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
        "smoke_all_generators": smoke_all_generators,
        "converted": False,
    }
    if not complete:
        raise RuntimeError(f"Collection completed with invalid summary: {summary}")
    manifest["complete"] = True
    mark_updated(manifest)
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(output / "collection_summary.json", summary)
    return summary
