"""Resumable six-task clean-scene oracle collection."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from sim_mujoco.collision import target_gripper_contact_count
from sim_mujoco.data_collection.conversions import policy_state_from_mujoco
from sim_mujoco.data_collection.oracle_controller import (
    OracleConfig,
    PlaceOracleConfig,
    PlaceRedPepperOracleController,
    ScriptedOracleController,
    oracle_config_for_task,
)
from sim_mujoco.data_collection.real_raw_recorder import RealRawEpisodeRecorder
from sim_mujoco.data_collection.task_success import (
    accepted_oracle_episode,
    simulation_is_finite,
    update_task_success,
)
from sim_mujoco.data_generation.artifacts import (
    write_diagnostic_visuals,
    write_episode_visuals,
)
from sim_mujoco.data_generation.config import PipelineConfig, TaskPlan
from sim_mujoco.data_generation.manifest import (
    atomic_write_json,
    initial_manifest,
    mark_updated,
)
from sim_mujoco.data_generation.safety import replace_authorized_roots
from sim_mujoco.data_generation.stability import (
    StabilitySample,
    evaluate_place_initial_grasp,
)
from sim_mujoco.environment import MuJoCoEnvironment
from sim_mujoco.remote_policy_observation import render_native_rgb
from sim_mujoco.task_scenes import TABLE_TOP_Z


def resolve_seed(task: TaskPlan, requested_episode_index: int, retry_index: int, stride: int) -> int:
    if requested_episode_index < 0 or retry_index < 0 or stride < 1:
        raise ValueError("Episode index, retry index, and stride must be valid")
    return task.base_seed + requested_episode_index + retry_index * stride


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _body_pose(environment: MuJoCoEnvironment, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = mujoco.mj_name2id(
        environment.context.model, mujoco.mjtObj.mjOBJ_BODY, body_name
    )
    return (
        np.asarray(environment.context.data.xpos[body_id], dtype=np.float64).copy(),
        np.asarray(environment.context.data.xquat[body_id], dtype=np.float64).copy(),
    )


def _tcp_pose(environment: MuJoCoEnvironment) -> tuple[np.ndarray, np.ndarray]:
    site_id = mujoco.mj_name2id(
        environment.context.model, mujoco.mjtObj.mjOBJ_SITE, "tool_center_point"
    )
    rotation = np.asarray(
        environment.context.data.site_xmat[site_id], dtype=np.float64
    ).reshape(3, 3)
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    return (
        np.asarray(environment.context.data.site_xpos[site_id], dtype=np.float64).copy(),
        quaternion,
    )


def _table_contact(collision: dict[str, Any], target_body: str) -> bool:
    return any(
        (row.get("body1") == target_body or row.get("body2") == target_body)
        and (row.get("geom1") == "table" or row.get("geom2") == "table")
        for row in collision.get("contacts") or ()
    )


def _validate_place_initial_grasp(
    environment: MuJoCoEnvironment, config: PipelineConfig
) -> tuple[dict[str, Any], dict[str, list[np.ndarray]]]:
    runtime = environment.task_runtime
    assert runtime is not None
    initial_object, initial_object_quaternion = _body_pose(
        environment, runtime.active_target_body
    )
    initial_tcp, initial_tcp_quaternion = _tcp_pose(environment)
    ring_position, _ = _body_pose(environment, str(runtime.spec["success"]["ring_body"]))
    action = policy_state_from_mujoco(
        environment.context.model, environment.context.data
    ).astype(np.float32)
    # Hold the randomized reset target, not the contact-deflected measured
    # joint state. Re-commanding the measured state unloads the arm servos and
    # can move the TCP relative to an otherwise settled physical grasp.
    initial_arm_target = np.asarray(
        environment.initial_conditions["initial_joint_positions"],
        dtype=np.float32,
    )
    if initial_arm_target.shape != (6,) or not np.isfinite(initial_arm_target).all():
        raise ValueError("Place initial arm target must be finite with shape (6,)")
    action[:6] = initial_arm_target
    action[6] = float(runtime.spec["initial_gripper_raw"])
    samples: list[StabilitySample] = []
    diagnostic_frames: dict[str, list[np.ndarray]] = {
        "realsense_0": [],
        "realsense_1": [],
        "realsense_2": [],
    }
    for _ in range(config.place_initial.steps):
        environment.apply_action(action)
        environment.step_physics(config.place_initial.action_dt_s)
        collision = environment.safety_diagnostics()["collision"]
        active_target_body = runtime.active_target_body
        object_position, _ = _body_pose(environment, active_target_body)
        tcp_position, _ = _tcp_pose(environment)
        ring_distance = float(np.linalg.norm(object_position[:2] - ring_position[:2]))
        samples.append(
            StabilitySample(
                simulation_time_s=float(environment.context.data.time),
                object_position_m=tuple(float(value) for value in object_position),
                tcp_position_m=tuple(float(value) for value in tcp_position),
                finite=bool(
                    simulation_is_finite(environment)
                    and np.isfinite(object_position).all()
                    and np.isfinite(tcp_position).all()
                ),
                table_contact=_table_contact(collision, active_target_body),
                forbidden_collision=bool(collision.get("forbidden")),
                inside_ring=ring_distance <= config.place.ring_radius_m,
                gripper_contact_count=target_gripper_contact_count(
                    collision, active_target_body
                ),
            )
        )
        for raw_name, camera_name in (
            ("realsense_0", "base_camera"),
            ("realsense_1", "wrist_camera"),
            ("realsense_2", "overview_camera"),
        ):
            diagnostic_frames[raw_name].append(
                render_native_rgb(
                    environment.context.renderer,
                    environment.context.data,
                    camera_name,
                ).copy()
            )
    result = evaluate_place_initial_grasp(
        samples,
        config=config.place_initial,
        table_top_z_m=TABLE_TOP_Z,
        initial_object_position_m=initial_object,
        initial_tcp_position_m=initial_tcp,
    )
    result.update(
        {
            "initial_tcp_position_m": initial_tcp.tolist(),
            "initial_tcp_orientation": initial_tcp_quaternion.tolist(),
            "initial_pepper_position_m": initial_object.tolist(),
            "initial_pepper_orientation": initial_object_quaternion.tolist(),
            "initial_pepper_to_tcp_transform": {
                "translation_m": (initial_object - initial_tcp).tolist(),
                "configured_translation_m": list(
                    config.place_initial.tcp_to_pepper_translation_m
                ),
                "configured_quaternion_wxyz": list(
                    config.place_initial.tcp_to_pepper_quaternion_wxyz
                ),
            },
            "initial_gripper_raw": float(runtime.spec["initial_gripper_raw"]),
            "initial_arm_hold_target": initial_arm_target.tolist(),
            "initialization_frames_recorded": 0,
            "object_identity": runtime.active_target_body,
            "released_object_identity": runtime.target_body,
            "release_uses_held_body_swap": True,
            "permanent_attachment": False,
        }
    )
    return result, diagnostic_frames


def _controller(environment: MuJoCoEnvironment, config: PipelineConfig):
    runtime = environment.task_runtime
    assert runtime is not None
    if runtime.spec["success"]["type"] == "place_in_ring":
        return PlaceRedPepperOracleController(
            environment,
            PlaceOracleConfig(
                action_dt_s=config.place.action_dt_s,
                verify_steps=config.place.steps,
                ring_radius_m=config.place.ring_radius_m,
                maximum_height_above_table_m=config.place.maximum_height_above_table_m,
                maximum_final_speed_mps=config.place.maximum_final_speed_mps,
                velocity_fit_samples=config.place.velocity_fit_samples,
            ),
        )
    task = next(task for task in config.tasks if task.task_id == environment.task)
    base = oracle_config_for_task(
        environment.task,
        action_dt_s=config.pick.action_dt_s,
        closed_gripper_raw=task.closed_gripper_raw,
        grasp_tcp_offset_from_object_m=task.grasp_tcp_offset_from_object_m,
    )
    values = asdict(base)
    values.update(
        {
            "verify_steps": config.pick.steps,
            "verification_entry_lift_height_m": config.pick.entry_lift_height_m,
            "verification_minimum_lift_height_m": config.pick.minimum_lift_height_m,
            "maximum_relative_downward_slip_m": config.pick.maximum_relative_downward_slip_m,
            "maximum_final_relative_downward_slip_m": (
                config.pick.maximum_final_relative_downward_slip_m
            ),
            "maximum_final_downward_speed_mps": config.pick.maximum_final_downward_speed_mps,
            "maximum_grasp_region_delta_m": config.pick.maximum_grasp_region_delta_m,
            "velocity_fit_samples": config.pick.velocity_fit_samples,
        }
    )
    return ScriptedOracleController(environment, OracleConfig(**values))


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
    place_initial: dict[str, Any] = {}
    if task.task_id == "place_red_pepper_in_ring":
        place_initial, initial_diagnostic_frames = _validate_place_initial_grasp(
            environment, config
        )
        if not place_initial["initial_grasp_success"]:
            staging_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "task_id": task.task_id,
                "task_prompt": task.prompt,
                "requested_episode_index": requested_episode_index,
                "base_seed": task.base_seed,
                "retry_index": retry_index,
                "resolved_seed": resolved_seed,
                "scene_variant": "clean",
                "success": False,
                "failure_reason": place_initial["initial_grasp_failure_reason"],
                "validation": {"place_initial_grasp": place_initial},
            }
            if requested_episode_index == 0 and retry_index == 0:
                failure["diagnostic_visuals"] = write_diagnostic_visuals(
                    staging_dir, initial_diagnostic_frames
                )
            atomic_write_json(staging_dir / "failure.json", failure)
            return False, failure

    controller = _controller(environment, config)
    recorder = RealRawEpisodeRecorder(
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
        environment=environment,
        save_hz=config.action_hz,
    )
    runtime = environment.task_runtime
    assert runtime is not None
    metrics = runtime.metrics()
    last_action = None
    while not controller.terminal:
        action = controller.next_action()
        if action is None:
            break
        recorder.record_observation(gripper_target=float(action[6]))
        environment.apply_action(action)
        environment.step_physics(1.0 / config.action_hz)
        metrics = update_task_success(environment)
        collision = environment.safety_diagnostics()["collision"]
        controller.notify_post_step(
            task_metrics=metrics,
            collision=collision,
            simulation_finite=simulation_is_finite(environment),
        )
        last_action = action
    recorder.record_observation(
        gripper_target=float(last_action[6]) if last_action is not None else None
    )
    stability = controller.stability_metadata()
    validation = (
        {"place_initial_grasp": place_initial, "stable_place": stability}
        if task.task_id == "place_red_pepper_in_ring"
        else {"stable_grasp": stability}
    )
    validation_success = bool(
        stability.get("stable_place_success")
        and place_initial.get("initial_grasp_success")
        and stability.get("release_detected")
        if task.task_id == "place_red_pepper_in_ring"
        else stability.get("stable_grasp_success")
    )
    success = accepted_oracle_episode(
        terminal_stage=controller.stage.value,
        task_metrics=metrics,
        failure_reason=controller.failure_reason,
        validation_success=validation_success,
    )
    meta = recorder.finalize(
        success=success,
        failure_reason=controller.failure_reason,
        initial_conditions=environment.initial_conditions,
        task_metrics=metrics,
        oracle_transitions=controller.transition_log(),
        oracle_plan=controller.plan.to_json(),
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
            }
            for task in config.tasks
        ],
        "total_target_episodes": 6 if smoke else config.total_episodes,
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
