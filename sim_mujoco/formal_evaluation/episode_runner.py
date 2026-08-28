"""Closed-loop formal episode execution with explicit validity accounting."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np
from policy_runtime.remote_policy_client import PolicyConnectionError
from policy_runtime.remote_policy_client import PolicyTimeoutError
from policy_runtime.remote_policy_client import RemotePolicyClient
from policy_runtime.safety import SafetyConfig
from policy_runtime.safety import validate_action_chunk

from sim_mujoco.collision import collision_diagnostics
from sim_mujoco.formal_evaluation.config import FormalProtocol
from sim_mujoco.formal_evaluation.config import TaskSpec
from sim_mujoco.formal_evaluation.failure_diagnosis import diagnose_episode_failure
from sim_mujoco.formal_evaluation.models import ModelSpec
from sim_mujoco.formal_evaluation.outputs import EPISODE_SCHEMA_VERSION
from sim_mujoco.formal_evaluation.outputs import validate_episode_result
from sim_mujoco.formal_evaluation.outputs import write_json
from sim_mujoco.formal_evaluation.representative_videos import retain_video_bundle
from sim_mujoco.formal_evaluation.representative_videos import (
    unrecorded_video_artifacts,
)
from sim_mujoco.formal_evaluation.rng import policy_rng_seed
from sim_mujoco.formal_evaluation.slip_trace import SlipTraceRecorder
from sim_mujoco.formal_evaluation.slip_trace import SlipTraceSettings
from sim_mujoco.gripper_slip_diagnostics import CommandContext
from sim_mujoco.gripper_slip_diagnostics import PhysicsTraceRecorder
from sim_mujoco.gripper_slip_diagnostics import reconstruct_network_action
from sim_mujoco.formal_evaluation.success import FormalTaskEvaluator
from sim_mujoco.formal_evaluation.success import validate_initial_place_grasp
from sim_mujoco.remote_policy_control import apply_safe_control_target
from sim_mujoco.remote_policy_control import compute_safe_control_target
from sim_mujoco.remote_policy_evaluation import VideoRecorder
from sim_mujoco.remote_policy_observation import arm_actuator_ctrl_limits
from sim_mujoco.remote_policy_observation import arm_joint_limits
from sim_mujoco.remote_policy_observation import build_openpi_observation
from sim_mujoco.remote_policy_observation import get_robot_state
from sim_mujoco.remote_policy_observation import gripper_raw_to_ctrl
from sim_mujoco.remote_policy_observation import initialize_scene
from sim_mujoco.remote_policy_observation import load_simulation
from sim_mujoco.task_scenes import configure_task_scene


@dataclass(frozen=True)
class EpisodeRequest:
    protocol: FormalProtocol
    model: ModelSpec
    provenance: dict[str, Any]
    task: TaskSpec
    seed: int
    output_dir: Path
    model_output_dir: Path
    record_video: bool


def _invalid_reason_for_action_error(exc: ValueError) -> str:
    message = str(exc).lower()
    return (
        "nonfinite_action"
        if "nan" in message or "inf" in message
        else "invalid_action_shape"
    )


def _canonical_limits(model: mujoco.MjModel) -> np.ndarray:
    model_limits = arm_joint_limits(model)
    actuator_limits = arm_actuator_ctrl_limits(model)
    return np.column_stack(
        (
            np.maximum(model_limits[:, 0], actuator_limits[:, 0]),
            np.minimum(model_limits[:, 1], actuator_limits[:, 1]),
        )
    )


def validate_formal_action_chunk(
    actions: np.ndarray,
    *,
    current_state: np.ndarray,
    joint_limits: np.ndarray,
    protocol: FormalProtocol,
) -> tuple[np.ndarray, Any]:
    """Validate all model outputs but safety-check only the executed prefix."""

    from policy_runtime.action_decoder import validate_policy_actions

    full_chunk = validate_policy_actions(
        np.asarray(actions, dtype=np.float32),
        action_horizon=protocol.policy_action_horizon,
        action_dim=7,
    )
    prefix_safety = validate_action_chunk(
        full_chunk[: protocol.execute_chunk_steps],
        current_state,
        joint_limits,
        SafetyConfig(max_joint_delta_rad=0.05, reject_if_clip_exceeds_rad=0.25),
    )
    return full_chunk, prefix_safety


def record_executed_target_clipping(safety: dict[str, Any], target: Any) -> None:
    """Account for every actually executed target, not only chunk action zero."""

    safety["executed_action_count"] += 1
    if not target.clipped:
        return
    safety["clipped_action_count"] += 1
    raw = np.asarray(target.raw_action, dtype=np.float64)
    safety["max_requested_vs_executed_delta"] = max(
        float(safety["max_requested_vs_executed_delta"]),
        float(np.max(np.abs(raw[:6] - target.arm_target_clamped))),
        abs(float(target.gripper_raw - target.gripper_raw_clamped)),
    )
    for component in range(6):
        if not np.isclose(raw[component], target.arm_target_clamped[component]):
            safety["per_dimension_clip_counts"][component] += 1
    if not np.isclose(target.gripper_raw, target.gripper_raw_clamped):
        safety["per_dimension_clip_counts"][6] += 1


def _action_quantiles(model: ModelSpec) -> tuple[np.ndarray, np.ndarray]:
    path = model.manager_directory / "assets" / model.norm_asset_id / "norm_stats.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    actions = value["norm_stats"]["actions"]
    q01 = np.asarray(actions["q01"], dtype=np.float64)
    q99 = np.asarray(actions["q99"], dtype=np.float64)
    if q01.shape != (7,) or q99.shape != (7,):
        raise ValueError(f"Expected 7D action quantiles in {path}")
    return q01, q99


def _bilateral_target_contact(collision: dict[str, Any], target_body: str) -> bool:
    left = False
    right = False
    for contact in collision.get("contacts") or ():
        bodies = {contact.get("body1"), contact.get("body2")}
        if target_body not in bodies:
            continue
        left = left or "left_finger" in bodies
        right = right or "right_finger" in bodies
    return left and right


def _apply_diagnostic_latch(
    *,
    target: Any,
    latch_raw: float,
    runtime: Any,
    context: Any,
) -> None:
    target.gripper_raw_clamped = float(latch_raw)
    physical_raw = runtime.physical_gripper_raw_target(float(latch_raw))
    target.gripper_ctrl_target = gripper_raw_to_ctrl(
        physical_raw,
        context.config,
    )
    target.ctrl_target[6] = target.gripper_ctrl_target
    context.data.ctrl[6] = target.gripper_ctrl_target


def _episode_result(
    *,
    request: EpisodeRequest,
    success: bool,
    valid: bool,
    termination_reason: str,
    invalid_reason: str | None,
    policy_steps: int,
    executed_actions: int,
    metrics: dict[str, Any],
    safety: dict[str, Any],
    initial_state: dict[str, Any],
    final_state: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    diagnosis = diagnose_episode_failure(
        task_id=request.task.task_id,
        success=success,
        valid=valid,
        metrics=metrics,
    )
    return {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "evaluation_protocol_version": request.protocol.protocol_version,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": request.model.to_json(),
        "episode": {
            "task": request.task.task_id,
            "prompt": request.task.prompt,
            "seed": int(request.seed),
            "success": bool(success),
            "valid": bool(valid),
            "termination_reason": termination_reason,
            "invalid_reason": invalid_reason,
            "failure_category": diagnosis.category,
            "failure_reason": diagnosis.reason,
            "failure_stage": diagnosis.stage,
            "policy_steps": int(policy_steps),
            "executed_actions": int(executed_actions),
        },
        "metrics": metrics,
        "failure_diagnostics": diagnosis.diagnostics,
        "safety": safety,
        "initial_state": initial_state,
        "final_state": final_state,
        "provenance": request.provenance,
        "artifacts": artifacts,
    }


def run_formal_episode(
    request: EpisodeRequest, *, policy: RemotePolicyClient
) -> dict[str, Any]:
    """Run one task/seed with fresh MuJoCo state and request-scoped RNG keys."""

    protocol = request.protocol
    context = None
    recorder: VideoRecorder | None = None
    slip_trace: SlipTraceRecorder | None = None
    physics_trace: PhysicsTraceRecorder | None = None
    action_q01: np.ndarray | None = None
    action_q99: np.ndarray | None = None
    slip_trace_settings = SlipTraceSettings.from_environment()
    temporary_video_metadata: dict[str, Any] | None = None
    initial_state: dict[str, Any] = {}
    final_state: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    safety = {
        "executed_action_count": 0,
        "clipped_action_count": 0,
        "clipped_action_fraction": 0.0,
        "per_dimension_clip_counts": [0] * 7,
        "max_requested_vs_executed_delta": 0.0,
        "invalid_action_count": 0,
        "policy_error_count": 0,
    }
    policy_steps = 0
    executed_actions = 0
    termination_reason = "max_policy_steps"
    invalid_reason: str | None = None
    scored_metrics: dict[str, Any] | None = None
    scored_safety: dict[str, Any] | None = None
    scored_policy_steps: int | None = None
    scored_executed_actions: int | None = None
    diagnostic_post_success_end_s: float | None = None
    diagnostic_stop_reason: str | None = None
    diagnostic_complete = False
    diagnostic_latch_engaged = False
    bilateral_contact_samples = 0
    started = time.perf_counter()

    try:
        context = load_simulation(protocol.robot_xml_path, protocol.camera_config_path)
        if not np.isclose(
            context.model.opt.timestep, protocol.expected_physics_timestep_s
        ):
            raise RuntimeError(
                f"Physics timestep {context.model.opt.timestep} differs from formal protocol "
                f"{protocol.expected_physics_timestep_s}"
            )
        initialize_scene(context.model, context.data, settle_steps=0)
        runtime, initial_conditions = configure_task_scene(
            context.model,
            context.data,
            task=request.task.task_id,
            seed=request.seed,
            object_xy_range=protocol.object_xy_range_m,
            object_yaw_range_deg=protocol.object_yaw_range_deg,
            joint_noise=protocol.joint_noise_rad,
            config_path=protocol.task_scene_config_path,
        )
        initial_state = dict(initial_conditions)
        if request.task.task_id == "place_red_pepper_in_ring":
            reset_validation = validate_initial_place_grasp(
                runtime=runtime,
                initial_conditions=initial_conditions,
                protocol=protocol,
            )
            initial_state["placement_reset_validation"] = reset_validation
            if not reset_validation["validated"]:
                termination_reason = "invalid_initial_placement_grasp"
                invalid_reason = "reset_validation_failed"
        evaluator = FormalTaskEvaluator(runtime, protocol, context.config)
        if slip_trace_settings.enabled:
            slip_trace = SlipTraceRecorder(
                output_dir=request.output_dir,
                target_body=runtime.active_target_body,
            )
            action_q01, action_q99 = _action_quantiles(request.model)
            physics_trace = PhysicsTraceRecorder(
                model=context.model,
                data=context.data,
                target_body=runtime.active_target_body,
                camera_config=context.config,
                initial_target_z_m=runtime.initial_target_z,
                trial={
                    "source": "learned_policy",
                    "model": request.model.to_json(),
                    "task": request.task.task_id,
                    "seed": request.seed,
                    "execute_chunk_steps": protocol.execute_chunk_steps,
                    "policy_action_horizon": protocol.policy_action_horizon,
                    "control_duration_s": protocol.control_duration_s,
                    "normalization_reconstruction": (
                        "inverse checkpoint quantile normalization after undoing "
                        "the absolute-arm output transform"
                    ),
                },
            )

        if request.record_video:
            recorder = VideoRecorder(request.output_dir / "temporary_video", fps=30)
            recorder.maybe_record(context)

        while invalid_reason is None and policy_steps < protocol.max_policy_steps:
            observation = build_openpi_observation(
                context.model,
                context.data,
                context.renderer,
                context.config,
                request.task.prompt,
            )
            # `adjust_observation` deliberately remains a no-op. Task geometry
            # is used only below for scoring/diagnostics and is never added here.
            rng_seed = policy_rng_seed(
                protocol_salt=protocol.rng_salt,
                task_id=request.task.task_id,
                evaluation_seed=request.seed,
                policy_step=policy_steps,
            )
            try:
                response = policy.infer(observation, rng_seed=rng_seed)
            except PolicyTimeoutError as exc:
                if scored_metrics is not None:
                    diagnostic_stop_reason = f"post_success_policy_timeout: {exc}"
                    break
                termination_reason, invalid_reason = "policy_timeout", "policy_timeout"
                safety["policy_error_count"] += 1
                safety["policy_error"] = str(exc)
                break
            except PolicyConnectionError as exc:
                if scored_metrics is not None:
                    diagnostic_stop_reason = f"post_success_server_error: {exc}"
                    break
                termination_reason, invalid_reason = (
                    "policy_connection_error",
                    "server_error",
                )
                safety["policy_error_count"] += 1
                safety["policy_error"] = str(exc)
                break
            except Exception as exc:
                if scored_metrics is not None:
                    diagnostic_stop_reason = f"post_success_policy_error: {exc!r}"
                    break
                termination_reason, invalid_reason = "policy_error", "policy_error"
                safety["policy_error_count"] += 1
                safety["policy_error"] = repr(exc)
                break

            try:
                actions, prefix_safety = validate_formal_action_chunk(
                    np.asarray(response["actions"], dtype=np.float32),
                    current_state=observation["observation/state"],
                    joint_limits=_canonical_limits(context.model),
                    protocol=protocol,
                )
            except (KeyError, TypeError, ValueError) as exc:
                if scored_metrics is not None:
                    diagnostic_stop_reason = (
                        f"post_success_invalid_policy_action: {exc}"
                    )
                    break
                termination_reason, invalid_reason = (
                    "invalid_policy_action",
                    _invalid_reason_for_action_error(ValueError(str(exc))),
                )
                safety["invalid_action_count"] += 1
                safety["invalid_action_error"] = str(exc)
                break

            # Full chunk shape/finite validation happened above. Safety clipping
            # and rejection intentionally apply only to actions actually run.
            executed_prefix = actions[: protocol.execute_chunk_steps]
            network_actions: np.ndarray | None = None
            if physics_trace is not None:
                assert action_q01 is not None and action_q99 is not None
                network_actions = np.stack(
                    [
                        reconstruct_network_action(
                            np.asarray(action, dtype=np.float64),
                            np.asarray(
                                observation["observation/state"], dtype=np.float64
                            ),
                            q01=action_q01,
                            q99=action_q99,
                        )
                        for action in actions
                    ]
                )
            if not prefix_safety.accepted:
                if scored_metrics is not None:
                    diagnostic_stop_reason = (
                        f"post_success_safety_rejection: {prefix_safety.reason}"
                    )
                    break
                termination_reason, invalid_reason = "unsafe_action", "safety_rejection"
                safety["invalid_action_count"] += 1
                safety["safety_rejection_reason"] = prefix_safety.reason
                break

            collision = collision_diagnostics(context.model, context.data)
            for action_index_in_chunk, action in enumerate(executed_prefix):
                target = compute_safe_control_target(
                    context.model,
                    context.data,
                    context.config,
                    action,
                    max_joint_step=0.05,
                    control_dt_s=protocol.control_duration_s,
                )
                runtime.release_if_requested(target.gripper_raw_clamped)
                physical_gripper_raw = runtime.physical_gripper_raw_target(
                    target.gripper_raw_clamped
                )
                target.gripper_ctrl_target = gripper_raw_to_ctrl(
                    physical_gripper_raw,
                    context.config,
                )
                target.ctrl_target[6] = target.gripper_ctrl_target
                executed_actions += 1
                record_executed_target_clipping(safety, target)
                if diagnostic_latch_engaged:
                    assert slip_trace_settings.diagnostic_latch_raw is not None
                    _apply_diagnostic_latch(
                        target=target,
                        latch_raw=slip_trace_settings.diagnostic_latch_raw,
                        runtime=runtime,
                        context=context,
                    )
                apply_safe_control_target(context.data, target)
                physics_steps = max(
                    1, round(protocol.control_duration_s / context.model.opt.timestep)
                )
                for _ in range(physics_steps):
                    mujoco.mj_step(context.model, context.data)
                    # Once the lift criterion is provisionally met, sample the
                    # target at physics cadence so a within-chunk downward slip
                    # cannot be hidden by the next c5 boundary observation.
                    evaluator.observe_post_success_hold()
                    if recorder is not None:
                        recorder.maybe_record(context)
                    collision = collision_diagnostics(context.model, context.data)
                    diagnostic_target_body = runtime.active_target_body
                    if slip_trace is not None:
                        slip_trace.set_target_body(diagnostic_target_body)
                    if physics_trace is not None:
                        physics_trace.set_target_body(diagnostic_target_body)
                    if _bilateral_target_contact(collision, diagnostic_target_body):
                        bilateral_contact_samples += 1
                    else:
                        bilateral_contact_samples = 0
                    if (
                        not diagnostic_latch_engaged
                        and slip_trace_settings.diagnostic_latch_raw is not None
                        and bilateral_contact_samples >= 5
                    ):
                        diagnostic_latch_engaged = True
                        _apply_diagnostic_latch(
                            target=target,
                            latch_raw=slip_trace_settings.diagnostic_latch_raw,
                            runtime=runtime,
                            context=context,
                        )
                    if slip_trace is not None:
                        slip_trace.sample(
                            model=context.model,
                            data=context.data,
                            camera_config=context.config,
                            policy_step=policy_steps,
                            executed_action_index=executed_actions - 1,
                            action_index_in_chunk=action_index_in_chunk,
                            gripper_raw_command=target.gripper_raw,
                            gripper_raw_command_clamped=target.gripper_raw_clamped,
                            gripper_ctrl_target=target.gripper_ctrl_target,
                            collision=collision,
                            original_v1_success_reached=scored_metrics is not None,
                            post_success_diagnostic=scored_metrics is not None,
                        )
                    if physics_trace is not None:
                        physics_trace.sample(
                            CommandContext(
                                source="learned_policy",
                                stage=(
                                    "POST_SUCCESS_DIAGNOSTIC"
                                    if scored_metrics is not None
                                    else "FORMAL_EVALUATION"
                                ),
                                action_step=executed_actions - 1,
                                inference_index=policy_steps,
                                chunk_index=policy_steps,
                                action_index_in_chunk=action_index_in_chunk,
                                gripper_network_normalized=(
                                    None
                                    if network_actions is None
                                    else float(
                                        network_actions[action_index_in_chunk, 6]
                                    )
                                ),
                                gripper_returned_raw=float(target.gripper_raw),
                                gripper_clamped_raw=float(target.gripper_raw_clamped),
                                gripper_ctrl=float(target.gripper_ctrl_target),
                                network_action=(
                                    None
                                    if network_actions is None
                                    else np.asarray(
                                        network_actions[action_index_in_chunk],
                                        dtype=np.float64,
                                    ).tolist()
                                ),
                                returned_action=np.asarray(
                                    action, dtype=np.float64
                                ).tolist(),
                                arm_target_clamped_rad=np.asarray(
                                    target.arm_target_clamped,
                                    dtype=np.float64,
                                ).tolist(),
                                ctrl_target=np.asarray(
                                    target.ctrl_target,
                                    dtype=np.float64,
                                ).tolist(),
                            )
                        )
                    if (
                        diagnostic_post_success_end_s is not None
                        and float(context.data.time) + 1e-12
                        >= diagnostic_post_success_end_s
                    ):
                        diagnostic_complete = True
                        diagnostic_stop_reason = "post_success_duration_complete"
                        break
                    if collision["forbidden"]:
                        break
                if collision["forbidden"] or diagnostic_complete:
                    break

            policy_steps += 1
            if scored_metrics is not None:
                if collision["forbidden"]:
                    diagnostic_stop_reason = str(collision["termination_reason"])
                    break
                if diagnostic_complete:
                    break
                continue
            if collision["forbidden"]:
                termination_reason, invalid_reason = (
                    str(collision["termination_reason"]),
                    "forbidden_collision",
                )
                break
            metrics = evaluator.update()
            metrics["last_policy_rng_seed"] = rng_seed
            if metrics["task_success"]:
                termination_reason = "task_success"
                if (
                    slip_trace is None
                    or slip_trace_settings.post_success_seconds <= 0.0
                ):
                    break
                # Freeze the exact original result at the first success point.
                # The following physics is diagnostic-only and cannot change
                # success, validity, termination, or safety accounting.
                scored_metrics = deepcopy(metrics)
                scored_safety = deepcopy(safety)
                scored_safety["clipped_action_fraction"] = (
                    scored_safety["clipped_action_count"]
                    / scored_safety["executed_action_count"]
                    if scored_safety["executed_action_count"]
                    else 0.0
                )
                scored_policy_steps = policy_steps
                scored_executed_actions = executed_actions
                diagnostic_post_success_end_s = (
                    float(context.data.time) + slip_trace_settings.post_success_seconds
                )

        if scored_metrics is not None and diagnostic_stop_reason is None:
            diagnostic_stop_reason = "post_success_max_policy_steps"

        safety["clipped_action_fraction"] = (
            safety["clipped_action_count"] / safety["executed_action_count"]
            if safety["executed_action_count"]
            else 0.0
        )
        if scored_metrics is not None:
            metrics = scored_metrics
            safety = scored_safety or safety
            policy_steps = scored_policy_steps or policy_steps
            executed_actions = scored_executed_actions or executed_actions
            termination_reason = "task_success"
            invalid_reason = None
        if context is not None:
            final_state = {
                "robot_state": get_robot_state(
                    context.model, context.data, context.config
                ).tolist(),
                "qpos": np.asarray(context.data.qpos, dtype=np.float64).tolist(),
                "sim_time_s": float(context.data.time),
                "wall_time_s": time.perf_counter() - started,
                "collision": collision_diagnostics(context.model, context.data),
            }
            if slip_trace is not None:
                final_state["slip_trace_diagnostic"] = {
                    "post_success_seconds_requested": slip_trace_settings.post_success_seconds,
                    "post_success_end_sim_time_s": diagnostic_post_success_end_s,
                    "stop_reason": diagnostic_stop_reason,
                    "physics_sample_count": len(slip_trace.rows),
                    "scientific_result_frozen_at_original_success": scored_metrics
                    is not None,
                    "diagnostic_latch_raw": slip_trace_settings.diagnostic_latch_raw,
                    "diagnostic_latch_engaged": diagnostic_latch_engaged,
                    "diagnostic_latch_bilateral_samples_required": 5,
                }
                if physics_trace is not None:
                    final_state["slip_trace_diagnostic"][
                        "detailed_physics_sample_count"
                    ] = len(physics_trace.rows)
    except (
        Exception
    ) as exc:  # Preserve partial state and make the episode visibly invalid.
        if scored_metrics is not None:
            # A diagnostic-only continuation failure cannot retroactively
            # alter the scientific outcome frozen at the original v1 success.
            diagnostic_stop_reason = f"post_success_environment_error: {exc!r}"
            metrics = scored_metrics
            safety = scored_safety or safety
            policy_steps = scored_policy_steps or policy_steps
            executed_actions = scored_executed_actions or executed_actions
            termination_reason, invalid_reason = "task_success", None
            artifacts["slip_trace_continuation_error"] = repr(exc)
        else:
            termination_reason, invalid_reason = (
                "environment_error",
                "environment_error",
            )
            safety["environment_error"] = repr(exc)
            safety["policy_error_count"] += 1
    finally:
        if recorder is not None:
            try:
                recorder.close()
                recorder.validate_outputs()
                temporary_video_metadata = recorder.metadata()
                artifacts.update(temporary_video_metadata)
            except Exception as exc:
                if scored_metrics is not None:
                    artifacts["post_success_video_error"] = repr(exc)
                else:
                    termination_reason, invalid_reason = (
                        "video_error",
                        "environment_error",
                    )
                    safety["video_error"] = repr(exc)
        if slip_trace is not None:
            try:
                trace_path = slip_trace.write()
                artifacts["slip_trace"] = {
                    "enabled": True,
                    "path": str(trace_path),
                    "physics_sample_count": len(slip_trace.rows),
                    "reference_defined": slip_trace.reference_relative_offset
                    is not None,
                    "post_success_seconds_requested": slip_trace_settings.post_success_seconds,
                    "diagnostic_stop_reason": diagnostic_stop_reason,
                }
            except Exception as exc:
                safety["slip_trace_write_error"] = repr(exc)
                artifacts["slip_trace"] = {
                    "enabled": True,
                    "status": "write_error",
                    "error": repr(exc),
                }
        if physics_trace is not None:
            try:
                paths = physics_trace.write(request.output_dir)
                artifacts["detailed_physics_trace"] = {
                    "enabled": True,
                    "paths": {key: str(value) for key, value in paths.items()},
                    "physics_sample_count": len(physics_trace.rows),
                    "event_count": len(physics_trace.events),
                    "network_output_semantics": (
                        "reconstructed pre-denormalization output; arm absolute transform "
                        "undone before inverse quantile normalization"
                    ),
                }
            except Exception as exc:
                safety["detailed_physics_trace_write_error"] = repr(exc)
                artifacts["detailed_physics_trace"] = {
                    "enabled": True,
                    "status": "write_error",
                    "error": repr(exc),
                }
        if context is not None:
            context.close()

    valid = invalid_reason is None
    success = bool(valid and metrics.get("task_success"))
    result = _episode_result(
        request=request,
        success=success,
        valid=valid,
        termination_reason=termination_reason,
        invalid_reason=invalid_reason,
        policy_steps=policy_steps,
        executed_actions=executed_actions,
        metrics=metrics,
        safety=safety,
        initial_state=initial_state,
        final_state=final_state,
        artifacts=artifacts,
    )
    validate_episode_result(result)
    result_path = request.output_dir / "result.json"
    # The first write makes the scientific result durable before any temporary
    # recording is moved or removed by the retention policy.
    write_json(result_path, result)
    if temporary_video_metadata is not None:
        try:
            result["artifacts"] = {
                **artifacts,
                **retain_video_bundle(
                    model_root=request.model_output_dir,
                    result_json_path=result_path,
                    result=result,
                    temporary_video_dir=request.output_dir / "temporary_video",
                    temporary_metadata=temporary_video_metadata,
                    video_policy=protocol.video_policy,
                ),
            }
        except Exception as exc:
            # Preserve the valid/invalid task outcome. A retention failure is
            # visible in artifacts and coverage validation rather than being
            # recast as a policy-performance result.
            result["artifacts"] = {
                **artifacts,
                "video_retention": {
                    "status": "temporary_video_finalization_error",
                    "video_policy": protocol.video_policy,
                },
                "video_finalization_error": repr(exc),
            }
    elif not request.record_video:
        result["artifacts"] = {
            **artifacts,
            **unrecorded_video_artifacts(
                result=result, video_policy=protocol.video_policy
            ),
        }
    validate_episode_result(result)
    write_json(result_path, result)
    return result
