from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from policy_runtime.action_decoder import action_prefix, decode_policy_response
from policy_runtime.environment_protocol import RobotEnvironment
from policy_runtime.episode_logging import EpisodeLogger
from policy_runtime.observation_builder import validate_policy_observation
from policy_runtime.remote_policy_client import PolicyTimeoutError
from policy_runtime.safety import SafetyConfig, validate_action_chunk


class PolicyClient(Protocol):
    last_inference_latency_s: float | None

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class DryLoopConfig:
    prompt: str
    iterations: int = 5
    output_dir: Path = Path("output/dry_loop")
    save_camera_debug: bool = False


@dataclass(frozen=True)
class ClosedLoopConfig:
    prompt: str
    max_policy_steps: int = 20
    execute_chunk_steps: int = 1
    control_period_s: float = 0.1
    output_dir: Path = Path("output/closed_loop")


def _with_prompt(observation: Any, prompt: str) -> Any:
    return observation if observation.prompt == prompt else replace(observation, prompt=prompt)


def run_dry_loop(
    environment: RobotEnvironment,
    policy: PolicyClient,
    config: DryLoopConfig,
    *,
    safety_config: SafetyConfig = SafetyConfig(),
) -> dict[str, Any]:
    if config.iterations < 1:
        raise ValueError("iterations must be at least 1")
    simulator = str(getattr(environment, "simulator_name", environment.__class__.__name__))
    logger = EpisodeLogger(config.output_dir, simulator=simulator)
    environment.reset()
    completed = 0
    for iteration in range(config.iterations):
        loop_started = time.perf_counter()
        observation = _with_prompt(environment.observe(), config.prompt)
        validate_policy_observation(observation)
        response = policy.infer(observation.as_openpi_dict())
        chunk = decode_policy_response(
            response,
            inference_latency_s=policy.last_inference_latency_s,
        )
        safety = validate_action_chunk(
            chunk.actions,
            observation.state,
            environment.joint_limits,
            safety_config,
        )
        logger.save_array(f"iteration_{iteration:03d}/state.npy", observation.state)
        if config.save_camera_debug:
            logger.save_image(
                f"iteration_{iteration:03d}/base_image.png", observation.base_image
            )
            logger.save_image(
                f"iteration_{iteration:03d}/wrist_image.png", observation.wrist_image
            )
        logger.save_array(f"iteration_{iteration:03d}/actions.npy", chunk.actions)
        logger.log(
            "dry_inference",
            iteration=iteration,
            state=observation.state,
            prompt=observation.prompt,
            image_shape=list(observation.base_image.shape),
            image_dtype=str(observation.base_image.dtype),
            base_image_range=[
                int(observation.base_image.min()),
                int(observation.base_image.max()),
            ],
            wrist_image_range=[
                int(observation.wrist_image.min()),
                int(observation.wrist_image.max()),
            ],
            color_order=observation.color_order,
            state_range=[
                float(observation.state.min()),
                float(observation.state.max()),
            ],
            action_shape=list(chunk.actions.shape),
            action_range=[float(chunk.actions.min()), float(chunk.actions.max())],
            actions_finite=bool(np.isfinite(chunk.actions).all()),
            safety=safety.to_json(),
            inference_latency_s=chunk.inference_latency_s,
            total_loop_latency_s=time.perf_counter() - loop_started,
            response_keys=sorted(response),
            raw_policy_response=response,
        )
        if not safety.accepted:
            logger.save_array(
                f"rejected_iteration_{iteration:03d}/state.npy",
                observation.state,
            )
            logger.save_image(
                f"rejected_iteration_{iteration:03d}/base_image.png",
                observation.base_image,
            )
            logger.save_image(
                f"rejected_iteration_{iteration:03d}/wrist_image.png",
                observation.wrist_image,
            )
            environment.hold_position()
            raise RuntimeError(f"Rejected policy action chunk: {safety.reason}")
        completed += 1
    result = {"termination_reason": "iterations_complete", "iterations": completed}
    logger.write_metadata(result)
    return result


def run_closed_loop(
    environment: RobotEnvironment,
    policy: PolicyClient,
    config: ClosedLoopConfig,
    *,
    safety_config: SafetyConfig = SafetyConfig(reject_if_clip_exceeds_rad=0.25),
    recorder: Any | None = None,
) -> dict[str, Any]:
    if config.max_policy_steps < 1:
        raise ValueError("max_policy_steps must be at least 1")
    if config.execute_chunk_steps < 1:
        raise ValueError("execute_chunk_steps must be at least 1")
    if config.control_period_s <= 0:
        raise ValueError("control_period_s must be positive")

    simulator = str(getattr(environment, "simulator_name", environment.__class__.__name__))
    logger = EpisodeLogger(config.output_dir, simulator=simulator)
    initial_observation = _with_prompt(environment.reset(), config.prompt)
    validate_policy_observation(initial_observation)
    logger.save_array("initial/state.npy", initial_observation.state)
    logger.save_image("initial/base_image.png", initial_observation.base_image)
    logger.save_image("initial/wrist_image.png", initial_observation.wrist_image)
    if recorder is not None:
        recorder.write(environment.recording_frames())
    termination = "max_policy_steps"
    completed = 0
    for step in range(config.max_policy_steps):
        observation = _with_prompt(environment.observe(), config.prompt)
        validate_policy_observation(observation)
        try:
            response = policy.infer(observation.as_openpi_dict())
        except PolicyTimeoutError as exc:
            environment.hold_position()
            logger.log("policy_timeout", step=step, error=str(exc))
            termination = "policy_timeout"
            break
        chunk = decode_policy_response(
            response,
            inference_latency_s=policy.last_inference_latency_s,
        )
        safety = validate_action_chunk(
            chunk.actions,
            observation.state,
            environment.joint_limits,
            safety_config,
        )
        logger.save_array(
            f"policy_step_{step:03d}/state_before.npy",
            observation.state,
        )
        logger.save_image(
            f"policy_step_{step:03d}/base_image_before.png",
            observation.base_image,
        )
        logger.save_image(
            f"policy_step_{step:03d}/wrist_image_before.png",
            observation.wrist_image,
        )
        logger.save_array(f"policy_step_{step:03d}/actions.npy", chunk.actions)
        logger.log(
            "policy_step",
            step=step,
            state=observation.state,
            safety=safety.to_json(),
            inference_latency_s=chunk.inference_latency_s,
        )
        if not safety.accepted:
            environment.hold_position()
            termination = "unsafe_action"
            break
        for action in action_prefix(safety.actions, config.execute_chunk_steps):
            environment.apply_action(action)
            environment.step_physics(config.control_period_s)
            if recorder is not None:
                environment.observe()
                recorder.write(environment.recording_frames())
            diagnostics = environment.safety_diagnostics()
            if diagnostics.get("real_time_factor_degraded"):
                logger.log(
                    "real_time_factor_warning",
                    step=step,
                    real_time_factor=diagnostics.get("real_time_factor"),
                )
            if not environment.is_safe():
                environment.hold_position()
                logger.log(
                    "environment_unsafe",
                    step=step,
                    diagnostics=diagnostics,
                )
                termination = "simulation_instability"
                break
        if termination == "simulation_instability":
            break
        completed += 1
    environment.hold_position()
    final_observation = _with_prompt(environment.observe(), config.prompt)
    validate_policy_observation(final_observation)
    logger.save_array("final/state.npy", final_observation.state)
    logger.save_image("final/base_image.png", final_observation.base_image)
    logger.save_image("final/wrist_image.png", final_observation.wrist_image)
    result = {
        "termination_reason": termination,
        "policy_steps": completed,
        "initial_state": initial_observation.state.tolist(),
        "final_state": final_observation.state.tolist(),
        "explicit_final_hold": True,
        "safety_diagnostics": environment.safety_diagnostics(),
    }
    if recorder is not None:
        recorder.close()
        result["recording"] = recorder.metadata()
    logger.write_metadata(result)
    return result
