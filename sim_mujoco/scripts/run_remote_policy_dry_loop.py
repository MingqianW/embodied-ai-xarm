from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policy_runtime.episode_logging import json_default, write_json
from policy_runtime.remote_policy_client import RemotePolicyClient, RemotePolicyConfig
from policy_runtime.safety import SafetyConfig, validate_action_chunk
from simulation.robot.control import (
    compute_safe_control_target,
    extract_first_action,
    validate_policy_actions,
)
from simulation.robot.gripper import actuator_ctrl_from_raw_hardware
from simulation.robot.model import arm_actuator_ctrl_limits
from simulation.robot.model import arm_joint_limits
from simulation.observation.policy import build_policy_observation
from policy_runtime.image_preprocessing import image_diagnostics
from simulation.runtime import initialize_scene
from simulation.runtime import load_simulation
from sim_mujoco.paths import mujoco_output_root


DEFAULT_OUTPUT_DIR = mujoco_output_root() / "remote_policy_dry_loop"


def hold_current_control(data, state: np.ndarray, config: dict[str, Any]) -> None:
    data.ctrl[:6] = state[:6]
    data.ctrl[6] = actuator_ctrl_from_raw_hardware(float(state[6]), config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    if args.iterations < 1:
        raise SystemExit("--iterations must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = RemotePolicyClient(
        RemotePolicyConfig(
            host=args.host,
            port=args.port,
            connect_timeout_s=args.timeout,
            inference_timeout_s=args.timeout,
        )
    )
    context = load_simulation()
    try:
        initialize_scene(context.model, context.data)
        for iteration in range(args.iterations):
            iteration_dir = args.output_dir / f"iteration_{iteration:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)

            observation = build_policy_observation(
                context.model,
                context.data,
                context.renderer,
                context.config,
                args.prompt,
            )
            state = np.asarray(observation["observation/state"], dtype=np.float32)

            Image.fromarray(observation["observation/image"]).save(iteration_dir / "base.png")
            Image.fromarray(observation["observation/wrist_image"]).save(iteration_dir / "wrist.png")
            write_json(
                iteration_dir / "observation.json",
                {
                    "prompt": observation["prompt"],
                    "state": state,
                    "base_image": image_diagnostics(observation["observation/image"]),
                    "wrist_image": image_diagnostics(observation["observation/wrist_image"]),
                },
            )

            start = time.perf_counter()
            result = policy.infer(observation)
            latency = time.perf_counter() - start
            actions = validate_policy_actions(np.asarray(result["actions"], dtype=np.float32))
            np.save(iteration_dir / "actions.npy", actions)
            model_limits = arm_joint_limits(context.model)
            actuator_limits = arm_actuator_ctrl_limits(context.model)
            limits = np.column_stack(
                (
                    np.maximum(model_limits[:, 0], actuator_limits[:, 0]),
                    np.minimum(model_limits[:, 1], actuator_limits[:, 1]),
                )
            )
            chunk_safety = validate_action_chunk(
                actions,
                state,
                limits,
                SafetyConfig(max_joint_delta_rad=0.05, reject_if_clip_exceeds_rad=0.25),
            )
            if not chunk_safety.accepted:
                hold_current_control(context.data, state, context.config)
                raise RuntimeError(f"Rejected action chunk: {chunk_safety.reason}")

            first_action = extract_first_action(actions)
            safe_target = compute_safe_control_target(
                context.model,
                context.data,
                context.config,
                first_action,
            )
            diagnostics = {
                "iteration": iteration,
                "current_state": state,
                "raw_first_action": first_action,
                "safe_target": safe_target.to_json(),
                "chunk_safety": chunk_safety.to_json(),
                "policy_timing": result.get("policy_timing"),
                "server_timing": result.get("server_timing"),
                "total_client_latency_seconds": latency,
                "response_keys": list(result.keys()),
            }
            write_json(iteration_dir / "diagnostics.json", diagnostics)

            print(f"[dry {iteration + 1}/{args.iterations}]")
            print("  current state:", state)
            print("  raw first action:", first_action)
            print("  clamped target:", safe_target.ctrl_target)
            print("  joint deltas:", safe_target.joint_delta_clamped)
            print("  gripper target raw/ctrl:", safe_target.gripper_raw_clamped, safe_target.gripper_ctrl_target)
            print(f"  latency: {latency:.3f} s")
            if safe_target.clip_messages:
                print("  clipping:")
                for message in safe_target.clip_messages:
                    print("   ", message)

            hold_current_control(context.data, state, context.config)
            for _ in range(max(1, int(0.02 / context.model.opt.timestep))):
                import mujoco

                mujoco.mj_step(context.model, context.data)
    finally:
        policy.close()
        context.close()


if __name__ == "__main__":
    main()
from sim_mujoco.paths import mujoco_output_root
