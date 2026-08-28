from __future__ import annotations

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
from simulation.robot.control import validate_policy_actions
from simulation.resources import DEFAULT_MODEL_PATH
from simulation.robot.model import arm_joint_limits
from simulation.observation.policy import build_policy_observation
from policy_runtime.image_preprocessing import image_diagnostics
from simulation.runtime import initialize_scene
from simulation.runtime import load_simulation
from sim_mujoco.paths import mujoco_output_root


OUTPUT_DIR = mujoco_output_root() / "remote_policy_test"
PROMPT = "pick up the object"


def save_result(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def image_check(name: str, image: np.ndarray) -> tuple[bool, str]:
    diagnostics = image_diagnostics(image)
    if diagnostics["max"] == 0:
        return False, f"{name}: all black"
    if diagnostics["std"] < 2.0:
        return False, f"{name}: nearly uniform std={diagnostics['std']:.3f}"
    if diagnostics["max"] <= diagnostics["min"]:
        return False, f"{name}: empty dynamic range"
    return True, f"{name}: non-empty std={diagnostics['std']:.3f}"


def write_validation_report(
    path: Path,
    observation: dict[str, Any],
    actions: np.ndarray,
    model,
) -> list[str]:
    lines: list[str] = []

    def add(label: str, passed: bool, detail: str) -> None:
        status = "PASS" if passed else "FAIL"
        lines.append(f"{status} {label}: {detail}")

    base_ok, base_detail = image_check("base_image", observation["observation/image"])
    wrist_ok, wrist_detail = image_check("wrist_image", observation["observation/wrist_image"])
    add("base image not empty/uniform/black", base_ok, base_detail)
    add("wrist image not empty/uniform/black", wrist_ok, wrist_detail)

    state = np.asarray(observation["observation/state"], dtype=np.float32)
    add("state finite", bool(np.isfinite(state).all()), state.tolist().__repr__())
    limits = arm_joint_limits(model)
    in_limits = bool(np.all(state[:6] >= limits[:, 0]) and np.all(state[:6] <= limits[:, 1]))
    add("state arm joints inside MuJoCo joint limits", in_limits, f"limits={limits.tolist()}")
    gripper_ok = bool(50.0 <= float(state[6]) <= 845.0)
    add("state gripper_raw inside training range", gripper_ok, f"gripper_raw={float(state[6]):.6f}")

    add("actions finite", bool(np.isfinite(actions).all()), f"shape={actions.shape}")
    target_limits_ok = bool(np.all(actions[:, :6] >= limits[:, 0] - 0.25) and np.all(actions[:, :6] <= limits[:, 1] + 0.25))
    add("predicted joint targets inside or near valid ranges", target_limits_ok, f"min={actions[:, :6].min(axis=0).tolist()} max={actions[:, :6].max(axis=0).tolist()}")
    gripper_near = bool(np.all(actions[:, 6] >= 0.0) and np.all(actions[:, 6] <= 895.0))
    add("predicted gripper within or near training range", gripper_near, f"min={float(actions[:, 6].min()):.6f} max={float(actions[:, 6].max()):.6f}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    context = load_simulation()
    try:
        initialize_scene(context.model, context.data)
        observation = build_policy_observation(
            context.model,
            context.data,
            context.renderer,
            context.config,
            PROMPT,
        )

        Image.fromarray(observation["observation/image"]).save(output_dir / "base_model_input.png")
        Image.fromarray(observation["observation/wrist_image"]).save(output_dir / "wrist_model_input.png")

        policy = RemotePolicyClient(
            RemotePolicyConfig(host="127.0.0.1", port=18000)
        )

        start = time.perf_counter()
        result = policy.infer(observation)
        total_latency = time.perf_counter() - start
        actions = validate_policy_actions(np.asarray(result["actions"], dtype=np.float32))

        state = observation["observation/state"]
        first_action = actions[0]
        payload = {
            "scene_path": context.model_path,
            "camera_config_path": context.camera_config_path,
            "observation": {
                "keys": list(observation.keys()),
                "base_image": image_diagnostics(observation["observation/image"]),
                "wrist_image": image_diagnostics(observation["observation/wrist_image"]),
                "state": state,
                "prompt": observation["prompt"],
            },
            "response_keys": list(result.keys()),
            "actions": actions,
            "first_action": first_action,
            "first_action_arm_target": first_action[:6],
            "first_action_gripper_target": float(first_action[6]),
            "first_action_joint_delta_relative_to_current_qpos": first_action[:6] - state[:6],
            "policy_timing": result.get("policy_timing"),
            "server_timing": result.get("server_timing"),
            "total_client_latency_seconds": total_latency,
        }
        save_result(output_dir / "result.json", payload)
        report_lines = write_validation_report(output_dir / "validation_report.txt", observation, actions, context.model)

        print("scene path:", DEFAULT_MODEL_PATH)
        print("observation keys:", list(observation.keys()))
        for key in ("observation/image", "observation/wrist_image"):
            image = observation[key]
            print(key, "shape:", image.shape)
            print(key, "dtype:", image.dtype)
            print(key, "min/max:", int(image.min()), int(image.max()))
        print("state shape:", state.shape)
        print("state dtype:", state.dtype)
        print("full state:", state)
        print("response keys:", list(result.keys()))
        print("action shape:", actions.shape)
        print("action dtype:", actions.dtype)
        print("all 10 actions:")
        print(actions)
        print("first action:", first_action)
        print("first-action arm target:", first_action[:6])
        print("first-action gripper target:", float(first_action[6]))
        print("first-action joint delta relative to current qpos:", first_action[:6] - state[:6])
        print("policy_timing:", result.get("policy_timing"))
        print("server_timing:", result.get("server_timing"))
        print(f"total client latency: {total_latency:.3f} s")
        print("validation report:")
        for line in report_lines:
            print(" ", line)
        policy.close()
    finally:
        context.close()


if __name__ == "__main__":
    main()
