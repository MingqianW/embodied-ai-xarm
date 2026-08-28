"""Compute-node diagnostic for the canonical Place initial-grasp validator."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_mujoco.data_generation.collection import (  # noqa: E402
    _validate_place_initial_grasp,
)
from sim_mujoco.data_generation.config import load_pipeline_config  # noqa: E402
from simulation.environment import MuJoCoEnvironment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "sim_mujoco/config/data_generation/clean_multitask_stable_v3.yaml"
        ),
    )
    parser.add_argument("--seed", type=int, default=600000)
    parser.add_argument("--settle-steps", type=int, required=True)
    parser.add_argument("--gripper-raw", type=float)
    parser.add_argument("--tcp-to-pepper-z-m", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.settle_steps < 0:
        raise SystemExit("--settle-steps must be nonnegative")

    config = load_pipeline_config(args.config)
    task_scene_config = config.task_scene_config
    if args.gripper_raw is not None or args.tcp_to_pepper_z_m is not None:
        task_scene_data = yaml.safe_load(
            config.task_scene_config.read_text(encoding="utf-8")
        )
        place_scene = task_scene_data["tasks"]["place_red_pepper_in_ring"]
        if args.gripper_raw is not None:
            place_scene["initial_gripper_raw"] = float(args.gripper_raw)
            config = replace(
                config,
                place_initial=replace(
                    config.place_initial,
                    gripper_raw=float(args.gripper_raw),
                ),
            )
        if args.tcp_to_pepper_z_m is not None:
            translation = list(place_scene["initial_tcp_to_object"]["translation_m"])
            translation[2] = float(args.tcp_to_pepper_z_m)
            place_scene["initial_tcp_to_object"]["translation_m"] = translation
            config = replace(
                config,
                place_initial=replace(
                    config.place_initial,
                    tcp_to_pepper_translation_m=tuple(translation),
                ),
            )
        task_scene_config = args.output.with_suffix(".task_scenes.yaml")
        task_scene_config.parent.mkdir(parents=True, exist_ok=True)
        task_scene_config.write_text(
            yaml.safe_dump(task_scene_data, sort_keys=False), encoding="utf-8"
        )
    task = next(
        task
        for task in config.tasks
        if task.task_id == "place_red_pepper_in_ring"
    )
    with MuJoCoEnvironment(
        task=task.task_id,
        prompt=task.prompt,
        camera_config_path=config.camera_config,
        task_scene_config_path=task_scene_config,
        settle_steps=args.settle_steps,
        object_xy_range=config.object_xy_range_m,
        object_yaw_range_deg=config.object_yaw_range_deg,
        joint_noise=config.joint_noise_rad,
        scene_variant="clean",
    ) as environment:
        environment.reset(seed=args.seed)
        result, _ = _validate_place_initial_grasp(environment, config)

    payload = {
        "seed": args.seed,
        "settle_steps": args.settle_steps,
        "settle_duration_s": args.settle_steps * 0.002,
        "gripper_raw": args.gripper_raw,
        "tcp_to_pepper_z_m": args.tcp_to_pepper_z_m,
        "resolved_task_scene_config": str(task_scene_config.resolve()),
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
