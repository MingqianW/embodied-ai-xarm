from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any


from policy_runtime.recording import VideoRecorder  # noqa: E402
from data.sim.generation.oracle import (  # noqa: E402
    OracleConfig,
    ScriptedOracleController,
)
from data.sim.generation.acceptance import (  # noqa: E402
    accepted_oracle_episode,
    simulation_is_finite,
    update_task_success,
)
from simulation.environment import MuJoCoEnvironment  # noqa: E402
from simulation.resources import output_root  # noqa: E402


DEFAULT_OUTPUT = output_root() / "scripted_oracle_test"


def run_one_episode(
    environment: MuJoCoEnvironment,
    *,
    seed: int,
    output_dir: Path,
    record_video: bool,
    closed_gripper_raw: float | None,
    grasp_tcp_offset_from_object_m: float | None,
) -> dict[str, Any]:
    environment.reset(seed=seed)
    controller = ScriptedOracleController(
        environment,
        OracleConfig(
            task=environment.task,
            action_dt_s=0.1,
            **({"closed_gripper_raw": closed_gripper_raw} if closed_gripper_raw is not None else {}),
            **({"grasp_tcp_offset_from_object_m": grasp_tcp_offset_from_object_m} if grasp_tcp_offset_from_object_m is not None else {}),
        ),
    )
    recorder = (
        VideoRecorder(output_dir=output_dir, fps=10)
        if record_video
        else None
    )
    task_metrics = environment.task_runtime.metrics()
    max_lift_height_m = float(task_metrics.get("lift_height_m") or 0.0)
    max_success_streak = int(task_metrics.get("success_streak") or 0)
    try:
        if recorder is not None:
            recorder.write(environment.recording_frames())
        while not controller.terminal:
            action = controller.next_action()
            if action is None:
                break
            environment.apply_action(action)
            environment.step_physics(controller.config.action_dt_s)
            task_metrics = update_task_success(environment)
            max_lift_height_m = max(
                max_lift_height_m,
                float(task_metrics.get("lift_height_m") or 0.0),
            )
            max_success_streak = max(
                max_success_streak,
                int(task_metrics.get("success_streak") or 0),
            )
            collision = environment.safety_diagnostics()["collision"]
            controller.notify_post_step(
                task_metrics=task_metrics,
                collision=collision,
                simulation_finite=simulation_is_finite(environment),
            )
            if recorder is not None:
                recorder.write(environment.recording_frames())
    finally:
        if recorder is not None:
            recorder.close()

    failure_reason = controller.failure_reason
    stability = controller.stability_metadata()
    success = accepted_oracle_episode(
        terminal_stage=controller.stage.value,
        task_metrics=task_metrics,
        failure_reason=failure_reason,
        validation_success=bool(stability.get("stable_grasp_success")),
    )
    result = {
        "seed": seed,
        "success": success,
        "terminal_stage": controller.stage.value,
        "failure_reason": failure_reason,
        "action_steps": controller.action_steps,
        "simulation_time_s": float(environment.context.data.time),
        "task_metrics": task_metrics,
        "max_lift_height_m": max_lift_height_m,
        "max_success_streak": max_success_streak,
        "transitions": controller.transition_log(),
        "plan": controller.plan.to_json(),
        "oracle_config": asdict(controller.config),
        "stable_grasp": stability,
    }
    if recorder is not None:
        result["video"] = recorder.metadata()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="red_block")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--object-xy-range", type=float, default=0.0)
    parser.add_argument("--object-yaw-range-deg", type=float, default=0.0)
    parser.add_argument("--joint-noise", type=float, default=0.0)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--closed-gripper-raw", type=float)
    parser.add_argument("--grasp-tcp-offset-from-object-m", type=float)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.episodes < 1:
        raise SystemExit("--episodes must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with MuJoCoEnvironment(
        task=args.task,
        object_xy_range=args.object_xy_range,
        object_yaw_range_deg=args.object_yaw_range_deg,
        joint_noise=args.joint_noise,
    ) as environment:
        for episode_offset in range(args.episodes):
            seed = args.seed_start + episode_offset * args.seed_stride
            episode_dir = (
                args.output_dir / f"episode_{episode_offset:06d}_seed_{seed}"
            )
            result = run_one_episode(
                environment,
                seed=seed,
                output_dir=episode_dir,
                record_video=args.record_video,
                closed_gripper_raw=args.closed_gripper_raw,
                grasp_tcp_offset_from_object_m=(
                    args.grasp_tcp_offset_from_object_m
                ),
            )
            results.append(result)
            print(
                f"[{episode_offset + 1}/{args.episodes}] seed={seed} "
                f"success={result['success']} "
                f"stage={result['terminal_stage']} "
                f"steps={result['action_steps']} "
                f"failure={result['failure_reason']}"
            )

    successes = sum(bool(result["success"]) for result in results)
    summary = {
        "task": args.task,
        "episodes": args.episodes,
        "successes": successes,
        "failures": args.episodes - successes,
        "success_rate": successes / args.episodes,
        "seed_start": args.seed_start,
        "seed_stride": args.seed_stride,
        "object_xy_range": args.object_xy_range,
        "object_yaw_range_deg": args.object_yaw_range_deg,
        "joint_noise": args.joint_noise,
        "closed_gripper_raw": args.closed_gripper_raw,
        "grasp_tcp_offset_from_object_m": (
            args.grasp_tcp_offset_from_object_m
        ),
        "max_lift_height_m": max(
            float(result["max_lift_height_m"]) for result in results
        ),
        "max_success_streak": max(
            int(result["max_success_streak"]) for result in results
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))

    randomized = any(
        value > 0.0
        for value in (
            args.object_xy_range,
            args.object_yaw_range_deg,
            args.joint_noise,
        )
    )
    required = (
        math.ceil(0.9 * args.episodes)
        if randomized
        else args.episodes
    )
    if successes < required:
        raise SystemExit(
            f"Oracle gate failed: {successes}/{args.episodes}; "
            f"required at least {required}"
        )


if __name__ == "__main__":
    main()
