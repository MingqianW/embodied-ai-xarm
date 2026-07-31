from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_runtime.config import load_yaml
from policy_runtime.evaluation import (
    EpisodeEvaluation,
    validate_label,
    write_evaluation_outputs,
)
from policy_runtime.episode_logging import write_json
from policy_runtime.remote_policy_client import (
    PolicyConnectionError,
    PolicyTimeoutError,
    RemotePolicyClient,
    RemotePolicyConfig,
)
from policy_runtime.runners import ClosedLoopConfig, run_closed_loop
from policy_runtime.safety import SafetyConfig
from sim_isaac.dependencies import IsaacDependencyError
from sim_isaac.environment import DEFAULT_CONFIG_DIR, IsaacEnvironment
from sim_isaac.recording import create_recorder
from sim_isaac.success_evaluation import evaluate_lift


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and label reproducible Isaac Sim policy evaluation episodes."
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-policy-steps", type=int, default=100)
    parser.add_argument("--execute-chunk-steps", type=int, choices=range(1, 6))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--prompt")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--scene-variant",
        choices=("auto", "clean", "distractors"),
        default="auto",
        help="Force clean or distractor scenes for exact dataset mixes.",
    )
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--inference-timeout", type=float)
    parser.add_argument("--task", default="pick_up_object")
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_CONFIG_DIR / "robot.yaml")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CONFIG_DIR / "cameras.yaml")
    parser.add_argument("--control-config", type=Path, default=DEFAULT_CONFIG_DIR / "control.yaml")
    parser.add_argument("--task-config", type=Path, default=DEFAULT_CONFIG_DIR / "tasks.yaml")
    return parser.parse_args()


def _human_label(
    automatic_success: bool, automatic_score: float
) -> tuple[str, float, str]:
    default = "success" if automatic_success else "failure"
    while True:
        try:
            raw = input(
                f"Label episode [success/failure/invalid] (default {default}): "
            ).strip()
        except EOFError:
            return "invalid", automatic_score, "interactive label unavailable (EOF)"
        try:
            label = validate_label(raw or default)
            break
        except ValueError as exc:
            print(exc)
    notes = ""
    score = 1.0 if label == "success" else automatic_score
    if label != "success":
        while True:
            try:
                raw_score = input(
                    f"Partial score 0-1 (default {automatic_score:.3f}): "
                ).strip()
            except EOFError:
                raw_score = ""
            try:
                score = automatic_score if not raw_score else float(raw_score)
                if not 0.0 <= score <= 1.0:
                    raise ValueError("score must be between 0 and 1")
                break
            except ValueError as exc:
                print(f"Invalid score: {exc}")
        try:
            notes = input("Failure/invalid notes (optional): ").strip()
        except EOFError:
            pass
    return label, score, notes


def main() -> int:
    args = parse_args()
    if args.episodes < 1:
        print("ERROR: --episodes must be at least 1", file=sys.stderr)
        return 2
    control = load_yaml(args.control_config)
    robot_config_snapshot = load_yaml(args.robot_config)
    camera_config_snapshot = load_yaml(args.camera_config)
    task_config_snapshot = load_yaml(args.task_config)
    policy_config = control["policy"]
    safety = control["safety"]
    recording = control["recording"]
    host = args.host or os.environ.get("OPENPI_POLICY_HOST") or str(policy_config["host"])
    port = int(
        args.port
        if args.port is not None
        else os.environ.get("OPENPI_POLICY_PORT", policy_config["port"])
    )
    prompt = args.prompt or str(policy_config["prompt"])
    timeout = float(args.inference_timeout or policy_config["timeout_s"])
    execute_steps = int(
        args.execute_chunk_steps or policy_config["execute_chunk_steps"]
    )
    output_dir = args.output_dir or Path(
        os.environ.get("ISAAC_OUTPUT_DIR", "sim_isaac/output")
    ) / "evaluation"
    checkpoint = args.checkpoint or os.environ.get("OPENPI_CHECKPOINT")
    rows: list[dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with IsaacEnvironment(
            robot_config_path=args.robot_config,
            camera_config_path=args.camera_config,
            control_config_path=args.control_config,
            task_config_path=args.task_config,
            task_name=args.task,
            prompt=prompt,
            headless=args.headless,
            seed=args.seed,
            scene_variant=args.scene_variant,
        ) as environment, RemotePolicyClient(
            RemotePolicyConfig(
                host=host,
                port=port,
                connect_timeout_s=args.connect_timeout,
                inference_timeout_s=timeout,
            )
        ) as policy:
            environment.require_safe("interactive evaluation")
            for episode_index in range(args.episodes):
                episode_seed = args.seed + episode_index
                environment.seed = episode_seed
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                episode_dir = (
                    output_dir / f"episode_{episode_index:04d}_{timestamp}"
                )
                recorder = (
                    create_recorder(
                        episode_dir / "recording",
                        fps=int(recording["fps"]),
                        max_frames=int(recording["max_frames"]),
                        fallback_to_frames=bool(
                            recording["fallback_to_frame_directory"]
                        ),
                    )
                    if args.record
                    else None
                )
                started_wall = time.perf_counter()
                started_iso = datetime.now(timezone.utc).isoformat()
                try:
                    result = run_closed_loop(
                        environment,
                        policy,
                        ClosedLoopConfig(
                            prompt=prompt,
                            max_policy_steps=args.max_policy_steps,
                            execute_chunk_steps=execute_steps,
                            control_period_s=1.0 / float(
                                policy_config["control_hz"]
                            ),
                            output_dir=episode_dir,
                        ),
                        safety_config=SafetyConfig(
                            action_mode=str(policy_config["action_mode"]),
                            max_joint_delta_rad=float(
                                safety["max_joint_delta_rad"]
                            ),
                            reject_if_clip_exceeds_rad=float(
                                safety["reject_if_clip_exceeds_rad"]
                            ),
                        ),
                        recorder=recorder,
                    )
                except KeyboardInterrupt:
                    environment.hold_position()
                    recording_metadata: dict[str, object] = {}
                    if recorder is not None:
                        recorder.close()
                        recording_metadata = recorder.metadata()
                    result = {
                        "termination_reason": "user_interrupt",
                        "policy_steps": 0,
                        "safety_diagnostics": environment.safety_diagnostics(),
                    }
                    if recording_metadata:
                        result["recording"] = recording_metadata
                    write_json(episode_dir / "episode.json", result)
                    print("Episode stopped by operator; continuing to labeling.")
                lift = evaluate_lift(
                    environment.scene.objects.position(),
                    environment.scene.objects.last_position_m,
                    target_lift_m=environment.task.success_lift_height_m,
                    partial_credit_lift_m=environment.task.partial_credit_height_m,
                )
                if args.non_interactive:
                    label = "success" if lift.success else "failure"
                    assigned_score = lift.score
                    notes = "automatic lift-threshold label"
                else:
                    label, assigned_score, notes = _human_label(
                        lift.success, lift.score
                    )
                recording_result = result.get("recording", {})
                evaluation = EpisodeEvaluation(
                    simulator="isaac",
                    task=args.task,
                    prompt=prompt,
                    seed=episode_seed,
                    checkpoint=checkpoint,
                    policy_server=f"ws://{host}:{port}",
                    start_time=started_iso,
                    duration_s=time.perf_counter() - started_wall,
                    success=lift.success,
                    score=assigned_score,
                    failure_reason=(
                        None if lift.success else str(result["termination_reason"])
                    ),
                    notes=notes,
                    video_path=(
                        str(recording_result.get("video_path"))
                        if recording_result
                        else None
                    ),
                    config_snapshot={
                        "control": control,
                        "robot": robot_config_snapshot,
                        "cameras": camera_config_snapshot,
                        "tasks": task_config_snapshot,
                        "camera_backend": environment.cameras.backend,
                    },
                    label=label,
                )
                row = evaluation.to_json()
                row.update(
                    {
                        "episode_index": episode_index,
                        "termination_reason": result["termination_reason"],
                        "policy_steps": result["policy_steps"],
                        "sim_time": environment.safety_diagnostics().get(
                            "simulation_time_s"
                        ),
                        "wall_time": evaluation.duration_s,
                        "lift_height_m": lift.lift_height_m,
                    }
                )
                write_json(episode_dir / "evaluation.json", row)
                # CSV fields must remain scalar; the complete config is in per-episode JSON.
                row["config_snapshot"] = str(row["config_snapshot"])
                rows.append(row)
                print(
                    f"episode={episode_index} label={label} "
                    f"score={assigned_score:.3f} termination={result['termination_reason']}"
                )
            # SimulationApp.close() may terminate Isaac's bundled process while
            # unwinding the context manager. Persist the aggregate while the
            # runtime is still alive.
            summary = write_evaluation_outputs(output_dir, rows)
            print(f"Evaluation summary: {output_dir / 'summary.json'}")
            print(summary)
    except (
        IsaacDependencyError,
        FileNotFoundError,
        PolicyConnectionError,
        PolicyTimeoutError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
