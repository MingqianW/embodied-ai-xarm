from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_mujoco.remote_policy_evaluation import (
    LABELS,
    read_episodes_csv,
    replay_video,
    validate_label,
    write_json,
    write_summary,
)
from sim_mujoco.scripts.run_remote_policy_closed_loop import (
    MAX_EXECUTE_CHUNK_STEPS,
    EpisodeConfig,
    run_episode,
    validate_execute_chunk_steps,
)
from sim_mujoco.task_scenes import task_names
from sim_mujoco.paths import mujoco_output_root


DEFAULT_OUTPUT_ROOT = mujoco_output_root() / "human_evaluation"


def run_id() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--task", choices=(*task_names(), "all"), default="red_block")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-policy-steps", type=int, default=80)
    parser.add_argument(
        "--execute-chunk-steps",
        type=int,
        default=1,
        help=f"Actions executed before re-observation and inference (1-{MAX_EXECUTE_CHUNK_STEPS}).",
    )
    parser.add_argument("--max-joint-step", type=float, default=0.05)
    parser.add_argument("--control-duration", type=float, default=0.02)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--object-xy-range", type=float, default=0.03)
    parser.add_argument("--object-yaw-range-deg", type=float, default=15.0)
    parser.add_argument("--joint-noise", type=float, default=0.01)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--auto-label-invalid",
        action="store_true",
        help="For unattended smoke tests only: label each attempted episode invalid without prompting.",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    return {
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "host": args.host,
        "port": args.port,
        "timeout": args.timeout,
        "task": args.task,
        "selected_tasks": list(task_names()) if args.task == "all" else [args.task],
        "prompt": args.prompt,
        "max_policy_steps": args.max_policy_steps,
        "execute_chunk_steps": args.execute_chunk_steps,
        "max_joint_step": args.max_joint_step,
        "control_duration": args.control_duration,
        "video_fps": args.video_fps,
        "object_xy_range": args.object_xy_range,
        "object_yaw_range_deg": args.object_yaw_range_deg,
        "joint_noise": args.joint_noise,
        "headless": args.headless,
        "run_dir": run_dir,
    }


def existing_labeled_indices(rows: list[dict[str, Any]]) -> set[int]:
    indices: set[int] = set()
    for row in rows:
        label = str(row.get("label") or "")
        if label in LABELS:
            indices.add(int(row["episode_index"]))
    return indices


def existing_labeled_keys(
    rows: list[dict[str, Any]],
    *,
    fallback_task: str,
) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for row in rows:
        label = str(row.get("label") or "")
        if label in LABELS:
            task = str(row.get("task") or fallback_task)
            keys.add((task, int(row["episode_index"])))
    return keys


def result_row(
    *,
    episode_index: int,
    seed: int,
    label: str,
    score: float | None,
    comment: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    initial = result.get("initial_conditions") or {}
    return {
        "episode_index": episode_index,
        "seed": seed,
        "task": result.get("task"),
        "prompt": result.get("prompt"),
        "automatic_task_success": result.get("task_success"),
        "simulator": "mujoco",
        "success": label == "success",
        "score": score,
        "label": label,
        "valid": label != "invalid",
        "comment": comment,
        "termination_reason": result.get("termination_reason"),
        "policy_steps": result.get("policy_steps"),
        "sim_time": result.get("sim_time"),
        "wall_time": result.get("wall_time"),
        "initial_object_x": initial.get("initial_object_x"),
        "initial_object_y": initial.get("initial_object_y"),
        "initial_object_yaw": initial.get("initial_object_yaw"),
        "video_frames": result.get("video_frames"),
        "video_fps": result.get("video_fps"),
        "combined_video_path": result.get("combined_video_path"),
    }


def prompt_for_label(
    result: dict[str, Any],
) -> tuple[str | None, float | None, str]:
    comment = ""
    combined_path = Path(str(result.get("combined_video_path") or ""))
    while True:
        print()
        print("[s] success  [f] failure  [i] invalid  [r] replay combined video  [c] comment  [q] save and quit")
        choice = input("label> ").strip().lower()
        if choice in ("s", "success"):
            return "success", 1.0, comment
        if choice in ("f", "failure"):
            while True:
                raw_score = input("partial score 0-1 [0]> ").strip()
                try:
                    score = 0.0 if not raw_score else float(raw_score)
                    if not 0.0 <= score <= 1.0:
                        raise ValueError
                    return "failure", score, comment
                except ValueError:
                    print("Score must be a number between 0 and 1.")
        if choice in ("i", "invalid"):
            return "invalid", None, comment
        if choice in ("r", "replay"):
            try:
                replay_video(combined_path)
                print(f"replay opened: {combined_path}")
            except Exception as exc:
                print(f"could not replay video: {exc}")
            continue
        if choice in ("c", "comment"):
            comment = input("comment> ").strip()
            print("comment saved for this prompt")
            continue
        if choice in ("q", "quit"):
            return None, None, comment
        print("Please choose s, f, i, r, c, or q.")


def print_episode_diagnostics(index: int, seed: int, result: dict[str, Any]) -> None:
    print()
    print(f"Episode {index} seed={seed}")
    print("---------------------")
    print("termination_reason:", result.get("termination_reason"))
    print("policy_steps:", result.get("policy_steps"))
    print("sim_time:", result.get("sim_time"))
    print("wall_time:", result.get("wall_time"))
    print("clipping_count:", result.get("clipping_count"))
    print("policy_error:", result.get("policy_error"))
    print("task:", result.get("task"))
    print("automatic_task_success:", result.get("task_success"))
    print("task_metrics:", result.get("task_metrics"))
    print("combined_video:", result.get("combined_video_path"))


def main() -> None:
    args = parse_args()
    try:
        validate_execute_chunk_steps(args.execute_chunk_steps)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    if args.task == "all" and args.prompt:
        raise SystemExit("--prompt cannot be combined with --task all; each task uses its training prompt")
    selected_tasks = task_names() if args.task == "all" else (args.task,)

    if args.resume:
        run_dir = args.output_dir
        if not run_dir.is_dir():
            raise SystemExit(f"--resume expects --output-dir to be an existing run directory: {run_dir}")
        rows: list[dict[str, Any]] = list(read_episodes_csv(run_dir / "episodes.csv"))
        if not (run_dir / "config.json").exists():
            write_json(run_dir / "config.json", make_config(args, run_dir))
    else:
        run_dir = args.output_dir / run_id()
        run_dir.mkdir(parents=True, exist_ok=False)
        rows = []
        write_json(run_dir / "config.json", make_config(args, run_dir))

    fallback_task = selected_tasks[0] if len(selected_tasks) == 1 else ""
    labeled = existing_labeled_keys(rows, fallback_task=fallback_task)
    preflight_done = bool(args.resume)
    try:
        for task in selected_tasks:
            task_root = run_dir / task if len(selected_tasks) > 1 else run_dir
            for episode_index in range(args.episodes):
                key = (task, episode_index)
                if key in labeled:
                    print(f"{task} episode {episode_index} already labeled; skipping")
                    continue

                seed = args.seed_start + episode_index
                episode_dir = task_root / f"episode_{episode_index:03d}_seed_{seed}"
                if episode_dir.exists() and (episode_dir / "result.json").exists():
                    confirm = input(f"{episode_dir} exists but is unlabeled. Overwrite? [y/N]> ").strip().lower()
                    if confirm != "y":
                        print("leaving existing episode untouched; stopping")
                        return
                episode_dir.mkdir(parents=True, exist_ok=True)

                result: dict[str, Any]
                try:
                    result = run_episode(
                        EpisodeConfig(
                            host=args.host,
                            port=args.port,
                            timeout=args.timeout,
                            task=task,
                            prompt=args.prompt,
                            max_policy_steps=args.max_policy_steps,
                            execute_chunk_steps=args.execute_chunk_steps,
                            max_joint_step=args.max_joint_step,
                            control_duration=args.control_duration,
                            headless=args.headless,
                            output_dir=episode_dir,
                            seed=seed,
                            object_xy_range=args.object_xy_range,
                            object_yaw_range_deg=args.object_yaw_range_deg,
                            joint_noise=args.joint_noise,
                            record_video=True,
                            video_fps=args.video_fps,
                            run_preflight=not preflight_done,
                        ),
                    )
                    preflight_done = True
                except KeyboardInterrupt:
                    print("Interrupted during episode; marking may continue if result.json exists.")
                    result_path = episode_dir / "result.json"
                    if not result_path.exists():
                        raise
                    import json

                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    result["termination_reason"] = "interrupted"
                except Exception as exc:
                    print(f"Episode infrastructure error: {exc}")
                    result_path = episode_dir / "result.json"
                    if result_path.exists():
                        import json

                        result = json.loads(result_path.read_text(encoding="utf-8"))
                    else:
                        result = {
                            "task": task,
                            "termination_reason": "error",
                            "policy_error": repr(exc),
                            "policy_steps": 0,
                            "sim_time": 0.0,
                            "wall_time": 0.0,
                            "initial_conditions": {"seed": seed, "task": task},
                        }
                        write_json(result_path, result)

                print_episode_diagnostics(episode_index, seed, result)
                if args.auto_label_invalid:
                    label, score, comment = (
                        "invalid",
                        None,
                        "auto-label-invalid smoke test",
                    )
                else:
                    label, score, comment = prompt_for_label(result)
                if label is None:
                    print("saving progress and quitting")
                    write_summary(run_dir, rows)
                    return

                label = validate_label(label)
                result["human_label"] = label
                result["human_score"] = score
                result["human_comment"] = comment
                write_json(episode_dir / "result.json", result)
                rows.append(
                    result_row(
                        episode_index=episode_index,
                        seed=seed,
                        label=label,
                        score=score,
                        comment=comment,
                        result=result,
                    )
                )
                labeled.add(key)
                summary = write_summary(run_dir, rows)
                print("saved label:", label)
                print("human-rated task success rate:", summary["human_rated_task_success_rate"])

    finally:
        write_summary(run_dir, rows)

    print()
    print("Evaluation complete")
    print("-------------------")
    summary = write_summary(run_dir, rows)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("run_dir:", run_dir)


if __name__ == "__main__":
    main()
