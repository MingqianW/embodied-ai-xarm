"""Headless automatic fixed-seed MuJoCo policy evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_mujoco.remote_policy_evaluation import write_json  # noqa: E402
from sim_mujoco.scripts.run_remote_policy_closed_loop import EpisodeConfig  # noqa: E402
from sim_mujoco.scripts.run_remote_policy_closed_loop import run_episode  # noqa: E402

INVALID_REASONS = {
    "error",
    "policy_timeout",
    "unsafe_action",
    "non_finite_simulation",
    "forbidden_collision",
}


def _diagnostics(episode_dir: Path) -> tuple[list[float], float | None]:
    latencies = []
    first_action_magnitude = None
    for path in sorted(episode_dir.glob("policy_step_*/diagnostics.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        latency = value.get("total_client_latency_seconds")
        if latency is not None:
            latencies.append(float(latency))
        if first_action_magnitude is None and value.get("raw_actions") is not None:
            actions = np.asarray(value["raw_actions"], dtype=np.float64)
            first_action_magnitude = float(np.linalg.norm(actions[0]))
    return latencies, first_action_magnitude


def _row(index: int, seed: int, episode_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    reason = str(result.get("termination_reason") or "")
    invalid = bool(result.get("policy_error")) or reason in INVALID_REASONS or reason.endswith("_collision")
    success = bool(result.get("task_success")) and not invalid
    label = "invalid" if invalid else ("success" if success else "failure")
    latencies, first_action = _diagnostics(episode_dir)
    return {
        "episode_index": index,
        "seed": seed,
        "label": label,
        "success": success,
        "valid": not invalid,
        "termination_reason": reason,
        "policy_steps": int(result.get("policy_steps") or 0),
        "mean_inference_latency_seconds": float(np.mean(latencies)) if latencies else None,
        "first_action_magnitude": first_action,
        "clipping_count": int(result.get("clipping_count") or 0),
        "combined_video_path": result.get("combined_video_path"),
        "result_path": str(episode_dir / "result.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-label", required=True)
    parser.add_argument(
        "--task",
        choices=(
            "red_pepper",
            "blue_block",
            "red_block",
            "smallest_block",
            "largest_block",
            "place_red_pepper_in_ring",
        ),
        default="red_block",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=50_000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-policy-steps", type=int, default=80)
    parser.add_argument(
        "--execute-chunk-steps",
        type=int,
        choices=range(1, 11),
        default=1,
        help="Actions executed before re-observation and inference (1-10).",
    )
    parser.add_argument("--control-duration", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.video_every < 1 or args.control_duration <= 0:
        raise ValueError("episodes, video-every, and control-duration must be positive")
    output = args.output_dir.resolve()
    if output.exists() and not args.resume and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output exists; pass --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)

    config = {
        "policy_label": args.policy_label,
        "episodes": args.episodes,
        "seeds": list(range(args.seed_start, args.seed_start + args.episodes)),
        "task": args.task,
        "host": args.host,
        "port": args.port,
        "max_policy_steps": args.max_policy_steps,
        "execute_chunk_steps": args.execute_chunk_steps,
        "control_duration": args.control_duration,
        "object_xy_range": 0.03,
        "object_yaw_range_deg": 15.0,
        "joint_noise": 0.01,
        "headless": True,
    }
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Resume configuration differs from the existing evaluation")
    write_json(config_path, config)

    rows = []
    preflight_done = False
    for index, seed in enumerate(config["seeds"]):
        episode_dir = output / f"episode_{index:03d}_seed_{seed}"
        result_path = episode_dir / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            preflight_done = True
        else:
            episode_dir.mkdir(parents=True, exist_ok=False)
            try:
                result = run_episode(
                    EpisodeConfig(
                        host=args.host,
                        port=args.port,
                        task=args.task,
                        max_policy_steps=args.max_policy_steps,
                        execute_chunk_steps=args.execute_chunk_steps,
                        control_duration=args.control_duration,
                        headless=True,
                        output_dir=episode_dir,
                        seed=seed,
                        object_xy_range=0.03,
                        object_yaw_range_deg=15.0,
                        joint_noise=0.01,
                        record_video=index % args.video_every == 0,
                        run_preflight=not preflight_done,
                        timeout=args.timeout,
                    )
                )
                preflight_done = True
            except Exception as exc:
                if result_path.is_file():
                    result = json.loads(result_path.read_text())
                else:
                    result = {
                        "termination_reason": "error",
                        "policy_error": repr(exc),
                        "task_success": False,
                        "policy_steps": 0,
                        "clipping_count": 0,
                    }
                    write_json(result_path, result)
        row = _row(index, seed, episode_dir, result)
        rows.append(row)
        print(
            f"[{index + 1}/{args.episodes}] seed={seed} label={row['label']} "
            f"steps={row['policy_steps']} reason={row['termination_reason']}"
        )
        write_json(output / "episodes.json", {"episodes": rows})

    successes = sum(row["label"] == "success" for row in rows)
    failures = sum(row["label"] == "failure" for row in rows)
    invalid = sum(row["label"] == "invalid" for row in rows)
    valid = successes + failures
    summary = {
        "policy_label": args.policy_label,
        "attempted": len(rows),
        "successes": successes,
        "failures": failures,
        "invalid": invalid,
        "success_rate_all": successes / len(rows),
        "success_rate_valid": successes / valid if valid else None,
        "episodes": rows,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}, indent=2))
    if args.fail_on_invalid and invalid:
        raise SystemExit(f"Evaluation produced {invalid} invalid episodes")


if __name__ == "__main__":
    main()
