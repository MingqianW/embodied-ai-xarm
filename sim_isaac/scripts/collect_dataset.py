"""Collect the configured multi-task Isaac Sim dataset mix.

The requested prompt counts are exact; scene-mix metadata is intentionally
ignored.  The policy server supplies actions; every
episode is saved by ``run_interactive_evaluation.py`` with state, images,
actions, recording, and evaluation metadata.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_runtime.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "sim_isaac" / "output" / "dataset_200")
    parser.add_argument("--task-config", type=Path, default=ROOT / "sim_isaac" / "config" / "tasks.yaml")
    parser.add_argument("--max-policy-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Write the exact plan without contacting the policy server.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.task_config)
    tasks = config.get("tasks", {})
    plan: list[dict[str, object]] = []
    for task_name, task in tasks.items():
        collection = task.get("collection", {})
        target = int(collection.get("target_episodes", 0))
        if target:
            plan.append(
                {
                    "task": task_name,
                    "prompt": str(task["prompt"]),
                    "scene_variant": "clean",
                    "episodes": target,
                    "output_dir": str(args.output_dir / task_name),
                }
            )
    total = sum(int(item["episodes"]) for item in plan)
    if total != 200:
        raise ValueError(f"Configured dataset total must be 200, got {total}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "planned" if args.dry_run else "starting",
        "total_episodes": total,
        "plan": plan,
    }
    manifest_path = args.output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset plan: {manifest_path} ({total} episodes)")
    if args.dry_run:
        return 0

    runner = Path(__file__).with_name("run_interactive_evaluation.py")
    for index, item in enumerate(plan):
        command = [
            sys.executable,
            str(runner),
            "--task", str(item["task"]),
            "--episodes", str(item["episodes"]),
            "--host", args.host,
            "--port", str(args.port),
            "--max-policy-steps", str(args.max_policy_steps),
            "--seed", str(args.seed + index * 1000),
            "--scene-variant", str(item["scene_variant"]),
            "--output-dir", str(item["output_dir"]),
            "--record",
            "--non-interactive",
            "--headless",
        ]
        print("Running:", " ".join(f'\"{part}\"' if " " in part else part for part in command))
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            manifest["status"] = "failed"
            manifest["failed_index"] = index
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            return completed.returncode
    manifest["status"] = "completed"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {total} episodes. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
