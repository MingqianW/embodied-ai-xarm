"""Validate locally collected Isaac Sim oracle episodes without launching Isaac."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_runtime.config import load_yaml
from policy_runtime.episode_logging import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "sim_isaac" / "output" / "oracle_dataset")
    parser.add_argument("--task-config", type=Path, default=ROOT / "sim_isaac" / "config" / "tasks.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--require-task-success", action="store_true", help="Reject episodes whose simulated object did not satisfy the task metric.")
    return parser.parse_args()


def _array_ok(path: Path, shape: tuple[int, ...]) -> tuple[bool, str | None]:
    if not path.is_file():
        return False, f"missing {path.name}"
    try:
        value = np.load(path)
    except Exception as exc:  # pragma: no cover - corrupt files are runtime inputs
        return False, f"cannot load {path.name}: {exc}"
    if value.shape != shape:
        return False, f"{path.name} shape={value.shape}, expected={shape}"
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        return False, f"{path.name} contains non-finite or non-numeric values"
    return True, None


def _image_ok(path: Path) -> tuple[bool, str | None]:
    if not path.is_file():
        return False, f"missing {path.name}"
    try:
        with Image.open(path) as image:
            if image.mode not in ("RGB", "RGBA") or image.width <= 0 or image.height <= 0:
                return False, f"invalid image mode/size for {path.name}"
    except Exception as exc:  # pragma: no cover
        return False, f"cannot read {path.name}: {exc}"
    return True, None


def validate_episode(path: Path, task_name: str, *, require_task_success: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    metadata_path = path / "episode.json"
    if not metadata_path.is_file():
        return {"task": task_name, "path": str(path), "quality_ok": False, "errors": ["missing episode.json"]}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "collected":
        errors.append(f"status={metadata.get('status')!r}")
    steps = int(metadata.get("steps", 0))
    if steps < 1:
        errors.append("no executed oracle steps")
    for state_path in (path / "initial" / "state.npy", path / "final" / "state.npy"):
        ok, reason = _array_ok(state_path, (7,))
        if not ok and reason:
            errors.append(reason)
    step_dirs = sorted(path.glob("step_*"))
    if len(step_dirs) != steps:
        errors.append(f"step directory count={len(step_dirs)}, metadata steps={steps}")
    for step_dir in step_dirs:
        for filename, shape in (("state.npy", (7,)), ("action.npy", (7,))):
            ok, reason = _array_ok(step_dir / filename, shape)
            if not ok and reason:
                errors.append(f"{step_dir.name}: {reason}")
        for filename in ("base_image.png", "wrist_image.png"):
            ok, reason = _image_ok(step_dir / filename)
            if not ok and reason:
                errors.append(f"{step_dir.name}: {reason}")
    for folder in ("initial", "final"):
        for filename in ("base_image.png", "wrist_image.png"):
            ok, reason = _image_ok(path / folder / filename)
            if not ok and reason:
                errors.append(f"{folder}: {reason}")
    stages = {str(item.get("stage")) for item in metadata.get("transitions", [])}
    required_stages = {"pregrasp", "descend", "close", "lift"}
    if task_name == "place_red_pepper_in_ring":
        required_stages |= {"preplace", "place", "release"}
    missing_stages = sorted(required_stages - stages)
    if missing_stages:
        errors.append(f"missing oracle stages={missing_stages}")
    safety = metadata.get("safety", {})
    for key in ("finite_state", "transforms_finite", "joints_within_limits", "time_advanced", "camera_fresh", "object_above_table"):
        if safety.get(key) is not True:
            errors.append(f"safety.{key}={safety.get(key)!r}")
    task_success = metadata.get("task_success")
    if require_task_success and task_success is not True:
        errors.append(f"task_success={task_success!r}")
    return {
        "task": task_name,
        "path": str(path),
        "steps": steps,
        "task_success": task_success,
        "quality_ok": not errors,
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    config = load_yaml(args.task_config)
    expected = {
        name: int(spec.get("collection", {}).get("target_episodes", 0))
        for name, spec in config.get("tasks", {}).items()
        if int(spec.get("collection", {}).get("target_episodes", 0)) > 0
    }
    rows: list[dict[str, Any]] = []
    for task_name, target in expected.items():
        for episode_dir in sorted((args.input_dir / task_name).glob("episode_*")):
            rows.append(validate_episode(episode_dir, task_name, require_task_success=args.require_task_success))
    expected_total = args.expected_total if args.expected_total is not None else sum(expected.values())
    valid = sum(bool(row["quality_ok"]) for row in rows)
    report = {
        "status": "passed" if len(rows) == expected_total and valid == len(rows) else "failed",
        "expected_total": expected_total,
        "found_total": len(rows),
        "valid_total": valid,
        "invalid_total": len(rows) - valid,
        "task_success_total": sum(row.get("task_success") is True for row in rows),
        "task_failure_total": sum(row.get("task_success") is not True for row in rows),
        "task_targets": expected,
        "episodes": rows,
    }
    output = args.output or args.input_dir / "quality_report.json"
    write_json(output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
