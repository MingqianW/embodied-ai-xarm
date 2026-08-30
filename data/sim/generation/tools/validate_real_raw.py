"""Fast, strict validation for a real-raw-compatible simulation dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


EXPECTED = {
    "pick_up_the_red_pepper": {"total": 50, "clean": 50, "distractors": 0},
    "pick_up_the_blue_block": {"total": 25, "clean": 15, "distractors": 10},
    "pick_up_the_red_block": {"total": 25, "clean": 15, "distractors": 10},
    "pick_up_the_smallest_block": {
        "total": 25,
        "clean": 15,
        "distractors": 10,
    },
    "pick up the largest block": {
        "total": 25,
        "clean": 15,
        "distractors": 10,
    },
    "place_the_red_pepper_in_the_ring": {
        "total": 50,
        "clean": 30,
        "distractors": 20,
    },
}
IMAGE_COLUMNS = (
    "realsense_0_file",
    "realsense_1_file",
    "realsense_2_file",
)
STATE_COLUMNS = (
    "j1_rad",
    "j2_rad",
    "j3_rad",
    "j4_rad",
    "j5_rad",
    "j6_rad",
    "gripper_mm",
)


def _load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or ()), list(reader)


def _check_image(path: Path) -> None:
    with Image.open(path) as image:
        if image.size != (640, 480) or image.mode != "RGB":
            raise ValueError(
                f"Expected RGB 640x480 image, got {image.mode} {image.size}: {path}"
            )
        image.verify()


def validate(raw_root: Path) -> dict[str, Any]:
    raw_root = raw_root.resolve()
    actual_tasks = {path.name for path in raw_root.iterdir() if path.is_dir()}
    if actual_tasks != set(EXPECTED):
        raise ValueError(
            f"Task folders differ. expected={sorted(EXPECTED)} "
            f"actual={sorted(actual_tasks)}"
        )

    task_summary: dict[str, Any] = {}
    total_rows = 0
    total_training_samples = 0
    total_images = 0
    sampled_images = 0
    for task_name, expected in EXPECTED.items():
        task_dir = raw_root / task_name
        episodes = sorted(
            path for path in task_dir.iterdir() if path.is_dir()
        )
        expected_names = [
            f"episode_{index:03d}" for index in range(expected["total"])
        ]
        if [path.name for path in episodes] != expected_names:
            raise ValueError(f"Episode indices are not contiguous for {task_name}")
        variants: Counter[str] = Counter()
        task_rows = 0
        task_samples = 0
        for episode_index, episode_dir in enumerate(episodes):
            meta_path = episode_dir / "meta.json"
            robot_log = episode_dir / "robot_log.csv"
            gripper_events = episode_dir / "gripper_events.csv"
            if not meta_path.is_file() or not robot_log.is_file():
                raise FileNotFoundError(f"Missing raw files in {episode_dir}")
            if not gripper_events.is_file():
                raise FileNotFoundError(gripper_events)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("task") != task_name:
                raise ValueError(f"Task mismatch in {meta_path}")
            if int(meta.get("episode_index", -1)) != episode_index:
                raise ValueError(f"Episode index mismatch in {meta_path}")
            simulation = meta.get("simulation") or {}
            if simulation.get("success") is not True:
                raise ValueError(f"Non-success episode in training raw: {episode_dir}")
            variant = str(simulation.get("scene_variant"))
            variants[variant] += 1

            fields, rows = _load_rows(robot_log)
            required = {"ts", *STATE_COLUMNS, *IMAGE_COLUMNS}
            if not required.issubset(fields):
                raise ValueError(f"Missing robot-log fields in {robot_log}")
            if len(rows) < 2:
                raise ValueError(f"Too few rows in {robot_log}")
            if int(simulation.get("robot_log_rows", -1)) != len(rows):
                raise ValueError(f"Row count mismatch in {episode_dir}")
            timestamps = np.asarray(
                [float(row["ts"]) for row in rows],
                dtype=np.float64,
            )
            if not np.allclose(np.diff(timestamps), 0.1, atol=2e-6, rtol=0.0):
                raise ValueError(f"Non-10-Hz timestamps in {robot_log}")
            state = np.asarray(
                [[float(row[column]) for column in STATE_COLUMNS] for row in rows],
                dtype=np.float64,
            )
            if state.shape != (len(rows), 7) or not np.isfinite(state).all():
                raise ValueError(f"Invalid state values in {robot_log}")
            if np.max(np.abs(state[:, :6])) > 2.0 * np.pi:
                raise ValueError(f"Joint range is not radians in {robot_log}")
            if state[:, 6].min() < -1e-3 or state[:, 6].max() > 850.001:
                raise ValueError(f"Gripper raw range invalid in {robot_log}")

            for camera_index, column in enumerate(IMAGE_COLUMNS):
                paths = [episode_dir / row[column] for row in rows]
                missing = [path for path in paths if not path.is_file()]
                if missing:
                    raise FileNotFoundError(missing[0])
                camera_dir = episode_dir / f"realsense_{camera_index}"
                if len(list(camera_dir.glob("*.png"))) != len(rows):
                    raise ValueError(f"Image count mismatch in {camera_dir}")
                for index in sorted({0, len(paths) // 2, len(paths) - 1}):
                    _check_image(paths[index])
                    sampled_images += 1
                total_images += len(paths)
            task_rows += len(rows)
            task_samples += len(rows) - 1

        expected_variants = Counter(
            {
                "clean": expected["clean"],
                "distractors": expected["distractors"],
            }
        )
        expected_variants += Counter()
        if variants != expected_variants:
            raise ValueError(
                f"Scene mix mismatch for {task_name}: "
                f"expected={expected_variants}, actual={variants}"
            )
        task_summary[task_name] = {
            "episodes": len(episodes),
            "scene_mix": dict(variants),
            "raw_rows": task_rows,
            "training_samples_after_real_converter": task_samples,
        }
        total_rows += task_rows
        total_training_samples += task_samples

    return {
        "passed": True,
        "raw_root": str(raw_root),
        "episodes": sum(value["total"] for value in EXPECTED.values()),
        "raw_rows": total_rows,
        "training_samples_after_real_converter": total_training_samples,
        "control_duration_s": total_training_samples / 10.0,
        "image_files": total_images,
        "sampled_images_decoded": sampled_images,
        "tasks": task_summary,
        "converted": False,
        "uploaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.raw_root)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
