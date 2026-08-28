"""Export a fixed per-task sample of LeRobot image rows as side-by-side MP4s."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


DATASETS = {
    "stable_v3": Path(
        "/work/nvme/bfmk/mw89/mujoco_datasets/local/xarm_mujoco_clean_multitask_stable_v3"
    ),
    "stable_v4_10x": Path(
        "/work/nvme/bfmk/mw89/mujoco_datasets/local/xarm_mujoco_clean_multitask_stable_v4_10x_real"
    ),
}
DEFAULT_OUTPUT = Path(
    "/work/nvme/bfmk/mw89/exports/mujoco_training_videos/per_task_2episodes_v1"
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _selection(dataset_root: Path, per_task: int) -> list[dict]:
    tasks = {int(row["task_index"]): str(row["task"]) for row in _read_jsonl(dataset_root / "meta/tasks.jsonl")}
    episodes = _read_jsonl(dataset_root / "meta/episodes.jsonl")
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in episodes:
        task_rows = row.get("tasks") or []
        if len(task_rows) != 1:
            raise ValueError(f"Expected exactly one task for episode {row}")
        by_task[str(task_rows[0])].append(row)
    selected: list[dict] = []
    for task_index, task in tasks.items():
        rows = sorted(by_task[task], key=lambda row: int(row["episode_index"]))[:per_task]
        if len(rows) != per_task:
            raise ValueError(f"Task {task!r} has only {len(rows)} episodes")
        selected.extend(
            {
                "task_index": task_index,
                "task": task,
                "episode_index": int(row["episode_index"]),
                "frames": int(row["length"]),
            }
            for row in rows
        )
    return selected


def _parquet_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def _decode(image: dict, *, label: str) -> np.ndarray:
    encoded = image.get("bytes")
    if encoded is None:
        raise ValueError(f"{label} has no embedded image bytes")
    frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Cannot decode {label}")
    return frame


def _write_episode(dataset_root: Path, item: dict, output: Path, fps: int) -> dict:
    parquet_path = _parquet_path(dataset_root, item["episode_index"])
    table = pq.read_table(parquet_path, columns=["image", "wrist_image"])
    base_rows = table.column("image").to_pylist()
    wrist_rows = table.column("wrist_image").to_pylist()
    if len(base_rows) != item["frames"] or len(wrist_rows) != item["frames"]:
        raise ValueError(f"Frame count mismatch in {parquet_path}")
    first_base = _decode(base_rows[0], label=f"{parquet_path}:image[0]")
    first_wrist = _decode(wrist_rows[0], label=f"{parquet_path}:wrist_image[0]")
    if first_base.shape[:2] != first_wrist.shape[:2]:
        first_wrist = cv2.resize(first_wrist, (first_base.shape[1], first_base.shape[0]))
    height, width = first_base.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width * 2, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open MP4 writer: {output}")
    try:
        for base_row, wrist_row in zip(base_rows, wrist_rows, strict=True):
            base = _decode(base_row, label=f"{parquet_path}:image")
            wrist = _decode(wrist_row, label=f"{parquet_path}:wrist_image")
            if base.shape[:2] != (height, width):
                base = cv2.resize(base, (width, height))
            if wrist.shape[:2] != (height, width):
                wrist = cv2.resize(wrist, (width, height))
            writer.write(np.concatenate((base, wrist), axis=1))
    finally:
        writer.release()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Video output is missing or empty: {output}")
    return {**item, "source_parquet": str(parquet_path), "video": str(output), "bytes": output.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-task", type=int, default=2)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.per_task != 2:
        raise ValueError("This approved export is fixed at two episodes per task")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing export root: {output}")
    selection = {name: _selection(root, args.per_task) for name, root in DATASETS.items()}
    if args.dry_run:
        print(json.dumps({"output": str(output), "selection": selection}, indent=2))
        return
    written = []
    for name, dataset_root in DATASETS.items():
        for item in selection[name]:
            filename = f"episode_{item['episode_index']:06d}_combined.mp4"
            video = output / name / f"task_{item['task_index']:02d}_{_safe_name(item['task'])}" / filename
            written.append(_write_episode(dataset_root, item, video, args.fps))
    manifest = {
        "schema_version": 1,
        "selection": "first two episode indices per task",
        "views": ["image", "wrist_image"],
        "layout": "side-by-side: image | wrist_image",
        "fps": args.fps,
        "videos": written,
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "video_count": len(written)}, indent=2))


if __name__ == "__main__":
    main()
