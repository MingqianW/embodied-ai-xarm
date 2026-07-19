"""Check xArm action outliers using the actual OpenPI data pipeline.

This simplified checker intentionally focuses only on OpenPI-normalized action
targets. It reproduces the training-side preprocessing, normalizes actions with
the supplied norm_stats.json, then reports the largest abs(normalized) values.

Example:
    python fine_tune/check_lerobot_openpi_outliers.py \
        --openpi-root "$HOME/repos/openpi" \
        --config-name pi05_xarm \
        --norm-stats "$HOME/repos/openpi/assets/pi05_xarm/local/xarm_pi05_20260703/norm_stats.json" \
        --joint-dims 6 \
        --normalized-warning 3 \
        --normalized-fail 20 \
        --top-k 100
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


JOINT_NAMES = (
    "j1_rad",
    "j2_rad",
    "j3_rad",
    "j4_rad",
    "j5_rad",
    "j6_rad",
)


@dataclass(frozen=True)
class SampleRecord:
    dataset_index: int
    episode_index: int | None
    frame_index: int | None
    task_index: int | None
    task: str
    transformed_actions: np.ndarray


@dataclass(frozen=True)
class NormalizedOutlier:
    abs_normalized: float
    normalized_value: float
    transformed_value: float
    dataset_index: int
    episode_index: int | None
    frame_index: int | None
    task_index: int | None
    task: str
    horizon: int
    dimension: int
    joint_name: str


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size != 1:
        return None
    return int(array.reshape(()))


def scalar_string(value: Any) -> str | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 1:
        return str(array.reshape(()).item())
    return str(value)


def load_norm_stats(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"norm_stats.json does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    stats = payload.get("norm_stats", payload)
    if not isinstance(stats, dict) or "actions" not in stats:
        raise SystemExit(f"Could not find actions stats in {path}. Available keys: {list(payload.keys())}")

    action_stats = stats["actions"]
    for key in ("q01", "q99"):
        if key not in action_stats:
            raise SystemExit(f"Missing actions.{key} in {path}")

    return stats


def add_openpi_to_path(openpi_root: Path) -> None:
    source_root = openpi_root.expanduser().resolve() / "src"
    if not source_root.exists():
        raise SystemExit(f"OpenPI source directory does not exist: {source_root}")
    sys.path.insert(0, str(source_root))


def build_openpi_dataset(config_name: str):
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    import openpi.training.config as openpi_config
    import openpi.transforms as transforms

    config = openpi_config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.repo_id is None:
        raise SystemExit(f"OpenPI config {config_name!r} does not define a repo_id")

    metadata = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    delta_timestamps = {
        key: [step / metadata.fps for step in range(config.model.action_horizon)]
        for key in data_config.action_sequence_keys
    }
    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id, delta_timestamps=delta_timestamps)

    prompt_transform = None
    if data_config.prompt_from_task:
        prompt_transform = transforms.PromptFromLeRobotTask(metadata.tasks)

    preprocessing = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ]
    )

    return config, data_config, metadata, dataset, prompt_transform, preprocessing


def resolve_task(raw_sample: dict[str, Any], prompt_sample: dict[str, Any], metadata_tasks: Any) -> tuple[int | None, str]:
    task_index = optional_int(raw_sample.get("task_index"))
    direct_task = (
        scalar_string(prompt_sample.get("prompt"))
        or scalar_string(raw_sample.get("task"))
        or scalar_string(raw_sample.get("prompt"))
    )
    if direct_task is not None:
        return task_index, direct_task
    if task_index is not None:
        try:
            return task_index, str(metadata_tasks[task_index])
        except (IndexError, KeyError, TypeError):
            pass
    return task_index, "__unknown_task__"


def collect_records(dataset: Any, prompt_transform: Any, preprocessing: Any, metadata_tasks: Any, action_horizon: int) -> list[SampleRecord]:
    records: list[SampleRecord] = []

    for dataset_index in range(len(dataset)):
        raw_sample = dataset[dataset_index]
        episode_index = optional_int(raw_sample.get("episode_index"))
        frame_index = optional_int(raw_sample.get("frame_index"))

        prompt_sample = copy.deepcopy(raw_sample)
        if prompt_transform is not None:
            prompt_sample = prompt_transform(prompt_sample)

        task_index, task = resolve_task(raw_sample, prompt_sample, metadata_tasks)
        transformed = preprocessing(copy.deepcopy(prompt_sample))
        if "actions" not in transformed:
            raise SystemExit(f"Transformed sample {dataset_index} does not contain 'actions'")

        actions = np.asarray(transformed["actions"], dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2:
            raise SystemExit(f"Expected actions shape (horizon, dim), got {actions.shape} at {dataset_index}")
        if actions.shape[0] != action_horizon:
            raise SystemExit(f"Expected action horizon {action_horizon}, got {actions.shape[0]} at {dataset_index}")
        if not np.all(np.isfinite(actions)):
            print(f"WARNING: skipping dataset_index={dataset_index}; transformed actions contain NaN/inf")
            continue

        records.append(
            SampleRecord(
                dataset_index=dataset_index,
                episode_index=episode_index,
                frame_index=frame_index,
                task_index=task_index,
                task=task,
                transformed_actions=actions,
            )
        )

    if not records:
        raise SystemExit("No valid transformed samples were collected.")
    return records


def quantile_normalize(values: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    dimensions = values.shape[-1]
    low = q01[..., :dimensions]
    high = q99[..., :dimensions]
    return (values - low) / (high - low + 1e-6) * 2.0 - 1.0


def insert_top_k(outliers: list[NormalizedOutlier], item: NormalizedOutlier, top_k: int) -> None:
    outliers.append(item)
    outliers.sort(key=lambda value: value.abs_normalized, reverse=True)
    del outliers[top_k:]


def inspect_normalized_outliers(
    records: list[SampleRecord],
    q01: np.ndarray,
    q99: np.ndarray,
    joint_dims: int,
    normalized_warning: float,
    normalized_fail: float,
    top_k: int,
) -> tuple[list[NormalizedOutlier], dict[str, Any]]:
    top_outliers: list[NormalizedOutlier] = []
    warning_count = 0
    fail_count = 0
    task_fail_counts: Counter[str] = Counter()
    episode_fail_counts: Counter[int] = Counter()

    for record in records:
        normalized = quantile_normalize(record.transformed_actions, q01, q99)

        for horizon in range(record.transformed_actions.shape[0]):
            for dimension in range(joint_dims):
                transformed_value = float(record.transformed_actions[horizon, dimension])
                normalized_value = float(normalized[horizon, dimension])
                abs_normalized = abs(normalized_value)

                if abs_normalized >= normalized_warning:
                    warning_count += 1
                    joint_name = JOINT_NAMES[dimension] if dimension < len(JOINT_NAMES) else f"dim_{dimension}"
                    insert_top_k(
                        top_outliers,
                        NormalizedOutlier(
                            abs_normalized=abs_normalized,
                            normalized_value=normalized_value,
                            transformed_value=transformed_value,
                            dataset_index=record.dataset_index,
                            episode_index=record.episode_index,
                            frame_index=record.frame_index,
                            task_index=record.task_index,
                            task=record.task,
                            horizon=horizon,
                            dimension=dimension,
                            joint_name=joint_name,
                        ),
                        top_k,
                    )

                if abs_normalized >= normalized_fail:
                    fail_count += 1
                    task_fail_counts[record.task] += 1
                    if record.episode_index is not None:
                        episode_fail_counts[record.episode_index] += 1

    return top_outliers, {
        "normalized_warning_values": warning_count,
        "normalized_fail_values": fail_count,
        "task_fail_counts": task_fail_counts,
        "episode_fail_counts": episode_fail_counts,
    }


def print_norm_stats(q01: np.ndarray, q99: np.ndarray, joint_dims: int) -> None:
    print("\nOPENPI ACTION q01/q99 NORMALIZATION")
    for dimension in range(joint_dims):
        name = JOINT_NAMES[dimension] if dimension < len(JOINT_NAMES) else f"dim_{dimension}"
        width = float(q99[dimension] - q01[dimension])
        print(
            f"  dim={dimension} name={name} "
            f"q01={q01[dimension]:.9f} q99={q99[dimension]:.9f} range={width:.9f}"
        )


def print_outliers(outliers: list[NormalizedOutlier]) -> None:
    print("\nTOP NORMALIZED OUTLIERS")
    if not outliers:
        print("none")
        return

    for item in outliers:
        print(
            f"abs_normalized={item.abs_normalized:.6f} "
            f"normalized={item.normalized_value:.6f} "
            f"transformed_action={item.transformed_value:.9f} "
            f"episode={item.episode_index} "
            f"frame={item.frame_index} "
            f"horizon={item.horizon} "
            f"dim={item.dimension} "
            f"joint={item.joint_name} "
            f"dataset_index={item.dataset_index} "
            f"task_index={item.task_index} "
            f"task={item.task!r}"
        )


def print_counter(title: str, counter: Counter[Any], limit: int) -> None:
    print(f"\n{title}")
    if not counter:
        print("none")
        return
    for key, count in counter.most_common(limit):
        print(f"  {count:8d}  {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", type=Path, default=Path.home() / "repos" / "openpi")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--joint-dims", type=int, default=6)
    parser.add_argument(
        "--normalized-warning",
        type=float,
        default=3.0,
        help="Print values whose abs(normalized) reaches this threshold.",
    )
    parser.add_argument(
        "--normalized-fail",
        type=float,
        default=20.0,
        help="Exit nonzero if any value has abs(normalized) at least this threshold.",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--summary-limit", type=int, default=30)
    args = parser.parse_args()

    if args.joint_dims <= 0:
        raise SystemExit("--joint-dims must be positive")
    if args.normalized_warning <= 0 or args.normalized_fail <= 0:
        raise SystemExit("normalized thresholds must be positive")
    if args.normalized_fail < args.normalized_warning:
        raise SystemExit("--normalized-fail must be >= --normalized-warning")

    openpi_root = args.openpi_root.expanduser().resolve()
    add_openpi_to_path(openpi_root)

    stats = load_norm_stats(args.norm_stats)
    action_stats = stats["actions"]
    q01 = np.asarray(action_stats["q01"], dtype=np.float64)
    q99 = np.asarray(action_stats["q99"], dtype=np.float64)

    config, data_config, metadata, dataset, prompt_transform, preprocessing = build_openpi_dataset(args.config_name)
    action_horizon = config.model.action_horizon

    print("OPENPI CONFIG")
    print(f"openpi_root: {openpi_root}")
    print(f"config_name: {args.config_name}")
    print(f"repo_id: {data_config.repo_id}")
    print(f"dataset_length: {len(dataset)}")
    print(f"fps: {metadata.fps}")
    print(f"action_horizon: {action_horizon}")
    print(f"action_sequence_keys: {tuple(data_config.action_sequence_keys)}")
    print(f"use_quantile_norm: {data_config.use_quantile_norm}")
    print(f"norm_stats: {args.norm_stats.expanduser().resolve()}")
    print(f"normalized_warning: {args.normalized_warning}")
    print(f"normalized_fail: {args.normalized_fail}")

    records = collect_records(dataset, prompt_transform, preprocessing, metadata.tasks, action_horizon)
    action_dimensions = records[0].transformed_actions.shape[-1]

    if args.joint_dims > action_dimensions:
        raise SystemExit(f"--joint-dims={args.joint_dims} exceeds transformed action dim {action_dimensions}")
    if args.joint_dims > len(q01):
        raise SystemExit(f"--joint-dims={args.joint_dims} exceeds norm stats dim {len(q01)}")

    print(f"valid_records: {len(records)}")
    print(f"transformed_action_dim: {action_dimensions}")
    print_norm_stats(q01, q99, args.joint_dims)

    outliers, counts = inspect_normalized_outliers(
        records=records,
        q01=q01,
        q99=q99,
        joint_dims=args.joint_dims,
        normalized_warning=args.normalized_warning,
        normalized_fail=args.normalized_fail,
        top_k=args.top_k,
    )

    checked_values = len(records) * action_horizon * args.joint_dims
    print("\nSUMMARY")
    print(f"checked_values: {checked_values}")
    print(f"normalized_warning_values: {counts['normalized_warning_values']}")
    print(f"normalized_fail_values: {counts['normalized_fail_values']}")

    print_counter("EPISODES WITH NORMALIZED FAIL VALUES", counts["episode_fail_counts"], args.summary_limit)
    print_counter("TASKS WITH NORMALIZED FAIL VALUES", counts["task_fail_counts"], args.summary_limit)
    print_outliers(outliers)

    if counts["normalized_fail_values"] > 0:
        raise SystemExit("\nFAILED: OpenPI-normalized action outliers were found.")

    print("\nPASSED: no OpenPI-normalized action values exceeded --normalized-fail.")


if __name__ == "__main__":
    main()
