"""Pre-training validator for LeRobot xArm action/state discontinuities.

This catches the failure mode where a small-looking raw joint jump becomes a
huge normalized OpenPI target because one action dimension has a tiny q01/q99
range.

Examples:
    python fine_tune/check_lerobot_action_jumps.py \
      --dataset-root ~/.cache/huggingface/lerobot/local/xarm_pickup_v260624

    python fine_tune/check_lerobot_action_jumps.py \
      --dataset-root ~/.cache/huggingface/lerobot/local/xarm_pickup_v260624 \
      --norm-stats ~/projects/openpi_xarm/openpi/assets/pi05_xarm_full_finetune/local/xarm_pickup_v260624/norm_stats.json \
      --action-horizon 10 \
      --fail-raw-delta 0.01 \
      --fail-normalized 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "local/xarm_pi05_data"


@dataclass(frozen=True)
class Outlier:
    score: float
    raw_delta: float
    normalized: float | None
    z_norm: float | None
    dim: int
    horizon: int
    episode: int | None
    start_frame: int | None
    target_frame: int | None
    task_index: int | None
    task: str | None
    state_value: float
    action_value: float
    parquet: Path
    mode: str


def default_dataset_root() -> Path:
    hf_lerobot_home = os.environ.get("HF_LEROBOT_HOME")
    if hf_lerobot_home:
        return Path(hf_lerobot_home).expanduser() / DEFAULT_REPO_ID
    return Path.home() / ".cache" / "huggingface" / "lerobot" / DEFAULT_REPO_ID


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_task_map(dataset_root: Path) -> dict[int, str]:
    task_map: dict[int, str] = {}
    for record in read_jsonl(dataset_root / "meta" / "tasks.jsonl"):
        if "task_index" in record and "task" in record:
            task_map[int(record["task_index"])] = str(record["task"])
    return task_map


def import_numpy_pyarrow():
    try:
        import numpy as np
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Missing dependency. Install these in the same environment as the LeRobot dataset:\n"
            "  pip install numpy pyarrow\n"
            f"Original import error: {exc}"
        ) from exc
    return np, pq


def find_action_stats(stats: Any) -> dict[str, Any] | None:
    if not isinstance(stats, dict):
        return None

    for key in ("actions", "action"):
        value = stats.get(key)
        if isinstance(value, dict) and any(k in value for k in ("q01", "q99", "mean", "std")):
            return value

    for key, value in stats.items():
        if isinstance(value, dict) and "action" in str(key).lower():
            if any(k in value for k in ("q01", "q99", "mean", "std")):
                return value
        found = find_action_stats(value)
        if found is not None:
            return found
    return None


def load_norm_stats(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.expanduser().open("r", encoding="utf-8") as f:
        stats = json.load(f)
    action_stats = find_action_stats(stats)
    if action_stats is None:
        raise SystemExit(f"Could not find action stats in {path}")
    return action_stats


def as_float_array(np: Any, value: Any) -> Any:
    return np.asarray(value, dtype=np.float32)


def stat_array(np: Any, stats: dict[str, Any], key: str) -> Any | None:
    value = stats.get(key)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def normalize_q01_q99(np: Any, x: Any, stats: dict[str, Any]) -> Any | None:
    q01 = stat_array(np, stats, "q01")
    q99 = stat_array(np, stats, "q99")
    if q01 is None or q99 is None:
        return None
    dims = min(len(x), len(q01), len(q99))
    denom = q99[:dims] - q01[:dims]
    out = np.full(len(x), np.nan, dtype=np.float32)
    valid = np.abs(denom) > 1e-12
    out[:dims][valid] = 2.0 * (x[:dims][valid] - q01[:dims][valid]) / denom[valid] - 1.0
    return out


def normalize_mean_std(np: Any, x: Any, stats: dict[str, Any]) -> Any | None:
    mean = stat_array(np, stats, "mean")
    std = stat_array(np, stats, "std")
    if mean is None or std is None:
        return None
    dims = min(len(x), len(mean), len(std))
    out = np.full(len(x), np.nan, dtype=np.float32)
    valid = np.abs(std[:dims]) > 1e-12
    out[:dims][valid] = (x[:dims][valid] - mean[:dims][valid]) / std[:dims][valid]
    return out


def scalar_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def read_episode_table(np: Any, pq: Any, parquet: Path) -> dict[str, Any]:
    schema = pq.read_schema(parquet)
    wanted = [c for c in ("state", "actions", "episode_index", "frame_index", "task_index", "task", "prompt") if c in schema.names]
    table = pq.read_table(parquet, columns=wanted)
    rows = table.to_pylist()
    if not rows:
        return {"rows": [], "states": np.empty((0, 0)), "actions": np.empty((0, 0))}
    states = np.stack([as_float_array(np, row["state"]) for row in rows])
    actions = np.stack([as_float_array(np, row["actions"]) for row in rows])
    return {"rows": rows, "states": states, "actions": actions}


def push_outlier(outliers: list[Outlier], item: Outlier, limit: int) -> None:
    outliers.append(item)
    outliers.sort(key=lambda x: x.score, reverse=True)
    del outliers[limit:]


def scan_parquets(
    *,
    dataset_root: Path,
    norm_stats: dict[str, Any] | None,
    action_horizon: int,
    joint_dims: int,
    fail_raw_delta: float,
    fail_normalized: float,
    top_k: int,
) -> tuple[list[Outlier], dict[str, Any]]:
    np, pq = import_numpy_pyarrow()
    parquets = sorted(dataset_root.glob("data/**/*.parquet"))
    if not parquets:
        raise SystemExit(f"No parquet files found under {dataset_root / 'data'}")

    task_map = load_task_map(dataset_root)
    top: list[Outlier] = []
    fail_count = 0
    row_count = 0
    episode_count = 0
    task_counts: dict[str, int] = {}

    for parquet in parquets:
        data = read_episode_table(np, pq, parquet)
        rows = data["rows"]
        states = data["states"]
        actions = data["actions"]
        if len(rows) == 0:
            continue

        episode_count += 1
        row_count += len(rows)
        dims = min(joint_dims, states.shape[1], actions.shape[1])

        for i, row in enumerate(rows):
            task_index = scalar_or_none(row.get("task_index"))
            task = row.get("task") or row.get("prompt")
            if task is None and task_index is not None:
                task = task_map.get(task_index)
            if task:
                task_counts[str(task)] = task_counts.get(str(task), 0) + 1

            for horizon in range(action_horizon):
                j = min(i + horizon, len(rows) - 1)
                transformed = np.zeros(max(actions.shape[1], 32), dtype=np.float32)
                transformed[: actions.shape[1]] = actions[j]
                transformed[:dims] = actions[j, :dims] - states[i, :dims]

                qnorm = normalize_q01_q99(np, transformed, norm_stats) if norm_stats else None
                znorm = normalize_mean_std(np, transformed, norm_stats) if norm_stats else None

                for dim in range(dims):
                    raw_delta = float(transformed[dim])
                    q_value = None if qnorm is None or math.isnan(float(qnorm[dim])) else float(qnorm[dim])
                    z_value = None if znorm is None or math.isnan(float(znorm[dim])) else float(znorm[dim])
                    norm_score = abs(q_value) if q_value is not None else 0.0
                    raw_score = abs(raw_delta) / fail_raw_delta if fail_raw_delta > 0 else abs(raw_delta)
                    score = max(norm_score, raw_score)

                    is_fail = abs(raw_delta) >= fail_raw_delta
                    if q_value is not None:
                        is_fail = is_fail or abs(q_value) >= fail_normalized
                    if is_fail:
                        fail_count += 1

                    if score > 1.0:
                        push_outlier(
                            top,
                            Outlier(
                                score=score,
                                raw_delta=raw_delta,
                                normalized=q_value,
                                z_norm=z_value,
                                dim=dim,
                                horizon=horizon,
                                episode=scalar_or_none(row.get("episode_index")),
                                start_frame=scalar_or_none(row.get("frame_index")),
                                target_frame=scalar_or_none(rows[j].get("frame_index")),
                                task_index=task_index,
                                task=str(task) if task is not None else None,
                                state_value=float(states[i, dim]),
                                action_value=float(actions[j, dim]),
                                parquet=parquet,
                                mode="horizon_delta" if horizon else "next_step_delta",
                            ),
                            top_k,
                        )

    summary = {
        "parquets": len(parquets),
        "episodes_seen": episode_count,
        "rows": row_count,
        "fail_count": fail_count,
        "task_counts": task_counts,
    }
    return top, summary


def print_outliers(outliers: list[Outlier], dataset_root: Path) -> None:
    print("\nTOP OUTLIERS")
    if not outliers:
        print("none")
        return

    for item in outliers:
        rel = item.parquet.relative_to(dataset_root) if item.parquet.is_relative_to(dataset_root) else item.parquet
        norm = "NA" if item.normalized is None else f"{item.normalized:.3f}"
        znorm = "NA" if item.z_norm is None else f"{item.z_norm:.3f}"
        print(
            "score={score:.3f} raw_delta={raw:.9f} qnorm={qnorm} znorm={znorm} "
            "dim={dim} horizon={horizon} episode={episode} start_frame={sf} target_frame={tf} "
            "task_index={task_index} task={task!r} state={state:.9f} action={action:.9f} file={file}".format(
                score=item.score,
                raw=item.raw_delta,
                qnorm=norm,
                znorm=znorm,
                dim=item.dim,
                horizon=item.horizon,
                episode=item.episode,
                sf=item.start_frame,
                tf=item.target_frame,
                task_index=item.task_index,
                task=item.task,
                state=item.state_value,
                action=item.action_value,
                file=rel,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="LeRobot dataset root. Defaults to $HF_LEROBOT_HOME/local/xarm_pi05_data.",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="Optional OpenPI norm_stats.json. Enables q01/q99 and mean/std normalized outlier checks.",
    )
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--joint-dims", type=int, default=6, help="Number of joint dimensions to check as deltas.")
    parser.add_argument(
        "--fail-raw-delta",
        type=float,
        default=0.01,
        help="Fail if any checked joint delta magnitude is at least this value.",
    )
    parser.add_argument(
        "--fail-normalized",
        type=float,
        default=20.0,
        help="Fail if q01/q99-normalized magnitude is at least this value.",
    )
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    dataset_root = (args.dataset_root or default_dataset_root()).expanduser()
    norm_stats = load_norm_stats(args.norm_stats)

    print(f"dataset_root: {dataset_root}")
    print(f"norm_stats: {args.norm_stats.expanduser() if args.norm_stats else '(not provided)'}")
    print(f"action_horizon: {args.action_horizon}")
    print(f"joint_dims: {args.joint_dims}")
    print(f"fail_raw_delta: {args.fail_raw_delta}")
    print(f"fail_normalized: {args.fail_normalized}")

    if norm_stats is not None:
        q01 = norm_stats.get("q01")
        q99 = norm_stats.get("q99")
        if q01 is not None and q99 is not None:
            ranges = [float(b) - float(a) for a, b in zip(q01[: args.joint_dims], q99[: args.joint_dims])]
            print(f"q99_minus_q01_first_{args.joint_dims}: {ranges}")

    outliers, summary = scan_parquets(
        dataset_root=dataset_root,
        norm_stats=norm_stats,
        action_horizon=args.action_horizon,
        joint_dims=args.joint_dims,
        fail_raw_delta=args.fail_raw_delta,
        fail_normalized=args.fail_normalized,
        top_k=args.top_k,
    )

    print("\nSUMMARY")
    print(f"parquets: {summary['parquets']}")
    print(f"episodes_seen: {summary['episodes_seen']}")
    print(f"rows: {summary['rows']}")
    print(f"fail_count: {summary['fail_count']}")
    print("task_counts:")
    for task, count in sorted(summary["task_counts"].items()):
        print(f"  {count:8d}  {task}")

    print_outliers(outliers, dataset_root)

    if summary["fail_count"]:
        raise SystemExit(
            "\nFAILED: action/state jump outliers were found. Inspect the listed episodes before training."
        )
    print("\nPASSED: no action/state jump outliers exceeded the configured thresholds.")


if __name__ == "__main__":
    main()
