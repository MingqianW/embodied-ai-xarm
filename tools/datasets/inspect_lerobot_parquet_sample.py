"""Inspect one sample from a processed LeRobot parquet dataset.

Examples:
    python tools/datasets/inspect_lerobot_parquet_sample.py

    python tools/datasets/inspect_lerobot_parquet_sample.py \
        --dataset-root "$HF_LEROBOT_HOME/local/xarm_pi05_data"

    python tools/datasets/inspect_lerobot_parquet_sample.py \
        --parquet /path/to/chunk.parquet \
        --row 5
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "local/xarm_pi05_data"

# These fields must be stored directly in each parquet row.
REQUIRED_DATA_KEYS = (
    "image",
    "wrist_image",
    "state",
    "actions",
)

# LeRobot versions may either store task text directly or store task_index
# and resolve it through meta/tasks.jsonl.
TASK_KEYS = ("task", "task_index")


def _default_dataset_root() -> Path:
    hf_lerobot_home = os.environ.get("HF_LEROBOT_HOME")
    if hf_lerobot_home:
        return Path(hf_lerobot_home).expanduser() / DEFAULT_REPO_ID

    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "lerobot"
        / DEFAULT_REPO_ID
    )


def _find_parquets(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.glob("data/**/*.parquet"))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise SystemExit(
                    f"Expected a JSON object in {path} at line "
                    f"{line_number}, got {type(record).__name__}"
                )

            records.append(record)

    return records


def _load_task_map(dataset_root: Path | None) -> dict[int, str]:
    """Load task_index -> task text from meta/tasks.jsonl."""

    if dataset_root is None:
        return {}

    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    records = _load_jsonl(tasks_path)

    task_map: dict[int, str] = {}

    for record in records:
        if "task_index" not in record or "task" not in record:
            continue

        try:
            task_index = int(record["task_index"])
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"Invalid task_index in {tasks_path}: "
                f"{record.get('task_index')!r}"
            ) from exc

        task_text = str(record["task"])
        task_map[task_index] = task_text

    return task_map


def _infer_dataset_root_from_parquet(parquet_path: Path) -> Path | None:
    """Walk upward and locate a dataset root containing meta/info.json."""

    resolved = parquet_path.expanduser().resolve()

    for parent in resolved.parents:
        if (parent / "meta" / "info.json").exists():
            return parent

        if (parent / "meta" / "tasks.jsonl").exists():
            return parent

    return None


def _shape(value: Any) -> str:
    if hasattr(value, "shape"):
        return "x".join(str(dim) for dim in value.shape)

    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return f"{len(value)}x{len(value[0])}"

        return str(len(value))

    return "scalar"


def _preview(value: Any, *, max_items: int = 8) -> str:
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                f"{key}: {type(item).__name__}"
                for key, item in value.items()
            )
            + "}"
        )

    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"

    if isinstance(value, (list, tuple)):
        items = list(value[:max_items])
        suffix = " ..." if len(value) > max_items else ""
        return f"{items}{suffix}"

    return repr(value)


def _image_summary(value: Any) -> str:
    if isinstance(value, dict):
        parts = []

        for key, item in value.items():
            if isinstance(item, bytes):
                parts.append(f"{key}=bytes({len(item)})")
            else:
                parts.append(f"{key}={item!r}")

        return ", ".join(parts)

    if isinstance(value, bytes):
        return f"bytes({len(value)})"

    return repr(value)


def _try_print_image_size(label: str, value: Any) -> None:
    image_bytes = None
    image_path = None

    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
    elif isinstance(value, bytes):
        image_bytes = value
    elif isinstance(value, str):
        image_path = value

    try:
        from PIL import Image
    except Exception:
        print(f"{label}.decoded_size: skipped (PIL is not installed)")
        return

    try:
        if image_bytes:
            with Image.open(io.BytesIO(image_bytes)) as image:
                print(
                    f"{label}.decoded_size: "
                    f"{image.size}, mode={image.mode}"
                )
        elif image_path and Path(image_path).exists():
            with Image.open(image_path) as image:
                print(
                    f"{label}.decoded_size: "
                    f"{image.size}, mode={image.mode}"
                )
        else:
            print(
                f"{label}.decoded_size: skipped "
                "(no embedded bytes or existing path)"
            )
    except Exception as exc:
        print(f"{label}.decoded_size: failed ({exc})")


def _resolve_task(
    sample: dict[str, Any],
    task_map: dict[int, str],
) -> tuple[str | None, int | None, str]:
    """Resolve task text from direct task or task_index metadata."""

    direct_task = sample.get("task")

    if direct_task is not None:
        task_index_value = sample.get("task_index")
        task_index = (
            int(task_index_value)
            if task_index_value is not None
            else None
        )
        return str(direct_task), task_index, "parquet.task"

    task_index_value = sample.get("task_index")

    if task_index_value is None:
        return None, None, "missing"

    try:
        task_index = int(task_index_value)
    except (TypeError, ValueError):
        return None, None, "invalid task_index"

    task_text = task_map.get(task_index)

    if task_text is None:
        return None, task_index, "unmapped task_index"

    return task_text, task_index, "meta/tasks.jsonl"


def inspect_sample(
    parquet_path: Path,
    row_index: int,
    dataset_root: Path | None,
) -> None:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Install it in the environment "
            "that has your LeRobot dataset:\n"
            "  pip install pyarrow\n"
            f"Original import error: {exc}"
        ) from exc

    parquet_path = parquet_path.expanduser().resolve()

    if dataset_root is None:
        dataset_root = _infer_dataset_root_from_parquet(parquet_path)

    task_map = _load_task_map(dataset_root)

    parquet_file = pq.ParquetFile(parquet_path)

    if row_index < 0 or row_index >= parquet_file.metadata.num_rows:
        raise SystemExit(
            f"--row {row_index} is out of range for {parquet_path} "
            f"(rows: {parquet_file.metadata.num_rows})"
        )

    table = parquet_file.read()
    sample = table.slice(row_index, 1).to_pylist()[0]

    print(
        f"dataset_root: "
        f"{dataset_root if dataset_root else '(could not infer)'}"
    )
    print(f"parquet: {parquet_path}")
    print(f"parquet_rows: {parquet_file.metadata.num_rows}")
    print(f"parquet_row_groups: {parquet_file.metadata.num_row_groups}")
    print(f"selected_row: {row_index}")
    print()

    print("schema:")
    print(table.schema)
    print()

    missing_data_keys = [
        key for key in REQUIRED_DATA_KEYS
        if key not in sample
    ]

    has_task_representation = any(
        key in sample for key in TASK_KEYS
    )

    print(
        "required_data_keys_missing: "
        f"{missing_data_keys if missing_data_keys else 'none'}"
    )
    print(
        "task_representation: "
        f"{'present' if has_task_representation else 'missing'}"
    )

    if not has_task_representation:
        print(
            "task_error: neither 'task' nor 'task_index' "
            "exists in the parquet row"
        )

    print(f"task_map_entries: {len(task_map)}")
    print()

    for key in (
        "episode_index",
        "frame_index",
        "timestamp",
        "index",
        "task_index",
        "task",
        "prompt",
    ):
        if key in sample:
            print(f"{key}: {_preview(sample[key])}")

    resolved_task, task_index, task_source = _resolve_task(
        sample,
        task_map,
    )

    print(f"resolved_task_index: {task_index}")
    print(f"resolved_task: {resolved_task!r}")
    print(f"resolved_task_source: {task_source}")

    if task_index is not None and resolved_task is None:
        print(
            "task_mapping_error: "
            f"task_index {task_index} was not found in "
            "meta/tasks.jsonl"
        )

    print()

    for key in ("state", "actions"):
        if key in sample:
            print(f"{key}.shape: {_shape(sample[key])}")
            print(f"{key}.preview: {_preview(sample[key])}")

    print()

    for key in ("image", "wrist_image"):
        if key in sample:
            print(f"{key}: {_image_summary(sample[key])}")
            _try_print_image_size(key, sample[key])

    print()
    print("all_columns:")

    for key in sorted(sample):
        print(f"  {key}: {type(sample[key]).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Root of the real LeRobot dataset. Defaults to "
            "$HF_LEROBOT_HOME/local/xarm_pi05_data."
        ),
    )

    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help=(
            "Inspect one parquet file directly instead of searching "
            "under --dataset-root."
        ),
    )

    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help=(
            "Row index to inspect. When inspecting all parquet files, "
            "this row is inspected in every file."
        ),
    )

    parser.add_argument(
        "--parquet-index",
        type=int,
        default=None,
        help=(
            "Inspect only one parquet file by its sorted index. "
            "If omitted, all parquet files are inspected."
        ),
    )

    args = parser.parse_args()

    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root
        else _default_dataset_root().resolve()
    )

    direct_parquet = (
        args.parquet.expanduser().resolve()
        if args.parquet
        else None
    )

    # Direct parquet mode: inspect one explicitly provided file.
    if direct_parquet is not None:
        if not direct_parquet.exists():
            raise SystemExit(f"Parquet file does not exist: {direct_parquet}")

        inferred_root = _infer_dataset_root_from_parquet(direct_parquet)
        if args.dataset_root is None and inferred_root is not None:
            dataset_root = inferred_root

        inspect_sample(
            parquet_path=direct_parquet,
            row_index=args.row,
            dataset_root=dataset_root,
        )
        return

    # Dataset-root mode.
    info = _load_json(dataset_root / "meta" / "info.json")

    if info:
        print("meta/info.json:")

        for key in (
            "repo_id",
            "robot_type",
            "fps",
            "total_episodes",
            "total_frames",
            "features",
        ):
            if key in info:
                print(f"  {key}: {_preview(info[key])}")

        print()

    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    task_map = _load_task_map(dataset_root)

    print(f"tasks_file: {tasks_path}")
    print(f"tasks_file_exists: {tasks_path.exists()}")
    print(f"task_map_entries: {len(task_map)}")

    if task_map:
        print("task_map_preview:")

        for task_index, task_text in list(sorted(task_map.items()))[:10]:
            print(f"  {task_index}: {task_text!r}")

    print()

    parquets = _find_parquets(dataset_root)

    if not parquets:
        raise SystemExit(
            f"No parquet files found under {dataset_root / 'data'}.\n"
            "This usually means you are pointing at the light "
            "JSONL/images export, not the real LeRobotDataset.\n"
            "Try:\n"
            f'  find "{dataset_root}" '
            '-path "*/data/*" -name "*.parquet" | head'
        )

    print(f"parquet_files_found: {len(parquets)}")
    print()

    # Inspect one selected parquet when --parquet-index is supplied.
    if args.parquet_index is not None:
        if args.parquet_index < 0 or args.parquet_index >= len(parquets):
            raise SystemExit(
                f"--parquet-index {args.parquet_index} is out of range; "
                f"found {len(parquets)} parquet files"
            )

        inspect_sample(
            parquet_path=parquets[args.parquet_index],
            row_index=args.row,
            dataset_root=dataset_root,
        )
        return

    # Default behavior: inspect every parquet file.
    inspected = 0
    skipped = 0
    failed = 0

    for parquet_index, parquet_path in enumerate(parquets):
        print()
        print("=" * 100)
        print(
            f"PARQUET {parquet_index + 1}/{len(parquets)} "
            f"(index={parquet_index})"
        )
        print("=" * 100)

        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(parquet_path)
            num_rows = parquet_file.metadata.num_rows

            if args.row < 0 or args.row >= num_rows:
                print(
                    f"SKIPPED: requested row {args.row}, "
                    f"but this parquet has {num_rows} rows"
                )
                skipped += 1
                continue

            inspect_sample(
                parquet_path=parquet_path,
                row_index=args.row,
                dataset_root=dataset_root,
            )
            inspected += 1

        except Exception as exc:
            failed += 1
            print(f"FAILED: {parquet_path}")
            print(f"error: {type(exc).__name__}: {exc}")

    print()
    print("=" * 100)
    print("INSPECTION SUMMARY")
    print("=" * 100)
    print(f"dataset_root: {dataset_root}")
    print(f"parquet_files_found: {len(parquets)}")
    print(f"successfully_inspected: {inspected}")
    print(f"skipped_due_to_row_range: {skipped}")
    print(f"failed: {failed}")

    if failed:
        raise SystemExit(
            f"Inspection completed with {failed} failed parquet file(s)."
        )

    print("Inspection completed successfully.")

if __name__ == "__main__":
    main()
