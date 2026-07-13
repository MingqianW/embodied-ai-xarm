"""Inspect one sample from a processed LeRobot parquet dataset.

Examples:
    python fine_tune/inspect_lerobot_parquet_sample.py
    python fine_tune/inspect_lerobot_parquet_sample.py --dataset-root "$HF_LEROBOT_HOME/local/xarm_pi05_data"
    python fine_tune/inspect_lerobot_parquet_sample.py --parquet /path/to/chunk.parquet --row 5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "local/xarm_pi05_data"
REQUIRED_KEYS = ("image", "wrist_image", "state", "actions", "task")


def _default_dataset_root() -> Path:
    hf_lerobot_home = os.environ.get("HF_LEROBOT_HOME")
    if hf_lerobot_home:
        return Path(hf_lerobot_home).expanduser() / DEFAULT_REPO_ID
    return Path.home() / ".cache" / "huggingface" / "lerobot" / DEFAULT_REPO_ID


def _find_parquets(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.glob("data/**/*.parquet"))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        return "{" + ", ".join(f"{k}: {type(v).__name__}" for k, v in value.items()) + "}"
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
        import io
    except Exception:
        print(f"{label}.decoded_size: skipped (PIL is not installed)")
        return

    try:
        if image_bytes:
            with Image.open(io.BytesIO(image_bytes)) as image:
                print(f"{label}.decoded_size: {image.size}, mode={image.mode}")
        elif image_path and Path(image_path).exists():
            with Image.open(image_path) as image:
                print(f"{label}.decoded_size: {image.size}, mode={image.mode}")
        else:
            print(f"{label}.decoded_size: skipped (no embedded bytes or existing path)")
    except Exception as exc:
        print(f"{label}.decoded_size: failed ({exc})")


def inspect_sample(parquet_path: Path, row_index: int, dataset_root: Path | None) -> None:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Install it in the environment that has your LeRobot dataset:\n"
            "  pip install pyarrow\n"
            f"Original import error: {exc}"
        ) from exc

    parquet_file = pq.ParquetFile(parquet_path)
    if row_index < 0 or row_index >= parquet_file.metadata.num_rows:
        raise SystemExit(
            f"--row {row_index} is out of range for {parquet_path} "
            f"(rows: {parquet_file.metadata.num_rows})"
        )

    table = parquet_file.read()
    sample = table.slice(row_index, 1).to_pylist()[0]

    print(f"dataset_root: {dataset_root if dataset_root else '(direct parquet)'}")
    print(f"parquet: {parquet_path}")
    print(f"parquet_rows: {parquet_file.metadata.num_rows}")
    print(f"parquet_row_groups: {parquet_file.metadata.num_row_groups}")
    print(f"selected_row: {row_index}")
    print()

    print("schema:")
    print(table.schema)
    print()

    missing = [key for key in REQUIRED_KEYS if key not in sample]
    print(f"required_keys_missing: {missing if missing else 'none'}")
    print()

    for key in ("episode_index", "frame_index", "timestamp", "task", "prompt"):
        if key in sample:
            print(f"{key}: {_preview(sample[key])}")

    for key in ("state", "actions"):
        if key in sample:
            print(f"{key}.shape: {_shape(sample[key])}")
            print(f"{key}.preview: {_preview(sample[key])}")

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
        help="Root of the real LeRobot dataset. Defaults to $HF_LEROBOT_HOME/local/xarm_pi05_data.",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Inspect this parquet file directly instead of searching under --dataset-root.",
    )
    parser.add_argument("--row", type=int, default=0, help="Row index inside the selected parquet file.")
    parser.add_argument(
        "--parquet-index",
        type=int,
        default=0,
        help="Which parquet file to inspect after sorting data/**/*.parquet.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser() if args.dataset_root else _default_dataset_root()
    parquet_path = args.parquet.expanduser() if args.parquet else None

    if parquet_path is None:
        info = _load_json(dataset_root / "meta" / "info.json")
        if info:
            print("meta/info.json:")
            for key in ("repo_id", "robot_type", "fps", "total_episodes", "total_frames", "features"):
                if key in info:
                    print(f"  {key}: {_preview(info[key])}")
            print()

        parquets = _find_parquets(dataset_root)
        if not parquets:
            raise SystemExit(
                f"No parquet files found under {dataset_root / 'data'}.\n"
                "This usually means you are pointing at the light JSONL/images export, not the real LeRobotDataset.\n"
                "Try: find \"$HF_LEROBOT_HOME/local/xarm_pi05_data\" -path \"*/data/*\" -name \"*.parquet\" | head"
            )
        if args.parquet_index < 0 or args.parquet_index >= len(parquets):
            raise SystemExit(f"--parquet-index {args.parquet_index} is out of range; found {len(parquets)} parquet files")
        parquet_path = parquets[args.parquet_index]

    inspect_sample(parquet_path, args.row, dataset_root if args.parquet is None else None)


if __name__ == "__main__":
    main()
