"""Trim the first and last episodes from a real LeRobot dataset.

Dry-run is the default. Pass --apply to modify the dataset.

Examples:
    python fine_tune/trim_lerobot_edges.py
    python fine_tune/trim_lerobot_edges.py --first 25 --last 25 --apply
    python fine_tune/trim_lerobot_edges.py --dataset-root "$HF_LEROBOT_HOME/local/xarm_pi05_data" --apply
    python fine_tune/trim_lerobot_edges.py --apply --trash-dir /content/drive/MyDrive/embodied_ai_xarm/trim_backup

The script removes the selected episode parquet files and matching media files,
then updates meta/episodes*.jsonl and common totals in meta/info.json. If a
parquet chunk contains both removed and kept episodes, the script rewrites that
parquet with only the kept rows. It does not renumber remaining episode indices.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "local/xarm_pi05_data"


def default_dataset_root() -> Path:
    hf_lerobot_home = os.environ.get("HF_LEROBOT_HOME")
    if hf_lerobot_home:
        return Path(hf_lerobot_home).expanduser() / DEFAULT_REPO_ID
    return Path.home() / ".cache" / "huggingface" / "lerobot" / DEFAULT_REPO_ID


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def import_pyarrow():
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Install it in the same environment as the dataset:\n"
            "  pip install pyarrow\n"
            f"Original import error: {exc}"
        ) from exc
    return pq


def find_episode_indices(dataset_root: Path) -> list[int]:
    episode_records = read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    if episode_records:
        indices = sorted({int(record["episode_index"]) for record in episode_records if "episode_index" in record})
        if indices:
            return indices

    pq = import_pyarrow()
    indices: set[int] = set()
    for parquet_path in sorted(dataset_root.glob("data/**/*.parquet")):
        schema = pq.read_schema(parquet_path)
        if "episode_index" not in schema.names:
            continue
        table = pq.read_table(parquet_path, columns=["episode_index"])
        for row in table.to_pylist():
            if row.get("episode_index") is not None:
                indices.add(int(row["episode_index"]))
    return sorted(indices)


def selected_edge_indices(indices: list[int], first: int, last: int) -> set[int]:
    if first < 0 or last < 0:
        raise SystemExit("--first and --last must be non-negative")
    if first + last >= len(indices):
        raise SystemExit(
            f"Refusing to remove {first + last} episode(s) from a dataset with only {len(indices)} episode(s)."
        )
    return set(indices[:first]) | (set(indices[-last:]) if last else set())


def parquet_episode_indices(parquet_path: Path) -> set[int]:
    pq = import_pyarrow()
    schema = pq.read_schema(parquet_path)
    if "episode_index" not in schema.names:
        return set()
    table = pq.read_table(parquet_path, columns=["episode_index"])
    return {int(row["episode_index"]) for row in table.to_pylist() if row.get("episode_index") is not None}


def rewrite_parquet_without_episodes(parquet_path: Path, remove_indices: set[int]) -> int:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Install it in the same environment as the dataset:\n"
            "  pip install pyarrow\n"
            f"Original import error: {exc}"
        ) from exc

    table = pq.read_table(parquet_path)
    if "episode_index" not in table.column_names:
        return 0

    episode_column = table["episode_index"]
    value_set = pa.array(sorted(remove_indices), type=episode_column.type)
    remove_mask = pc.is_in(episode_column, value_set=value_set)
    keep_mask = pc.invert(remove_mask)
    filtered = table.filter(keep_mask)
    removed_rows = table.num_rows - filtered.num_rows
    if removed_rows:
        pq.write_table(filtered, parquet_path)
    return removed_rows


def episode_name(index: int) -> str:
    return f"episode_{index:06d}"


def related_media_files(dataset_root: Path, episode_indices: set[int]) -> list[Path]:
    names = {episode_name(index) for index in episode_indices}
    related: list[Path] = []
    for top in ("videos", "images"):
        base = dataset_root / top
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.stem in names:
                related.append(path)
    return sorted(related)


def backup_or_delete(path: Path, dataset_root: Path, trash_dir: Path | None) -> None:
    if trash_dir is None:
        path.unlink()
        return
    dst = trash_dir / path.relative_to(dataset_root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dst))


def backup_file(path: Path, dataset_root: Path, trash_dir: Path | None) -> None:
    if trash_dir is None or not path.exists():
        return
    dst = trash_dir / path.relative_to(dataset_root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)


def filter_jsonl_by_episode(path: Path, remove_indices: set[int], dataset_root: Path, trash_dir: Path | None) -> int:
    records = read_jsonl(path)
    if not records:
        return 0
    kept = []
    removed = 0
    for record in records:
        episode_index = record.get("episode_index")
        if episode_index is not None and int(episode_index) in remove_indices:
            removed += 1
        else:
            kept.append(record)
    if removed:
        backup_file(path, dataset_root, trash_dir)
        write_jsonl(path, kept)
    return removed


def update_info_json(dataset_root: Path, remaining_parquets: list[Path], trash_dir: Path | None) -> None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return

    pq = import_pyarrow()
    info = read_json(info_path)
    total_frames = 0
    remaining_episode_indices: set[int] = set()
    remaining_video_files = list((dataset_root / "videos").rglob("*")) if (dataset_root / "videos").exists() else []

    for parquet_path in remaining_parquets:
        parquet_file = pq.ParquetFile(parquet_path)
        total_frames += parquet_file.metadata.num_rows
        schema = pq.read_schema(parquet_path)
        if "episode_index" in schema.names:
            table = pq.read_table(parquet_path, columns=["episode_index"])
            remaining_episode_indices.update(
                int(row["episode_index"]) for row in table.to_pylist() if row.get("episode_index") is not None
            )

    for key in ("total_frames", "num_frames"):
        if key in info:
            info[key] = total_frames
    for key in ("total_episodes", "num_episodes"):
        if key in info:
            info[key] = len(remaining_episode_indices)
    if "total_videos" in info:
        info["total_videos"] = len([path for path in remaining_video_files if path.is_file()])
    if "total_chunks" in info:
        chunks = {path.parent.relative_to(dataset_root / "data").as_posix() for path in remaining_parquets}
        info["total_chunks"] = len(chunks)

    backup_file(info_path, dataset_root, trash_dir)
    write_json(info_path, info)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Real LeRobot dataset root. Defaults to $HF_LEROBOT_HOME/local/xarm_pi05_data.",
    )
    parser.add_argument("--first", type=int, default=25, help="Number of earliest episode indices to remove.")
    parser.add_argument("--last", type=int, default=25, help="Number of latest episode indices to remove.")
    parser.add_argument("--apply", action="store_true", help="Actually modify the dataset.")
    parser.add_argument("--trash-dir", type=Path, default=None, help="Move removed files and metadata backups here.")
    args = parser.parse_args()

    dataset_root = (args.dataset_root or default_dataset_root()).expanduser()
    trash_dir = args.trash_dir.expanduser() if args.trash_dir else None
    parquets = sorted(dataset_root.glob("data/**/*.parquet"))
    if not parquets:
        raise SystemExit(f"No parquet files found under {dataset_root / 'data'}")

    all_indices = find_episode_indices(dataset_root)
    if not all_indices:
        raise SystemExit(f"No episode_index values found in {dataset_root}")
    remove_indices = selected_edge_indices(all_indices, args.first, args.last)

    parquet_indices = {path: parquet_episode_indices(path) for path in parquets}
    parquet_paths_to_delete = [path for path, indices in parquet_indices.items() if indices and indices <= remove_indices]
    parquet_paths_to_rewrite = [
        path
        for path, indices in parquet_indices.items()
        if indices & remove_indices and not indices <= remove_indices
    ]
    related_paths_to_delete = related_media_files(dataset_root, remove_indices)
    remaining_parquets = [path for path in parquets if path not in set(parquet_paths_to_delete)]

    print(f"dataset_root: {dataset_root}")
    print(f"mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"episodes before: {len(all_indices)}")
    print(f"episodes to remove: {len(remove_indices)}")
    print(f"first removed episode indices: {sorted(remove_indices)[:10]}")
    print(f"last removed episode indices: {sorted(remove_indices)[-10:]}")
    print(f"parquet files before: {len(parquets)}")
    print(f"parquet files to remove: {len(parquet_paths_to_delete)}")
    print(f"parquet files to rewrite: {len(parquet_paths_to_rewrite)}")
    print(f"remaining parquet files: {len(remaining_parquets)}")
    print(f"related media files to remove: {len(related_paths_to_delete)}")
    print("first parquet files to remove:")
    for path in parquet_paths_to_delete[:20]:
        print(f"  {path.relative_to(dataset_root)}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to modify the dataset.")
        print("Recommended: add --trash-dir so removed files and metadata backups are recoverable.")
        return

    if trash_dir is not None:
        trash_dir.mkdir(parents=True, exist_ok=True)
        print(f"trash_dir: {trash_dir}")

    for path in parquet_paths_to_delete + related_paths_to_delete:
        if path.exists():
            backup_or_delete(path, dataset_root, trash_dir)

    rewritten_rows = 0
    for path in parquet_paths_to_rewrite:
        if path.exists():
            backup_file(path, dataset_root, trash_dir)
            rewritten_rows += rewrite_parquet_without_episodes(path, remove_indices)
    if rewritten_rows:
        print(f"rewrote mixed parquet files: removed {rewritten_rows} row(s)")

    for meta_name in ("episodes.jsonl", "episodes_stats.jsonl"):
        removed = filter_jsonl_by_episode(dataset_root / "meta" / meta_name, remove_indices, dataset_root, trash_dir)
        if removed:
            print(f"updated meta/{meta_name}: removed {removed} record(s)")

    update_info_json(dataset_root, remaining_parquets, trash_dir)
    print("updated meta/info.json totals")
    print("done")


if __name__ == "__main__":
    main()
