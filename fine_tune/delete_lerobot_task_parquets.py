"""Delete LeRobot parquet files for selected LeRobot tasks.

This is intentionally dry-run by default. Pass --apply to modify the dataset.

Examples:
    python fine_tune/delete_lerobot_task_parquets.py
    python fine_tune/delete_lerobot_task_parquets.py --task "pick up the red pepper"
    python fine_tune/delete_lerobot_task_parquets.py --task-file tasks_to_delete.txt --apply
    python fine_tune/delete_lerobot_task_parquets.py --task-contains largest --task-contains smallest --apply
    python fine_tune/delete_lerobot_task_parquets.py --blocked-word largest --blocked-word smallest

After deletion, remaining episodes are reindexed to 0..N-1 by default because
some LeRobot loaders derive expected parquet paths from contiguous episode
indices. Pass --no-reindex only if you know your loader supports sparse indices.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "local/xarm_pi05_data"
DEFAULT_TASK_CONTAINS = ("largest", "smallest")


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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_task_map(dataset_root: Path) -> dict[int, str]:
    task_map: dict[int, str] = {}
    for record in read_jsonl(dataset_root / "meta" / "tasks.jsonl"):
        if "task_index" in record and "task" in record:
            task_map[int(record["task_index"])] = str(record["task"])
    return task_map


def normalize_task(text: str) -> str:
    return " ".join(text.strip().lower().split())


def load_task_file(path: Path) -> list[str]:
    tasks = []
    with path.expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tasks.append(line)
    return tasks


def task_matches(
    task: str | None,
    *,
    exact_tasks: set[str],
    contains_terms: tuple[str, ...],
) -> bool:
    if task is None:
        return False
    normalized = normalize_task(task)
    if normalized in exact_tasks:
        return True
    return any(normalize_task(term) in normalized for term in contains_terms)


def read_parquet_summary(parquet_path: Path, task_map: dict[int, str]) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Install it in the same environment as the dataset:\n"
            "  pip install pyarrow\n"
            f"Original import error: {exc}"
        ) from exc

    schema = pq.read_schema(parquet_path)
    columns = [c for c in ("task", "prompt", "task_index", "episode_index") if c in schema.names]
    table = pq.read_table(parquet_path, columns=columns)
    rows = table.to_pylist()
    tasks: Counter[str | None] = Counter()
    episode_indices: set[int] = set()

    for row in rows:
        if "episode_index" in row and row["episode_index"] is not None:
            episode_indices.add(int(row["episode_index"]))

        task = row.get("task")
        if task is None:
            task = row.get("prompt")
        if task is None and row.get("task_index") is not None:
            task = task_map.get(int(row["task_index"]))
        tasks[task] += 1

    return {
        "path": parquet_path,
        "rows": len(rows),
        "tasks": tasks,
        "episode_indices": episode_indices,
    }


def import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Install it in the same environment as the dataset:\n"
            "  pip install pyarrow\n"
            f"Original import error: {exc}"
        ) from exc
    return pa, pq


def episode_number(episode_index: int) -> str:
    return f"episode_{episode_index:06d}"


def related_episode_files(dataset_root: Path, episode_indices: set[int]) -> list[Path]:
    related: list[Path] = []
    names = {episode_number(index) for index in episode_indices}
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

    destination = trash_dir / path.relative_to(dataset_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))


def backup_file(path: Path, dataset_root: Path, trash_dir: Path | None) -> None:
    if trash_dir is None or not path.exists():
        return
    destination = trash_dir / path.relative_to(dataset_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def filter_meta_file(path: Path, episode_indices_to_delete: set[int], *, apply: bool, trash_dir: Path | None) -> int:
    records = read_jsonl(path)
    if not records:
        return 0

    kept = []
    removed = 0
    for record in records:
        episode_index = record.get("episode_index")
        if episode_index is not None and int(episode_index) in episode_indices_to_delete:
            removed += 1
        else:
            kept.append(record)

    if apply and removed:
        if trash_dir is not None:
            backup_path = trash_dir / path.relative_to(path.parents[1])
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_path)
        write_jsonl(path, kept)
    return removed


def replace_episode_strings(value: Any, index_map: dict[int, int]) -> Any:
    if isinstance(value, str):
        updated = value
        for old_index, new_index in index_map.items():
            updated = updated.replace(episode_number(old_index), episode_number(new_index))
        return updated
    if isinstance(value, list):
        return [replace_episode_strings(item, index_map) for item in value]
    if isinstance(value, dict):
        return {key: replace_episode_strings(item, index_map) for key, item in value.items()}
    return value


def rewrite_meta_jsonl(path: Path, index_map: dict[int, int], dataset_root: Path, trash_dir: Path | None) -> int:
    records = read_jsonl(path)
    if not records:
        return 0

    changed = 0
    updated_records = []
    for record in records:
        updated = replace_episode_strings(record, index_map)
        episode_index = updated.get("episode_index")
        if episode_index is not None and int(episode_index) in index_map:
            updated["episode_index"] = index_map[int(episode_index)]
        if updated != record:
            changed += 1
        updated_records.append(updated)

    if changed:
        backup_file(path, dataset_root, trash_dir)
        write_jsonl(path, updated_records)
    return changed


def rewrite_parquet_episode_index(parquet_path: Path, old_index: int, new_index: int) -> None:
    pa, pq = import_pyarrow()
    table = pq.read_table(parquet_path)
    if "episode_index" not in table.column_names:
        return

    column_index = table.column_names.index("episode_index")
    old_column = table["episode_index"]
    updated_values = [
        new_index if value is not None and int(value) == old_index else value
        for value in old_column.to_pylist()
    ]
    updated_column = pa.array(updated_values, type=old_column.type)
    table = table.set_column(column_index, "episode_index", updated_column)
    pq.write_table(table, parquet_path)


def rename_paths_by_episode_stem(paths: list[Path], index_map: dict[int, int]) -> dict[Path, Path]:
    renames: dict[Path, Path] = {}
    for path in paths:
        for old_index, new_index in index_map.items():
            if path.stem == episode_number(old_index):
                renames[path] = path.with_name(f"{episode_number(new_index)}{path.suffix}")
                break
    return renames


def apply_two_phase_renames(renames: dict[Path, Path]) -> None:
    temp_paths: dict[Path, Path] = {}
    for src, dst in renames.items():
        if src == dst or not src.exists():
            continue
        tmp = src.with_name(f".{src.name}.reindex_tmp")
        if tmp.exists():
            raise SystemExit(f"Temporary reindex path already exists: {tmp}")
        temp_paths[tmp] = dst
        src.rename(tmp)

    for tmp, dst in temp_paths.items():
        if dst.exists():
            raise SystemExit(f"Refusing to overwrite existing reindex target: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(dst)


def reindex_remaining_dataset(
    dataset_root: Path,
    *,
    remaining_parquets: list[Path],
    trash_dir: Path | None,
) -> list[Path]:
    parquet_summaries = [read_parquet_summary(path, load_task_map(dataset_root)) for path in remaining_parquets]
    old_indices = sorted({index for summary in parquet_summaries for index in summary["episode_indices"]})
    index_map = {old_index: new_index for new_index, old_index in enumerate(old_indices)}
    if all(old_index == new_index for old_index, new_index in index_map.items()):
        print("episode indices already contiguous from 0; no reindex needed")
        return remaining_parquets

    print(f"reindexing remaining episodes: {len(index_map)} episode(s)")
    print(f"first reindex pairs: {list(index_map.items())[:10]}")

    parquet_renames: dict[Path, Path] = {}
    for summary in parquet_summaries:
        indices = summary["episode_indices"]
        if len(indices) != 1:
            raise SystemExit(
                f"Cannot safely reindex mixed-episode parquet: {summary['path']}\n"
                "Re-convert with one episode per parquet, or add a custom parquet split step."
            )
        old_index = next(iter(indices))
        new_index = index_map[old_index]
        path = summary["path"]
        backup_file(path, dataset_root, trash_dir)
        rewrite_parquet_episode_index(path, old_index, new_index)
        parquet_renames[path] = path.with_name(f"{episode_number(new_index)}{path.suffix}")

    media_paths = [
        path
        for top in ("videos", "images")
        for base in [dataset_root / top]
        if base.exists()
        for path in base.rglob("*")
        if path.is_file()
    ]
    media_renames = rename_paths_by_episode_stem(media_paths, index_map)
    for path in media_renames:
        backup_file(path, dataset_root, trash_dir)

    apply_two_phase_renames(parquet_renames)
    apply_two_phase_renames(media_renames)

    for meta_path in sorted((dataset_root / "meta").glob("*.jsonl")):
        changed = rewrite_meta_jsonl(meta_path, index_map, dataset_root, trash_dir)
        if changed:
            print(f"reindexed meta/{meta_path.name}: updated {changed} record(s)")

    return [parquet_renames.get(path, path) for path in remaining_parquets]


def update_info_json(dataset_root: Path, *, remaining_parquets: list[Path], apply: bool, trash_dir: Path | None) -> None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return

    try:
        import pyarrow.parquet as pq
    except Exception:
        return

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    total_frames = 0
    episode_indices: set[int] = set()
    for parquet in remaining_parquets:
        parquet_file = pq.ParquetFile(parquet)
        total_frames += parquet_file.metadata.num_rows
        schema = pq.read_schema(parquet)
        if "episode_index" in schema.names:
            table = pq.read_table(parquet, columns=["episode_index"])
            for row in table.to_pylist():
                if row.get("episode_index") is not None:
                    episode_indices.add(int(row["episode_index"]))

    for key in ("total_frames", "num_frames"):
        if key in info:
            info[key] = total_frames
    for key in ("total_episodes", "num_episodes"):
        if key in info:
            info[key] = len(episode_indices) if episode_indices else len(remaining_parquets)

    if apply:
        if trash_dir is not None:
            backup_path = trash_dir / "meta" / "info.json"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(info_path, backup_path)
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Real LeRobot dataset root. Defaults to $HF_LEROBOT_HOME/local/xarm_pi05_data.",
    )
    parser.add_argument("--task", action="append", default=None, help="Exact task text to remove. Repeat as needed.")
    parser.add_argument(
        "--task-file",
        type=Path,
        default=None,
        help="Text file with one exact task per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--task-contains",
        action="append",
        default=None,
        help="Case-insensitive substring to match task text. Repeat as needed.",
    )
    parser.add_argument(
        "--blocked-word",
        action="append",
        default=None,
        help="Deprecated alias for --task-contains. Repeat for multiple substrings.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually delete or move matching files.")
    parser.add_argument(
        "--no-reindex",
        action="store_true",
        help="Do not renumber remaining episodes to 0..N-1 after deletion.",
    )
    parser.add_argument(
        "--trash-dir",
        type=Path,
        default=None,
        help="Move removed files here instead of permanently deleting them.",
    )
    parser.add_argument(
        "--keep-meta",
        action="store_true",
        help="Do not update meta/episodes*.jsonl or meta/info.json.",
    )
    args = parser.parse_args()

    dataset_root = (args.dataset_root or default_dataset_root()).expanduser()
    exact_task_values = list(args.task or [])
    if args.task_file is not None:
        exact_task_values.extend(load_task_file(args.task_file))
    exact_tasks = {normalize_task(task) for task in exact_task_values}
    contains_terms = tuple(args.task_contains or args.blocked_word or ())
    if args.task_contains and args.blocked_word:
        contains_terms = tuple(args.task_contains + args.blocked_word)
    if not exact_tasks and not contains_terms:
        contains_terms = DEFAULT_TASK_CONTAINS

    trash_dir = args.trash_dir.expanduser() if args.trash_dir else None

    parquets = sorted(dataset_root.glob("data/**/*.parquet"))
    if not parquets:
        raise SystemExit(f"No parquet files found under {dataset_root / 'data'}")

    task_map = load_task_map(dataset_root)
    summaries = [read_parquet_summary(path, task_map) for path in parquets]
    to_remove = [
        summary
        for summary in summaries
        if any(task_matches(task, exact_tasks=exact_tasks, contains_terms=contains_terms) for task in summary["tasks"])
    ]
    episode_indices_to_delete = {
        episode_index for summary in to_remove for episode_index in summary["episode_indices"]
    }
    parquet_paths_to_delete = [summary["path"] for summary in to_remove]
    related_paths_to_delete = related_episode_files(dataset_root, episode_indices_to_delete)
    remaining_parquets = [path for path in parquets if path not in set(parquet_paths_to_delete)]

    print(f"dataset_root: {dataset_root}")
    print(f"exact_tasks: {sorted(exact_task_values)}")
    print(f"task_contains: {contains_terms}")
    print(f"mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"parquet files before: {len(parquets)}")
    print(f"matching parquet files: {len(parquet_paths_to_delete)}")
    print(f"remaining parquet files: {len(remaining_parquets)}")
    print(f"matching episode indices: {len(episode_indices_to_delete)}")
    print(f"related image/video files: {len(related_paths_to_delete)}")
    if remaining_parquets and not args.no_reindex:
        remaining_indices = sorted(
            episode_index
            for summary in summaries
            if summary["path"] in set(remaining_parquets)
            for episode_index in summary["episode_indices"]
        )
        needs_reindex = remaining_indices != list(range(len(remaining_indices)))
        print(f"will reindex remaining episodes: {needs_reindex}")

    task_counts: Counter[str | None] = Counter()
    row_count = 0
    for summary in to_remove:
        task_counts.update(summary["tasks"])
        row_count += int(summary["rows"])
    print(f"rows to remove: {row_count}")
    print("tasks to remove:")
    for task, count in task_counts.most_common():
        print(f"  {count:8d}  {task}")

    print("first matching parquet files:")
    for path in parquet_paths_to_delete[:20]:
        print(f"  {path.relative_to(dataset_root)}")
    if len(parquet_paths_to_delete) > 20:
        print(f"  ... {len(parquet_paths_to_delete) - 20} more")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to remove these files.")
        print("Safer option: add --trash-dir /content/drive/MyDrive/embodied_ai_xarm/pruned_lerobot_backup")
        return

    if trash_dir is not None:
        trash_dir.mkdir(parents=True, exist_ok=True)
        print(f"moving removed files to: {trash_dir}")

    for path in parquet_paths_to_delete + related_paths_to_delete:
        if path.exists():
            backup_or_delete(path, dataset_root, trash_dir)

    if not args.keep_meta:
        for meta_name in ("episodes.jsonl", "episodes_stats.jsonl"):
            removed = filter_meta_file(
                dataset_root / "meta" / meta_name,
                episode_indices_to_delete,
                apply=True,
                trash_dir=trash_dir,
            )
            if removed:
                print(f"updated meta/{meta_name}: removed {removed} record(s)")
        if not args.no_reindex:
            remaining_parquets = reindex_remaining_dataset(
                dataset_root,
                remaining_parquets=remaining_parquets,
                trash_dir=trash_dir,
            )
        update_info_json(dataset_root, remaining_parquets=remaining_parquets, apply=True, trash_dir=trash_dir)
        print("updated meta/info.json totals when total_* or num_* keys were present")

    print("done")


if __name__ == "__main__":
    main()
