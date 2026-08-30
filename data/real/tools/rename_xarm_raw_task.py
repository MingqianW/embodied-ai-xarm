#!/usr/bin/env python3
"""Rename a raw xArm task folder and update episode metadata.

This updates the raw layout consumed by the canonical real-data utilities:

    raw/<task>/episode_xxx/meta.json

Example:

    python -m data.real.tools.rename_xarm_raw_task \
        --old-task pick_up_blue_block \
        --new-task pick_up_blue_cube

The default invocation is a preview. Pass ``--apply`` only after reviewing the
reported rename/move and metadata operations.

If Windows refuses to rename the folder because it is locked by another
process, use --copy to create the renamed raw task folder and update the copied
metadata while leaving the original folder untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from data.real.config import get_raw_data_root


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_task_name(task: str) -> None:
    if not task:
        raise ValueError("task name cannot be empty")
    if task in {".", ".."}:
        raise ValueError(f"invalid task name: {task!r}")
    if any(char in task for char in ("/", "\\", ":")):
        raise ValueError(f"task name must be a single folder name: {task!r}")


def find_episode_dirs(task_dir: Path) -> list[Path]:
    return sorted(path.parent for path in task_dir.glob("*/meta.json"))


def update_meta_files(
    task_dir: Path,
    *,
    old_task: str,
    new_task: str,
    dry_run: bool,
    display_task_dir: Path | None = None,
) -> tuple[int, int]:
    changed = 0
    mismatched = 0

    for episode_dir in find_episode_dirs(task_dir):
        meta_path = episode_dir / "meta.json"
        meta = read_json(meta_path)
        current_task = str(meta.get("task") or "")
        if current_task and current_task != old_task and current_task != new_task:
            mismatched += 1
            print(
                f"WARN {episode_dir.name}: meta task is {current_task!r}, "
                f"expected {old_task!r}; changing it to {new_task!r}"
            )

        if meta.get("task") != new_task:
            changed += 1
            if dry_run:
                display_meta_path = meta_path
                if display_task_dir is not None:
                    display_meta_path = display_task_dir / meta_path.relative_to(task_dir)
                print(f"DRY-RUN update {display_meta_path}: task {current_task!r} -> {new_task!r}")
            else:
                meta["task"] = new_task
                write_json(meta_path, meta)

    return changed, mismatched


def merge_task_dirs(old_dir: Path, new_dir: Path, *, dry_run: bool) -> int:
    episode_dirs = sorted(path for path in old_dir.iterdir() if path.is_dir())
    collisions = [path.name for path in episode_dirs if (new_dir / path.name).exists()]
    if collisions:
        preview = ", ".join(collisions[:10])
        raise SystemExit(f"cannot merge; target already has episode folder(s): {preview}")

    for episode_dir in episode_dirs:
        target = new_dir / episode_dir.name
        if dry_run:
            print(f"DRY-RUN move {episode_dir} -> {target}")
        else:
            shutil.move(str(episode_dir), str(target))

    if not dry_run:
        try:
            old_dir.rmdir()
        except OSError:
            print(f"WARN old task folder is not empty after merge: {old_dir}")

    return len(episode_dirs)


def copy_task_dir(old_dir: Path, new_dir: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"DRY-RUN copy {old_dir} -> {new_dir}")
        return
    shutil.copytree(old_dir, new_dir)


def permission_error_message(old_dir: Path, new_dir: Path, exc: PermissionError) -> str:
    return (
        f"permission denied while renaming {old_dir} -> {new_dir}: {exc}\n"
        "Close any Explorer, editor, Python, image viewer, or sync process that may be using that task folder, "
        "then rerun the command. You can also rerun with --copy to create the new task folder and update the "
        "copied metadata while leaving the original folder in place."
    )


def rename_task(
    raw_root: Path,
    *,
    old_task: str,
    new_task: str,
    merge: bool,
    copy: bool,
    dry_run: bool,
) -> None:
    validate_task_name(old_task)
    validate_task_name(new_task)
    if old_task == new_task:
        raise SystemExit("old task and new task are the same")

    old_dir = raw_root / old_task
    new_dir = raw_root / new_task
    if not raw_root.exists():
        raise SystemExit(f"raw root does not exist: {raw_root}")
    if not old_dir.exists():
        raise SystemExit(f"old task folder does not exist: {old_dir}")

    if new_dir.exists() and copy:
        raise SystemExit(f"new task folder already exists: {new_dir}; remove it or choose another name before --copy")
    if new_dir.exists() and not merge:
        raise SystemExit(f"new task folder already exists: {new_dir} (use --merge to combine episode folders)")

    print(f"raw root: {raw_root}")
    print(f"rename: {old_task!r} -> {new_task!r}")
    print(f"mode: {'dry-run' if dry_run else 'write'}")

    if copy:
        copy_task_dir(old_dir, new_dir, dry_run=dry_run)
        task_dir = old_dir if dry_run else new_dir
        display_task_dir = new_dir if dry_run else None
    elif new_dir.exists():
        moved = merge_task_dirs(old_dir, new_dir, dry_run=dry_run)
        task_dir = new_dir
        display_task_dir = None
        print(f"{'would move' if dry_run else 'moved'} {moved} episode folder(s)")
    else:
        if dry_run:
            print(f"DRY-RUN rename {old_dir} -> {new_dir}")
            task_dir = old_dir
            display_task_dir = new_dir
        else:
            try:
                old_dir.rename(new_dir)
            except PermissionError as exc:
                raise SystemExit(permission_error_message(old_dir, new_dir, exc)) from exc
            task_dir = new_dir
            display_task_dir = None

    changed, mismatched = update_meta_files(
        task_dir,
        old_task=old_task,
        new_task=new_task,
        dry_run=dry_run,
        display_task_dir=display_task_dir,
    )
    print(f"{'would update' if dry_run else 'updated'} {changed} meta.json file(s)")
    if mismatched:
        print(f"metadata task mismatches corrected: {mismatched}")
    if copy and not dry_run:
        print(f"copied raw task remains at old path: {old_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=None, help="Override raw data root from xarm_data_config.json.")
    parser.add_argument("--old-task", required=True, help="Existing task folder/name to rename.")
    parser.add_argument("--new-task", required=True, help="New task folder/name.")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Move episodes into an existing new-task folder if there are no episode name collisions.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy old-task to new-task and update copied metadata, leaving the original task folder untouched.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the previewed filesystem and metadata changes. The default is dry-run.",
    )
    args = parser.parse_args()

    if args.merge and args.copy:
        raise SystemExit("--merge and --copy cannot be used together")

    rename_task(
        get_raw_data_root(args.raw_root),
        old_task=args.old_task,
        new_task=args.new_task,
        merge=args.merge,
        copy=args.copy,
        dry_run=not args.apply,
    )


if __name__ == "__main__":
    main()
