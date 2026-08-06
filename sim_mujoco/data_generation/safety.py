"""Exact-root output replacement and group-permission safety helpers."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUTHORIZED_ROOTS = frozenset(
    {
        Path("/work/nvme/bfmk/mw89/mujoco_datasets/raw/xarm_mujoco_clean_multitask_stable_v3"),
        Path("/work/nvme/bfmk/mw89/mujoco_datasets/local/xarm_mujoco_clean_multitask_stable_v3"),
        Path("/work/nvme/bfmk/mw89/mujoco_datasets/smoke/xarm_mujoco_clean_multitask_stable_v3"),
        Path("/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v3"),
    }
)
REJECTED_ROOTS = frozenset(
    Path(value)
    for value in (
        "/",
        "/work",
        "/work/nvme",
        "/work/nvme/bfmk",
        "/work/nvme/bfmk/mw89",
    )
)
LOG_PARENT = Path("/work/nvme/bfmk/mw89/logs")


def validate_authorized_root(path: Path) -> Path:
    raw = Path(path).expanduser()
    if not str(raw).strip():
        raise ValueError("Output path must be nonempty")
    if raw.is_symlink():
        raise ValueError(f"Output root must not be a symbolic link: {raw}")
    resolved = raw.resolve(strict=False)
    if resolved in REJECTED_ROOTS or resolved not in AUTHORIZED_ROOTS:
        raise ValueError(f"Output root is not an exact authorized v3 root: {resolved}")
    if resolved.exists() and resolved.is_symlink():
        raise ValueError(f"Resolved output root must not be a symbolic link: {resolved}")
    return resolved


def path_inventory(path: Path) -> dict[str, Any]:
    resolved = validate_authorized_root(path)
    files = 0
    directories = 0
    bytes_total = 0
    if resolved.exists():
        for current, dirnames, filenames in os.walk(resolved, followlinks=False):
            directories += len(dirnames)
            for name in filenames:
                candidate = Path(current) / name
                if candidate.is_symlink():
                    continue
                files += 1
                bytes_total += candidate.stat().st_size
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "files": files,
        "directories": directories,
        "bytes": bytes_total,
    }


def _write_preoverwrite_inventory(rows: list[dict[str, Any]], timestamp: str) -> Path:
    LOG_PARENT.mkdir(parents=True, exist_ok=True)
    safe_timestamp = timestamp.replace(":", "").replace("+", "_")
    path = LOG_PARENT / (
        "xarm_mujoco_clean_multitask_stable_v3_preoverwrite_inventory_"
        f"{safe_timestamp}.txt"
    )
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite inventory: {path}")
    lines = [f"timestamp={timestamp}"]
    for row in rows:
        lines.append(json.dumps(row, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def replace_authorized_roots(
    paths: list[Path],
    *,
    overwrite: bool,
    git_sha: str,
    config_path: Path,
) -> dict[str, Any]:
    if not overwrite:
        raise ValueError("Explicit --overwrite authorization is required")
    resolved = [validate_authorized_root(path) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Duplicate output roots are not allowed")
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = [path_inventory(path) for path in resolved]
    inventory = _write_preoverwrite_inventory(rows, timestamp)
    for path in resolved:
        print(f"AUTHORIZED_OVERWRITE path={path} current={path_inventory(path)}")
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=False)
        os.chmod(path, 0o2770)
        try:
            shutil.chown(path, group="delta_bfmk")
        except LookupError as exc:
            raise RuntimeError("Required group delta_bfmk does not exist") from exc
        if shutil.which("setfacl"):
            subprocess.run(
                [
                    "setfacl",
                    "-m",
                    "d:g:delta_bfmk:rx,d:m::rwx",
                    str(path),
                ],
                check=True,
            )
        marker = {
            "schema_version": 1,
            "overwritten_utc": timestamp,
            "hostname": os.uname().nodename,
            "git_sha": git_sha,
            "config_path": str(Path(config_path).resolve()),
            "removed_and_recreated_path": str(path),
            "preoverwrite_inventory": str(inventory),
        }
        (path / "OVERWRITE_MARKER.json").write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8"
        )
        print(f"RECREATED path={path}")
    return {
        "timestamp": timestamp,
        "paths": [str(path) for path in resolved],
        "preoverwrite_inventory": str(inventory),
    }


def apply_group_permissions(paths: list[Path]) -> None:
    resolved = [validate_authorized_root(path) for path in paths]
    for root in resolved:
        if not root.exists():
            raise FileNotFoundError(root)
        for current, dirnames, filenames in os.walk(root, followlinks=False):
            directory = Path(current)
            shutil.chown(directory, group="delta_bfmk")
            os.chmod(directory, os.stat(directory).st_mode | stat.S_IRGRP | stat.S_IXGRP | stat.S_ISGID)
            for name in filenames:
                path = directory / name
                if path.is_symlink():
                    continue
                shutil.chown(path, group="delta_bfmk")
                os.chmod(path, os.stat(path).st_mode | stat.S_IRGRP)
    if shutil.which("setfacl"):
        for root in resolved:
            for current, dirnames, _ in os.walk(root, followlinks=False):
                for directory in (Path(current), *(Path(current) / name for name in dirnames)):
                    subprocess.run(
                        [
                            "setfacl",
                            "-m",
                            "d:g:delta_bfmk:rx,d:m::rwx",
                            str(directory),
                        ],
                        check=True,
                    )
