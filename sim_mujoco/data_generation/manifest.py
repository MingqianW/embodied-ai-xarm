"""Atomic JSON state for resumable collection phases."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def config_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def initial_manifest(dataset_version: str, run_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "xarm_mujoco_clean_collection_manifest_v1",
        "dataset_version": dataset_version,
        "run_config_sha256": config_sha256(run_config),
        "complete": False,
        "completed": [],
        "failed_attempts": [],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }


def mark_updated(manifest: dict[str, Any]) -> None:
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
