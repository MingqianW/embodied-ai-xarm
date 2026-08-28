"""Durable human-readable and machine-readable phase status."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.sim.generation.manifest import atomic_write_json


def git_sha(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def write_status(log_root: Path, payload: dict[str, Any]) -> None:
    value = dict(payload)
    value["timestamp"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(Path(log_root) / "CODEX_STATUS.json", value)
    lines = [
        "# Codex Pipeline Status",
        "",
        *(f"- **{key.replace('_', ' ')}:** `{json.dumps(item, ensure_ascii=False)}`"
          for key, item in value.items()),
        "",
    ]
    (Path(log_root) / "CODEX_STATUS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
