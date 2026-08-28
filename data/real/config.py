"""Shared data path configuration for xArm fine-tuning utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/data/real/xarm_data_config.json"
DEFAULT_RAW_DATA_ROOT = Path("fine_tune/data/xarm_pi05_data/raw")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {config_path}")
    return data


def resolve_from_repo(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return repo_root() / value


def get_raw_data_root(raw_root_override: Path | None = None) -> Path:
    if raw_root_override is not None:
        return resolve_from_repo(raw_root_override)

    config = read_config()
    raw_data_root = config.get("raw_data_root", DEFAULT_RAW_DATA_ROOT)
    if not isinstance(raw_data_root, str):
        raise ValueError("xarm_data_config.json field 'raw_data_root' must be a string")
    return resolve_from_repo(raw_data_root)
