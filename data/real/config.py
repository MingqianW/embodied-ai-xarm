"""Shared path configuration for externally collected xArm data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import warnings


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/data/real/xarm_data_config.json"
DEFAULT_RAW_DATA_ROOT = "datasets/real/raw"
LEGACY_RAW_DATA_ROOT = "fine_tune/data/xarm_pi05_data/raw"


class RealDataPathWarning(UserWarning):
    """Warn about a compatibility fallback or unresolved real-data root."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_config(config_path: Path | None = None) -> dict[str, Any]:
    config_path = CONFIG_PATH if config_path is None else config_path
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
    if "raw_data_root" in config:
        raw_data_root = config["raw_data_root"]
        if not isinstance(raw_data_root, str):
            raise ValueError("xarm_data_config.json field 'raw_data_root' must be a string")
        return resolve_from_repo(raw_data_root)

    canonical = resolve_from_repo(DEFAULT_RAW_DATA_ROOT)
    if canonical.exists():
        return canonical

    legacy = resolve_from_repo(LEGACY_RAW_DATA_ROOT)
    if legacy.exists():
        warnings.warn(
            f"Using legacy ignored real-data path {legacy}. Migrate data to "
            f"{canonical} or set raw_data_root in {CONFIG_PATH}; no data was moved.",
            RealDataPathWarning,
            stacklevel=2,
        )
        return legacy

    warnings.warn(
        f"Real-data root does not exist. Expected canonical path {canonical}; "
        f"create it, pass --raw-root, or set raw_data_root in {CONFIG_PATH}.",
        RealDataPathWarning,
        stacklevel=2,
    )
    return canonical
