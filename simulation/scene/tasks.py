from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from simulation.resources import task_config_path


TASK_CONFIG_PATH = task_config_path()
TABLE_TOP_Z = 0.05


def _normalized_name(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def load_task_scene_config(path: Path = TASK_CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("tasks"), dict):
        raise ValueError(f"Invalid task scene config: {path}")
    return config


def task_names(path: Path = TASK_CONFIG_PATH) -> tuple[str, ...]:
    return tuple(load_task_scene_config(path)["tasks"])


def resolve_task(
    task: str, path: Path = TASK_CONFIG_PATH
) -> tuple[str, dict[str, Any]]:
    config = load_task_scene_config(path)
    requested = _normalized_name(task)
    for task_name, spec in config["tasks"].items():
        candidates = [task_name, spec.get("prompt", ""), *(spec.get("aliases") or [])]
        if requested in {_normalized_name(candidate) for candidate in candidates}:
            resolved = dict(spec)
            resolved["name"] = task_name
            return task_name, resolved
    available = ", ".join(config["tasks"])
    raise ValueError(f"Unknown MuJoCo task {task!r}. Available tasks: {available}")
