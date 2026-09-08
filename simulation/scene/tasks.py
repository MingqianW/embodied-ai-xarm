from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from simulation.resources import task_config_path


TASK_CONFIG_PATH = task_config_path()
TABLE_TOP_Z = 0.05


@dataclass(frozen=True)
class SceneRandomizationProfile:
    """Canonical reset randomization selected by generation and evaluation."""

    name: str
    object_xy_range_m: float
    object_yaw_range_deg: float
    joint_noise_rad: float
    layout_profile: str


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


def resolve_scene_randomization_profile(
    name: str, path: Path = TASK_CONFIG_PATH
) -> SceneRandomizationProfile:
    config = load_task_scene_config(path)
    catalog = config.get("catalog") or {}
    profiles = catalog.get("randomization_profiles") or {}
    profile = profiles.get(name)
    if not isinstance(profile, dict):
        available = ", ".join(sorted(str(value) for value in profiles))
        raise ValueError(
            f"Unknown scene randomization profile {name!r}; available: {available}"
        )
    layout_profile = str(profile.get("layout_profile", ""))
    if layout_profile not in (catalog.get("layout_profiles") or {}):
        raise ValueError(
            f"Randomization profile {name!r} references unknown layout "
            f"profile {layout_profile!r}"
        )
    values = (
        float(profile.get("object_xy_range_m", -1.0)),
        float(profile.get("object_yaw_range_deg", -1.0)),
        float(profile.get("joint_noise_rad", -1.0)),
    )
    if any(value < 0.0 for value in values):
        raise ValueError(f"Randomization profile {name!r} has a negative range")
    return SceneRandomizationProfile(name, *values, layout_profile)


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
