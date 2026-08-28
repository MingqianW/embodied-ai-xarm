"""Load validated, package-owned simulation configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from simulation.resources import camera_config_path
from simulation.resources import gripper_config_path


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a mapping: {resolved}")
    return value


def load_camera_calibration(path: Path | None = None) -> dict[str, Any]:
    resolved = camera_config_path() if path is None else Path(path).resolve()
    config = _load_mapping(resolved, label="Camera calibration config")
    for camera_name in ("base_camera", "wrist_camera"):
        parameters = config.get(camera_name)
        if not isinstance(parameters, dict):
            raise ValueError(f"Missing {camera_name} mapping in {resolved}")
        missing = {
            "position",
            "target",
            "fovy_deg",
        } - parameters.keys()
        if missing:
            raise ValueError(
                f"Missing {camera_name} fields in {resolved}: {sorted(missing)}"
            )
    if not isinstance(config.get("render"), dict):
        raise ValueError(f"Missing render mapping in {resolved}")
    return config


def load_gripper_mapping(path: Path | None = None) -> dict[str, Any]:
    resolved = gripper_config_path() if path is None else Path(path).resolve()
    config = _load_mapping(resolved, label="Gripper mapping config")
    mapping = config.get("gripper_mapping", config)
    if not isinstance(mapping, dict):
        raise ValueError(f"Missing gripper_mapping in {resolved}")
    required = {"raw_closed", "raw_open"}
    missing = required - mapping.keys()
    if missing:
        raise ValueError(f"Missing gripper fields in {resolved}: {sorted(missing)}")
    return deepcopy(mapping)


def load_simulation_config(
    camera_path: Path | None = None,
    gripper_path: Path | None = None,
) -> dict[str, Any]:
    """Return the runtime config consumed by existing workflow adapters.

    Frozen diagnostic camera configs historically embed their gripper mapping.
    That explicitly supported legacy form takes precedence only when present;
    canonical configuration keeps camera and gripper ownership separate.
    """

    camera = load_camera_calibration(camera_path)
    embedded_mapping = camera.pop("gripper_mapping", None)
    if embedded_mapping is not None and gripper_path is None:
        mapping = embedded_mapping
    else:
        mapping = load_gripper_mapping(gripper_path)
    if not isinstance(mapping, dict):
        raise ValueError("gripper_mapping must be a mapping")
    camera["gripper_mapping"] = deepcopy(mapping)
    return camera
