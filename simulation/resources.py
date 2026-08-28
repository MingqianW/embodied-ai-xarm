"""Portable lookup of canonical simulation resources.

Resource defaults are anchored to the installed ``simulation`` package, never
the process working directory. Environment overrides remain available for
cluster runs and model-variant evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DEFAULT_MODEL_PATH = PACKAGE_ROOT / "assets" / "xarm6" / "xarm6_pick_scene.xml"
DEFAULT_CAMERA_CONFIG_PATH = PACKAGE_ROOT / "config" / "camera_calibration.yaml"
DEFAULT_GRIPPER_CONFIG_PATH = PACKAGE_ROOT / "config" / "gripper_mapping.yaml"
DEFAULT_TASK_CONFIG_PATH = PACKAGE_ROOT / "config" / "task_scenes.yaml"


def _from_environment(
    name: str,
    default: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    values = os.environ if environ is None else environ
    configured = values.get(name)
    path = Path(configured).expanduser() if configured else default
    return path.resolve()


def package_root() -> Path:
    """Return the physical root of the canonical simulation package."""

    return PACKAGE_ROOT


def repository_root(environ: Mapping[str, str] | None = None) -> Path:
    """Return the repository root used by development and workflow tools."""

    return _from_environment("EMBODIED_AI_ROOT", REPOSITORY_ROOT, environ=environ)


def asset_path(*parts: str) -> Path:
    """Resolve a path beneath the package-owned asset directory."""

    path = (PACKAGE_ROOT / "assets").joinpath(*parts).resolve()
    asset_root = (PACKAGE_ROOT / "assets").resolve()
    if path != asset_root and asset_root not in path.parents:
        raise ValueError(f"Simulation asset path escapes asset root: {parts!r}")
    return path


def model_path(environ: Mapping[str, str] | None = None) -> Path:
    return _from_environment(
        "XARM_MUJOCO_MODEL_PATH",
        DEFAULT_MODEL_PATH,
        environ=environ,
    )


def camera_config_path(environ: Mapping[str, str] | None = None) -> Path:
    return _from_environment(
        "XARM_CAMERA_CONFIG_PATH",
        DEFAULT_CAMERA_CONFIG_PATH,
        environ=environ,
    )


def gripper_config_path(environ: Mapping[str, str] | None = None) -> Path:
    return _from_environment(
        "XARM_GRIPPER_CONFIG_PATH",
        DEFAULT_GRIPPER_CONFIG_PATH,
        environ=environ,
    )


def task_config_path(environ: Mapping[str, str] | None = None) -> Path:
    return _from_environment(
        "XARM_TASK_CONFIG_PATH",
        DEFAULT_TASK_CONFIG_PATH,
        environ=environ,
    )
