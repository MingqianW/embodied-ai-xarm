"""Portable path resolution for MuJoCo workflows.

Environment variables override repository-relative defaults. No directory is
created merely by importing this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(
    name: str,
    default: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    env = os.environ if environ is None else environ
    value = env.get(name)
    if not value:
        return default.resolve()
    return Path(value).expanduser().resolve()


def repository_root(environ: Mapping[str, str] | None = None) -> Path:
    return _path_from_env("EMBODIED_AI_ROOT", SOURCE_ROOT, environ=environ)


def mujoco_output_root(environ: Mapping[str, str] | None = None) -> Path:
    root = repository_root(environ)
    return _path_from_env(
        "MUJOCO_OUTPUT_ROOT",
        root / "sim_mujoco" / "output",
        environ=environ,
    )


def mujoco_dataset_root(environ: Mapping[str, str] | None = None) -> Path:
    return _path_from_env(
        "MUJOCO_DATASET_ROOT",
        mujoco_output_root(environ) / "datasets",
        environ=environ,
    )


def openpi_root(environ: Mapping[str, str] | None = None) -> Path:
    return _path_from_env(
        "OPENPI_ROOT",
        repository_root(environ) / "third_party" / "openpi",
        environ=environ,
    )


def openpi_checkpoint_root(environ: Mapping[str, str] | None = None) -> Path | None:
    env = os.environ if environ is None else environ
    value = env.get("OPENPI_CHECKPOINT_ROOT")
    return Path(value).expanduser().resolve() if value else None


def simulation_root(environ: Mapping[str, str] | None = None) -> Path:
    return repository_root(environ) / "sim_mujoco"


def active_model_path(environ: Mapping[str, str] | None = None) -> Path:
    return simulation_root(environ) / "assets" / "xarm6" / "xarm6_pick_scene.xml"


def camera_config_path(environ: Mapping[str, str] | None = None) -> Path:
    return simulation_root(environ) / "config" / "camera_calibration.yaml"


def task_config_path(environ: Mapping[str, str] | None = None) -> Path:
    return simulation_root(environ) / "config" / "task_scenes.yaml"
