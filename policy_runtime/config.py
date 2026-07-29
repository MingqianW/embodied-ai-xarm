from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to load runtime configuration") from exc
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def environment_overrides(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    result: dict[str, Any] = {}
    if env.get("OPENPI_POLICY_HOST"):
        result.setdefault("policy", {})["host"] = env["OPENPI_POLICY_HOST"]
    if env.get("OPENPI_POLICY_PORT"):
        try:
            result.setdefault("policy", {})["port"] = int(env["OPENPI_POLICY_PORT"])
        except ValueError as exc:
            raise ValueError("OPENPI_POLICY_PORT must be an integer") from exc
    if env.get("OPENPI_CHECKPOINT"):
        result.setdefault("policy", {})["checkpoint"] = env["OPENPI_CHECKPOINT"]
    if env.get("ISAAC_SIM_PATH"):
        result.setdefault("isaac", {})["installation_path"] = env["ISAAC_SIM_PATH"]
    if env.get("XARM_ASSET_PATH"):
        result.setdefault("robot", {})["asset_path"] = env["XARM_ASSET_PATH"]
    if env.get("ISAAC_OUTPUT_DIR"):
        result.setdefault("output", {})["root"] = env["ISAAC_OUTPUT_DIR"]
    return result


def resolve_config(
    defaults: Mapping[str, Any],
    *,
    local_config: Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve CLI -> environment -> local file -> defaults precedence."""

    value = deepcopy(dict(defaults))
    if local_config is not None:
        value = deep_merge(value, load_yaml(local_config))
    value = deep_merge(value, environment_overrides(environ))
    if cli_overrides:
        value = deep_merge(value, cli_overrides)
    return value


def repository_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(project_root) / path
