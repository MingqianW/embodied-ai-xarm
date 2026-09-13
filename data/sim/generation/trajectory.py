"""Versioned trajectory-profile parsing, sampling, and stable paired seeds."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


_PICK_PARAMETERS = frozenset(
    {
        "pregrasp_offset_xy_m", "approach_waypoint_offset_xy_m", "lift_offset_xy_m",
        "tcp_yaw_offset_deg", "pregrasp_clearance_from_object_m",
        "lift_clearance_from_object_m", "approach_speed_scale", "lift_speed_scale",
    }
)
_PLACE_PARAMETERS = frozenset(
    {"preplace_offset_xy_m", "preplace_pepper_height_m", "transfer_speed_scale"}
)


@dataclass(frozen=True)
class Distribution:
    kind: str
    fixed: Any = None
    low: Any = None
    high: Any = None
    mean: Any = None
    std: Any = None


@dataclass(frozen=True)
class TrajectoryProfile:
    name: str
    version: int
    defaults: Mapping[str, Distribution]
    task_overrides: Mapping[str, Mapping[str, Distribution]]
    member_overrides: Mapping[tuple[str, str], Mapping[str, Distribution]]


def stable_seed(*parts: object) -> int:
    """Return a process-independent positive seed from stable identifiers."""

    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**31 - 1) + 1


def scene_seed(base_seed: int, task_id: str, scene_index: int) -> int:
    return stable_seed("paired-scene-v1", base_seed, task_id, scene_index, "scene")


def trajectory_seed(
    base_seed: int, task_id: str, scene_index: int, member_id: str, retry_index: int
) -> int:
    return stable_seed(
        "paired-trajectory-v1", base_seed, task_id, scene_index, member_id, retry_index
    )


def _numeric(value: Any, name: str) -> Any:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        return float(value)
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"{name} vector values must contain exactly two entries")
        return tuple(_numeric(item, name) for item in value)
    raise ValueError(f"{name} must be a number or two-number vector")


def _shape(value: Any) -> tuple[int, ...]:
    return (2,) if isinstance(value, tuple) else ()


def _componentwise(left: Any, right: Any, predicate) -> bool:
    if isinstance(left, tuple):
        return all(predicate(a, b) for a, b in zip(left, right, strict=True))
    return bool(predicate(left, right))


def parse_distribution(value: Any, name: str) -> Distribution:
    """Parse fixed, uniform, or truncated-normal configuration values strictly."""

    if not isinstance(value, Mapping):
        return Distribution("fixed", fixed=_numeric(value, name))
    unknown = set(value) - {"fixed", "uniform", "truncated_normal"}
    if unknown or len(value) != 1:
        raise ValueError(f"{name} must specify exactly one supported distribution")
    if "fixed" in value:
        return Distribution("fixed", fixed=_numeric(value["fixed"], name))
    if "uniform" in value:
        row = value["uniform"]
        if not isinstance(row, Mapping) or set(row) != {"min", "max"}:
            raise ValueError(f"{name}.uniform requires only min and max")
        low, high = _numeric(row["min"], name), _numeric(row["max"], name)
        if _shape(low) != _shape(high) or not _componentwise(low, high, lambda a, b: a <= b):
            raise ValueError(f"{name}.uniform has invalid bounds")
        return Distribution("uniform", low=low, high=high)
    row = value["truncated_normal"]
    if not isinstance(row, Mapping) or set(row) != {"mean", "std", "min", "max"}:
        raise ValueError(f"{name}.truncated_normal requires mean, std, min, and max")
    mean, std = _numeric(row["mean"], name), _numeric(row["std"], name)
    low, high = _numeric(row["min"], name), _numeric(row["max"], name)
    if len({_shape(mean), _shape(std), _shape(low), _shape(high)}) != 1:
        raise ValueError(f"{name}.truncated_normal values must have matching shapes")
    if (not _componentwise(std, std, lambda a, _: a > 0) or
            not _componentwise(low, high, lambda a, b: a <= b) or
            not _componentwise(low, mean, lambda a, b: a <= b) or
            not _componentwise(mean, high, lambda a, b: a <= b)):
        raise ValueError(f"{name}.truncated_normal has invalid bounds")
    return Distribution("truncated_normal", low=low, high=high, mean=mean, std=std)


def _validate_parameter(task_id: str, name: str, distribution: Distribution) -> None:
    allowed = _PLACE_PARAMETERS if task_id == "place_red_pepper_in_ring" else _PICK_PARAMETERS
    if name not in allowed:
        raise ValueError(f"Unsupported trajectory parameter {name!r} for {task_id}")
    vector = name.endswith("_offset_xy_m")
    candidate = distribution.fixed if distribution.kind == "fixed" else distribution.low
    if (isinstance(candidate, tuple)) != vector:
        raise ValueError(f"{name} has the wrong units/shape")
    if name.endswith("_offset_xy_m"):
        extrema = [distribution.fixed] if distribution.kind == "fixed" else [distribution.low, distribution.high]
        if any(max(abs(float(item)) for item in value) > 0.04 for value in extrema):
            raise ValueError(f"{name} must remain within +/-0.04 m")
    elif name == "tcp_yaw_offset_deg":
        extrema = [distribution.fixed] if distribution.kind == "fixed" else [distribution.low, distribution.high]
        if any(abs(float(item)) > 25.0 for item in extrema):
            raise ValueError("tcp_yaw_offset_deg must remain within +/-25 degrees")
    elif name.endswith("clearance_from_object_m"):
        extrema = [distribution.fixed] if distribution.kind == "fixed" else [distribution.low, distribution.high]
        if any(not 0.04 <= float(item) <= 0.16 for item in extrema):
            raise ValueError(f"{name} must be in meters within [0.04, 0.16]")
    elif name == "preplace_pepper_height_m":
        extrema = [distribution.fixed] if distribution.kind == "fixed" else [distribution.low, distribution.high]
        if any(not 0.16 <= float(item) <= 0.28 for item in extrema):
            raise ValueError("preplace_pepper_height_m must be in meters within [0.16, 0.28]")
    elif name.endswith("speed_scale"):
        extrema = [distribution.fixed] if distribution.kind == "fixed" else [distribution.low, distribution.high]
        if any(not 0.5 <= float(item) <= 1.5 for item in extrema):
            raise ValueError(f"{name} must be within [0.5, 1.5]")


def parse_trajectory_profile(value: Any, *, task_ids: set[str], member_ids: Mapping[str, set[str]]) -> TrajectoryProfile:
    if not isinstance(value, Mapping):
        raise ValueError("trajectory_profile must be a mapping")
    if set(value) - {"name", "version", "defaults", "tasks"}:
        raise ValueError("trajectory_profile has unknown fields")
    name, version = str(value.get("name", "")), value.get("version")
    if not name or not isinstance(version, int) or version < 1:
        raise ValueError("trajectory_profile requires a name and positive integer version")
    defaults_row = value.get("defaults", {})
    if not isinstance(defaults_row, Mapping):
        raise ValueError("trajectory_profile.defaults must be a mapping")
    defaults = {str(key): parse_distribution(item, f"trajectory_profile.defaults.{key}") for key, item in defaults_row.items()}
    tasks_row = value.get("tasks", {})
    if not isinstance(tasks_row, Mapping) or set(tasks_row) - task_ids:
        raise ValueError("trajectory_profile.tasks has an unknown task")
    task_overrides: dict[str, Mapping[str, Distribution]] = {}
    member_overrides: dict[tuple[str, str], Mapping[str, Distribution]] = {}
    for task_id, row in tasks_row.items():
        if not isinstance(row, Mapping) or set(row) - {"parameters", "members"}:
            raise ValueError(f"trajectory_profile.tasks.{task_id} has unknown fields")
        parameters = row.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError(f"trajectory_profile.tasks.{task_id}.parameters must be a mapping")
        parsed = {str(key): parse_distribution(item, f"trajectory_profile.tasks.{task_id}.parameters.{key}") for key, item in parameters.items()}
        task_overrides[str(task_id)] = parsed
        members = row.get("members", {})
        if not isinstance(members, Mapping) or set(members) - member_ids[str(task_id)]:
            raise ValueError(f"trajectory_profile.tasks.{task_id}.members has an unknown member")
        for member_id, member in members.items():
            if not isinstance(member, Mapping) or set(member) != {"parameters"} or not isinstance(member["parameters"], Mapping):
                raise ValueError(f"trajectory_profile member {task_id}/{member_id} must contain only parameters")
            member_overrides[(str(task_id), str(member_id))] = {
                str(key): parse_distribution(item, f"trajectory_profile.tasks.{task_id}.members.{member_id}.parameters.{key}")
                for key, item in member["parameters"].items()
            }
    for task_id in task_ids:
        for name_, distribution in defaults.items():
            _validate_parameter(task_id, name_, distribution)
        for name_, distribution in task_overrides.get(task_id, {}).items():
            _validate_parameter(task_id, name_, distribution)
        for member_id in member_ids[task_id]:
            for name_, distribution in member_overrides.get((task_id, member_id), {}).items():
                _validate_parameter(task_id, name_, distribution)
    return TrajectoryProfile(name, version, defaults, task_overrides, member_overrides)


def _sample_value(distribution: Distribution, generator: np.random.Generator) -> Any:
    if distribution.kind == "fixed":
        return distribution.fixed
    low, high = np.asarray(distribution.low), np.asarray(distribution.high)
    if distribution.kind == "uniform":
        value = generator.uniform(low, high)
    else:
        mean, std = np.asarray(distribution.mean), np.asarray(distribution.std)
        value = generator.normal(mean, std)
        # Rejection avoids an additional dependency and keeps values truly bounded.
        for _ in range(32):
            invalid = (value < low) | (value > high)
            if not np.any(invalid):
                break
            value = np.where(invalid, generator.normal(mean, std), value)
        value = np.clip(value, low, high)
    array = np.asarray(value)
    return tuple(float(item) for item in array) if array.ndim else float(array)


def resolve_parameters(profile: TrajectoryProfile | None, *, task_id: str, member_id: str, seed: int) -> dict[str, Any]:
    if profile is None:
        return {}
    values: dict[str, Distribution] = dict(profile.defaults)
    values.update(profile.task_overrides.get(task_id, {}))
    values.update(profile.member_overrides.get((task_id, member_id), {}))
    generator = np.random.default_rng(seed)
    return {key: _sample_value(distribution, generator) for key, distribution in sorted(values.items())}
