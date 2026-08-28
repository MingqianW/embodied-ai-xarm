"""Backend-independent validation for the shared xArm training contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from data.common.schema import (
    TRAINING_ACTION_KEY,
    TRAINING_IMAGE_KEY,
    TRAINING_REQUIRED_KEYS,
    TRAINING_STATE_KEY,
    TRAINING_TASK_KEY,
    TRAINING_WRIST_IMAGE_KEY,
    XARM_ACTION_SHAPE,
    XARM_IMAGE_SHAPE,
    XARM_STATE_SHAPE,
)


ImageValue = str | Path | np.ndarray


def validate_policy_vector(
    values: Any,
    *,
    label: str,
    shape: tuple[int, ...] = XARM_STATE_SHAPE,
) -> np.ndarray:
    """Return an exact-shape finite float32 vector without changing values."""

    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} contains NaN or Inf")
    return np.ascontiguousarray(vector)


def validate_image_reference(value: Any, *, label: str) -> ImageValue:
    if isinstance(value, np.ndarray):
        if value.shape != XARM_IMAGE_SHAPE:
            raise ValueError(
                f"{label} must have shape {XARM_IMAGE_SHAPE}, got {value.shape}"
            )
        if value.dtype != np.uint8:
            raise ValueError(f"{label} must have dtype uint8, got {value.dtype}")
        return np.ascontiguousarray(value)
    if isinstance(value, (str, Path)):
        return value
    raise TypeError(f"{label} must be a path or RGB uint8 array")


def validate_training_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize only the five model-facing record fields."""

    missing = [key for key in TRAINING_REQUIRED_KEYS if key not in record]
    if missing:
        raise ValueError(f"Training record is missing fields: {missing}")
    task = str(record[TRAINING_TASK_KEY])
    if not task.strip():
        raise ValueError("task must be a non-empty string")
    return {
        TRAINING_IMAGE_KEY: validate_image_reference(
            record[TRAINING_IMAGE_KEY], label=TRAINING_IMAGE_KEY
        ),
        TRAINING_WRIST_IMAGE_KEY: validate_image_reference(
            record[TRAINING_WRIST_IMAGE_KEY], label=TRAINING_WRIST_IMAGE_KEY
        ),
        TRAINING_STATE_KEY: validate_policy_vector(
            record[TRAINING_STATE_KEY], label=TRAINING_STATE_KEY
        ),
        TRAINING_ACTION_KEY: validate_policy_vector(
            record[TRAINING_ACTION_KEY],
            label=TRAINING_ACTION_KEY,
            shape=XARM_ACTION_SHAPE,
        ),
        TRAINING_TASK_KEY: task,
    }


def validate_nonnegative_index(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
    try:
        index = int(value)
        exact = float(value) == float(index)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{label} must be a non-negative integer, got {value!r}"
        ) from exc
    if index < 0 or not exact:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
    return index


def validate_timestamp(value: Any) -> float:
    timestamp = float(value)
    if not math.isfinite(timestamp):
        raise ValueError("timestamp must be finite")
    return timestamp
