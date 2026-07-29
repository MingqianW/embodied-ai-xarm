from __future__ import annotations

import time
from typing import Any

import numpy as np

from policy_runtime.image_preprocessing import (
    ImagePreprocessingConfig,
    preprocess_policy_image,
)
from policy_runtime.schemas import CANONICAL_STATE_DIM, PolicyObservation


OPENPI_OBSERVATION_KEYS = frozenset(
    {"observation/image", "observation/wrist_image", "observation/state", "prompt"}
)


def validate_canonical_state(state: np.ndarray) -> np.ndarray:
    value = np.asarray(state, dtype=np.float32)
    if value.shape != (CANONICAL_STATE_DIM,):
        raise ValueError(f"Canonical state must have shape (7,), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Canonical state contains NaN or Inf")
    return np.ascontiguousarray(value)


def build_policy_observation(
    base_image: np.ndarray,
    wrist_image: np.ndarray,
    state: np.ndarray,
    prompt: str,
    *,
    base_preprocessing: ImagePreprocessingConfig = ImagePreprocessingConfig(),
    wrist_preprocessing: ImagePreprocessingConfig = ImagePreprocessingConfig(),
    timestamp_s: float | None = None,
    frame_ids: dict[str, int | str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> PolicyObservation:
    if not isinstance(prompt, str):
        raise TypeError(f"Prompt must be str, got {type(prompt).__name__}")
    observation = PolicyObservation(
        base_image=preprocess_policy_image(base_image, base_preprocessing),
        wrist_image=preprocess_policy_image(wrist_image, wrist_preprocessing),
        state=validate_canonical_state(state),
        prompt=prompt,
        timestamp_s=time.time() if timestamp_s is None else float(timestamp_s),
        color_order="RGB",
        frame_ids=dict(frame_ids or {}),
        metadata=dict(metadata or {}),
    )
    validate_policy_observation(observation)
    return observation


def validate_policy_observation(observation: PolicyObservation | dict[str, Any]) -> None:
    if isinstance(observation, PolicyObservation):
        payload = observation.as_openpi_dict()
        color_order = observation.color_order
        timestamp_s = observation.timestamp_s
    else:
        payload = observation
        color_order = "RGB"
        timestamp_s = None

    if set(payload) != OPENPI_OBSERVATION_KEYS:
        raise ValueError(
            f"Observation keys must be {sorted(OPENPI_OBSERVATION_KEYS)}, "
            f"got {sorted(payload)}"
        )
    for key in ("observation/image", "observation/wrist_image"):
        image = payload[key]
        if not isinstance(image, np.ndarray) or image.shape != (224, 224, 3):
            raise ValueError(f"{key} must have shape (224, 224, 3)")
        if image.dtype != np.uint8:
            raise ValueError(f"{key} must have dtype uint8, got {image.dtype}")
    validate_canonical_state(payload["observation/state"])
    if not isinstance(payload["prompt"], str):
        raise ValueError("prompt must be str")
    if color_order != "RGB":
        raise ValueError(f"Policy-facing images must be RGB, got {color_order}")
    if timestamp_s is not None and not np.isfinite(timestamp_s):
        raise ValueError("Observation timestamp must be finite")
