from __future__ import annotations

from typing import Any

import numpy as np

from policy_runtime.schemas import (
    CANONICAL_ACTION_DIM,
    PolicyActionChunk,
)


DEFAULT_ACTION_HORIZON = 10


def validate_policy_actions(
    actions: np.ndarray,
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    action_dim: int = CANONICAL_ACTION_DIM,
) -> np.ndarray:
    value = np.asarray(actions, dtype=np.float32)
    expected = (int(action_horizon), int(action_dim))
    if value.shape != expected:
        raise ValueError(f"Expected policy actions shape {expected}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Policy actions contain NaN or Inf")
    return np.ascontiguousarray(value)


def decode_policy_response(
    response: dict[str, Any],
    *,
    inference_latency_s: float | None = None,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    action_dim: int = CANONICAL_ACTION_DIM,
) -> PolicyActionChunk:
    if not isinstance(response, dict):
        raise TypeError(f"Policy response must be a mapping, got {type(response).__name__}")
    if "actions" not in response:
        raise KeyError(f"Policy response has no 'actions' key; keys={sorted(response)}")
    actions = validate_policy_actions(
        response["actions"],
        action_horizon=action_horizon,
        action_dim=action_dim,
    )
    return PolicyActionChunk(
        actions=actions,
        inference_latency_s=inference_latency_s,
        raw_response=response,
    )


def action_prefix(chunk: PolicyActionChunk | np.ndarray, count: int) -> np.ndarray:
    actions = chunk.actions if isinstance(chunk, PolicyActionChunk) else np.asarray(chunk)
    if count < 1:
        raise ValueError("Action prefix count must be at least 1")
    if count > actions.shape[0]:
        raise ValueError(f"Requested {count} actions from a chunk of length {actions.shape[0]}")
    return np.asarray(actions[:count], dtype=np.float32).copy()
