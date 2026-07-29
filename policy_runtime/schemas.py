from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


POLICY_SCHEMA_VERSION = "1.0"
CANONICAL_STATE_DIM = 7
CANONICAL_ACTION_DIM = 7


@dataclass(frozen=True)
class PolicyObservation:
    """Canonical policy-facing observation.

    Images are RGB uint8 HWC arrays after the shared preprocessing stage.
    State ordering is six joint angles in radians followed by the xArm gripper
    SDK position convention used by the training dataset.
    """

    base_image: np.ndarray
    wrist_image: np.ndarray
    state: np.ndarray
    prompt: str
    timestamp_s: float
    color_order: str = "RGB"
    frame_ids: dict[str, int | str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = POLICY_SCHEMA_VERSION

    def as_openpi_dict(self) -> dict[str, Any]:
        return {
            "observation/image": self.base_image,
            "observation/wrist_image": self.wrist_image,
            "observation/state": self.state,
            "prompt": self.prompt,
        }


@dataclass(frozen=True)
class PolicyActionChunk:
    actions: np.ndarray
    inference_latency_s: float | None = None
    raw_response: dict[str, Any] | None = None
    schema_version: str = POLICY_SCHEMA_VERSION


@dataclass(frozen=True)
class SafetyResult:
    accepted: bool
    clipped: bool
    reason: str | None
    actions: np.ndarray
    messages: tuple[str, ...] = ()
    rejected_indices: tuple[int, ...] = ()
    schema_version: str = POLICY_SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "accepted": self.accepted,
            "clipped": self.clipped,
            "reason": self.reason,
            "actions": self.actions.tolist(),
            "messages": list(self.messages),
            "rejected_indices": list(self.rejected_indices),
        }
