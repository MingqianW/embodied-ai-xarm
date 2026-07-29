from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from policy_runtime.schemas import PolicyObservation


@runtime_checkable
class RobotEnvironment(Protocol):
    """Boundary implemented by each simulator adapter."""

    @property
    def joint_limits(self) -> np.ndarray:
        """Return canonical arm joint limits with shape (6, 2), in radians."""

    def reset(self, seed: int | None = None) -> PolicyObservation: ...

    def observe(self) -> PolicyObservation: ...

    def apply_action(self, action: np.ndarray) -> None: ...

    def step_physics(self, duration_s: float) -> None: ...

    def hold_position(self) -> None: ...

    def is_safe(self) -> bool: ...

    def safety_diagnostics(self) -> dict[str, Any]: ...

    def recording_frames(self) -> dict[str, np.ndarray]: ...

    def close(self) -> None: ...

    def __enter__(self) -> "RobotEnvironment": ...

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None: ...
