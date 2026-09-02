"""Minimal generator boundary between task motion intent and shared collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class GeneratorContext:
    """Inputs fixed for one reset, before a generator emits any action."""

    environment: Any
    pipeline_config: Any
    task: Any
    requested_episode_index: int
    retry_index: int
    seed: int


@dataclass(frozen=True)
class GeneratorInitialization:
    success: bool = True
    metadata: dict[str, Any] | None = None
    diagnostic_frames: dict[str, list[np.ndarray]] | None = None
    failure_reason: str | None = None


class EpisodeGenerator(Protocol):
    """Task-owned motion intent that emits canonical seven-dimensional actions."""

    generator_id: str
    generator_version: str
    initialization: GeneratorInitialization

    @property
    def terminal(self) -> bool: ...

    @property
    def stage(self) -> Any: ...

    @property
    def failure_reason(self) -> str | None: ...

    def next_action(self) -> np.ndarray | None: ...

    def notify_post_step(
        self, *, task_metrics: dict[str, Any], collision: dict[str, Any], simulation_finite: bool
    ) -> None: ...

    def stability_metadata(self) -> dict[str, Any]: ...

    def transition_log(self) -> list[dict[str, Any]]: ...

    def plan_metadata(self) -> dict[str, Any]: ...

    def validation_metadata(self) -> dict[str, Any]: ...

    def accepted(self) -> bool: ...


class ControllerEpisodeGenerator:
    """Adapter for the existing reusable Pick/Place controller state machines."""

    def __init__(
        self,
        controller: Any,
        *,
        generator_id: str,
        generator_version: str = "v1",
        kind: str,
        initialization: GeneratorInitialization | None = None,
    ) -> None:
        self._controller = controller
        self.generator_id = generator_id
        self.generator_version = generator_version
        self._kind = kind
        self.initialization = initialization or GeneratorInitialization()

    @property
    def terminal(self) -> bool:
        return bool(self._controller.terminal)

    @property
    def stage(self) -> Any:
        return self._controller.stage

    @property
    def failure_reason(self) -> str | None:
        return self._controller.failure_reason

    def next_action(self) -> np.ndarray | None:
        action = self._controller.next_action()
        if action is None:
            return None
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Episode generators must emit one finite canonical [j1..j6, gripper_raw] action")
        return action

    def notify_post_step(
        self, *, task_metrics: dict[str, Any], collision: dict[str, Any], simulation_finite: bool
    ) -> None:
        self._controller.notify_post_step(
            task_metrics=task_metrics, collision=collision, simulation_finite=simulation_finite
        )

    def stability_metadata(self) -> dict[str, Any]:
        return self._controller.stability_metadata()

    def transition_log(self) -> list[dict[str, Any]]:
        return self._controller.transition_log()

    def plan_metadata(self) -> dict[str, Any]:
        return self._controller.plan.to_json()

    def validation_metadata(self) -> dict[str, Any]:
        stability = self.stability_metadata()
        if self._kind == "place":
            return {"place_initial_grasp": self.initialization.metadata or {}, "stable_place": stability}
        return {"stable_grasp": stability}

    def accepted(self) -> bool:
        stability = self.stability_metadata()
        if self._kind == "place":
            return bool(
                stability.get("stable_place_success")
                and (self.initialization.metadata or {}).get("initial_grasp_success")
                and stability.get("release_detected")
            )
        return bool(stability.get("stable_grasp_success"))


class RejectedEpisodeGenerator:
    """Reports a deterministic pre-controller failure without touching recording."""

    def __init__(self, *, generator_id: str, initialization: GeneratorInitialization) -> None:
        self.generator_id = generator_id
        self.generator_version = "v1"
        self.initialization = initialization
        self.stage = "FAILED"
        self.failure_reason = initialization.failure_reason

    @property
    def terminal(self) -> bool:
        return True

    def next_action(self) -> None:
        return None

    def notify_post_step(self, **_: Any) -> None:
        return None

    def stability_metadata(self) -> dict[str, Any]:
        return {}

    def transition_log(self) -> list[dict[str, Any]]:
        return []

    def plan_metadata(self) -> dict[str, Any]:
        return {}

    def validation_metadata(self) -> dict[str, Any]:
        return {"initialization": self.initialization.metadata or {}}

    def accepted(self) -> bool:
        return False
