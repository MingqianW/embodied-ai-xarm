"""Success and safety checks shared by oracle testing and collection."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim_mujoco.environment import MuJoCoEnvironment


def simulation_is_finite(environment: MuJoCoEnvironment) -> bool:
    data = environment.context.data
    return bool(
        np.isfinite(data.time)
        and np.isfinite(data.qpos).all()
        and np.isfinite(data.qvel).all()
        and np.isfinite(data.ctrl).all()
    )


def update_task_success(environment: MuJoCoEnvironment) -> dict[str, Any]:
    runtime = environment.task_runtime
    if runtime is None:
        raise RuntimeError("Environment has not been reset with a task runtime")
    return runtime.update_success()


def accepted_oracle_episode(
    *,
    terminal_stage: str,
    task_metrics: dict[str, Any],
    failure_reason: str | None,
    validation_success: bool | None = None,
) -> bool:
    return bool(
        terminal_stage == "COMPLETE"
        and (
            bool(validation_success)
            if validation_success is not None
            else task_metrics.get("task_success")
        )
        and failure_reason is None
    )
