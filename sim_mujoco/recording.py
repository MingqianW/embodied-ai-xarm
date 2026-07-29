from __future__ import annotations

from typing import Any

import numpy as np

from sim_mujoco.environment import MuJoCoEnvironment


def capture_recording_frames(environment: MuJoCoEnvironment) -> dict[str, np.ndarray]:
    """MuJoCo-specific capture hook consumed by the shared recorder."""

    return environment.recording_frames()


def recording_diagnostics(environment: MuJoCoEnvironment) -> dict[str, Any]:
    frames = environment.recording_frames()
    return {
        name: {"shape": list(frame.shape), "dtype": str(frame.dtype)}
        for name, frame in frames.items()
    }
