from __future__ import annotations

from pathlib import Path

from policy_runtime.recording import VideoRecorder


def create_recorder(
    output_dir: Path,
    *,
    fps: int = 30,
    max_frames: int = 18_000,
    fallback_to_frames: bool = True,
) -> VideoRecorder:
    """Create the simulator-independent recorder for an Isaac episode."""

    return VideoRecorder(
        output_dir=output_dir,
        fps=fps,
        max_frames=max_frames,
        fallback_to_frames=fallback_to_frames,
    )
