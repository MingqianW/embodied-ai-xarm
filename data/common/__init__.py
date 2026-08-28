"""Training-facing contracts shared by real and simulated xArm data."""

from data.common.records import EpisodeRecord, FrameRecord, SourceBackend
from data.common.schema import XARM_IMAGE_SHAPE, XARM_STATE_COLUMNS

__all__ = [
    "EpisodeRecord",
    "FrameRecord",
    "SourceBackend",
    "XARM_IMAGE_SHAPE",
    "XARM_STATE_COLUMNS",
]

