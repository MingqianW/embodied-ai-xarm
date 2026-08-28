"""Canonical in-memory frame and episode records shared by both backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from data.common.validation import (
    ImageValue,
    validate_nonnegative_index,
    validate_timestamp,
    validate_training_record,
)


class SourceBackend(str, Enum):
    REAL = "real"
    SIM = "sim"


@dataclass(frozen=True)
class FrameRecord:
    """One aligned training sample plus non-model indexing/provenance fields.

    ``actions`` is the absolute target at this frame.  For the tracked raw xArm
    format and the compatible MuJoCo format it is state from raw row ``t + 1``;
    the final raw row therefore creates no training frame.
    """

    image: ImageValue
    wrist_image: ImageValue
    state: np.ndarray
    actions: np.ndarray
    task: str
    episode_index: int
    frame_index: int
    timestamp: float
    source: SourceBackend
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "FrameRecord":
        """Build the canonical frame while retaining all non-contract metadata."""

        contract_keys = {
            "image",
            "wrist_image",
            "state",
            "actions",
            "task",
            "episode_index",
            "frame_index",
            "timestamp",
            "source",
        }
        return cls(
            image=record["image"],
            wrist_image=record["wrist_image"],
            state=record["state"],
            actions=record["actions"],
            task=str(record["task"]),
            episode_index=record["episode_index"],
            frame_index=record["frame_index"],
            timestamp=record["timestamp"],
            source=SourceBackend(record["source"]),
            metadata={
                key: value for key, value in record.items() if key not in contract_keys
            },
        )

    def __post_init__(self) -> None:
        training = validate_training_record(self.as_unvalidated_writer_record())
        object.__setattr__(self, "image", training["image"])
        object.__setattr__(self, "wrist_image", training["wrist_image"])
        object.__setattr__(self, "state", training["state"])
        object.__setattr__(self, "actions", training["actions"])
        object.__setattr__(self, "task", training["task"])
        object.__setattr__(
            self,
            "episode_index",
            validate_nonnegative_index(self.episode_index, label="episode_index"),
        )
        object.__setattr__(
            self,
            "frame_index",
            validate_nonnegative_index(self.frame_index, label="frame_index"),
        )
        object.__setattr__(self, "timestamp", validate_timestamp(self.timestamp))
        object.__setattr__(self, "source", SourceBackend(self.source))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def as_unvalidated_writer_record(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "wrist_image": self.wrist_image,
            "state": self.state,
            "actions": self.actions,
            "task": self.task,
        }

    def as_writer_record(self) -> dict[str, Any]:
        return validate_training_record(self.as_unvalidated_writer_record())

    def as_record(self) -> dict[str, Any]:
        """Return model fields plus indexing, provenance, and backend metadata."""

        return {
            **self.as_writer_record(),
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "source": self.source.value,
            **self.metadata,
        }


@dataclass(frozen=True)
class EpisodeRecord:
    """A non-empty, contiguous sequence of frames from one backend."""

    episode_index: int
    source: SourceBackend
    frames: tuple[FrameRecord, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        episode_index = validate_nonnegative_index(
            self.episode_index, label="episode_index"
        )
        source = SourceBackend(self.source)
        frames = tuple(self.frames)
        if not frames:
            raise ValueError("Canonical episode must contain at least one frame")
        for expected_frame_index, frame in enumerate(frames):
            if frame.episode_index != episode_index:
                raise ValueError("Frame episode_index does not match its episode")
            if frame.frame_index != expected_frame_index:
                raise ValueError("Frame indices must be contiguous from zero")
            if frame.source is not source:
                raise ValueError("Frame source does not match its episode")
            if (
                expected_frame_index
                and frame.timestamp < frames[expected_frame_index - 1].timestamp
            ):
                raise ValueError("Frame timestamps must be monotonic")
        object.__setattr__(self, "episode_index", episode_index)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def as_writer_records(self) -> list[dict[str, Any]]:
        return [frame.as_writer_record() for frame in self.frames]

    def as_records(self) -> list[dict[str, Any]]:
        return [frame.as_record() for frame in self.frames]
