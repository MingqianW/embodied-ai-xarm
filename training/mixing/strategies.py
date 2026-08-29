"""Scientifically distinct source-sampling and shuffling policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from data.common.records import SourceBackend


class MixingMode(str, Enum):
    SINGLE_SOURCE = "single_source"
    FIXED_PER_BATCH = "fixed_per_batch"
    FIXED_SAMPLE_SCHEDULE = "fixed_sample_schedule"
    GLOBAL_TRAJECTORY_SHUFFLE = "global_trajectory_shuffle"


@dataclass(frozen=True)
class MixingStrategy:
    """Sampling policy independent of the number of stored trajectories."""

    mode: MixingMode
    source: SourceBackend | None = None
    real_per_batch: int | None = None
    sim_per_batch: int | None = None
    schedule: tuple[SourceBackend, ...] = ()
    shuffle_within_batch: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", MixingMode(self.mode))
        if self.source is not None:
            object.__setattr__(self, "source", SourceBackend(self.source))
        object.__setattr__(self, "schedule", tuple(SourceBackend(x) for x in self.schedule))
        if self.mode is MixingMode.SINGLE_SOURCE:
            if self.source is None:
                raise ValueError("single_source requires source")
        elif self.mode is MixingMode.FIXED_PER_BATCH:
            if not self.real_per_batch or not self.sim_per_batch:
                raise ValueError("fixed_per_batch requires positive real and sim counts")
        elif self.mode is MixingMode.FIXED_SAMPLE_SCHEDULE:
            if not self.schedule or set(self.schedule) != {SourceBackend.REAL, SourceBackend.SIM}:
                raise ValueError("fixed_sample_schedule must contain both real and sim")
        elif self.mode is MixingMode.GLOBAL_TRAJECTORY_SHUFFLE:
            if any(value is not None for value in (self.source, self.real_per_batch, self.sim_per_batch)):
                raise ValueError("global trajectory shuffle cannot enforce a source ratio")

    @classmethod
    def single(cls, source: SourceBackend, *, seed: int = 42) -> "MixingStrategy":
        return cls(MixingMode.SINGLE_SOURCE, source=source, seed=seed)

    @classmethod
    def per_batch(
        cls, real: int, sim: int, *, seed: int = 42, shuffle_within_batch: bool = True
    ) -> "MixingStrategy":
        return cls(
            MixingMode.FIXED_PER_BATCH,
            real_per_batch=real,
            sim_per_batch=sim,
            shuffle_within_batch=shuffle_within_batch,
            seed=seed,
        )

    @classmethod
    def sample_ratio(cls, real: int, sim: int, *, seed: int = 42) -> "MixingStrategy":
        if real <= 0 or sim <= 0:
            raise ValueError("sample ratio counts must be positive")
        return cls(
            MixingMode.FIXED_SAMPLE_SCHEDULE,
            schedule=(SourceBackend.REAL,) * real + (SourceBackend.SIM,) * sim,
            seed=seed,
        )

    @classmethod
    def trajectory_shuffle(cls, *, seed: int = 42) -> "MixingStrategy":
        return cls(MixingMode.GLOBAL_TRAJECTORY_SHUFFLE, seed=seed)

    def validate_for_batch_size(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.mode is MixingMode.FIXED_PER_BATCH:
            expected = int(self.real_per_batch or 0) + int(self.sim_per_batch or 0)
            if expected != batch_size:
                raise ValueError(f"Per-batch source counts total {expected}, not batch_size {batch_size}")
