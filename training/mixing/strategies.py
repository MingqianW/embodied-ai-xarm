"""Named-source sampling policies for OpenPI training.

The stored data source is deliberately separate from a mixing policy. A source
may be ``real``/``sim`` for the existing xArm experiments, or any other stable
name supplied by a future experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, TypeAlias

from data.common.records import SourceBackend


SourceName: TypeAlias = str | SourceBackend


def normalize_source_name(value: SourceName) -> SourceName:
    """Keep legacy real/sim enum identity while accepting arbitrary names."""

    if isinstance(value, SourceBackend):
        return value
    name = str(value).strip()
    if not name:
        raise ValueError("source name must be non-empty")
    try:
        return SourceBackend(name)
    except ValueError:
        return name


class MixingMode(str, Enum):
    SINGLE_SOURCE = "single_source"
    FIXED_PER_BATCH = "fixed_per_batch"
    WEIGHTED_SAMPLE_STREAM = "weighted_sample_stream"
    # Compatibility spelling retained for existing experiment metadata.
    FIXED_SAMPLE_SCHEDULE = "weighted_sample_stream"
    GLOBAL_TRAJECTORY_SHUFFLE = "global_trajectory_shuffle"


def _named_counts(
    values: Mapping[SourceName, int] | tuple[tuple[SourceName, int], ...],
) -> tuple[tuple[SourceName, int], ...]:
    entries = tuple(values.items()) if isinstance(values, Mapping) else tuple(values)
    if not entries:
        raise ValueError("at least one source weight is required")
    normalized: list[tuple[SourceName, int]] = []
    seen: set[SourceName] = set()
    for source, count in entries:
        source = normalize_source_name(source)
        if source in seen:
            raise ValueError(f"source {source!r} appears more than once")
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"source {source!r} requires a positive integer count")
        seen.add(source)
        normalized.append((source, count))
    return tuple(normalized)


@dataclass(frozen=True)
class MixingStrategy:
    """A deterministic policy independent of physical dataset sizes.

    ``composition`` controls exact source counts in every global batch.
    ``weights`` controls a deterministic weighted sample stream. Its repeated
    schedule contains each source exactly ``weight`` times, preserving the
    configured ratio in every complete schedule cycle.
    """

    mode: MixingMode
    source: SourceName | None = None
    composition: tuple[tuple[SourceName, int], ...] = ()
    weights: tuple[tuple[SourceName, int], ...] = ()
    # Legacy public fields remain for existing configs and callers.
    real_per_batch: int | None = None
    sim_per_batch: int | None = None
    schedule: tuple[SourceName, ...] = ()
    shuffle_within_batch: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", MixingMode(self.mode))
        if self.source is not None:
            object.__setattr__(self, "source", normalize_source_name(self.source))
        object.__setattr__(
            self,
            "schedule",
            tuple(normalize_source_name(value) for value in self.schedule),
        )
        if self.composition:
            object.__setattr__(self, "composition", _named_counts(self.composition))
        if self.weights:
            object.__setattr__(self, "weights", _named_counts(self.weights))
        if self.mode is MixingMode.SINGLE_SOURCE:
            if self.source is None:
                raise ValueError("single_source requires source")
            if self.composition or self.weights or self.schedule:
                raise ValueError("single_source cannot define composition or weights")
        elif self.mode is MixingMode.FIXED_PER_BATCH:
            composition = self.composition
            if not composition:
                if not self.real_per_batch or not self.sim_per_batch:
                    raise ValueError("fixed_per_batch requires named composition")
                composition = (
                    (SourceBackend.REAL, self.real_per_batch),
                    (SourceBackend.SIM, self.sim_per_batch),
                )
                object.__setattr__(self, "composition", composition)
            if self.source is not None or self.weights or self.schedule:
                raise ValueError("fixed_per_batch cannot define source or weighted schedule")
            counts = dict(composition)
            if set(counts) == {SourceBackend.REAL, SourceBackend.SIM}:
                object.__setattr__(self, "real_per_batch", counts[SourceBackend.REAL])
                object.__setattr__(self, "sim_per_batch", counts[SourceBackend.SIM])
        elif self.mode is MixingMode.WEIGHTED_SAMPLE_STREAM:
            weights = self.weights
            if not weights and self.schedule:
                counts: dict[SourceName, int] = {}
                for source in self.schedule:
                    counts[source] = counts.get(source, 0) + 1
                weights = _named_counts(tuple(counts.items()))
                object.__setattr__(self, "weights", weights)
            if not weights:
                raise ValueError("weighted_sample_stream requires named weights")
            if self.source is not None or self.composition:
                raise ValueError("weighted_sample_stream cannot define source or composition")
            if not self.schedule:
                object.__setattr__(
                    self,
                    "schedule",
                    tuple(source for source, count in weights for _ in range(count)),
                )
        elif self.mode is MixingMode.GLOBAL_TRAJECTORY_SHUFFLE:
            if any(value is not None for value in (self.source, self.real_per_batch, self.sim_per_batch)):
                raise ValueError("global trajectory shuffle cannot enforce a source ratio")
            if self.composition or self.weights or self.schedule:
                raise ValueError("global trajectory shuffle cannot define source weights")
        if self.seed < 0:
            raise ValueError("mixing seed must be non-negative")

    @classmethod
    def single(cls, source: SourceName, *, seed: int = 42) -> "MixingStrategy":
        return cls(MixingMode.SINGLE_SOURCE, source=source, seed=seed)

    @classmethod
    def per_source_batch(
        cls,
        composition: Mapping[SourceName, int],
        *,
        seed: int = 42,
        shuffle_within_batch: bool = True,
    ) -> "MixingStrategy":
        return cls(
            MixingMode.FIXED_PER_BATCH,
            composition=_named_counts(composition),
            shuffle_within_batch=shuffle_within_batch,
            seed=seed,
        )

    @classmethod
    def per_batch(
        cls, real: int, sim: int, *, seed: int = 42, shuffle_within_batch: bool = True
    ) -> "MixingStrategy":
        """Compatibility helper for the existing A-style real/sim experiment."""

        return cls(
            MixingMode.FIXED_PER_BATCH,
            composition=((SourceBackend.REAL, real), (SourceBackend.SIM, sim)),
            real_per_batch=real,
            sim_per_batch=sim,
            shuffle_within_batch=shuffle_within_batch,
            seed=seed,
        )

    @classmethod
    def weighted_stream(
        cls, weights: Mapping[SourceName, int], *, seed: int = 42
    ) -> "MixingStrategy":
        return cls(MixingMode.WEIGHTED_SAMPLE_STREAM, weights=_named_counts(weights), seed=seed)

    @classmethod
    def sample_ratio(cls, real: int, sim: int, *, seed: int = 42) -> "MixingStrategy":
        """Compatibility helper for the existing B-style 1:10 stream."""

        return cls.weighted_stream({SourceBackend.REAL: real, SourceBackend.SIM: sim}, seed=seed)

    @classmethod
    def trajectory_shuffle(cls, *, seed: int = 42) -> "MixingStrategy":
        return cls(MixingMode.GLOBAL_TRAJECTORY_SHUFFLE, seed=seed)

    @property
    def required_sources(self) -> tuple[SourceName, ...]:
        if self.mode is MixingMode.SINGLE_SOURCE:
            return (self.source,)  # type: ignore[return-value]
        if self.mode is MixingMode.FIXED_PER_BATCH:
            return tuple(source for source, _ in self.composition)
        if self.mode is MixingMode.WEIGHTED_SAMPLE_STREAM:
            return tuple(source for source, _ in self.weights)
        return ()

    def validate_for_batch_size(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.mode is MixingMode.FIXED_PER_BATCH:
            expected = sum(count for _, count in self.composition)
            if expected != batch_size:
                raise ValueError(f"Per-batch source counts total {expected}, not batch_size {batch_size}")
