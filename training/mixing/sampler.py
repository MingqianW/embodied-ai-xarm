"""Deterministic source streams shared by preflight and the OpenPI bridge."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Generic, Iterable, Mapping, Sequence, TypeVar

from training.mixing.strategies import MixingMode, MixingStrategy, SourceName, normalize_source_name


T = TypeVar("T")


@dataclass(frozen=True)
class SampleRef:
    dataset_id: str
    source: SourceName
    episode_index: int
    frame_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", normalize_source_name(self.source))


@dataclass(frozen=True)
class TrajectoryRef:
    dataset_id: str
    source: SourceName
    episode_index: int
    samples: tuple[SampleRef, ...]

    def __post_init__(self) -> None:
        source = normalize_source_name(self.source)
        object.__setattr__(self, "source", source)
        if not self.samples:
            raise ValueError("TrajectoryRef cannot be empty")
        if any(
            item.dataset_id != self.dataset_id
            or item.source != source
            or item.episode_index != self.episode_index
            for item in self.samples
        ):
            raise ValueError("Trajectory samples must match trajectory identity")


def _stable_seed(seed: int, label: str, epoch: int) -> int:
    digest = hashlib.sha256(f"{seed}:{label}:{epoch}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _source_label(source: SourceName) -> str:
    return source.value if hasattr(source, "value") else str(source)


class DeterministicSourceStream(Generic[T]):
    """Random-access deterministic source stream.

    Physical source pools are shuffled only within their own cycles. The
    source-selection policy stays exact and is independent of dataset size.
    """

    def __init__(
        self,
        pools: Mapping[SourceName, Sequence[T]],
        strategy: MixingStrategy,
        *,
        batch_size: int,
        trajectories: Sequence[Sequence[T]] = (),
    ) -> None:
        strategy.validate_for_batch_size(batch_size)
        self._strategy = strategy
        self._batch_size = batch_size
        self._pools = {
            normalize_source_name(source): tuple(values)
            for source, values in pools.items()
        }
        self._trajectories = tuple(tuple(values) for values in trajectories)
        if strategy.mode is MixingMode.GLOBAL_TRAJECTORY_SHUFFLE:
            if not self._trajectories or any(not values for values in self._trajectories):
                raise ValueError("global trajectory shuffle requires non-empty trajectories")
            self._trajectory_epoch_size = sum(len(values) for values in self._trajectories)
        else:
            self._trajectory_epoch_size = 0
            for source in strategy.required_sources:
                if not self._pools.get(source):
                    raise ValueError(f"No samples available for source {source!r}")

    def _permutation_index(self, source: SourceName, ordinal: int) -> int:
        pool = self._pools[source]
        epoch, offset = divmod(ordinal, len(pool))
        order = list(range(len(pool)))
        random.Random(_stable_seed(self._strategy.seed, _source_label(source), epoch)).shuffle(order)
        return order[offset]

    def _batch_sources(self, batch_index: int) -> tuple[SourceName, ...]:
        values = [source for source, count in self._strategy.composition for _ in range(count)]
        if self._strategy.shuffle_within_batch:
            random.Random(_stable_seed(self._strategy.seed, "batch", batch_index)).shuffle(values)
        return tuple(values)

    def _source_and_ordinal(self, position: int) -> tuple[SourceName, int]:
        if position < 0:
            raise IndexError("stream position must be non-negative")
        mode = self._strategy.mode
        if mode is MixingMode.SINGLE_SOURCE:
            return self._strategy.source, position  # type: ignore[return-value]
        if mode is MixingMode.FIXED_PER_BATCH:
            batch_index, within_batch = divmod(position, self._batch_size)
            sources = self._batch_sources(batch_index)
            source = sources[within_batch]
            per_batch = sources.count(source)
            before_batches = batch_index * per_batch
            within = sum(item == source for item in sources[:within_batch])
            return source, before_batches + within
        if mode is MixingMode.WEIGHTED_SAMPLE_STREAM:
            schedule = self._strategy.schedule
            cycle, offset = divmod(position, len(schedule))
            source = schedule[offset]
            weight = schedule.count(source)
            within = sum(item == source for item in schedule[:offset])
            return source, cycle * weight + within
        raise ValueError("trajectory shuffle has no source-only stream")

    def _trajectory_item(self, position: int) -> T:
        epoch, offset = divmod(position, self._trajectory_epoch_size)
        order = list(range(len(self._trajectories)))
        random.Random(_stable_seed(self._strategy.seed, "trajectories", epoch)).shuffle(order)
        for index in order:
            trajectory = self._trajectories[index]
            if offset < len(trajectory):
                return trajectory[offset]
            offset -= len(trajectory)
        raise AssertionError("trajectory offset escaped epoch")

    def item_at(self, position: int) -> T:
        if self._strategy.mode is MixingMode.GLOBAL_TRAJECTORY_SHUFFLE:
            return self._trajectory_item(position)
        source, ordinal = self._source_and_ordinal(position)
        return self._pools[source][self._permutation_index(source, ordinal)]

    def batch_at(self, batch_index: int) -> tuple[T, ...]:
        if batch_index < 0:
            raise IndexError("batch index must be non-negative")
        offset = batch_index * self._batch_size
        return tuple(self.item_at(offset + index) for index in range(self._batch_size))


def _source_pools(samples: Iterable[SampleRef]) -> dict[SourceName, tuple[SampleRef, ...]]:
    pools: dict[SourceName, list[SampleRef]] = {}
    for sample in samples:
        pools.setdefault(sample.source, []).append(sample)
    return {source: tuple(values) for source, values in pools.items()}


def sample_batches(
    samples: Sequence[SampleRef], strategy: MixingStrategy, *, batch_size: int, num_batches: int
) -> list[tuple[SampleRef, ...]]:
    """Observe a finite prefix of a deterministic source sample stream."""

    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    if strategy.mode is MixingMode.GLOBAL_TRAJECTORY_SHUFFLE:
        raise ValueError("Use trajectory_batches for global trajectory shuffle")
    stream = DeterministicSourceStream(_source_pools(samples), strategy, batch_size=batch_size)
    return [stream.batch_at(batch_index) for batch_index in range(num_batches)]


def trajectory_batches(
    trajectories: Sequence[TrajectoryRef], strategy: MixingStrategy, *, batch_size: int, num_batches: int
) -> list[tuple[SampleRef, ...]]:
    """Shuffle whole trajectories, then flatten without reordering frames."""

    if strategy.mode is not MixingMode.GLOBAL_TRAJECTORY_SHUFFLE:
        raise ValueError("trajectory_batches requires global_trajectory_shuffle")
    if not trajectories or batch_size <= 0 or num_batches <= 0:
        raise ValueError("trajectories, batch_size, and num_batches must be non-empty/positive")
    stream = DeterministicSourceStream(
        {},
        strategy,
        batch_size=batch_size,
        trajectories=[trajectory.samples for trajectory in trajectories],
    )
    return [stream.batch_at(batch_index) for batch_index in range(num_batches)]
