"""Dependency-free deterministic samplers used by preflight and OpenPI glue."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Iterable, Iterator, Mapping, Sequence

from data.common.records import SourceBackend
from training.mixing.strategies import MixingMode, MixingStrategy


@dataclass(frozen=True)
class SampleRef:
    dataset_id: str
    source: SourceBackend
    episode_index: int
    frame_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", SourceBackend(self.source))


@dataclass(frozen=True)
class TrajectoryRef:
    dataset_id: str
    source: SourceBackend
    episode_index: int
    samples: tuple[SampleRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", SourceBackend(self.source))
        if not self.samples:
            raise ValueError("TrajectoryRef cannot be empty")
        if any(
            item.dataset_id != self.dataset_id
            or item.source is not self.source
            or item.episode_index != self.episode_index
            for item in self.samples
        ):
            raise ValueError("Trajectory samples must match trajectory identity")


def _stable_seed(seed: int, label: str, epoch: int) -> int:
    digest = hashlib.sha256(f"{seed}:{label}:{epoch}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class _Cycler:
    def __init__(self, values: Sequence[SampleRef], *, seed: int, label: str):
        if not values:
            raise ValueError(f"No samples available for {label}")
        self._values = tuple(values)
        self._seed = seed
        self._label = label
        self._offset = 0
        self._epoch = -1
        self._permutation: list[int] = []

    def take(self, count: int) -> list[SampleRef]:
        output: list[SampleRef] = []
        for _ in range(count):
            epoch, index = divmod(self._offset, len(self._values))
            if epoch != self._epoch:
                self._epoch = epoch
                self._permutation = list(range(len(self._values)))
                random.Random(_stable_seed(self._seed, self._label, epoch)).shuffle(self._permutation)
            output.append(self._values[self._permutation[index]])
            self._offset += 1
        return output


def _source_pools(samples: Iterable[SampleRef]) -> dict[SourceBackend, tuple[SampleRef, ...]]:
    pools = {SourceBackend.REAL: [], SourceBackend.SIM: []}
    for sample in samples:
        pools[sample.source].append(sample)
    return {source: tuple(values) for source, values in pools.items()}


def sample_batches(
    samples: Sequence[SampleRef], strategy: MixingStrategy, *, batch_size: int, num_batches: int
) -> list[tuple[SampleRef, ...]]:
    """Observe a finite prefix of a single/per-batch/sample-schedule stream."""

    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    strategy.validate_for_batch_size(batch_size)
    if strategy.mode is MixingMode.GLOBAL_TRAJECTORY_SHUFFLE:
        raise ValueError("Use trajectory_batches for global trajectory shuffle")
    pools = _source_pools(samples)
    needed_sources = (
        (strategy.source,)
        if strategy.mode is MixingMode.SINGLE_SOURCE
        else (SourceBackend.REAL, SourceBackend.SIM)
    )
    cyclers = {
        source: _Cycler(pools[source], seed=strategy.seed, label=source.value)
        for source in needed_sources
        if source is not None
    }
    batches: list[tuple[SampleRef, ...]] = []
    stream_offset = 0
    for batch_index in range(num_batches):
        if strategy.mode is MixingMode.SINGLE_SOURCE:
            batch = cyclers[strategy.source].take(batch_size)  # type: ignore[index]
        elif strategy.mode is MixingMode.FIXED_PER_BATCH:
            batch = cyclers[SourceBackend.REAL].take(int(strategy.real_per_batch))
            batch += cyclers[SourceBackend.SIM].take(int(strategy.sim_per_batch))
            if strategy.shuffle_within_batch:
                random.Random(_stable_seed(strategy.seed, "batch", batch_index)).shuffle(batch)
        else:
            batch = []
            for _ in range(batch_size):
                source = strategy.schedule[stream_offset % len(strategy.schedule)]
                batch.extend(cyclers[source].take(1))
                stream_offset += 1
        batches.append(tuple(batch))
    return batches


def trajectory_batches(
    trajectories: Sequence[TrajectoryRef], strategy: MixingStrategy, *, batch_size: int, num_batches: int
) -> list[tuple[SampleRef, ...]]:
    """Shuffle whole trajectories, then flatten without reordering frames."""

    if strategy.mode is not MixingMode.GLOBAL_TRAJECTORY_SHUFFLE:
        raise ValueError("trajectory_batches requires global_trajectory_shuffle")
    if not trajectories or batch_size <= 0 or num_batches <= 0:
        raise ValueError("trajectories, batch_size, and num_batches must be non-empty/positive")

    def stream() -> Iterator[SampleRef]:
        epoch = 0
        while True:
            order = list(range(len(trajectories)))
            random.Random(_stable_seed(strategy.seed, "trajectories", epoch)).shuffle(order)
            for index in order:
                yield from trajectories[index].samples
            epoch += 1

    iterator = stream()
    return [tuple(next(iterator) for _ in range(batch_size)) for _ in range(num_batches)]
