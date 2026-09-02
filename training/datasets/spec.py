"""Dataset identity and selection, deliberately separate from sampling policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data.common.schema import TRAINING_REQUIRED_KEYS
from data.common.task_identity import TASK_BY_ID, resolve_task_id
from training.mixing.strategies import SourceName, normalize_source_name


CANONICAL_CONTRACT = "data.common:xarm_training_v1"


@dataclass(frozen=True)
class EpisodeSelection:
    """Physical trajectory selection before any training-time sampling."""

    mode: str = "all"
    limit: int | None = None
    description: str = "all accepted episodes"

    def __post_init__(self) -> None:
        if self.mode not in {"all", "first_by_episode_index", "explicit"}:
            raise ValueError(f"Unsupported episode selection mode: {self.mode}")
        if self.mode == "all" and self.limit is not None:
            raise ValueError("An all-episode selection cannot have a limit")
        if self.mode != "all" and (self.limit is None or self.limit <= 0):
            raise ValueError(f"{self.mode} requires a positive limit")


@dataclass(frozen=True)
class DatasetSpec:
    """One immutable data source; it intentionally contains no mixing weight."""

    dataset_id: str
    repo_id: str
    source: SourceName
    tasks: tuple[str, ...]
    revision: str | None = None
    local_path: Path | None = None
    selection: EpisodeSelection = EpisodeSelection()
    expected_episodes: int | None = None
    role: str = "training"
    contract: str = CANONICAL_CONTRACT

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.repo_id.strip():
            raise ValueError("dataset_id and repo_id must be non-empty")
        object.__setattr__(self, "source", normalize_source_name(self.source))
        canonical_tasks = tuple(resolve_task_id(task) for task in self.tasks)
        if not canonical_tasks or len(set(canonical_tasks)) != len(canonical_tasks):
            raise ValueError("tasks must be a non-empty unique canonical task set")
        object.__setattr__(self, "tasks", canonical_tasks)
        if self.expected_episodes is not None and self.expected_episodes <= 0:
            raise ValueError("expected_episodes must be positive")
        if self.contract != CANONICAL_CONTRACT:
            raise ValueError(f"Unsupported training contract: {self.contract}")
        if self.local_path is not None:
            object.__setattr__(self, "local_path", Path(self.local_path))

    @property
    def prompts(self) -> tuple[str, ...]:
        return tuple(TASK_BY_ID[task].prompt for task in self.tasks)

    @property
    def required_fields(self) -> tuple[str, ...]:
        return TRAINING_REQUIRED_KEYS


@dataclass(frozen=True)
class DatasetSet:
    """Datasets physically present in an experiment, before sampling."""

    datasets: tuple[DatasetSpec, ...]

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("DatasetSet cannot be empty")
        ids = [dataset.dataset_id for dataset in self.datasets]
        if len(set(ids)) != len(ids):
            raise ValueError("Dataset ids must be unique")

    @property
    def sources(self) -> frozenset[SourceName]:
        return frozenset(dataset.source for dataset in self.datasets)

    def for_source(self, source: SourceName) -> tuple[DatasetSpec, ...]:
        source = normalize_source_name(source)
        return tuple(dataset for dataset in self.datasets if dataset.source == source)

    def by_id(self, dataset_id: str) -> DatasetSpec:
        for dataset in self.datasets:
            if dataset.dataset_id == dataset_id:
                return dataset
        raise KeyError(dataset_id)
