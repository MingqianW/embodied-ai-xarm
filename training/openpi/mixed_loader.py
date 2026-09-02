"""Project-owned multi-LeRobot bridge for the upstream OpenPI training loop.

This module intentionally owns only source resolution, deterministic index
selection, and mixed-pool normalization. It delegates transforms, batching,
JAX/PyTorch conversion, optimization, distributed sampler behavior, and
checkpointing to the installed OpenPI revision.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from training.configs.experiments import ExperimentConfig
from training.datasets.spec import DatasetSpec
from training.mixing.sampler import DeterministicSourceStream
from training.mixing.strategies import MixingMode, SourceName
from training.normalization import NormalizationMode


MIXED_NORMALIZATION_MANIFEST = "mixed_normalization_manifest.json"


@dataclass(frozen=True)
class OpenPITrainingRuntime:
    """The project metadata needed to intercept one upstream TrainConfig."""

    experiment: ExperimentConfig
    openpi_config: Any
    dataset_paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        expected = {dataset.dataset_id for dataset in self.experiment.datasets.datasets}
        provided = set(self.dataset_paths)
        if expected != provided:
            raise ValueError(f"runtime dataset paths {sorted(provided)} do not match experiment datasets {sorted(expected)}")
        object.__setattr__(
            self,
            "dataset_paths",
            MappingProxyType({key: Path(value).resolve() for key, value in self.dataset_paths.items()}),
        )


@dataclass(frozen=True)
class DatasetFrameRef:
    dataset_id: str
    source: SourceName
    dataset_index: int
    episode_index: int
    frame_index: int


@dataclass
class _LoadedDataset:
    spec: DatasetSpec
    path: Path
    dataset: Any
    frames: tuple[DatasetFrameRef, ...]
    trajectories: tuple[tuple[DatasetFrameRef, ...], ...]


def _openpi_modules() -> tuple[Any, Any, Any]:
    """Import only after the adapter has placed OpenPI's src on sys.path."""

    import openpi.shared.normalize as normalize
    import openpi.training.data_loader as data_loader
    import openpi.transforms as transforms

    return data_loader, normalize, transforms


def _lerobot_dataset_class() -> type:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def _dataset_tasks(dataset: Any) -> Any:
    for candidate in (
        getattr(getattr(dataset, "meta", None), "tasks", None),
        getattr(dataset, "tasks", None),
    ):
        if candidate is not None:
            return candidate
    raise ValueError("LeRobot dataset does not expose a task catalog required for prompt_from_task=True")


class _PromptDataset:
    def __init__(self, dataset: Any, tasks: Any, transforms: Any):
        self._dataset = dataset
        self._prompt = transforms.PromptFromLeRobotTask(tasks)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._prompt(self._dataset[index])


def _episode_ranges(dataset: Any, frame_count: int) -> tuple[tuple[int, int], ...] | None:
    """Return LeRobot episode frame ranges without reading every sample."""

    table = getattr(dataset, "episode_data_index", None)
    if table is None:
        table = getattr(getattr(dataset, "meta", None), "episode_data_index", None)
    if table is None:
        return None
    try:
        starts = np.asarray(table["from"], dtype=np.int64).reshape(-1)
        ends = np.asarray(table["to"], dtype=np.int64).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("LeRobot episode_data_index must provide 'from' and 'to' arrays") from exc
    if len(starts) != len(ends) or np.any(starts < 0) or np.any(ends < starts) or np.any(ends > frame_count):
        raise ValueError("LeRobot episode_data_index contains invalid ranges")
    return tuple((int(start), int(end)) for start, end in zip(starts, ends, strict=True) if end > start)


def _selected_ranges(spec: DatasetSpec, ranges: tuple[tuple[int, int], ...] | None, frame_count: int) -> tuple[tuple[int, int], ...]:
    selection = spec.selection
    if selection.mode == "all":
        return ranges or ((0, frame_count),)
    if selection.mode == "first_by_episode_index":
        if ranges is None:
            raise ValueError(f"Dataset {spec.dataset_id} cannot select episodes without episode_data_index metadata")
        return ranges[: int(selection.limit)]
    raise ValueError(
        f"Dataset {spec.dataset_id} uses explicit episode selection, but this repository has no explicit episode id list"
    )


def _load_datasets(runtime: OpenPITrainingRuntime, data_config: Any, model_config: Any, transforms: Any) -> tuple[_LoadedDataset, ...]:
    LeRobotDataset = _lerobot_dataset_class()
    loaded: list[_LoadedDataset] = []
    for spec in runtime.experiment.datasets.datasets:
        path = runtime.dataset_paths[spec.dataset_id]
        raw_dataset = LeRobotDataset(
            path.name,
            root=path,
            delta_timestamps={
                key: [step / 10.0 for step in range(model_config.action_horizon)]
                for key in data_config.action_sequence_keys
            },
            download_videos=False,
        )
        frame_count = len(raw_dataset)
        episode_ranges = _episode_ranges(raw_dataset, frame_count)
        dataset = raw_dataset
        if data_config.prompt_from_task:
            dataset = _PromptDataset(raw_dataset, _dataset_tasks(raw_dataset), transforms)
        ranges = _selected_ranges(spec, episode_ranges, frame_count)
        frames: list[DatasetFrameRef] = []
        trajectories: list[tuple[DatasetFrameRef, ...]] = []
        for episode_index, (start, end) in enumerate(ranges):
            trajectory = tuple(
                DatasetFrameRef(spec.dataset_id, spec.source, index, episode_index, index - start)
                for index in range(start, end)
            )
            if trajectory:
                trajectories.append(trajectory)
                frames.extend(trajectory)
        if not frames:
            raise ValueError(f"Dataset {spec.dataset_id} has no selected training frames")
        loaded.append(_LoadedDataset(spec, path, dataset, tuple(frames), tuple(trajectories)))
    return tuple(loaded)


class _PhysicalPoolDataset:
    """Every selected frame exactly once; used for unbiased normalization."""

    def __init__(self, loaded: Sequence[_LoadedDataset]):
        self._datasets = {item.spec.dataset_id: item.dataset for item in loaded}
        self._frames = tuple(frame for item in loaded for frame in item.frames)

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self._frames[index]
        return self._datasets[ref.dataset_id][ref.dataset_index]


class _MixedStreamDataset:
    """Virtual random-access dataset whose consecutive items follow a policy."""

    def __init__(
        self,
        loaded: Sequence[_LoadedDataset],
        runtime: OpenPITrainingRuntime,
        *,
        virtual_length: int,
        position_offset: int,
    ) -> None:
        pools: dict[SourceName, list[DatasetFrameRef]] = {}
        for item in loaded:
            pools.setdefault(item.spec.source, []).extend(item.frames)
        self._stream = DeterministicSourceStream(
            pools,
            runtime.experiment.mixing,
            batch_size=runtime.openpi_config.batch_size,
            trajectories=[trajectory for item in loaded for trajectory in item.trajectories],
        )
        self._datasets = {item.spec.dataset_id: item.dataset for item in loaded}
        self._virtual_length = virtual_length
        self._position_offset = position_offset

    def __len__(self) -> int:
        return self._virtual_length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self._virtual_length:
            raise IndexError(index)
        ref = self._stream.item_at(self._position_offset + index)
        return self._datasets[ref.dataset_id][ref.dataset_index]


def _resume_batch_offset(config: Any) -> int:
    """Align a resumed deterministic stream with upstream's saved step number."""

    if not getattr(config, "resume", False):
        return 0
    try:
        steps = [int(path.name) for path in Path(config.checkpoint_dir).iterdir() if path.name.isdigit()]
    except OSError:
        return 0
    return (max(steps) + 1) * int(config.batch_size) if steps else 0


def _virtual_length(config: Any, num_batches: int | None) -> int:
    batches = num_batches if num_batches is not None else int(config.num_train_steps) + 2
    if batches <= 0:
        raise ValueError("num_batches must be positive")
    return batches * int(config.batch_size)


def create_mixed_data_loader(
    runtime: OpenPITrainingRuntime,
    config: Any,
    *,
    sharding: Any = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: str = "jax",
) -> Any:
    """Create an upstream-shaped loader using deterministic mixed source indices."""

    del shuffle  # Mixing itself defines the only allowed ordering policy.
    data_loader, _, transforms = _openpi_modules()
    data_config = config.data.create(config.assets_dirs, config.model)
    loaded = _load_datasets(runtime, data_config, config.model, transforms)
    dataset = _MixedStreamDataset(
        loaded,
        runtime,
        virtual_length=_virtual_length(config, num_batches),
        position_offset=_resume_batch_offset(config),
    )
    dataset = data_loader.transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    sampler = None
    if framework == "pytorch":
        if data_loader.torch.distributed.is_initialized():
            sampler = data_loader.torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=data_loader.torch.distributed.get_world_size(),
                rank=data_loader.torch.distributed.get_rank(),
                shuffle=False,
                drop_last=True,
            )
            local_batch_size = config.batch_size // data_loader.torch.distributed.get_world_size()
        else:
            local_batch_size = config.batch_size
    else:
        local_batch_size = config.batch_size // data_loader.jax.process_count()
    if local_batch_size <= 0:
        raise ValueError("global batch size is smaller than the active process count")
    logging.info(
        "Using deterministic mixed loader: experiment=%s strategy=%s local_batch_size=%s",
        runtime.experiment.name,
        runtime.experiment.mixing.mode.value,
        local_batch_size,
    )
    torch_loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=False,
        sampler=sampler,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        framework=framework,
    )
    return data_loader.DataLoaderImpl(data_config, torch_loader)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalization_manifest(runtime: OpenPITrainingRuntime, asset_id: str) -> dict[str, Any]:
    datasets = []
    for spec in runtime.experiment.datasets.datasets:
        path = runtime.dataset_paths[spec.dataset_id]
        info = path / "meta" / "info.json"
        datasets.append(
            {
                "dataset_id": spec.dataset_id,
                "source": getattr(spec.source, "value", spec.source),
                "repo_id": spec.repo_id,
                "path": str(path),
                "selection": {
                    "mode": spec.selection.mode,
                    "limit": spec.selection.limit,
                },
                "info_sha256": _hash_file(info) if info.is_file() else None,
            }
        )
    return {
        "format": "xarm_mixed_normalization_v1",
        "experiment": runtime.experiment.name,
        "asset_id": asset_id,
        "pool": datasets,
        "state_semantics": runtime.experiment.normalization.state_semantics,
        "action_semantics": runtime.experiment.normalization.action_semantics,
    }


def _remove_strings(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not np.issubdtype(np.asarray(value).dtype, np.str_)
    }


def compute_mixed_normalization(
    runtime: OpenPITrainingRuntime,
    *,
    max_frames: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Compute norm statistics over the selected physical mixed pool once each.

    This deliberately does *not* replay the training mixing stream. Repeating
    a small source to satisfy a training ratio would bias its normalization.
    """

    if runtime.experiment.normalization.mode is not NormalizationMode.COMPUTE_FROM_DATASETS:
        raise ValueError("mixed normalization is only valid for compute_from_datasets experiments")
    data_loader, normalize, transforms = _openpi_modules()
    config = runtime.openpi_config
    data_config = config.data.create(config.assets_dirs, config.model)
    asset_id = data_config.asset_id
    if not asset_id:
        raise ValueError("OpenPI data config has no asset id for normalization")
    configured_assets_dir = config.data.assets.assets_dir
    if configured_assets_dir and "://" in configured_assets_dir:
        raise ValueError("compute_from_datasets requires a writable local OpenPI assets directory")
    output_dir = Path(configured_assets_dir or config.assets_dirs) / asset_id
    stats_path = output_dir / "norm_stats.json"
    manifest_path = output_dir / MIXED_NORMALIZATION_MANIFEST
    expected_manifest = _normalization_manifest(runtime, asset_id)
    if stats_path.is_file() and not overwrite:
        if manifest_path.is_file():
            try:
                observed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot read mixed normalization manifest {manifest_path}: {exc}") from exc
            if observed_manifest == expected_manifest:
                return output_dir
        raise FileExistsError(
            f"Existing normalization stats at {stats_path} do not match this mixed dataset pool; "
            "rerun with --recompute-norm after reviewing the selected sources"
        )
    loaded = _load_datasets(runtime, data_config, config.model, transforms)
    dataset = _PhysicalPoolDataset(loaded)
    if max_frames is not None:
        if max_frames <= 1:
            raise ValueError("max_frames must be at least 2")
        frame_count = min(len(dataset), max_frames)
    else:
        frame_count = len(dataset)
    transformed = data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _remove_strings,
        ],
    )
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    pending: dict[str, list[np.ndarray]] = {key: [] for key in stats}
    for index in range(frame_count):
        sample = transformed[index]
        for key, running in stats.items():
            pending[key].append(np.asarray(sample[key]))
            if len(pending[key]) == 512:
                running.update(np.stack(pending[key]))
                pending[key].clear()
    for key, running in stats.items():
        if pending[key]:
            running.update(np.stack(pending[key]))
    normalize.save(output_dir, {key: running.get_statistics() for key, running in stats.items()})
    manifest_path.write_text(json.dumps(expected_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_dir


def ensure_mixed_normalization(runtime: OpenPITrainingRuntime, *, recompute: bool = False) -> Path | None:
    """Compute or verify a selected-pool asset before upstream training starts."""

    if runtime.experiment.normalization.mode is not NormalizationMode.COMPUTE_FROM_DATASETS:
        return None
    return compute_mixed_normalization(runtime, overwrite=recompute)


@contextmanager
def install_mixed_loader(runtime: OpenPITrainingRuntime) -> Iterator[None]:
    """Temporarily route this config through the bridge and leave OpenPI untouched."""

    data_loader, _, _ = _openpi_modules()
    original = data_loader.create_data_loader

    def create(config: Any, **kwargs: Any) -> Any:
        if config is runtime.openpi_config:
            return create_mixed_data_loader(runtime, config, **kwargs)
        return original(config, **kwargs)

    data_loader.create_data_loader = create
    try:
        yield
    finally:
        data_loader.create_data_loader = original
