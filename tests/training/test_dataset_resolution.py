from __future__ import annotations

from pathlib import Path

import pytest

from training.datasets.resolution import DatasetResolutionError, resolve_dataset_paths
from training.datasets.spec import DatasetSet, DatasetSpec


def _dataset(dataset_id: str, repo_id: str, source: str) -> DatasetSpec:
    return DatasetSpec(dataset_id, repo_id, source, ("red_block",))


def test_each_named_source_resolves_to_its_own_explicit_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    sim = tmp_path / "sim"
    real.mkdir()
    sim.mkdir()
    datasets = DatasetSet((_dataset("human", "local/human", "human_demo"), _dataset("sim", "local/sim", "synthetic")))
    resolved = resolve_dataset_paths(datasets, {"human": real, "sim": sim})
    assert resolved == {"human": real.resolve(), "sim": sim.resolve()}


def test_unknown_named_source_override_is_rejected(tmp_path: Path) -> None:
    datasets = DatasetSet((_dataset("human", "local/human", "human_demo"),))
    with pytest.raises(DatasetResolutionError, match="Unknown dataset"):
        resolve_dataset_paths(datasets, {"unknown": tmp_path})
