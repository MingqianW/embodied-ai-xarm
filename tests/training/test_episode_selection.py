from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.configs.experiments import get_experiment
from training.datasets.spec import EpisodeSelection
from training.openpi.mixed_loader import _selected_ranges, _episode_ranges, _normalization_manifest, OpenPITrainingRuntime


def test_explicit_holdout_selection_retains_original_episode_identity():
    spec = replace(get_experiment("pi05_xarm").datasets.datasets[0],
                   selection=EpisodeSelection("explicit", 2, "training split", (0, 2)))
    assert _selected_ranges(spec, ((0, 2), (2, 4), (4, 7)), 7) == ((0, 0, 2), (2, 4, 7))
    with pytest.raises(ValueError, match="missing or empty"):
        _selected_ranges(spec, ((0, 2), (2, 4)), 4)
    with pytest.raises(ValueError, match="metadata"):
        _selected_ranges(spec, None, 7)


def test_empty_episode_does_not_shift_later_episode_identity():
    dataset = SimpleNamespace(episode_data_index={"from": [0, 2, 2], "to": [2, 2, 5]})
    spec = replace(get_experiment("pi05_xarm").datasets.datasets[0],
                   selection=EpisodeSelection("explicit", 1, "training", (2,)))
    assert _selected_ranges(spec, _episode_ranges(dataset, 5), 5) == ((2, 2, 5),)


@pytest.mark.parametrize("indices", [(0, 0), (0, -1), (0, 1.5)])
def test_explicit_episode_ids_reject_ambiguous_selection(indices):
    with pytest.raises(ValueError):
        EpisodeSelection("explicit", 2, episode_indices=indices)


def test_normalization_identity_changes_with_training_split(tmp_path: Path):
    from training.datasets.spec import DatasetSet
    config = get_experiment("pi05_xarm")
    dataset = config.datasets.datasets[0]
    def manifest(indices):
        selected = replace(dataset, selection=EpisodeSelection("explicit", 2, "train", indices))
        runtime = OpenPITrainingRuntime(replace(config, datasets=DatasetSet((selected,))), object(), {dataset.dataset_id: tmp_path})
        return _normalization_manifest(runtime, "new_norm")
    assert manifest((0, 2)) != manifest((0, 3))
