from __future__ import annotations

import pytest

from data.common.records import SourceBackend
from data.common.schema import TRAINING_REQUIRED_KEYS
from training.datasets.spec import DatasetSet, DatasetSpec, EpisodeSelection


def test_dataset_identity_reuses_canonical_tasks_and_contract() -> None:
    spec = DatasetSpec(
        "real_fixture",
        "local/fixture",
        SourceBackend.REAL,
        ("red pepper", "red_block"),
        revision="v1",
    )
    assert spec.tasks == ("red_pepper", "red_block")
    assert spec.required_fields == TRAINING_REQUIRED_KEYS
    assert spec.prompts == ("pick up the red pepper", "pick up the red block")


def test_dataset_set_rejects_duplicate_identity() -> None:
    spec = DatasetSpec("same", "local/a", SourceBackend.REAL, ("red_block",))
    with pytest.raises(ValueError, match="unique"):
        DatasetSet((spec, spec))


def test_episode_selection_does_not_encode_sampling_ratio() -> None:
    selected = EpisodeSelection("first_by_episode_index", 198, "successful episodes")
    assert selected.limit == 198
    assert not hasattr(selected, "weight")
    with pytest.raises(ValueError, match="positive"):
        EpisodeSelection("explicit", 0)
