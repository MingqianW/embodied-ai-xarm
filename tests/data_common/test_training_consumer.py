from __future__ import annotations

from data.common.schema import OPENPI_FIELD_MAPPING, XARM_STATE_COLUMNS
from fine_tune.openpi_xarm_config import XARM_CONFIG_SNIPPET


def test_openpi_consumer_matches_canonical_storage_keys_and_normalization() -> None:
    assert tuple(OPENPI_FIELD_MAPPING) == (
        "image",
        "wrist_image",
        "state",
        "actions",
        "task",
    )
    assert len(XARM_STATE_COLUMNS) == 7
    for model_key, stored_key in (
        ("observation/image", "image"),
        ("observation/wrist_image", "wrist_image"),
        ("observation/state", "state"),
        ("actions", "actions"),
        ("prompt", "prompt"),
    ):
        assert f'"{model_key}": "{stored_key}"' in XARM_CONFIG_SNIPPET
    assert "make_bool_mask(6, -1)" in XARM_CONFIG_SNIPPET
    assert 'action_sequence_keys=("actions",)' in XARM_CONFIG_SNIPPET
