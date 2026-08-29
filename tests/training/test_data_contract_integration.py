from __future__ import annotations

from data.common.schema import OPENPI_FIELD_MAPPING, XARM_STATE_COLUMNS
from training.datasets.adapter import (
    XARM_ACTION_SEQUENCE_KEYS,
    XARM_DELTA_ACTION_MASK,
    XARM_OPENPI_REPACK_MAPPING,
)


def test_openpi_consumer_matches_canonical_storage_keys_and_normalization() -> None:
    assert tuple(OPENPI_FIELD_MAPPING) == (
        "image",
        "wrist_image",
        "state",
        "actions",
        "task",
    )
    assert len(XARM_STATE_COLUMNS) == 7
    assert XARM_OPENPI_REPACK_MAPPING == {
        "observation/image": "image",
        "observation/wrist_image": "wrist_image",
        "observation/state": "state",
        "actions": "actions",
        "prompt": "prompt",
    }
    assert XARM_DELTA_ACTION_MASK == (True,) * 6 + (False,)
    assert XARM_ACTION_SEQUENCE_KEYS == ("actions",)
