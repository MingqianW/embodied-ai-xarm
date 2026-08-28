from __future__ import annotations

import numpy as np
import pytest

from data.common.records import EpisodeRecord, FrameRecord, SourceBackend
from data.common.schema import (
    OPENPI_FIELD_MAPPING,
    XARM_ACTION_SHAPE,
    XARM_IMAGE_SHAPE,
    XARM_STATE_COLUMNS,
    XARM_STATE_SHAPE,
)
from data.common.validation import validate_training_record


def _frame(frame_index: int, *, source: SourceBackend = SourceBackend.SIM) -> FrameRecord:
    state = np.arange(7, dtype=np.float64) + frame_index
    actions = state + 1
    return FrameRecord(
        image=np.zeros(XARM_IMAGE_SHAPE, dtype=np.uint8),
        wrist_image=np.ones(XARM_IMAGE_SHAPE, dtype=np.uint8),
        state=state,
        actions=actions,
        task="pick up the red block",
        episode_index=3,
        frame_index=frame_index,
        timestamp=100.0 + frame_index / 10.0,
        source=source,
        metadata={"backend_only": True},
    )


def test_exact_training_shape_order_and_openpi_mapping() -> None:
    assert XARM_STATE_COLUMNS == (
        "j1_rad",
        "j2_rad",
        "j3_rad",
        "j4_rad",
        "j5_rad",
        "j6_rad",
        "gripper_mm",
    )
    assert XARM_STATE_SHAPE == XARM_ACTION_SHAPE == (7,)
    assert XARM_IMAGE_SHAPE == (480, 640, 3)
    assert OPENPI_FIELD_MAPPING == {
        "image": "observation/image",
        "wrist_image": "observation/wrist_image",
        "state": "observation/state",
        "actions": "action",
        "task": "task",
    }


def test_frame_preserves_values_and_excludes_provenance_from_writer_record() -> None:
    frame = _frame(0)
    writer_record = frame.as_writer_record()
    assert set(writer_record) == {"image", "wrist_image", "state", "actions", "task"}
    np.testing.assert_array_equal(writer_record["state"], np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(writer_record["actions"], np.arange(1, 8, dtype=np.float32))
    assert frame.source is SourceBackend.SIM
    assert frame.metadata == {"backend_only": True}


def test_episode_requires_contiguous_indices_backend_and_monotonic_time() -> None:
    episode = EpisodeRecord(
        episode_index=3,
        source=SourceBackend.SIM,
        frames=(_frame(0), _frame(1)),
        metadata={"seed": 7},
    )
    assert len(episode.as_writer_records()) == 2
    with pytest.raises(ValueError, match="contiguous"):
        EpisodeRecord(3, SourceBackend.SIM, (_frame(1),))
    with pytest.raises(ValueError, match="source"):
        EpisodeRecord(3, SourceBackend.SIM, (_frame(0, source=SourceBackend.REAL),))


def test_contract_rejects_wrong_shape_dtype_and_nonfinite_vectors() -> None:
    good = _frame(0).as_writer_record()
    with pytest.raises(ValueError, match="shape"):
        validate_training_record({**good, "state": np.zeros(6)})
    with pytest.raises(ValueError, match="uint8"):
        validate_training_record({**good, "image": np.zeros(XARM_IMAGE_SHAPE)})
    with pytest.raises(ValueError, match="NaN"):
        validate_training_record({**good, "actions": np.asarray([0] * 6 + [np.nan])})

