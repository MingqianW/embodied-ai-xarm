from __future__ import annotations

import numpy as np

from training.datasets.adapter import openpi_facing_batch, openpi_facing_record


def test_canonical_record_to_openpi_sample_excludes_source_provenance() -> None:
    record = {
        "image": np.zeros((480, 640, 3), dtype=np.uint8),
        "wrist_image": np.ones((480, 640, 3), dtype=np.uint8),
        "state": np.arange(7, dtype=np.float32),
        "actions": np.arange(7, dtype=np.float32),
        "task": "pick up the red block",
        "source": "sim",
        "episode_index": 3,
    }
    sample = openpi_facing_record(record)
    assert tuple(sample) == (
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "actions",
        "prompt",
    )
    assert sample["observation/state"].shape == (7,)
    assert sample["actions"].shape == (7,)
    assert "source" not in sample and "episode_index" not in sample


def test_small_canonical_batch_has_openpi_facing_keys_shapes_and_types() -> None:
    records = []
    for source in ("real", "sim"):
        records.append(
            {
                "image": np.zeros((480, 640, 3), dtype=np.uint8),
                "wrist_image": np.ones((480, 640, 3), dtype=np.uint8),
                "state": np.zeros(7, dtype=np.float32),
                "actions": np.ones(7, dtype=np.float32),
                "task": "pick up the red block",
                "source": source,
            }
        )
    batch = openpi_facing_batch(records)
    assert batch["observation/image"].shape == (2, 480, 640, 3)
    assert batch["observation/image"].dtype == np.uint8
    assert batch["observation/state"].shape == (2, 7)
    assert batch["observation/state"].dtype == np.float32
    assert batch["actions"].shape == (2, 7)
    assert batch["prompt"] == ("pick up the red block",) * 2
    assert "source" not in batch
