"""Thin adapter from ``data.common`` records to OpenPI's repacked keys."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from data.common.validation import validate_training_record


# OpenPI's RepackTransform mapping is destination key -> stored dataset key.
XARM_OPENPI_REPACK_MAPPING = {
    "observation/image": "image",
    "observation/wrist_image": "wrist_image",
    "observation/state": "state",
    "actions": "actions",
    "prompt": "prompt",
}
XARM_DELTA_ACTION_MASK = (True, True, True, True, True, True, False)
XARM_ACTION_SEQUENCE_KEYS = ("actions",)


def openpi_facing_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate canonical model fields and remove all provenance metadata.

    OpenPI's prompt transform normally derives ``prompt`` from the LeRobot
    task.  This fixture adapter performs that final rename directly so the
    source-metadata exclusion boundary can be tested without OpenPI installed.
    """

    canonical = validate_training_record(record)
    return {
        "observation/image": canonical["image"],
        "observation/wrist_image": canonical["wrist_image"],
        "observation/state": canonical["state"],
        "actions": canonical["actions"],
        "prompt": canonical["task"],
    }


def openpi_facing_batch(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate validated synthetic/local records without retaining provenance."""

    samples = [openpi_facing_record(record) for record in records]
    if not samples:
        raise ValueError("Cannot collate an empty OpenPI-facing batch")
    return {
        key: (
            tuple(sample[key] for sample in samples)
            if key == "prompt"
            else np.stack([sample[key] for sample in samples], axis=0)
        )
        for key in samples[0]
    }
