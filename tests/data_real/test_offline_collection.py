from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
import numpy as np

from data.common.schema import XARM_STATE_COLUMNS
from data.real.collection.raw_episodes import discover_raw_episodes, state_from_row
from data.real.conversion import convert_xarm_raw_to_lerobot as converter


def _raw_fixture(root: Path) -> Path:
    episode = root / "pick_up_the_red_block" / "episode_000"
    episode.mkdir(parents=True)
    (episode / "meta.json").write_text(
        json.dumps({"task": "pick_up_the_red_block", "created_ts": 123.0}),
        encoding="utf-8",
    )
    for camera in ("realsense_0", "realsense_1"):
        (episode / camera).mkdir()
        for index in range(2):
            Image.fromarray(np.full((480, 640, 3), index, dtype=np.uint8)).save(
                episode / camera / f"{index}.png"
            )
    fields = (
        "ts",
        *XARM_STATE_COLUMNS,
        "realsense_0_file",
        "realsense_1_file",
    )
    with (episode / "robot_log.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(2):
            writer.writerow(
                {
                    "ts": 123.0 + index / 10.0,
                    **{
                        name: index + offset / 10.0
                        for offset, name in enumerate(XARM_STATE_COLUMNS)
                    },
                    "realsense_0_file": f"realsense_0/{index}.png",
                    "realsense_1_file": f"realsense_1/{index}.png",
                }
            )
    return episode


def test_real_raw_discovery_preserves_order_units_and_timestamp_rows(tmp_path: Path) -> None:
    _raw_fixture(tmp_path)
    episodes = discover_raw_episodes(tmp_path)
    assert len(episodes) == 1
    assert episodes[0].raw_id == "pick_up_the_red_block/episode_000"
    assert state_from_row(episodes[0].rows[0]) == [index / 10.0 for index in range(7)]
    assert [float(row["ts"]) for row in episodes[0].rows] == [123.0, 123.1]


def test_offline_real_conversion_matches_shared_training_contract(
    tmp_path: Path, monkeypatch
) -> None:
    raw = tmp_path / "raw"
    _raw_fixture(raw)
    output = tmp_path / "converted"
    manifest = tmp_path / "manifest.json"
    captured = []

    monkeypatch.setattr(converter, "_try_write_hf_dataset", lambda *args, **kwargs: False)

    def fake_lerobot(records_by_episode, **kwargs):
        captured.extend(records_by_episode)
        return True

    monkeypatch.setattr(converter, "_try_write_lerobot_dataset", fake_lerobot)
    converter.convert(
        raw,
        output,
        repo_id="local/test",
        robot_type="xarm6",
        fps=10,
        push_to_hub=False,
        hub_private=True,
        overwrite=False,
        append_new=False,
        manifest_path=manifest,
        skip_light_image_copy=False,
        skip_hf_dataset=True,
        image_writer_threads=0,
        image_writer_processes=0,
    )

    assert len(captured) == 1 and len(captured[0].frames) == 1
    record = captured[0].as_records()[0]
    assert record["episode_index"] == 0
    assert record["frame_index"] == 0
    assert record["timestamp"] == 123.0
    assert record["source"] == "real"
    assert record["task"] == "pick up the red block"
    np.testing.assert_array_equal(
        record["state"],
        np.asarray([index / 10.0 for index in range(7)], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        record["actions"],
        np.asarray([1 + index / 10.0 for index in range(7)], dtype=np.float32),
    )
    assert json.loads((output / "meta/info.json").read_text())["source_backend"] == "real"
    manifest_data = json.loads(manifest.read_text())
    assert manifest_data["source_backend"] == "real"
    assert manifest_data["converted_raw_episodes"] == [
        "pick_up_the_red_block/episode_000"
    ]
