from __future__ import annotations

import json
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from data.common import lerobot_writer
from data.common.records import EpisodeRecord, FrameRecord, SourceBackend
from data.sim.generation.legacy.episode_recorder import (
    RAW_SCHEMA_VERSION,
    REAL_TRAINING_PROMPT,
)
from data.sim.generation.legacy.lerobot_adapter import (
    discover_successful_episodes,
    load_episode_records,
    validate_temporal_alignment,
)
from data.sim.generation.legacy.convert_mujoco_to_lerobot import convert
from data.sim.generation.tools.validate_lerobot_dataset import (
    EXPECTED_FEATURES,
    _selected_raw_episodes,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _raw_fixture(root: Path, *, successful: bool = True) -> Path:
    collection = root / "raw"
    episode = collection / "episodes" / "episode_000000"
    episode.mkdir(parents=True)
    _write_json(
        collection / "run_config.json",
        {"task": "red_block", "action_hz": 10},
    )
    entry = {
        "episode_index": 0,
        "path": "episodes/episode_000000",
        "success": successful,
    }
    _write_json(
        collection / "manifest.json",
        {"completed_episodes": [entry]},
    )
    _write_json(
        episode / "metadata.json",
        {
            "schema_version": RAW_SCHEMA_VERSION,
            "task": "red_block",
            "prompt": REAL_TRAINING_PROMPT,
            "success": successful,
            "failure_reason": None if successful else "test_failure",
            "fps": 10,
            "seed": 7,
            "number_of_samples": 2,
        },
    )
    for key, color in (("base_images", (255, 0, 0)), ("wrist_images", (0, 255, 0))):
        directory = episode / key
        directory.mkdir()
        for index in range(2):
            Image.new("RGB", (640, 480), color).save(
                directory / f"frame_{index:06d}.png"
            )
    state = np.asarray(
        [
            [0, -0.6, -1.2, 0, 1.8, 0, 845],
            [0.01, -0.59, -1.19, 0, 1.79, 0, 800],
        ],
        dtype=np.float32,
    )
    actions = np.asarray(
        [
            [0.02, -0.58, -1.18, 0, 1.78, 0, 780],
            [0.03, -0.57, -1.17, 0, 1.77, 0, 760],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        episode / "observations.npz",
        image=np.asarray(
            [
                "base_images/frame_000000.png",
                "base_images/frame_000001.png",
            ]
        ),
        wrist_image=np.asarray(
            [
                "wrist_images/frame_000000.png",
                "wrist_images/frame_000001.png",
            ]
        ),
        state=state,
        actions=actions,
        task=np.asarray([REAL_TRAINING_PROMPT] * 2),
        timestamp=np.asarray([0.0, 0.1]),
    )
    return collection


class MuJoCoLeRobotAdapterTests(unittest.TestCase):
    def test_discovers_only_manifest_approved_successes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            collection = _raw_fixture(Path(directory))
            failed = collection / "failed_attempts" / "attempt_000001"
            failed.mkdir(parents=True)
            (failed / "observations.npz").write_bytes(b"not training data")
            episodes = discover_successful_episodes(collection)
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0].episode_index, 0)
            self.assertNotIn("failed_attempts", episodes[0].relative_path)

    def test_rejects_failed_episode_in_completed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            collection = _raw_fixture(Path(directory), successful=False)
            with self.assertRaises(ValueError):
                discover_successful_episodes(collection)

    def test_loads_rgb_float32_records_and_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            collection = _raw_fixture(Path(directory))
            episode = discover_successful_episodes(collection)[0]
            records = load_episode_records(episode)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["state"].dtype, np.float32)
            self.assertEqual(records[0]["actions"].shape, (7,))
            self.assertEqual(records[0]["task"], REAL_TRAINING_PROMPT)
            report = validate_temporal_alignment(records)
            self.assertEqual(report["overall_moved_toward_fraction"], 1.0)

    def test_same_raw_episode_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            collection = _raw_fixture(Path(directory))
            episode = discover_successful_episodes(collection)[0]
            first = load_episode_records(episode)
            second = load_episode_records(episode)
            for left, right in zip(first, second, strict=True):
                np.testing.assert_array_equal(left["state"], right["state"])
                np.testing.assert_array_equal(left["actions"], right["actions"])
                self.assertEqual(left["image"], right["image"])

    def test_validation_reconstructs_recorded_episode_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection = _raw_fixture(root)
            dataset = root / "dataset"
            _write_json(
                dataset / "meta" / "mujoco_conversion_manifest.json",
                {
                    "source_selection": {
                        "strategy": "first_successful_by_episode_index",
                        "episode_limit": 1,
                    }
                },
            )
            selected = _selected_raw_episodes(dataset, collection)
            self.assertEqual([episode.episode_index for episode in selected], [0])


class _FakeMeta:
    def __init__(self) -> None:
        self.total_episodes = 0


class _FakeLeRobotDataset:
    created_kwargs = None
    latest = None

    def __init__(self, repo_id: str, root: Path | None = None) -> None:
        self.repo_id = repo_id
        self.root = root
        self.meta = _FakeMeta()
        self.frames = []
        self.saved = 0
        self.stopped = False
        _FakeLeRobotDataset.latest = self

    @classmethod
    def create(cls, **kwargs):
        cls.created_kwargs = kwargs
        value = cls(kwargs["repo_id"], kwargs["root"])
        return value

    def start_image_writer(self, num_processes=0, num_threads=4) -> None:
        self.writer = (num_processes, num_threads)

    def stop_image_writer(self) -> None:
        self.stopped = True

    def add_frame(self, frame) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.saved += 1
        self.meta.total_episodes += 1

    def push_to_hub(self, **kwargs) -> None:
        raise AssertionError("Unit test must never upload")


class SharedWriterTests(unittest.TestCase):
    def test_writer_uses_exact_real_schema_and_never_uploads(self) -> None:
        episode = EpisodeRecord(
            episode_index=0,
            source=SourceBackend.SIM,
            frames=(
                FrameRecord(
                    image=np.zeros((480, 640, 3), dtype=np.uint8),
                    wrist_image=np.zeros((480, 640, 3), dtype=np.uint8),
                    state=np.zeros(7, dtype=np.float32),
                    actions=np.ones(7, dtype=np.float32),
                    task=REAL_TRAINING_PROMPT,
                    episode_index=0,
                    frame_index=0,
                    timestamp=0.0,
                    source=SourceBackend.SIM,
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            with mock.patch.object(
                lerobot_writer,
                "_lerobot_imports",
                return_value=(Path(directory), _FakeLeRobotDataset),
            ):
                result = lerobot_writer.write_xarm_lerobot_dataset(
                    [episode],
                    repo_id="local/test",
                    output_path=output,
                    robot_type="xarm6",
                    fps=10,
                    overwrite=False,
                    resume=False,
                    push_to_hub=False,
                )
        features = _FakeLeRobotDataset.created_kwargs["features"]
        self.assertEqual(
            list(features),
            ["image", "wrist_image", "state", "actions"],
        )
        self.assertEqual(features["image"]["shape"], (480, 640, 3))
        self.assertEqual(features["state"]["shape"], (7,))
        self.assertEqual(features["state"]["dtype"], "float32")
        self.assertEqual(result["written_frames"], 1)
        self.assertEqual(_FakeLeRobotDataset.latest.saved, 1)
        self.assertTrue(_FakeLeRobotDataset.latest.stopped)
        default_features = {
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
        }
        normalized_custom = {
            key: {
                "dtype": value["dtype"],
                "shape": list(value["shape"]),
                "names": value["names"],
            }
            for key, value in features.items()
        }
        self.assertEqual(
            {**normalized_custom, **default_features},
            EXPECTED_FEATURES,
        )


class MuJoCoConverterResumeTests(unittest.TestCase):
    def _args(self, input_dir: Path, output_dir: Path, **overrides):
        values = {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "repo_id": "local/test",
            "dataset_name": "test",
            "resume": False,
            "overwrite": False,
            "validate_only": False,
            "copy_videos": False,
            "episode_limit": None,
            "num_workers": 1,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_resume_skips_converted_episode_and_rejects_identity_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = _raw_fixture(root)
            output = root / "canonical"

            def fake_write(records_by_episode, **kwargs):
                output.mkdir(parents=True, exist_ok=True)
                return {
                    "output_path": str(output),
                    "repo_id": kwargs["repo_id"],
                    "fps": 10,
                    "starting_episodes": 0,
                    "written_episodes": len(records_by_episode),
                    "ending_episodes": len(records_by_episode),
                    "written_frames": sum(map(len, records_by_episode)),
                }

            with mock.patch(
                "data.sim.generation.legacy.convert_mujoco_to_lerobot.write_xarm_lerobot_dataset",
                side_effect=fake_write,
            ) as writer:
                first = convert(self._args(raw, output))
                self.assertEqual(first["written_episodes"], 1)
                self.assertEqual(writer.call_count, 1)

            with mock.patch(
                "data.sim.generation.legacy.convert_mujoco_to_lerobot.write_xarm_lerobot_dataset"
            ) as writer:
                resumed = convert(
                    self._args(raw, output, resume=True)
                )
                self.assertEqual(resumed["written_episodes"], 0)
                writer.assert_not_called()

            with self.assertRaises(ValueError):
                convert(
                    self._args(
                        raw,
                        output,
                        resume=True,
                        repo_id="local/different",
                    )
                )

    def test_episode_limit_is_recorded_and_cannot_exceed_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = _raw_fixture(root)
            output = root / "canonical"

            def fake_write(records_by_episode, **kwargs):
                output.mkdir(parents=True, exist_ok=True)
                return {
                    "output_path": str(output),
                    "repo_id": kwargs["repo_id"],
                    "fps": 10,
                    "starting_episodes": 0,
                    "written_episodes": len(records_by_episode),
                    "ending_episodes": len(records_by_episode),
                    "written_frames": sum(map(len, records_by_episode)),
                }

            with mock.patch(
                "data.sim.generation.legacy.convert_mujoco_to_lerobot.write_xarm_lerobot_dataset",
                side_effect=fake_write,
            ):
                convert(self._args(raw, output, episode_limit=1))

            manifest = json.loads(
                (
                    output
                    / "meta"
                    / "mujoco_conversion_manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["source_selection"],
                {
                    "strategy": "first_successful_by_episode_index",
                    "episode_limit": 1,
                },
            )
            self.assertEqual(len(manifest["episodes"]), 1)

            with self.assertRaisesRegex(ValueError, "only 1 are available"):
                convert(
                    self._args(
                        raw,
                        root / "too_many",
                        episode_limit=2,
                        validate_only=True,
                    )
                )


if __name__ == "__main__":
    unittest.main()
