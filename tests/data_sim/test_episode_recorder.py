from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from policy_runtime.image_preprocessing import (
    ImagePreprocessingConfig,
    preprocess_policy_image,
)
from data.sim.generation.legacy.episode_recorder import (
    EpisodeRecorder,
    EpisodeRecorderConfig,
    REAL_TRAINING_PROMPT,
)
from data.sim.generation.oracle import (
    ScriptedOracleController,
)
from simulation.environment import MuJoCoEnvironment
from data.sim.generation.legacy.collect_oracle_data import (
    _record_attempt,
    _run_config_from_args,
    load_or_initialize_run,
    should_record_success_video,
)


def _args(**overrides):
    values = {
        "task": "red_block",
        "episodes": 3,
        "seed_start": 0,
        "action_hz": 10,
        "object_xy_range": 0.0,
        "object_yaw_range_deg": 0.0,
        "joint_noise": 0.0,
        "save_only_success": True,
        "record_video": False,
        "video_every": 1,
        "max_attempts": 9,
        "headless": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class EpisodeRecorderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = MuJoCoEnvironment(task="red_block", settle_steps=50)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.environment.close()

    def test_pre_action_alignment_and_rgb_images(self) -> None:
        self.environment.reset(seed=0)
        controller = ScriptedOracleController(self.environment)
        action = controller.next_action()
        self.assertIsNotNone(action)
        state_before = self.environment.observe().state.copy()
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(
                EpisodeRecorderConfig(
                    output_dir=Path(directory),
                    task="red_block",
                    prompt=REAL_TRAINING_PROMPT,
                    seed=0,
                    fps=10,
                ),
                self.environment,
            )
            recorder.record_pre_action(
                action=action,
                oracle_stage=controller.stage.value,
            )
            recorder.finalize(
                success=False,
                failure_reason="unit_test_stop",
                task_metrics=self.environment.task_runtime.metrics(),
                initial_conditions=self.environment.initial_conditions,
                randomization={
                    "object_xy_range_m": 0.0,
                    "object_yaw_range_deg": 0.0,
                    "joint_noise_rad": 0.0,
                },
                transitions=controller.transition_log(),
                oracle_plan=controller.plan.to_json(),
            )
            with np.load(Path(directory) / "observations.npz") as payload:
                np.testing.assert_allclose(payload["state"][0], state_before)
                np.testing.assert_allclose(payload["actions"][0], action)
                self.assertEqual(float(payload["timestamp"][0]), 0.0)
                image_path = str(payload["image"][0])
                policy_image_path = str(payload["policy_image"][0])

            base = np.asarray(
                Image.open(Path(directory) / image_path).convert("RGB")
            )
            policy = np.asarray(
                Image.open(
                    Path(directory) / policy_image_path
                ).convert("RGB")
            )
            self.assertEqual(base.shape, (480, 640, 3))
            self.assertEqual(policy.shape, (224, 224, 3))
            expected_policy = preprocess_policy_image(
                base,
                ImagePreprocessingConfig(
                    width=224,
                    height=224,
                    input_color_order="RGB",
                ),
            )
            np.testing.assert_array_equal(policy, expected_policy)

    def test_resume_accepts_identical_and_rejects_changed_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = _run_config_from_args(_args())
            load_or_initialize_run(output, config, resume=False)
            loaded, manifest = load_or_initialize_run(
                output,
                config,
                resume=True,
            )
            self.assertEqual(loaded, config)
            self.assertEqual(manifest["completed_episodes"], [])

            changed = dict(config)
            changed["action_hz"] = 20
            with self.assertRaises(ValueError):
                load_or_initialize_run(output, changed, resume=True)

    def test_existing_nonempty_output_requires_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "unrelated.txt").write_text("do not overwrite")
            with self.assertRaises(FileExistsError):
                load_or_initialize_run(
                    output,
                    _run_config_from_args(_args()),
                    resume=False,
                )
            self.assertEqual(
                (output / "unrelated.txt").read_text(),
                "do not overwrite",
            )

    def test_video_every_uses_one_based_success_count(self) -> None:
        recorded = [
            index
            for index in range(12)
            if should_record_success_video(
                record_video=True,
                prospective_episode_index=index,
                video_every=5,
            )
        ]
        self.assertEqual(recorded, [4, 9])
        self.assertFalse(
            should_record_success_video(
                record_video=False,
                prospective_episode_index=4,
                video_every=5,
            )
        )

    def test_metadata_separates_training_and_debug_fields(self) -> None:
        from simulation.resources import output_root

        metadata_path = (
            output_root()
            / "collection_smoke_one"
            / "episodes"
            / "episode_000000"
            / "metadata.json"
        )
        if not metadata_path.exists():
            self.skipTest("collection smoke artifact is unavailable")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(metadata["training_fields"]),
            {"image", "wrist_image", "state", "actions", "task"},
        )
        self.assertIn("mujoco_qpos", metadata["debug_fields"])
        self.assertTrue(metadata["success"])

    def test_same_seed_records_identical_training_data_and_images(self) -> None:
        args = _args(
            episodes=1,
            seed_start=77,
            object_xy_range=0.01,
            object_yaw_range_deg=5.0,
            joint_noise=0.005,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = []
            successes = []
            for name in ("first", "second"):
                success, metadata = _record_attempt(
                    self.environment,
                    seed=77,
                    attempt_dir=root / name,
                    args=args,
                    record_video=False,
                )
                successes.append(success)
                results.append(metadata)
            self.assertEqual(successes[0], successes[1])
            self.assertEqual(
                results[0]["failure_reason"], results[1]["failure_reason"]
            )
            keys = (
                "state",
                "actions",
                "task",
                "timestamp",
                "oracle_stage",
                "object_position",
                "tcp_position",
            )
            with np.load(root / "first" / "observations.npz") as first:
                with np.load(root / "second" / "observations.npz") as second:
                    for key in keys:
                        np.testing.assert_array_equal(first[key], second[key])
                    image_names = [
                        *first["image"].tolist(),
                        *first["wrist_image"].tolist(),
                        *first["policy_image"].tolist(),
                        *first["policy_wrist_image"].tolist(),
                    ]
            for relative in image_names:
                self.assertEqual(
                    (root / "first" / relative).read_bytes(),
                    (root / "second" / relative).read_bytes(),
                )
            self.assertEqual(results[0]["oracle_plan"], results[1]["oracle_plan"])


if __name__ == "__main__":
    unittest.main()
