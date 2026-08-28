from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_mujoco.remote_policy_evaluation import (
    VideoRecorder,
    apply_initial_randomization,
    pad_to_aspect,
    read_episodes_csv,
    summarize_episode_rows,
    tile_video_frame,
    validate_label,
    write_episodes_csv,
    write_json,
    write_summary,
)
from simulation.runtime import initialize_scene
from simulation.runtime import load_simulation


class LabelAndSummaryTests(unittest.TestCase):
    def test_label_validation(self) -> None:
        self.assertEqual(validate_label("success"), "success")
        self.assertEqual(validate_label(" failure "), "failure")
        with self.assertRaises(ValueError):
            validate_label("skipped")

    def test_success_rate_excludes_invalid_denominator(self) -> None:
        rows = [
            {"label": "success", "termination_reason": "max_policy_steps", "policy_steps": 10, "sim_time": 1, "wall_time": 2},
            {"label": "failure", "termination_reason": "max_policy_steps", "policy_steps": 8, "sim_time": 1, "wall_time": 2},
            {"label": "invalid", "termination_reason": "error", "policy_steps": 0, "sim_time": 0, "wall_time": 1},
        ]
        summary = summarize_episode_rows(rows)
        self.assertEqual(summary["attempted_episodes"], 3)
        self.assertEqual(summary["invalid_episodes"], 1)
        self.assertAlmostEqual(summary["human_rated_task_success_rate"], 0.5)
        self.assertAlmostEqual(summary["end_to_end_success_rate"], 1 / 3)

    def test_csv_json_serialization_and_resume_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {
                    "episode_index": 0,
                    "seed": 7,
                    "label": "invalid",
                    "valid": False,
                    "comment": "connection failed",
                    "termination_reason": "error",
                    "policy_steps": 0,
                    "sim_time": 0.0,
                    "wall_time": 0.1,
                    "combined_video_path": "combined.mp4",
                }
            ]
            write_episodes_csv(root / "episodes.csv", rows)
            loaded = read_episodes_csv(root / "episodes.csv")
            self.assertEqual(loaded[0]["label"], "invalid")
            self.assertEqual(loaded[0]["episode_index"], "0")
            write_json(root / "result.json", {"array": np.asarray([1, 2], dtype=np.float32)})
            self.assertIn("array", (root / "result.json").read_text(encoding="utf-8"))

    def test_written_summary_contains_per_task_breakdown(self) -> None:
        rows = [
            {"task": "red_block", "label": "success", "policy_steps": 2},
            {"task": "blue_block", "label": "failure", "policy_steps": 3},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_summary(Path(tmp), rows)
        self.assertEqual(set(summary["task_breakdown"]), {"red_block", "blue_block"})
        self.assertEqual(summary["task_breakdown"]["red_block"]["successes"], 1)


class VideoUtilityTests(unittest.TestCase):
    def test_aspect_ratio_padding_shape(self) -> None:
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        padded = pad_to_aspect(image, 224, 224)
        self.assertEqual(padded.shape, (224, 224, 3))
        self.assertEqual(padded.dtype, np.uint8)

    def test_tiled_video_frame_shape(self) -> None:
        overview = np.ones((480, 640, 3), dtype=np.uint8)
        base = np.ones((224, 224, 3), dtype=np.uint8) * 2
        wrist = np.ones((224, 224, 3), dtype=np.uint8) * 3
        tiled = tile_video_frame(overview, base, wrist)
        self.assertEqual(tiled.shape, (480, 640, 3))
        self.assertEqual(tiled.dtype, np.uint8)

    def test_video_finalization_writes_nonempty_files(self) -> None:
        context = load_simulation()
        try:
            initialize_scene(context.model, context.data)
            with tempfile.TemporaryDirectory() as tmp:
                recorder = VideoRecorder(Path(tmp), fps=5)
                try:
                    recorder.record(context)
                finally:
                    recorder.close()
                recorder.validate_outputs()
                self.assertGreater(recorder.frame_count, 0)
        finally:
            context.close()


class RandomizationTests(unittest.TestCase):
    def test_initial_randomization_is_deterministic(self) -> None:
        context_a = load_simulation()
        context_b = load_simulation()
        try:
            initialize_scene(context_a.model, context_a.data)
            initialize_scene(context_b.model, context_b.data)
            info_a = apply_initial_randomization(
                context_a.model,
                context_a.data,
                seed=123,
                object_xy_range=0.03,
                object_yaw_range_deg=15,
                joint_noise=0.01,
            )
            info_b = apply_initial_randomization(
                context_b.model,
                context_b.data,
                seed=123,
                object_xy_range=0.03,
                object_yaw_range_deg=15,
                joint_noise=0.01,
            )
            self.assertEqual(info_a["object_xy_delta"], info_b["object_xy_delta"])
            np.testing.assert_allclose(context_a.data.qpos[:8], context_b.data.qpos[:8])
        finally:
            context_a.close()
            context_b.close()


if __name__ == "__main__":
    unittest.main()
