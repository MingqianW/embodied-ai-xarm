from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

from evaluation.common.legacy_policy_results import (
    EpisodeEvaluation,
    summarize_episode_rows,
    validate_label,
    write_evaluation_outputs,
)


class EvaluationTests(unittest.TestCase):
    def test_windows_piped_label_bom_is_accepted(self) -> None:
        self.assertEqual(validate_label("\ufefffailure"), "failure")

    def test_partial_scores_and_success_rates(self) -> None:
        rows = [
            {"label": "success", "score": 1.0},
            {"label": "failure", "score": 0.25},
            {"label": "invalid", "score": ""},
        ]
        summary = summarize_episode_rows(rows)
        self.assertAlmostEqual(summary["human_rated_task_success_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_score"], 0.625)

    def test_episode_schema_and_outputs(self) -> None:
        episode = EpisodeEvaluation(
            simulator="isaac",
            task="pick",
            prompt="pick",
            seed=0,
            checkpoint=None,
            policy_server="127.0.0.1:18000",
            start_time="2026-07-24T00:00:00Z",
            duration_s=1.0,
            success=False,
            score=0.5,
            failure_reason="missed",
            label="failure",
        )
        payload = episode.to_json()
        self.assertEqual(payload["simulator"], "isaac")
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_evaluation_outputs(Path(tmp), [payload])
            self.assertTrue((Path(tmp) / "episodes.csv").is_file())
            self.assertEqual(summary["failures"], 1)

    def test_legacy_schema_does_not_follow_policy_transport_version(self) -> None:
        import evaluation.common.legacy_policy_results as legacy_results
        import policy_runtime.schemas as policy_schemas

        original = policy_schemas.POLICY_SCHEMA_VERSION
        try:
            policy_schemas.POLICY_SCHEMA_VERSION = "future-transport-version"
            reloaded = importlib.reload(legacy_results)
            episode = reloaded.EpisodeEvaluation(
                simulator="mujoco",
                task="pick",
                prompt="pick",
                seed=0,
                checkpoint=None,
                policy_server="127.0.0.1:18000",
                start_time="2026-07-24T00:00:00Z",
                duration_s=1.0,
                success=True,
            )
            self.assertEqual(episode.schema_version, "1.0")
            self.assertEqual(
                reloaded.summarize_episode_rows([episode.to_json()])["schema_version"],
                "1.0",
            )
        finally:
            policy_schemas.POLICY_SCHEMA_VERSION = original
            importlib.reload(legacy_results)


if __name__ == "__main__":
    unittest.main()
