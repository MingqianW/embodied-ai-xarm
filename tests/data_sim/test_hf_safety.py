from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.datasets.prepare_mujoco_hf_ready import prepare
from tools.datasets.upload_mujoco_dataset_to_hf import (
    build_upload_plan,
    execute,
)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_json(path: Path, value) -> None:
    _write(path, json.dumps(value))


class HuggingFaceSafetyTests(unittest.TestCase):
    def _prepared(self, root: Path) -> tuple[Path, str]:
        dataset = root / "canonical"
        raw = root / "raw"
        output = root / "hf_ready"
        repo_id = "test/xarm_mujoco_red_block_v1"
        _write_json(
            dataset / "meta" / "info.json",
            {
                "codebase_version": "v2.1",
                "fps": 10,
                "total_episodes": 1,
                "total_frames": 2,
                "splits": {"train": "0:1"},
            },
        )
        _write(dataset / "meta" / "tasks.jsonl", "{}\n")
        _write(dataset / "meta" / "episodes.jsonl", "{}\n")
        _write(dataset / "meta" / "episodes_stats.jsonl", "{}\n")
        _write(dataset / "data" / "chunk-000" / "episode_000000.parquet", "fixture")
        _write_json(
            raw / "run_config.json",
            {
                "object_xy_range_m": 0.03,
                "object_yaw_range_deg": 15.0,
                "joint_noise_rad": 0.01,
            },
        )
        _write_json(
            raw / "manifest.json",
            {
                "completed_episodes": [{"episode_index": 0}],
                "failed_attempts": [],
            },
        )
        prepare(
            argparse.Namespace(
                dataset_dir=dataset,
                raw_input_dir=raw,
                output_dir=output,
                repo_id=repo_id,
                overwrite=False,
            )
        )
        return output, repo_id

    def test_prepare_hashes_and_dry_run_never_uploads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output, repo_id = self._prepared(Path(directory))
            plan = build_upload_plan(
                output,
                repo_id=repo_id,
                private=True,
                commit_message="test",
                repo_status={
                    "exists": False,
                    "authenticated_check": True,
                    "detail": "fixture",
                },
            )
            self.assertTrue(plan["dry_run"])
            self.assertTrue(plan["local_metadata_consistent"])
            self.assertGreater(plan["total_file_count"], 0)

    def test_tampered_file_fails_manifest_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output, repo_id = self._prepared(Path(directory))
            (output / "DATASET_CARD.md").write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_upload_plan(
                    output,
                    repo_id=repo_id,
                    private=True,
                    commit_message="test",
                    repo_status={"exists": False},
                )

    def test_upload_requires_upload_yes_and_no_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output, repo_id = self._prepared(Path(directory))
            base = {
                "local_dir": output,
                "repo_id": repo_id,
                "private": True,
                "commit_message": "test",
            }
            with mock.patch(
                "tools.datasets.upload_mujoco_dataset_to_hf._repo_status",
                return_value={"exists": False, "authenticated_check": True},
            ):
                with self.assertRaises(ValueError):
                    execute(
                        argparse.Namespace(
                            **base,
                            upload=True,
                            yes=False,
                            dry_run=False,
                        )
                    )
                with self.assertRaises(ValueError):
                    execute(
                        argparse.Namespace(
                            **base,
                            upload=True,
                            yes=True,
                            dry_run=True,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
