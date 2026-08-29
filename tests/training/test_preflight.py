from __future__ import annotations

import json
from pathlib import Path

from training.configs.experiments import get_experiment
from training.validation.preflight import preflight


def _dataset(root: Path, *, state_shape: list[int] = [7]) -> Path:
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "features": {
                    "image": {"shape": [480, 640, 3]},
                    "wrist_image": {"shape": [480, 640, 3]},
                    "state": {"shape": state_shape},
                    "actions": {"shape": [7]},
                    "task": {"dtype": "string"},
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def test_preflight_validates_local_contract_and_reports_external_assets(tmp_path: Path) -> None:
    config = get_experiment("pi05_xarm_legacy_snippet_20001")
    dataset = _dataset(tmp_path / "dataset")
    report = preflight(
        config,
        dataset_paths={config.datasets.datasets[0].dataset_id: dataset},
        check_openpi=False,
    )
    assert report.passed
    assert any("local dataset metadata" in check for check in report.checks)
    assert not report.launch_ready
    assert any("must be computed" in warning for warning in report.warnings)


def test_preflight_rejects_wrong_state_shape(tmp_path: Path) -> None:
    config = get_experiment("pi05_xarm_legacy_snippet_20001")
    dataset = _dataset(tmp_path / "bad", state_shape=[8])
    report = preflight(
        config,
        dataset_paths={config.datasets.datasets[0].dataset_id: dataset},
        check_openpi=False,
    )
    assert not report.passed
    assert any("shape (8,)" in error for error in report.errors)


def test_preflight_rejects_incomplete_local_task_catalog(tmp_path: Path) -> None:
    config = get_experiment("pi05_xarm_legacy_snippet_20001")
    dataset = _dataset(tmp_path / "tasks")
    (dataset / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick up the red block"}) + "\n",
        encoding="utf-8",
    )
    report = preflight(
        config,
        dataset_paths={config.datasets.datasets[0].dataset_id: dataset},
        check_openpi=False,
    )
    assert not report.passed
    assert any("missing configured prompts" in error for error in report.errors)


def test_historical_multi_source_preflight_is_honest_about_missing_adapter() -> None:
    config = get_experiment("pi05_xarm_real50_sim50_stratified")
    report = preflight(config, check_openpi=False)
    assert report.passed
    assert not report.launch_ready
    assert any("external" in item for item in report.unresolved)
