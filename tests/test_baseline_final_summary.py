import json
from pathlib import Path
import sys

import pytest

from sim_mujoco.scripts import summarize_baseline_final


def _write_evaluation(root: Path, label: str, *, control_duration: float = 0.02) -> None:
    root.mkdir()
    seeds = list(range(50_000, 50_020))
    config = {
        "policy_label": label,
        "episodes": 20,
        "seeds": seeds,
        "task": "red_block",
        "host": "127.0.0.1",
        "port": 8000,
        "max_policy_steps": 80,
        "execute_chunk_steps": 1,
        "control_duration": control_duration,
        "object_xy_range": 0.03,
        "object_yaw_range_deg": 15.0,
        "joint_noise": 0.01,
        "headless": True,
    }
    rows = [
        {
            "seed": seed,
            "label": "success" if index < 10 else "failure",
        }
        for index, seed in enumerate(seeds)
    ]
    summary = {
        "attempted": 20,
        "successes": 10,
        "failures": 10,
        "invalid": 0,
        "success_rate_all": 0.5,
        "success_rate_valid": 0.5,
        "episodes": rows,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_summary_requires_identical_real_configs(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    final = tmp_path / "final"
    output = tmp_path / "output"
    _write_evaluation(baseline, "baseline")
    _write_evaluation(final, "final")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize",
            "--baseline",
            str(baseline),
            "--final",
            str(final),
            "--output-dir",
            str(output),
        ],
    )
    summarize_baseline_final.main()
    report = json.loads((output / "POST_TRAINING_EVALUATION.json").read_text())
    assert report["passed"]
    assert report["identical_seeds"]
    assert len(report["seeds"]) == 20
    assert report["shared_settings"]["control_duration"] == 0.02


def test_summary_rejects_different_control_settings(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    final = tmp_path / "final"
    _write_evaluation(baseline, "baseline")
    _write_evaluation(final, "final", control_duration=0.03)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize",
            "--baseline",
            str(baseline),
            "--final",
            str(final),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    with pytest.raises(RuntimeError, match="configurations differ"):
        summarize_baseline_final.main()
