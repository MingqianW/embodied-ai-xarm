from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from data.real import config as real_config
from data.real.tools import rename_xarm_raw_task


def _configure_temp_repo(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(real_config, "repo_root", lambda: root)
    monkeypatch.setattr(
        real_config,
        "CONFIG_PATH",
        root / "configs/data/real/xarm_data_config.json",
    )


def test_explicit_real_data_override_wins(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_repo(monkeypatch, tmp_path)
    override = tmp_path / "operator-selected"
    assert real_config.get_raw_data_root(override) == override


def test_explicit_real_data_config_wins_even_when_absent(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_repo(monkeypatch, tmp_path)
    real_config.CONFIG_PATH.parent.mkdir(parents=True)
    real_config.CONFIG_PATH.write_text(
        json.dumps({"raw_data_root": "external/site-data"}), encoding="utf-8"
    )
    assert real_config.get_raw_data_root() == tmp_path / "external/site-data"


def test_legacy_real_data_fallback_warns_without_mutating(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_repo(monkeypatch, tmp_path)
    legacy = tmp_path / real_config.LEGACY_RAW_DATA_ROOT
    legacy.mkdir(parents=True)

    with pytest.warns(real_config.RealDataPathWarning, match="no data was moved"):
        selected = real_config.get_raw_data_root()

    assert selected == legacy
    assert not (tmp_path / real_config.DEFAULT_RAW_DATA_ROOT).exists()


def test_missing_real_data_root_reports_canonical_and_override(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_repo(monkeypatch, tmp_path)
    with pytest.warns(real_config.RealDataPathWarning, match="pass --raw-root"):
        selected = real_config.get_raw_data_root()
    assert selected == tmp_path / real_config.DEFAULT_RAW_DATA_ROOT


def _raw_task(root: Path) -> Path:
    episode = root / "old_task" / "episode_000"
    episode.mkdir(parents=True)
    (episode / "meta.json").write_text(
        json.dumps({"task": "old_task"}), encoding="utf-8"
    )
    return episode


def test_rename_cli_defaults_to_preview(monkeypatch, tmp_path: Path, capsys) -> None:
    _raw_task(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rename_xarm_raw_task",
            "--raw-root",
            str(tmp_path),
            "--old-task",
            "old_task",
            "--new-task",
            "new_task",
        ],
    )

    rename_xarm_raw_task.main()

    assert (tmp_path / "old_task").is_dir()
    assert not (tmp_path / "new_task").exists()
    assert "mode: dry-run" in capsys.readouterr().out


def test_rename_cli_requires_apply_to_write(monkeypatch, tmp_path: Path) -> None:
    _raw_task(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rename_xarm_raw_task",
            "--raw-root",
            str(tmp_path),
            "--old-task",
            "old_task",
            "--new-task",
            "new_task",
            "--apply",
        ],
    )

    rename_xarm_raw_task.main()

    metadata = json.loads(
        (tmp_path / "new_task/episode_000/meta.json").read_text(encoding="utf-8")
    )
    assert metadata["task"] == "new_task"
    assert not (tmp_path / "old_task").exists()
