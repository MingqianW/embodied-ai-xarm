from dataclasses import replace
import json
from pathlib import Path

import pytest

from cluster.cli import _resolved, _sbatch_command, _workflow_json
from cluster.config import ClusterSettings
from cluster.workflows import get_workflow


def test_farmshare_resources_change_scheduling_only(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = Path(__file__).resolve().parents[2]
    config = repository / "cluster" / "farmshare" / "resources.json"
    monkeypatch.setenv("XARM_SLURM_RESOURCE_CONFIG", str(config))
    monkeypatch.setenv("XARM_SLURM_QOS", "gpu")
    monkeypatch.setenv("XARM_SLURM_ACCOUNT", "verified_account")
    monkeypatch.setenv("XARM_SLURM_PARTITION", "gpu")
    supplied = {"model_spec": "model.json", "host": "127.0.0.1"}
    settings, workflow, params, commands = _resolved("formal-sim-evaluation", supplied)
    original = get_workflow("formal-sim-evaluation")
    assert workflow.build(settings, params) == original.build(settings, params)
    assert (workflow.resources.cpus, workflow.resources.memory) == (16, "64G")
    resolved = _workflow_json(settings, workflow, params, commands)
    assert resolved["resources"]["memory"] == "64G"
    sbatch = _sbatch_command(settings, original, supplied)
    assert "--mem=64G" in sbatch
    assert "--cpus-per-task=16" in sbatch
    assert "--qos=gpu" in sbatch
    assert "--partition=gpu" in sbatch
    assert "XARM_SLURM_RESOURCE_CONFIG=" in next(x for x in sbatch if x.startswith("--export="))


@pytest.mark.parametrize("override", [
    {"cpus": 0}, {"cpus": True}, {"gpus": -1}, {"gpus": "1"},
    {"memory": "0G"}, {"time": "12:99:00"}, {"unknown": 1}, [],
])
def test_invalid_resource_overrides_fail_before_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: object,
) -> None:
    config = tmp_path / "resources.json"
    config.write_text(json.dumps({"environment-check": override}), encoding="utf-8")
    monkeypatch.setenv("XARM_SLURM_RESOURCE_CONFIG", str(config))
    with pytest.raises(ValueError, match="Resource"):
        _resolved("environment-check", {})


def test_absent_site_config_preserves_deltaai_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("XARM_SLURM_RESOURCE_CONFIG", "XARM_SLURM_QOS"):
        monkeypatch.delenv(name, raising=False)
    settings = ClusterSettings.from_environment()
    workflow = get_workflow("formal-sim-evaluation")
    supplied = {"model_spec": "model.json", "host": "policy-node"}
    sbatch = _sbatch_command(settings, workflow, supplied)
    assert "--mem=128G" in sbatch
    assert not any(x.startswith("--qos=") for x in sbatch)
    assert "XARM_SLURM_RESOURCE_CONFIG" not in settings.runtime_environment()
    assert _sbatch_command(replace(settings, resource_config=None, qos=None), workflow, supplied) == sbatch
