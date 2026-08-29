from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

import cluster.cli as cluster_cli
from cluster.cli import _parameters, _run, _sbatch_command
from cluster.config import ClusterSettings
from cluster.workflows import Command, Resources, WORKFLOWS, Workflow, get_workflow


def _slash(value: object) -> str:
    return str(value).replace("\\", "/")


def test_deltaai_defaults_are_centralized_and_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "XARM_WORK_ROOT", "OPENPI_ROOT", "XARM_PYTHON",
        "XARM_CLUSTER_LOG_ROOT", "XARM_SLURM_ACCOUNT", "XARM_SLURM_PARTITION",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = ClusterSettings.from_environment()
    assert str(settings.work_root) == "/work/nvme/bfmk/mw89"
    assert str(settings.python) == "/u/mw89/repos/openpi/.venv/bin/python"
    assert settings.account == "bfmk-dtai-gh"
    assert settings.partition == "ghx4"


def test_environment_overrides_every_deployment_location(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XARM_WORK_ROOT", "/srv/xarm")
    monkeypatch.setenv("OPENPI_ROOT", "/opt/openpi")
    monkeypatch.setenv("XARM_PYTHON", "/opt/python")
    monkeypatch.setenv("XARM_CLUSTER_LOG_ROOT", "/logs/xarm")
    monkeypatch.setenv("XARM_SLURM_ACCOUNT", "project")
    monkeypatch.setenv("XARM_SLURM_PARTITION", "gpu")
    settings = ClusterSettings.from_environment()
    assert _slash(settings.work_root).endswith("/srv/xarm")
    assert _slash(settings.openpi_root).endswith("/opt/openpi")
    assert _slash(settings.python).endswith("/opt/python")
    assert _slash(settings.log_root).endswith("/logs/xarm")
    assert (settings.account, settings.partition) == ("project", "gpu")
    runtime = settings.runtime_environment()
    assert _slash(runtime["MUJOCO_OUTPUT_ROOT"]).endswith("/srv/xarm/mujoco_outputs")
    assert _slash(runtime["MUJOCO_DATASET_ROOT"]).endswith("/srv/xarm/mujoco_datasets")


def test_posix_deployment_overrides_are_not_mangled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XARM_REPOSITORY", "/srv/repos/embodied-ai-xarm")
    monkeypatch.setenv("XARM_WORK_ROOT", "/srv/work/xarm")
    settings = ClusterSettings.from_environment()
    assert str(settings.repository) == "/srv/repos/embodied-ai-xarm"
    assert str(settings.work_root) == "/srv/work/xarm"


def test_workflows_delegate_without_inline_python() -> None:
    settings = ClusterSettings.from_environment()
    required_examples = {
        "physics-consistency": {"output": "/tmp/physics.json"},
        "formal-sim-evaluation": {
            "model_spec": "model.json",
            "host": "policy-node",
        },
        "training-preflight": {"config": "pi05_xarm", "dataset_paths": "real=/data/real"},
        "training": {"config": "pi05_xarm", "exp_name": "audit"},
    }
    for name, workflow in WORKFLOWS.items():
        params = workflow.parameters(required_examples.get(name, {}))
        commands = workflow.build(settings, params)
        assert commands
        assert all("-c" not in command.argv for command in commands)
        assert all(
            "sim_mujoco.data_generation" not in " ".join(command.argv)
            for command in commands
        )
    evaluation = get_workflow("formal-sim-evaluation")
    evaluation_commands = evaluation.build(
        settings,
        evaluation.parameters(required_examples["formal-sim-evaluation"]),
    )
    training = get_workflow("training")
    training_commands = training.build(
        settings,
        training.parameters(required_examples["training"]),
    )
    assert any("evaluation.sim.cli" in item.argv for item in evaluation_commands)
    assert any("training.cli" in item.argv for item in training_commands)


def test_simulation_data_workflows_preserve_phase_order() -> None:
    settings = ClusterSettings.from_environment()
    expected = {
        "smoke": ["environment", "generate-smoke", "audit-smoke", "permissions"],
        "generate": ["generate", "audit-raw", "permissions"],
        "convert": ["convert", "permissions"],
        "audit": [
            "audit-final",
            "permissions",
            "raw-path-permissions",
            "converted-path-permissions",
            "raw-acl",
            "converted-acl",
            "handoff",
        ],
    }
    for version in ("v3", "v4-10x"):
        for phase, labels in expected.items():
            workflow = get_workflow(f"sim-data-{phase}")
            commands = workflow.build(settings, {"plan": version})
            assert [command.label for command in commands] == labels


def test_sbatch_uses_only_the_generic_runner() -> None:
    settings = ClusterSettings.from_environment()
    workflow = get_workflow("sim-data-smoke")
    command = _sbatch_command(
        settings,
        workflow,
        {"plan": "v4-10x"},
        dependency="afterok:12345",
    )
    assert command[0] == "sbatch"
    runner_index = next(
        index
        for index, value in enumerate(command)
        if _slash(value).endswith("cluster/jobs/run_workflow.sbatch")
    )
    assert command[runner_index + 1] == workflow.name
    assert "--job-name=xarm-sim-data-smoke-v4-10x" in command
    assert "--dependency=afterok:12345" in command
    assert not any("slurm/simulation_data" in value for value in command)


def test_invalid_data_plan_is_rejected() -> None:
    workflow = get_workflow("sim-data-preflight")
    with pytest.raises(ValueError, match="Unsupported simulation data plan"):
        workflow.build(ClusterSettings.from_environment(), {"plan": "obsolete"})


def test_training_outputs_default_outside_repository() -> None:
    settings = ClusterSettings.from_environment()
    workflow = get_workflow("training")
    params = workflow.parameters({"config": "pi05_xarm", "exp_name": "test"})
    command = workflow.build(settings, params)[0].argv
    assert str(settings.work_root / "openpi_assets") in command
    assert str(settings.work_root / "openpi_checkpoints") in command


def test_parameter_parser_rejects_duplicates() -> None:
    assert _parameters(["a=1", "b=two=2"]) == {"a": "1", "b": "two=2"}
    with pytest.raises(ValueError, match="unique"):
        _parameters(["a=1", "a=2"])


def test_generic_job_has_no_embedded_scientific_entrypoint() -> None:
    text = Path("cluster/jobs/run_workflow.sbatch").read_text(encoding="utf-8")
    assert "cluster.cli run" in text
    for forbidden in ("data.sim", "evaluation.sim", "training.cli", "diagnostics.simulation"):
        assert forbidden not in text


def test_runner_writes_auditable_step_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = ClusterSettings(
        repository=Path.cwd(),
        work_root=tmp_path,
        openpi_root=tmp_path / "openpi",
        python=Path(sys.executable),
        account="test",
        partition="test",
        log_root=tmp_path / "logs",
    )
    workflow = Workflow(
        "test-workflow",
        "test only",
        Resources("00:01:00", 1, "1G", 0),
        lambda _settings, _params: (),
    )
    commands = (Command("success", (sys.executable, "-c", "print('ok')")),)
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setattr(
        cluster_cli,
        "_resolved",
        lambda _name, _supplied: (settings, workflow, {}, commands),
    )
    monkeypatch.setattr(cluster_cli, "_optional_output", lambda *_args: None)
    assert _run("test-workflow", {}, allow_local=False) == 0
    record = json.loads(
        (settings.log_root / "runs" / "test-workflow" / "42.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["state"] == "COMPLETED"
    assert record["job"]["id"] == "42"
    assert record["steps"][0]["label"] == "success"
    assert record["steps"][0]["returncode"] == 0
