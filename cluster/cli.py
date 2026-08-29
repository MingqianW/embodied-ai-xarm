"""List, inspect, run, and submit auditable DeltaAI/Slurm workflows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
from typing import Any, Iterable

from cluster.config import ClusterSettings
from cluster.workflows import Command, WORKFLOWS, Workflow, get_workflow


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parameters(values: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Parameter must be NAME=VALUE, got {value!r}")
        name, item = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"Parameter name must be nonempty and unique: {name!r}")
        result[name] = item
    return result


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _command_json(command: Command) -> dict[str, Any]:
    return {
        "label": command.label,
        "argv": list(command.argv),
        "record_output": command.record_output,
        "append_output": command.append_output,
    }


def _resolved(
    name: str,
    supplied: dict[str, str],
) -> tuple[ClusterSettings, Workflow, dict[str, str], tuple[Command, ...]]:
    settings = ClusterSettings.from_environment()
    workflow = get_workflow(name)
    params = workflow.parameters(supplied)
    commands = workflow.build(settings, params)
    if not commands or any(not command.argv for command in commands):
        raise ValueError(f"Workflow {name} resolved to an empty command")
    return settings, workflow, params, commands


def _workflow_json(
    settings: ClusterSettings,
    workflow: Workflow,
    params: dict[str, str],
    commands: tuple[Command, ...],
) -> dict[str, Any]:
    return {
        "name": workflow.name,
        "description": workflow.description,
        "resources": vars(workflow.resources),
        "required_parameters": list(workflow.required_parameters),
        "parameters": params,
        "commands": [_command_json(command) for command in commands],
        "settings": {
            "repository": str(settings.repository),
            "work_root": str(settings.work_root),
            "openpi_root": str(settings.openpi_root),
            "python": str(settings.python),
            "account": settings.account,
            "partition": settings.partition,
            "log_root": str(settings.log_root),
        },
    }


def _sbatch_command(
    settings: ClusterSettings,
    workflow: Workflow,
    supplied: dict[str, str],
    dependency: str | None = None,
) -> list[str]:
    resource = workflow.resources
    slurm_logs = settings.log_root / "slurm"
    params = workflow.parameters(supplied)
    identity = params.get("plan") or params.get("config")
    if not identity and params.get("model_spec"):
        identity = Path(params["model_spec"]).stem
    identity_suffix = (
        "-" + re.sub(r"[^A-Za-z0-9_-]+", "-", identity).strip("-")
        if identity
        else ""
    )
    if dependency and not re.fullmatch(r"[A-Za-z0-9_,:?-]+", dependency):
        raise ValueError(
            "Slurm dependency contains unsupported characters; use a value "
            "such as afterok:12345"
        )
    exported = settings.runtime_environment()
    selected = {
        name: value
        for name, value in exported.items()
        if name.startswith("XARM_") or name == "OPENPI_ROOT"
    }
    if dependency:
        selected["XARM_SLURM_DEPENDENCY"] = dependency
    if any("," in value or "\n" in value for value in selected.values()):
        raise ValueError("Slurm export values must not contain commas or newlines")
    export_arg = "ALL," + ",".join(
        f"{name}={value}" for name, value in sorted(selected.items())
    )
    command = [
        "sbatch",
        "--parsable",
        f"--job-name=xarm-{workflow.name}{identity_suffix}",
        f"--account={settings.account}",
        f"--partition={settings.partition}",
        f"--time={resource.time}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={resource.cpus}",
        f"--mem={resource.memory}",
        f"--output={slurm_logs / '%x-%j.out'}",
        f"--error={slurm_logs / '%x-%j.err'}",
        f"--export={export_arg}",
    ]
    if resource.gpus:
        command.append(f"--gpus-per-node={resource.gpus}")
    if dependency:
        command.append(f"--dependency={dependency}")
    command.extend(
        (
            str(settings.repository / "cluster" / "jobs" / "run_workflow.sbatch"),
            workflow.name,
        )
    )
    for name, value in supplied.items():
        command.extend(("--param", f"{name}={value}"))
    return command


def _git(repository: os.PathLike[str], *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _optional_output(*command: str) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json(value), encoding="utf-8")
    temporary.replace(path)


def _record_generation_status(
    settings: ClusterSettings,
    workflow: Workflow,
    params: dict[str, str],
    commands: tuple[Command, ...],
    state: str,
    failure: str | None,
) -> None:
    if workflow.phase is None:
        return
    flattened = [value for command in commands for value in command.argv]
    config = flattened[flattened.index("--config") + 1]
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    resume_command = f"python -m cluster.cli submit {workflow.name}"
    if params.get("plan"):
        resume_command += f" --param plan={params['plan']}"
    status = [
        str(settings.python),
        "-m",
        "data.sim.generation.cli",
        "status",
        "--config",
        config,
        "--phase",
        workflow.phase,
        "--job-id",
        job_id,
        "--job-state",
        state,
        "--next-action",
        workflow.next_action or "Inspect the cluster run record.",
        "--resume-command",
        resume_command,
    ]
    if state == "COMPLETED" and workflow.completed_work:
        status.extend(("--completed-work", workflow.completed_work))
    if failure:
        status.extend(("--known-failure", failure))
    subprocess.run(
        status,
        cwd=settings.repository,
        env={**os.environ, **settings.runtime_environment()},
        check=True,
    )


def _run(name: str, supplied: dict[str, str], *, allow_local: bool) -> int:
    if not os.environ.get("SLURM_JOB_ID") and not allow_local:
        raise RuntimeError(
            "cluster workflows must run inside Slurm; use --allow-local only "
            "for deliberate local testing"
        )
    settings, workflow, params, commands = _resolved(name, supplied)
    environment = {**os.environ, **settings.runtime_environment()}
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    record_path = settings.log_root / "runs" / name / f"{job_id}.json"
    record = {
        "schema_version": 1,
        "state": "RUNNING",
        "started_utc": _utc(),
        "job": {
            "id": job_id,
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node_list": os.environ.get("SLURM_NODELIST"),
            "dependency": os.environ.get("XARM_SLURM_DEPENDENCY")
            or os.environ.get("SLURM_JOB_DEPENDENCY"),
        },
        "workflow": _workflow_json(settings, workflow, params, commands),
        "provenance": {
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "repository_sha": _git(settings.repository, "rev-parse", "HEAD"),
            "repository_branch": _git(settings.repository, "branch", "--show-current"),
            "repository_status": _git(settings.repository, "status", "--short"),
            "openpi_sha": _git(settings.openpi_root, "rev-parse", "HEAD"),
            "gpu_inventory": _optional_output(
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ),
            "runtime_environment": settings.runtime_environment(),
        },
        "steps": [],
    }
    _write(record_path, record)
    print(
        _json(
            {"run_record": str(record_path), "workflow": name, "job_id": job_id}
        ),
        end="",
    )
    failure: str | None = None
    exit_code = 0
    for command in commands:
        step = {"label": command.label, "argv": list(command.argv), "started_utc": _utc()}
        record["steps"].append(step)
        _write(record_path, record)
        print(
            f"CLUSTER_STEP label={command.label} "
            f"command={shlex.join(command.argv)}",
            flush=True,
        )
        if command.record_output:
            result = subprocess.run(
                command.argv,
                cwd=settings.repository,
                env=environment,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            output = result.stdout or ""
            print(output, end="", flush=True)
            target = Path(command.record_output)
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if command.append_output else "w"
            with target.open(mode, encoding="utf-8") as stream:
                stream.write(output)
        else:
            result = subprocess.run(
                command.argv,
                cwd=settings.repository,
                env=environment,
                check=False,
            )
        step.update({"finished_utc": _utc(), "returncode": result.returncode})
        _write(record_path, record)
        if result.returncode:
            exit_code = result.returncode
            failure = f"{command.label} exited with code {result.returncode}"
            break
    state = "COMPLETED" if exit_code == 0 else "FAILED"
    try:
        _record_generation_status(
            settings,
            workflow,
            params,
            commands,
            state,
            failure,
        )
    except Exception as exc:
        record["status_record_error"] = f"{type(exc).__name__}: {exc}"
        if exit_code == 0:
            exit_code = 1
            state = "FAILED"
            failure = record["status_record_error"]
    record.update({"state": state, "finished_utc": _utc(), "failure": failure})
    _write(record_path, record)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list supported workflows")
    for command in ("show", "command", "submit", "run"):
        item = subparsers.add_parser(command)
        item.add_argument("workflow", choices=sorted(WORKFLOWS))
        item.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
        if command == "submit":
            item.add_argument("--dry-run", action="store_true")
            item.add_argument(
                "--dependency",
                help="Explicit Slurm dependency, for example afterok:12345.",
            )
        if command == "run":
            item.add_argument("--allow-local", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "list":
        for name, workflow in sorted(WORKFLOWS.items()):
            print(f"{name}\t{workflow.description}")
        return 0
    try:
        supplied = _parameters(args.param)
        settings, workflow, params, commands = _resolved(args.workflow, supplied)
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "show":
        print(_json(_workflow_json(settings, workflow, params, commands)), end="")
        return 0
    if args.command == "command":
        for command in commands:
            print(shlex.join(command.argv))
        return 0
    if args.command == "run":
        return _run(args.workflow, supplied, allow_local=args.allow_local)
    try:
        submit = _sbatch_command(
            settings,
            workflow,
            supplied,
            dependency=args.dependency,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(shlex.join(submit))
    if args.dry_run:
        return 0
    (settings.log_root / "slurm").mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        submit,
        cwd=settings.repository,
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"submitted_job_id={result.stdout.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
