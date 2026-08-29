"""Declarative Slurm workflows that call canonical project CLIs only."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import PurePath
from typing import Callable, Mapping

from cluster.config import ClusterSettings


@dataclass(frozen=True)
class Resources:
    time: str
    cpus: int
    memory: str
    gpus: int = 1


@dataclass(frozen=True)
class Command:
    label: str
    argv: tuple[str, ...]
    record_output: str | None = None
    append_output: bool = False


Builder = Callable[[ClusterSettings, Mapping[str, str]], tuple[Command, ...]]


@dataclass(frozen=True)
class Workflow:
    name: str
    description: str
    resources: Resources
    build: Builder
    required_parameters: tuple[str, ...] = ()
    defaults: Mapping[str, str] = field(default_factory=dict)
    phase: str | None = None
    next_action: str | None = None
    completed_work: str | None = None

    def parameters(self, supplied: Mapping[str, str]) -> dict[str, str]:
        unknown = set(supplied) - set(self.required_parameters) - set(self.defaults)
        if unknown:
            raise ValueError(
                f"Unknown parameters for {self.name}: "
                f"{', '.join(sorted(unknown))}"
            )
        result = {**self.defaults, **supplied}
        missing = [name for name in self.required_parameters if not result.get(name)]
        if missing:
            raise ValueError(f"Missing parameters for {self.name}: {', '.join(missing)}")
        return result


STANDARD = Resources("06:00:00", 8, "64G")
DIAGNOSTIC = Resources("02:00:00", 4, "24G")
EVALUATION = Resources("12:00:00", 16, "128G")
TRAINING = Resources("12:00:00", 16, "220G")
CPU_PREFLIGHT = Resources("00:30:00", 4, "24G", 0)
EXPORT = Resources("02:00:00", 8, "64G")
DATA_PLANS = frozenset({"v3", "v4-10x"})


def _python(settings: ClusterSettings, *args: object) -> tuple[str, ...]:
    return (str(settings.python), *(str(value) for value in args))


def _dataset(settings: ClusterSettings, version: str) -> dict[str, PurePath]:
    try:
        suffix = {
            "v3": "xarm_mujoco_clean_multitask_stable_v3",
            "v4-10x": "xarm_mujoco_clean_multitask_stable_v4_10x_real",
        }[version]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported simulation data plan {version!r}; "
            f"choose one of {', '.join(sorted(DATA_PLANS))}"
        ) from exc
    return {
        "config": (
            settings.repository
            / "configs"
            / "data"
            / "sim"
            / "generation"
            / (
                "clean_multitask_stable_v3.yaml"
                if version == "v3"
                else "clean_multitask_stable_v4_10x_real.yaml"
            )
        ),
        "raw": settings.work_root / "mujoco_datasets" / "raw" / suffix,
        "converted": settings.work_root / "mujoco_datasets" / "local" / suffix,
        "smoke": settings.work_root / "mujoco_datasets" / "smoke" / suffix,
        "log": settings.work_root / "logs" / suffix,
    }


def _generation(phase: str) -> Builder:
    def build(
        settings: ClusterSettings,
        params: Mapping[str, str],
    ) -> tuple[Command, ...]:
        version = params["plan"]
        p = _dataset(settings, version)
        cli = ("-m", "data.sim.generation.cli")
        config = ("--config", p["config"])
        if phase == "preflight":
            return (
                Command(
                    "inspect-config",
                    _python(settings, *cli, "inspect", *config),
                ),
            )
        if phase == "smoke":
            return (
                Command(
                    "environment",
                    _python(
                        settings,
                        "-m",
                        "diagnostics.simulation.environment.check",
                        "--require-egl",
                        "--json-output",
                        p["log"] / "DELTAI_MUJOCO_ENVIRONMENT.json",
                    ),
                ),
                Command(
                    "generate-smoke",
                    _python(
                        settings,
                        *cli,
                        "generate",
                        *config,
                        "--output",
                        p["smoke"],
                        "--smoke",
                        "--overwrite",
                    ),
                ),
                Command(
                    "audit-smoke",
                    _python(
                        settings,
                        *cli,
                        "audit",
                        *config,
                        "--raw",
                        p["smoke"],
                        "--report-dir",
                        p["log"],
                        "--decode-all-images",
                        "--smoke",
                    ),
                ),
                Command(
                    "permissions",
                    _python(
                        settings,
                        *cli,
                        "permissions",
                        *config,
                        p["smoke"],
                        p["log"],
                    ),
                ),
            )
        if phase == "generate":
            return (
                Command(
                    "generate",
                    _python(
                        settings,
                        *cli,
                        "generate",
                        *config,
                        "--output",
                        p["raw"],
                        "--overwrite",
                    ),
                ),
                Command(
                    "audit-raw",
                    _python(
                        settings,
                        *cli,
                        "audit",
                        *config,
                        "--raw",
                        p["raw"],
                        "--report-dir",
                        p["log"],
                        "--decode-all-images",
                    ),
                ),
                Command(
                    "permissions",
                    _python(
                        settings,
                        *cli,
                        "permissions",
                        *config,
                        p["raw"],
                        p["log"],
                    ),
                ),
            )
        if phase == "convert":
            return (
                Command(
                    "convert",
                    _python(
                        settings,
                        *cli,
                        "convert",
                        *config,
                        "--raw",
                        p["raw"],
                        "--output",
                        p["converted"],
                        "--overwrite",
                    ),
                ),
                Command(
                    "permissions",
                    _python(
                        settings,
                        *cli,
                        "permissions",
                        *config,
                        p["converted"],
                        p["log"],
                    ),
                ),
            )
        permissions_report = str(p["log"] / "PERMISSIONS_REPORT.txt")
        return (
            Command(
                "audit-final",
                _python(
                    settings,
                    *cli,
                    "audit",
                    *config,
                    "--raw",
                    p["raw"],
                    "--converted",
                    p["converted"],
                    "--report-dir",
                    p["log"],
                    "--decode-all-images",
                ),
            ),
            Command(
                "permissions",
                _python(
                    settings,
                    *cli,
                    "permissions",
                    *config,
                    p["raw"],
                    p["converted"],
                    p["smoke"],
                    p["log"],
                ),
            ),
            Command(
                "raw-path-permissions",
                ("namei", "-l", str(p["raw"])),
                permissions_report,
            ),
            Command(
                "converted-path-permissions",
                ("namei", "-l", str(p["converted"])),
                permissions_report,
                True,
            ),
            Command("raw-acl", ("getfacl", str(p["raw"])), permissions_report, True),
            Command("converted-acl", ("getfacl", str(p["converted"])), permissions_report, True),
            Command("handoff", _python(settings, *cli, "handoff", *config)),
        )

    return build


def _initialize(
    settings: ClusterSettings,
    params: Mapping[str, str],
) -> tuple[Command, ...]:
    version = params["plan"]
    p = _dataset(settings, version)
    return (
        Command(
            "initialize-log-root",
            _python(
                settings,
                "-m",
                "data.sim.generation.cli",
                "inspect",
                "--config",
                p["config"],
                "--initialize-log-root",
                "--overwrite",
            ),
        ),
    )


def _environment(settings: ClusterSettings, _: Mapping[str, str]) -> tuple[Command, ...]:
    job_id = os.environ.get("SLURM_JOB_ID", "%j")
    output = settings.log_root / "diagnostics" / f"environment-{job_id}.json"
    return (
        Command(
            "environment",
            _python(
                settings,
                "-m",
                "diagnostics.simulation.environment.check",
                "--require-egl",
                "--json-output",
                output,
            ),
        ),
    )


def _physics(settings: ClusterSettings, params: Mapping[str, str]) -> tuple[Command, ...]:
    return (
        Command(
            "physics-consistency",
            _python(
                settings,
                "-m",
                "diagnostics.simulation.physics.consistency",
                "--output",
                params["output"],
            ),
        ),
    )


def _videos(settings: ClusterSettings, _: Mapping[str, str]) -> tuple[Command, ...]:
    local = settings.work_root / "mujoco_datasets" / "local"
    output = settings.work_root / "exports" / "mujoco_training_videos" / "per_task_2episodes_v1"
    return (
        Command(
            "export-videos",
            _python(
                settings,
                "-m",
                "tools.datasets.export_lerobot_training_videos",
                "--dataset",
                f"stable_v3={local / 'xarm_mujoco_clean_multitask_stable_v3'}",
                "--dataset",
                "stable_v4_10x="
                f"{local / 'xarm_mujoco_clean_multitask_stable_v4_10x_real'}",
                "--output",
                output,
                "--per-task",
                "2",
                "--fps",
                "10",
            ),
        ),
    )


def _evaluation(settings: ClusterSettings, params: Mapping[str, str]) -> tuple[Command, ...]:
    args: list[object] = [
        "-m",
        "evaluation.sim.cli",
        "--model-spec",
        params["model_spec"],
        "--protocol",
        params["protocol"],
        "--openpi-root",
        settings.openpi_root,
        "--embodied-ai-root",
        settings.repository,
        "--host",
        params["host"],
        "--port",
        params["port"],
        "--timeout",
        params["timeout"],
    ]
    if params["output_root"]:
        args.extend(("--output-root", params["output_root"]))
    if params["resume"].lower() in {"1", "true", "yes"}:
        args.append("--resume")
    return (Command("formal-simulation-evaluation", _python(settings, *args)),)


def _training_preflight(
    settings: ClusterSettings,
    params: Mapping[str, str],
) -> tuple[Command, ...]:
    args: list[object] = [
        "-m",
        "training.cli",
        "preflight",
        params["config"],
        "--openpi-root",
        settings.openpi_root,
    ]
    for value in params["dataset_paths"].split(";"):
        if value.strip():
            args.extend(("--dataset-path", value.strip()))
    return (Command("training-preflight", _python(settings, *args)),)


def _training(settings: ClusterSettings, params: Mapping[str, str]) -> tuple[Command, ...]:
    assets_dir = params["assets_dir"] or str(settings.work_root / "openpi_assets")
    checkpoint_dir = params["checkpoint_dir"] or str(
        settings.work_root / "openpi_checkpoints"
    )
    return (
        Command(
            "training",
            _python(
                settings,
                "-m",
                "training.cli",
                "train",
                params["config"],
                "--exp-name",
                params["exp_name"],
                "--openpi-root",
                settings.openpi_root,
                "--assets-base-dir",
                assets_dir,
                "--checkpoint-base-dir",
                checkpoint_dir,
                "--execute",
            ),
        ),
    )


def _workflow(
    name: str,
    phase: str,
    resources: Resources,
    next_action: str,
    completed: str,
) -> Workflow:
    return Workflow(
        name,
        f"simulation data {phase}",
        resources,
        _generation(phase),
        defaults={"plan": "v3"},
        phase=phase,
        next_action=next_action,
        completed_work=completed,
    )


WORKFLOWS = {
    workflow.name: workflow
    for workflow in (
        Workflow(
            "sim-data-preflight",
            "Resolve and inspect a simulation data plan without writing outputs",
            CPU_PREFLIGHT,
            _generation("preflight"),
            defaults={"plan": "v3"},
        ),
        Workflow(
            "sim-data-initialize",
            "Initialize the selected plan's exact log root after explicit submission",
            Resources("00:30:00", 2, "8G", 0),
            _initialize,
            defaults={"plan": "v3"},
        ),
        _workflow(
            "sim-data-smoke",
            "smoke",
            Resources("02:00:00", 8, "64G"),
            "Review SMOKE_AUDIT.md and contact sheets before full generation.",
            "Six-task smoke generation and automated audit passed.",
        ),
        _workflow(
            "sim-data-generate",
            "generate",
            Resources("12:00:00", 8, "64G"),
            "Confirm collection_summary.json complete=true, then run conversion.",
            "The selected plan's accepted episodes and raw audit completed.",
        ),
        _workflow(
            "sim-data-convert",
            "convert",
            STANDARD,
            "Confirm conversion metadata matches the plan, then run final audit.",
            "Canonical conversion completed for the selected data plan.",
        ),
        _workflow(
            "sim-data-audit",
            "audit",
            STANDARD,
            "Review DATASET_AUDIT.md and FINAL_HANDOFF.md.",
            "Raw and converted audits passed and the final handoff was written.",
        ),
        Workflow(
            "environment-check",
            "Validate Python, MuJoCo, EGL, assets, and writable roots",
            DIAGNOSTIC,
            _environment,
        ),
        Workflow(
            "physics-consistency",
            "Compare generation and evaluation MuJoCo physics",
            CPU_PREFLIGHT,
            _physics,
            required_parameters=("output",),
        ),
        Workflow(
            "export-training-videos",
            "Export the approved two-per-task training videos",
            EXPORT,
            _videos,
        ),
        Workflow(
            "formal-sim-evaluation",
            "Run formal simulation evaluation against a verified policy server",
            EVALUATION,
            _evaluation,
            required_parameters=("model_spec", "host"),
            defaults={
                "protocol": (
                    "configs/evaluation/sim/protocols/"
                    "formal_xarm_pi05_eval_v2.json"
                ),
                "output_root": "",
                "port": "8000",
                "timeout": "120",
                "resume": "false",
            },
        ),
        Workflow(
            "training-preflight",
            "Run canonical training preflight without launching training",
            CPU_PREFLIGHT,
            _training_preflight,
            required_parameters=("config", "dataset_paths"),
        ),
        Workflow(
            "training",
            "Explicitly delegate one supported experiment to OpenPI",
            TRAINING,
            _training,
            required_parameters=("config", "exp_name"),
            defaults={
                "assets_dir": "",
                "checkpoint_dir": "",
            },
        ),
    )
}


def get_workflow(name: str) -> Workflow:
    try:
        return WORKFLOWS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown cluster workflow: {name}") from exc
