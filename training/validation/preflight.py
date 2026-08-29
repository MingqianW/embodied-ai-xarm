"""Static, local-dataset, sampler, normalization, and OpenPI preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from data.common.records import SourceBackend
from data.common.schema import (
    TRAINING_ACTION_KEY,
    TRAINING_IMAGE_KEY,
    TRAINING_STATE_KEY,
    TRAINING_WRIST_IMAGE_KEY,
    XARM_ACTION_SHAPE,
    XARM_IMAGE_SHAPE,
    XARM_STATE_SHAPE,
)
from training.configs.experiments import ExperimentConfig, LaunchSupport
from training.datasets.spec import DatasetSpec
from training.mixing.sampler import SampleRef, sample_batches
from training.mixing.strategies import MixingMode
from training.normalization import NormalizationMode
from training.openpi.adapter import OpenPIUnavailable, build_openpi_train_config, probe_openpi


@dataclass
class PreflightReport:
    experiment: str
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    openpi: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def launch_ready(self) -> bool:
        return self.passed and not self.unresolved and bool(self.openpi.get("available"))

    def as_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["passed"] = self.passed
        output["launch_ready"] = self.launch_ready
        return output


def _shape_from_feature(feature: Mapping[str, Any]) -> tuple[int, ...] | None:
    shape = feature.get("shape")
    if isinstance(shape, list) and all(isinstance(value, int) for value in shape):
        return tuple(shape)
    return None


def _validate_local_dataset(path: Path, dataset: DatasetSpec, report: PreflightReport) -> None:
    info_path = path / "meta" / "info.json"
    if not info_path.is_file():
        report.errors.append(f"Not a LeRobot dataset (missing {info_path})")
        return
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.errors.append(f"Cannot read LeRobot metadata {info_path}: {exc}")
        return
    features = info.get("features", {})
    if not isinstance(features, dict):
        report.errors.append(f"LeRobot features are missing from {info_path}")
        return
    expected = {
        TRAINING_IMAGE_KEY: XARM_IMAGE_SHAPE,
        TRAINING_WRIST_IMAGE_KEY: XARM_IMAGE_SHAPE,
        TRAINING_STATE_KEY: XARM_STATE_SHAPE,
        TRAINING_ACTION_KEY: XARM_ACTION_SHAPE,
    }
    for key, shape in expected.items():
        if key not in features:
            report.errors.append(f"Dataset {path} is missing feature {key!r}")
            continue
        observed = _shape_from_feature(features[key])
        if observed is not None and observed != shape:
            report.errors.append(f"Dataset {path} feature {key!r} shape {observed} != {shape}")
    if "task" not in features:
        report.errors.append(f"Dataset {path} is missing feature 'task'")
    fps = info.get("fps")
    if fps is not None and fps != 10:
        report.errors.append(f"Dataset {path} fps is {fps}, expected 10")
    observed_episodes = info.get("total_episodes")
    if dataset.expected_episodes is not None:
        if observed_episodes is None:
            report.warnings.append(
                f"Dataset {path} does not expose total_episodes; expected {dataset.expected_episodes}"
            )
        elif observed_episodes != dataset.expected_episodes:
            report.errors.append(
                f"Dataset {path} has {observed_episodes} episodes, expected {dataset.expected_episodes}"
            )
    tasks_path = path / "meta" / "tasks.jsonl"
    if tasks_path.is_file():
        try:
            tasks = {
                str(json.loads(line).get("task", ""))
                for line in tasks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        except (OSError, json.JSONDecodeError) as exc:
            report.errors.append(f"Cannot read task catalog {tasks_path}: {exc}")
        else:
            missing = set(dataset.prompts) - tasks
            if missing:
                report.errors.append(f"Dataset {path} is missing configured prompts: {sorted(missing)}")
            else:
                report.checks.append(f"task catalog covers {len(dataset.tasks)} configured tasks")
    else:
        report.warnings.append(f"Dataset task catalog not locally inspectable: {tasks_path}")
    report.checks.append(f"local dataset metadata: {path}")


def _sampler_check(config: ExperimentConfig, report: PreflightReport) -> None:
    if config.mixing.mode is MixingMode.GLOBAL_TRAJECTORY_SHUFFLE:
        report.checks.append("global trajectory shuffle strategy validated structurally")
        return
    samples = []
    for dataset in config.datasets.datasets:
        for index in range(3):
            samples.append(SampleRef(dataset.dataset_id, dataset.source, index, 0))
    try:
        batches = sample_batches(
            samples,
            config.mixing,
            batch_size=config.optimization.batch_size,
            num_batches=3,
        )
    except ValueError as exc:
        report.errors.append(f"Sampler construction failed: {exc}")
        return
    if any(len(batch) != config.optimization.batch_size for batch in batches):
        report.errors.append("Sampler produced an incorrectly sized batch")
    else:
        report.checks.append("synthetic sampler produced three complete batches")


def preflight(
    config: ExperimentConfig,
    *,
    dataset_paths: Mapping[str, Path] | None = None,
    openpi_root: Path | None = None,
    check_openpi: bool = True,
) -> PreflightReport:
    """Validate all locally knowable facts without model initialization/training."""

    report = PreflightReport(config.name)
    paths = dataset_paths or {}
    report.checks.append("experiment dataclasses and source/mixing constraints")
    if config.unverified_fields:
        report.unresolved.extend(
            f"historical field not independently verified: {field}"
            for field in config.unverified_fields
        )
    _sampler_check(config, report)
    for dataset in config.datasets.datasets:
        path = paths.get(dataset.dataset_id, dataset.local_path)
        if path is None:
            report.unresolved.append(
                f"dataset {dataset.dataset_id} ({dataset.repo_id}) has no local path; remote resolution not attempted"
            )
        else:
            _validate_local_dataset(Path(path), dataset, report)
    norm = config.normalization
    if norm.mode is NormalizationMode.COMPUTE_FROM_DATASETS:
        report.warnings.append(
            f"normalization asset {norm.asset_id!r} must be computed by OpenPI before training; preflight did not compute it"
        )
    elif norm.assets_dir and not norm.assets_dir.startswith(("gs://", "s3://")):
        norm_path = Path(norm.assets_dir) / norm.asset_id / "norm_stats.json"
        if not norm_path.is_file():
            report.unresolved.append(f"normalization statistics not found locally: {norm_path}")
        else:
            report.checks.append(f"normalization statistics: {norm_path}")
    else:
        report.unresolved.append(
            f"normalization asset {norm.asset_id!r} has no locally verifiable assets directory"
        )
    if not config.checkpoint.path.startswith(("gs://", "s3://")) and not Path(config.checkpoint.path).exists():
        report.unresolved.append(f"checkpoint path does not exist locally: {config.checkpoint.path}")
    else:
        report.checks.append(f"checkpoint initialization syntax: {config.checkpoint.mode.value}")
    if config.launch_support is LaunchSupport.EXTERNAL_MULTI_LEROBOT_ADAPTER:
        report.unresolved.append(
            "historical multi-LeRobot execution adapter is external and absent from this repository"
        )
    elif config.launch_support is LaunchSupport.HISTORICAL_CONFIG_INCOMPLETE:
        report.unresolved.append(
            "historical source config is incomplete; execution is disabled until exact fields are recovered"
        )
    if check_openpi:
        report.openpi = probe_openpi(openpi_root)
        if report.openpi.get("available") and config.launch_support is LaunchSupport.VENDORED_OPENPI:
            try:
                build_openpi_train_config(config, openpi_root=openpi_root)
            except (OpenPIUnavailable, TypeError, ValueError) as exc:
                report.errors.append(f"OpenPI config construction failed: {exc}")
            else:
                report.checks.append("OpenPI TrainConfig constructed without model initialization")
        elif not report.openpi.get("available"):
            report.unresolved.append(str(report.openpi.get("reason", "OpenPI unavailable")))
    return report
