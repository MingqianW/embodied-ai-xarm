"""Translate project configs to OpenPI objects without owning optimization."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any

from training.configs.experiments import (
    CheckpointMode,
    ExperimentConfig,
    LaunchSupport,
)
from training.normalization import NormalizationMode
from training.openpi.data_config import make_xarm_data_config_class


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPENPI_ROOT = PROJECT_ROOT / "third_party" / "openpi"


class OpenPIUnavailable(RuntimeError):
    pass


def _imports(openpi_root: Path | None = None) -> dict[str, Any]:
    root = (openpi_root or DEFAULT_OPENPI_ROOT).resolve()
    source = root / "src"
    if not source.is_dir():
        raise OpenPIUnavailable(f"OpenPI source directory does not exist: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        return {
            "config": importlib.import_module("openpi.training.config"),
            "optimizer": importlib.import_module("openpi.training.optimizer"),
            "weights": importlib.import_module("openpi.training.weight_loaders"),
            "pi0": importlib.import_module("openpi.models.pi0_config"),
            "model": importlib.import_module("openpi.models.model"),
            "libero": importlib.import_module("openpi.policies.libero_policy"),
            "transforms": importlib.import_module("openpi.transforms"),
        }
    except ModuleNotFoundError as exc:
        raise OpenPIUnavailable(f"OpenPI runtime dependency unavailable: {exc.name}") from exc


def probe_openpi(openpi_root: Path | None = None) -> dict[str, Any]:
    root = (openpi_root or DEFAULT_OPENPI_ROOT).resolve()
    try:
        modules = _imports(root)
    except OpenPIUnavailable as exc:
        return {"available": False, "root": str(root), "reason": str(exc)}
    return {
        "available": True,
        "root": str(root),
        "train_config": modules["config"].TrainConfig.__name__,
    }


def build_openpi_train_config(
    experiment: ExperimentConfig,
    *,
    exp_name: str = "local",
    assets_base_dir: str = "./assets",
    checkpoint_base_dir: str = "./checkpoints",
    openpi_root: Path | None = None,
) -> Any:
    """Construct an OpenPI config; project loading may supply multiple datasets.

    The first repo id remains required by upstream ``DataConfig``. The caller
    installs the project bridge, which resolves and selects every declared
    dataset without flattening the configured mixing policy.
    """

    if experiment.launch_support is LaunchSupport.HISTORICAL_CONFIG_INCOMPLETE:
        raise OpenPIUnavailable(
            f"{experiment.name} cannot launch because its original optimization config is not tracked"
        )
    modules = _imports(openpi_root)
    config = modules["config"]
    optimizer = modules["optimizer"]
    weights = modules["weights"]
    pi0 = modules["pi0"]
    dataset = experiment.datasets.datasets[0]
    normalization = experiment.normalization
    XArmDataConfig = make_xarm_data_config_class(
        config, modules["model"], modules["libero"], modules["transforms"]
    )
    assets = config.AssetsConfig(
        assets_dir=normalization.assets_dir,
        asset_id=normalization.asset_id,
    )
    model_config = pi0.Pi0Config(
        pi05=experiment.model.pi05,
        action_dim=experiment.model.action_dim,
        action_horizon=experiment.model.action_horizon,
        discrete_state_input=experiment.model.discrete_state_input,
        **(
            {"paligemma_variant": experiment.model.paligemma_variant}
            if experiment.model.paligemma_variant
            else {}
        ),
        **(
            {"action_expert_variant": experiment.model.action_expert_variant}
            if experiment.model.action_expert_variant
            else {}
        ),
    )
    lr = experiment.optimization.lr
    opt = experiment.optimization.optimizer
    kwargs: dict[str, Any] = {
        "name": experiment.name,
        "exp_name": exp_name,
        "model": model_config,
        "data": XArmDataConfig(
            repo_id=dataset.repo_id,
            assets=assets,
            base_config=config.DataConfig(prompt_from_task=True),
        ),
        "lr_schedule": optimizer.CosineDecaySchedule(
            warmup_steps=lr.warmup_steps,
            peak_lr=lr.peak_lr,
            decay_steps=lr.decay_steps,
            decay_lr=lr.decay_lr,
        ),
        "optimizer": optimizer.AdamW(
            b1=opt.b1,
            b2=opt.b2,
            eps=opt.eps,
            weight_decay=opt.weight_decay,
            clip_gradient_norm=opt.clip_gradient_norm,
        ),
        "ema_decay": experiment.optimization.ema_decay,
        "assets_base_dir": assets_base_dir,
        "checkpoint_base_dir": checkpoint_base_dir,
        "seed": experiment.optimization.seed,
        "batch_size": experiment.optimization.batch_size,
        "num_workers": experiment.optimization.num_workers,
        "num_train_steps": experiment.optimization.num_train_steps,
        "log_interval": experiment.optimization.log_interval,
        "save_interval": experiment.optimization.save_interval,
        "keep_period": experiment.optimization.keep_period,
        "wandb_enabled": experiment.optimization.wandb_enabled,
    }
    if experiment.checkpoint.mode in {CheckpointMode.BASE_WEIGHTS, CheckpointMode.WARM_START}:
        kwargs["weight_loader"] = weights.CheckpointWeightLoader(experiment.checkpoint.path)
    else:
        kwargs["resume"] = True
    if experiment.model.freeze_lora_base:
        kwargs["freeze_filter"] = model_config.get_freeze_filter()
    return config.TrainConfig(**kwargs)
