"""Resolved, inspectable training experiments without importing OpenPI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any

from data.common.records import SourceBackend
from data.common.task_identity import TASKS
from training.datasets.spec import DatasetSet, DatasetSpec, EpisodeSelection
from training.mixing.strategies import MixingStrategy
from training.normalization import NormalizationMode, NormalizationSpec


ALL_TASKS = tuple(task.task_id for task in TASKS)
PI05_BASE_PARAMS = "gs://openpi-assets/checkpoints/pi05_base/params"


class CheckpointMode(str, Enum):
    BASE_WEIGHTS = "base_weights"
    WARM_START = "warm_start"
    RESUME_STATE = "resume_state"


@dataclass(frozen=True)
class OpenPIModelConfig:
    family: str = "pi0.5"
    pi05: bool = True
    action_dim: int = 32
    action_horizon: int = 10
    discrete_state_input: bool = False
    paligemma_variant: str | None = None
    action_expert_variant: str | None = None
    freeze_lora_base: bool = False


@dataclass(frozen=True)
class CheckpointSpec:
    mode: CheckpointMode
    path: str
    restore_optimizer: bool = False
    restore_ema: bool = False
    restore_step: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", CheckpointMode(self.mode))
        if not self.path.strip():
            raise ValueError("checkpoint path cannot be empty")
        if self.mode is CheckpointMode.RESUME_STATE:
            if not (self.restore_optimizer and self.restore_ema and self.restore_step):
                raise ValueError("resume_state must restore optimizer, EMA, and step")
        elif self.restore_optimizer or self.restore_ema or self.restore_step:
            raise ValueError("weight initialization cannot claim training-state restore")


@dataclass(frozen=True)
class LRScheduleSpec:
    kind: str = "cosine_decay"
    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6


@dataclass(frozen=True)
class OptimizerSpec:
    kind: str = "adamw"
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0


@dataclass(frozen=True)
class OptimizationSpec:
    batch_size: int
    num_train_steps: int
    save_interval: int
    log_interval: int = 100
    keep_period: int | None = 5_000
    seed: int = 42
    num_workers: int = 2
    ema_decay: float | None = 0.999
    wandb_enabled: bool = True
    lr: LRScheduleSpec = LRScheduleSpec()
    optimizer: OptimizerSpec = OptimizerSpec()

    def __post_init__(self) -> None:
        for name in ("batch_size", "num_train_steps", "save_interval", "log_interval"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


class LaunchSupport(str, Enum):
    VENDORED_OPENPI = "vendored_openpi"
    EXTERNAL_MULTI_LEROBOT_ADAPTER = "external_multi_lerobot_adapter"
    HISTORICAL_CONFIG_INCOMPLETE = "historical_config_incomplete"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    datasets: DatasetSet
    mixing: MixingStrategy
    normalization: NormalizationSpec
    checkpoint: CheckpointSpec
    optimization: OptimizationSpec
    model: OpenPIModelConfig = OpenPIModelConfig()
    launch_support: LaunchSupport = LaunchSupport.VENDORED_OPENPI
    historical_alias: str | None = None
    evidence: tuple[str, ...] = ()
    unverified_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("experiment name cannot be empty")
        self.mixing.validate_for_batch_size(self.optimization.batch_size)
        required = set(self.mixing.required_sources)
        if not required.issubset(self.datasets.sources):
            raise ValueError("mixing strategy references a source absent from the dataset set")

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable resolved scientific configuration."""

        return asdict(self)


REAL_LEGACY = DatasetSpec(
    "real_xarm_pi05_data",
    "local/xarm_pi05_data",
    SourceBackend.REAL,
    ALL_TASKS,
    revision="historical-local",
)
REAL_V1_20260626 = DatasetSpec(
    "real_xarm_pi05_v1_20260626",
    "local/xarm_pi05_data",
    SourceBackend.REAL,
    ("red_pepper",),
    revision="20260626-v1",
    expected_episodes=50,
)
REAL_V2_20260703 = DatasetSpec(
    "real_xarm_pi05_v2_20260703",
    "local/xarm_pi05_data",
    SourceBackend.REAL,
    ("red_pepper", "blue_block", "red_block", "smallest_block", "largest_block"),
    revision="20260703-v2",
    expected_episodes=150,
)
REAL_20260703 = DatasetSpec(
    "real_xarm_pi05_20260703",
    "local/xarm_pi05_20260703",
    SourceBackend.REAL,
    ALL_TASKS,
    revision="20260703",
    expected_episodes=198,
)
SIM_STABLE_V3 = DatasetSpec(
    "sim_mujoco_stable_v3_1x",
    "local/xarm_mujoco_clean_multitask_stable_v3",
    SourceBackend.SIM,
    ALL_TASKS,
    revision="stable_v3",
    expected_episodes=200,
)
SIM_STABLE_V4_10X = DatasetSpec(
    "sim_mujoco_stable_v4_10x_real",
    "local/xarm_mujoco_clean_multitask_stable_v4_10x_real",
    SourceBackend.SIM,
    ALL_TASKS,
    revision="stable_v4_10x_real",
    expected_episodes=1980,
)
SIM_RED_BLOCK_EP198 = DatasetSpec(
    "sim_red_block_first_198",
    "local/xarm_mujoco_red_block_v1_ep198",
    SourceBackend.SIM,
    ("red_block",),
    revision="historical-continuation",
    selection=EpisodeSelection("first_by_episode_index", 198, "first 198 successful episode indices"),
    expected_episodes=198,
)


BASE_MODEL = OpenPIModelConfig()
BASE_WEIGHTS = CheckpointSpec(CheckpointMode.BASE_WEIGHTS, PI05_BASE_PARAMS)
DEFAULT_LR = LRScheduleSpec()
DEFAULT_OPTIMIZER = OptimizerSpec()


def _base(
    name: str,
    description: str,
    datasets: tuple[DatasetSpec, ...],
    mixing: MixingStrategy,
    normalization: NormalizationSpec,
    *,
    steps: int,
    save_interval: int,
    ema_decay: float | None = 0.999,
    wandb: bool = True,
    model: OpenPIModelConfig = BASE_MODEL,
    lr: LRScheduleSpec = DEFAULT_LR,
    launch_support: LaunchSupport = LaunchSupport.VENDORED_OPENPI,
    historical_alias: str | None = None,
    evidence: tuple[str, ...] = (),
    unverified_fields: tuple[str, ...] = (),
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        description=description,
        datasets=DatasetSet(datasets),
        mixing=mixing,
        normalization=normalization,
        checkpoint=BASE_WEIGHTS,
        optimization=OptimizationSpec(
            batch_size=16,
            num_train_steps=steps,
            save_interval=save_interval,
            ema_decay=ema_decay,
            wandb_enabled=wandb,
            lr=lr,
        ),
        model=model,
        launch_support=launch_support,
        historical_alias=historical_alias,
        evidence=evidence,
        unverified_fields=unverified_fields,
    )


_full = _base(
    "pi05_xarm_full_finetune",
    "Historical full-parameter Pi0.5 xArm fine-tune",
    (REAL_V1_20260626,),
    MixingStrategy.single(SourceBackend.REAL),
    NormalizationSpec(NormalizationMode.COMPUTE_FROM_DATASETS, REAL_V1_20260626.repo_id),
    steps=30_000,
    save_interval=5_000,
    ema_decay=0.99,
    evidence=("docs/experiments/training/training_data_tracker_260703.md", "removed legacy config snippet in Git history", "upstream OpenPI defaults at vendored commit 15a9616"),
)
_legacy_snippet = _base(
    "pi05_xarm_legacy_snippet_20001",
    "Removed 20,001-step real-only config template; retained for structured equivalence",
    (REAL_LEGACY,),
    MixingStrategy.single(SourceBackend.REAL),
    NormalizationSpec(NormalizationMode.COMPUTE_FROM_DATASETS, REAL_LEGACY.repo_id),
    steps=20_001,
    save_interval=5_000,
    evidence=("removed legacy config snippet in Git history",),
)
_v2_warm_start = replace(
    _base(
        "pi05_xarm_v2_warm_start_20260703",
        "Historical 150-episode five-task continuation from the v1 parameter checkpoint",
        (REAL_V2_20260703,),
        MixingStrategy.single(SourceBackend.REAL),
        NormalizationSpec(NormalizationMode.PRECOMPUTED_ASSET, REAL_V2_20260703.repo_id),
        steps=20_001,
        save_interval=5_000,
        lr=LRScheduleSpec(warmup_steps=500, peak_lr=1e-5, decay_steps=20_000, decay_lr=1e-6),
        evidence=("docs/experiments/training/training_data_tracker_260703.md",),
    ),
    checkpoint=CheckpointSpec(
        CheckpointMode.WARM_START,
        "/content/drive/MyDrive/embodied_ai_xarm/openpi_checkpoints/"
        "pi05_xarm_full_finetune/pi05_xarm_full_finetune/25000/params",
    ),
)
_real = _base(
    "pi05_xarm",
    "Latest audited real-only Delta Pi0.5 xArm training config",
    (REAL_20260703,),
    MixingStrategy.single(SourceBackend.REAL),
    NormalizationSpec(NormalizationMode.PRECOMPUTED_ASSET, REAL_20260703.repo_id),
    steps=30_001,
    save_interval=10_000,
    evidence=(
        "docs/experiments/migrations/training/REAL_SIM_CONTINUATION_AUDIT.json",
        "upstream OpenPI defaults at vendored commit 15a9616",
    ),
)
_smoke_model = replace(
    BASE_MODEL,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
    freeze_lora_base=True,
)
_smoke = _base(
    "pi05_xarm_colab_smoke",
    "Historical bounded LoRA smoke run",
    (REAL_LEGACY,),
    MixingStrategy.single(SourceBackend.REAL),
    NormalizationSpec(NormalizationMode.COMPUTE_FROM_DATASETS, REAL_LEGACY.repo_id),
    steps=1_000,
    save_interval=500,
    ema_decay=None,
    wandb=False,
    model=_smoke_model,
    lr=LRScheduleSpec(warmup_steps=100, peak_lr=5e-5, decay_steps=10_000, decay_lr=5e-5),
    evidence=("removed legacy config snippet in Git history",),
)


def _historical_multi(
    name: str,
    alias: str,
    description: str,
    sim: DatasetSpec,
    mixing: MixingStrategy,
    norm_asset: str,
) -> ExperimentConfig:
    return _base(
        name,
        description,
        (REAL_20260703, sim),
        mixing,
        NormalizationSpec(NormalizationMode.COMPUTE_FROM_DATASETS, norm_asset),
        steps=15_001,
        save_interval=5_000,
        launch_support=LaunchSupport.VENDORED_OPENPI,
        historical_alias=alias,
        evidence=(
            f"configs/evaluation/sim/models/{alias}.json",
            "The project-owned deterministic multi-LeRobot bridge implements the documented mixing semantics.",
        ),
    )


_a = _historical_multi(
    "pi05_xarm_real50_sim50_stratified",
    "A",
    "Real plus stable-v3 simulation; exactly 8 real and 8 simulation samples in every batch",
    SIM_STABLE_V3,
    MixingStrategy.per_batch(8, 8),
    "xarm_pi05_real_v3sim_1x",
)
_b = _historical_multi(
    "pi05_xarm_real1_sim10_stratified",
    "B",
    "Real plus 10x stable-v4 simulation; global sample stream repeats 1 real then 10 simulation",
    SIM_STABLE_V4_10X,
    MixingStrategy.sample_ratio(1, 10),
    "xarm_pi05_real_v4sim_10x",
)
_c = _historical_multi(
    "pi05_xarm_full_real_full_sim_trajectory_shuffle",
    "C",
    "Same full real/v4 pool as B; globally shuffled trajectories with natural source composition",
    SIM_STABLE_V4_10X,
    MixingStrategy.trajectory_shuffle(),
    "xarm_pi05_real_v4sim_10x",
)
_d = _base(
    "pi05_xarm_d_simonly_v3_1x",
    "Pi0.5 base initialized stable-v3 simulation-only experiment",
    (SIM_STABLE_V3,),
    MixingStrategy.single(SourceBackend.SIM),
    NormalizationSpec(
        NormalizationMode.COMPUTE_FROM_DATASETS,
        "local/xarm_mujoco_clean_multitask_stable_v3_simnorm_v1",
    ),
    steps=15_001,
    save_interval=5_000,
    historical_alias="D",
    evidence=("configs/evaluation/sim/models/D.json",),
    launch_support=LaunchSupport.HISTORICAL_CONFIG_INCOMPLETE,
    unverified_fields=(
        "optimization fields shown as project compatibility defaults",
        "num_train_steps inferred from checkpoint manager 15000 and run suffix 15001",
    ),
)

_continuation = ExperimentConfig(
    name="pi05_xarm_real_sim_50_50_continue",
    description="Completed historical true-state continuation from real checkpoint 30000 to manager 50000",
    datasets=DatasetSet((REAL_20260703, SIM_RED_BLOCK_EP198)),
    mixing=MixingStrategy.per_batch(8, 8),
    normalization=NormalizationSpec(
        NormalizationMode.PRESERVE_CHECKPOINT,
        "local/xarm_pi05_20260703",
        assets_dir=(
            "/work/nvme/bfmk/mw89/openpi_checkpoints/pi05_xarm/"
            "xarm_pi05_20260703_run1/30000/assets"
        ),
        sha256="63a3d8456b509600fde0e9a3546fa2c01145cb0c3527d46ead649b4c35b37fc4",
    ),
    checkpoint=CheckpointSpec(
        CheckpointMode.RESUME_STATE,
        "/work/nvme/bfmk/mw89/openpi_checkpoints/pi05_xarm/xarm_pi05_20260703_run1/30000",
        restore_optimizer=True,
        restore_ema=True,
        restore_step=True,
    ),
    optimization=OptimizationSpec(
        batch_size=16,
        num_train_steps=50_001,
        save_interval=5_000,
        ema_decay=0.999,
    ),
    launch_support=LaunchSupport.EXTERNAL_MULTI_LEROBOT_ADAPTER,
    evidence=(
        "docs/experiments/migrations/training/REAL_SIM_EXECUTION_PLAN.md",
        "docs/experiments/migrations/training/CHATGPT_REAL_SIM_PIPELINE_STATUS.txt",
        "docs/experiments/migrations/training/REAL_SIM_DISTRIBUTION_REPORT.json",
    ),
)

EXPERIMENTS = {
    config.name: config
    for config in (
        _full,
        _legacy_snippet,
        _v2_warm_start,
        _real,
        _smoke,
        _a,
        _b,
        _c,
        _d,
        _continuation,
    )
}


def get_experiment(name: str) -> ExperimentConfig:
    try:
        return EXPERIMENTS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(EXPERIMENTS))
        raise KeyError(f"Unknown training experiment {name!r}; choose one of: {choices}") from exc
