from __future__ import annotations

from data.common.records import SourceBackend
import pytest

from training.configs.experiments import CheckpointMode, CheckpointSpec, LaunchSupport, get_experiment
from training.mixing.strategies import MixingMode
from training.normalization import NormalizationMode


def test_legacy_snippet_resolves_old_explicit_and_upstream_default_semantics() -> None:
    config = get_experiment("pi05_xarm_legacy_snippet_20001")
    assert config.model.pi05 is True
    assert (config.model.action_dim, config.model.action_horizon) == (32, 10)
    assert config.model.discrete_state_input is False
    assert config.datasets.datasets[0].repo_id == "local/xarm_pi05_data"
    assert config.mixing == config.mixing.single(SourceBackend.REAL)
    assert config.optimization.batch_size == 16
    assert config.optimization.num_train_steps == 20_001
    assert config.optimization.save_interval == 5_000
    assert config.optimization.ema_decay == 0.999
    assert config.optimization.lr.warmup_steps == 1_000
    assert config.optimization.lr.peak_lr == 2.5e-5
    assert config.optimization.lr.decay_steps == 30_000
    assert config.optimization.lr.decay_lr == 2.5e-6
    assert config.optimization.optimizer.clip_gradient_norm == 1.0
    assert config.checkpoint.mode is CheckpointMode.BASE_WEIGHTS


def test_latest_audited_pi05_name_is_not_conflated_with_older_runs() -> None:
    config = get_experiment("pi05_xarm")
    dataset = config.datasets.datasets[0]
    assert dataset.repo_id == "local/xarm_pi05_20260703"
    assert dataset.expected_episodes == 198
    assert config.optimization.num_train_steps == 30_001
    assert config.optimization.save_interval == 10_000
    assert config.optimization.ema_decay == 0.999
    assert config.optimization.optimizer.b2 == 0.95
    assert config.optimization.optimizer.weight_decay == 1e-10
    assert config.optimization.lr == config.optimization.lr.__class__()
    assert config.checkpoint.mode is CheckpointMode.BASE_WEIGHTS
    assert config.normalization.mode is NormalizationMode.PRECOMPUTED_ASSET


def test_v2_tracker_warm_start_retains_custom_lr_and_checkpoint() -> None:
    config = get_experiment("pi05_xarm_v2_warm_start_20260703")
    assert config.checkpoint.mode is CheckpointMode.WARM_START
    assert config.checkpoint.path.endswith("/25000/params")
    assert config.datasets.datasets[0].expected_episodes == 150
    assert config.optimization.lr.warmup_steps == 500
    assert config.optimization.lr.peak_lr == 1e-5
    assert config.optimization.lr.decay_steps == 20_000
    assert config.optimization.lr.decay_lr == 1e-6


def test_full_and_colab_smoke_retain_their_meaning() -> None:
    full = get_experiment("pi05_xarm_full_finetune")
    smoke = get_experiment("pi05_xarm_colab_smoke")
    assert full.optimization.ema_decay == 0.99
    assert full.optimization.num_train_steps == 30_000
    assert full.datasets.datasets[0].expected_episodes == 50
    assert full.datasets.datasets[0].tasks == ("red_pepper",)
    assert smoke.model.freeze_lora_base
    assert smoke.model.paligemma_variant == "gemma_2b_lora"
    assert smoke.optimization.ema_decay is None
    assert smoke.optimization.lr.warmup_steps == 100
    assert smoke.optimization.lr.peak_lr == 5e-5
    assert not smoke.optimization.wandb_enabled


def test_historical_abc_make_the_scientific_difference_explicit() -> None:
    a = get_experiment("pi05_xarm_real50_sim50_stratified")
    b = get_experiment("pi05_xarm_real1_sim10_stratified")
    c = get_experiment("pi05_xarm_full_real_full_sim_trajectory_shuffle")
    assert a.historical_alias == "A" and a.mixing.mode is MixingMode.FIXED_PER_BATCH
    assert (a.mixing.real_per_batch, a.mixing.sim_per_batch) == (8, 8)
    assert b.historical_alias == "B" and b.mixing.schedule == (SourceBackend.REAL,) + (SourceBackend.SIM,) * 10
    assert c.historical_alias == "C" and c.mixing.mode is MixingMode.GLOBAL_TRAJECTORY_SHUFFLE
    assert b.datasets == c.datasets
    assert b.normalization == c.normalization
    assert a.launch_support is LaunchSupport.VENDORED_OPENPI
    assert b.normalization.mode is NormalizationMode.COMPUTE_FROM_DATASETS


def test_sim_only_d_has_fresh_sim_normalization_identity() -> None:
    config = get_experiment("pi05_xarm_d_simonly_v3_1x")
    assert config.historical_alias == "D"
    assert config.datasets.sources == {SourceBackend.SIM}
    assert config.normalization.mode is NormalizationMode.COMPUTE_FROM_DATASETS
    assert config.normalization.asset_id == "local/xarm_mujoco_clean_multitask_stable_v3_simnorm_v1"


def test_completed_continuation_is_true_resume_not_warm_start() -> None:
    config = get_experiment("pi05_xarm_real_sim_50_50_continue")
    assert config.checkpoint.mode is CheckpointMode.RESUME_STATE
    assert config.checkpoint.restore_optimizer
    assert config.checkpoint.restore_ema
    assert config.checkpoint.restore_step
    assert config.optimization.num_train_steps == 50_001
    assert config.optimization.save_interval == 5_000
    assert config.normalization.mode is NormalizationMode.PRESERVE_CHECKPOINT


def test_resume_mode_cannot_omit_optimizer_ema_or_step() -> None:
    with pytest.raises(ValueError, match="optimizer, EMA, and step"):
        CheckpointSpec(CheckpointMode.RESUME_STATE, "/checkpoint", restore_optimizer=True)
