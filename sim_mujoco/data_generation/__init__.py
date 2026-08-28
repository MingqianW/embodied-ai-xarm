"""Reusable configuration-driven MuJoCo dataset generation pipeline."""

from sim_mujoco.data_generation.config import PipelineConfig, load_pipeline_config
from sim_mujoco.data_generation.registry import TASKS, canonical_prompt, resolve_task_id

__all__ = [
    "PipelineConfig",
    "TASKS",
    "canonical_prompt",
    "load_pipeline_config",
    "resolve_task_id",
]
