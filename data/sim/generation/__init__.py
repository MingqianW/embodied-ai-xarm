"""Reusable configuration-driven MuJoCo dataset generation pipeline."""

from data.common.task_identity import TASKS, canonical_prompt, resolve_task_id
from data.sim.generation.config import PipelineConfig, load_pipeline_config

__all__ = [
    "PipelineConfig",
    "TASKS",
    "canonical_prompt",
    "load_pipeline_config",
    "resolve_task_id",
]
