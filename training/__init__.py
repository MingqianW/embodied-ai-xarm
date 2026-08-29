"""Repository-owned xArm training configuration and data orchestration."""

from training.configs.experiments import EXPERIMENTS, ExperimentConfig, get_experiment

__all__ = ["EXPERIMENTS", "ExperimentConfig", "get_experiment"]
