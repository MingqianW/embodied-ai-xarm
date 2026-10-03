"""Resolve project-owned inference configuration against unmodified OpenPI."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from training.configs.experiments import get_experiment
from training.normalization import NormalizationMode, NormalizationSpec
from training.openpi.adapter import _imports, build_openpi_train_config


HF_XARM_CONFIG = "pi05_xarm_hf_20260703"
HF_XARM_ASSET = "local/xarm_pi05_20260703"


def hf_xarm_inference_experiment() -> Any:
    # The supplied original pi05_xarm inference fields match this historical
    # variant, not the later Delta configuration that reused the same name.
    original = get_experiment("pi05_xarm_legacy_snippet_20001")
    return replace(
        original,
        name=HF_XARM_CONFIG,
        description="Inference configuration from operator-supplied original xArm snippet",
        normalization=NormalizationSpec(NormalizationMode.PRECOMPUTED_ASSET, HF_XARM_ASSET),
    )


def resolve_inference_config(name: str, *, openpi_root: Path) -> Any:
    if name == HF_XARM_CONFIG:
        return build_openpi_train_config(hf_xarm_inference_experiment(), openpi_root=openpi_root)
    # Preserve normal upstream resolution for existing externally registered
    # configurations, including failure for absent config names.
    return _imports(openpi_root)["config"].get_config(name)
