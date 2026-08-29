from __future__ import annotations

import pytest

from training.configs.experiments import get_experiment
from training.openpi.adapter import OpenPIUnavailable, build_openpi_train_config, probe_openpi


def test_openpi_probe_is_structured_even_when_runtime_dependencies_are_missing() -> None:
    result = probe_openpi()
    assert isinstance(result["available"], bool)
    assert "root" in result
    if not result["available"]:
        assert result["reason"]


def test_adapter_refuses_to_flatten_historical_multi_source_config() -> None:
    config = get_experiment("pi05_xarm_real50_sim50_stratified")
    with pytest.raises(OpenPIUnavailable, match="multi-LeRobot"):
        build_openpi_train_config(config)
