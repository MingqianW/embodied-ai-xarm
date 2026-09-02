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


def test_adapter_no_longer_rejects_multi_source_config_before_openpi_import() -> None:
    config = get_experiment("pi05_xarm_real50_sim50_stratified")
    result = probe_openpi()
    if result["available"]:
        assert build_openpi_train_config(config).name == config.name
    else:
        with pytest.raises(OpenPIUnavailable) as exc_info:
            build_openpi_train_config(config)
        assert "multi-LeRobot" not in str(exc_info.value)
