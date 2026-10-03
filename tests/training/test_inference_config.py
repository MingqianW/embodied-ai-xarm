from pathlib import Path

import pytest

from training.configs.experiments import get_experiment
import training.openpi.inference as inference


def test_hf_inference_preserves_original_snippet_fields_and_uses_embedded_asset() -> None:
    original = get_experiment("pi05_xarm_legacy_snippet_20001")
    resolved = inference.hf_xarm_inference_experiment()
    assert resolved.model == original.model
    assert (resolved.model.pi05, resolved.model.action_dim, resolved.model.action_horizon) == (True, 32, 10)
    assert resolved.model.discrete_state_input is False
    assert resolved.optimization == original.optimization
    assert resolved.datasets == original.datasets
    assert resolved.normalization.asset_id == "local/xarm_pi05_20260703"
    assert original.normalization.asset_id != resolved.normalization.asset_id


def test_existing_config_resolution_still_delegates_to_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    class Registry:
        def get_config(self, name: str) -> str:
            if name == "external":
                return "existing upstream config"
            raise ValueError("unknown upstream config")

    monkeypatch.setattr(inference, "_imports", lambda _: {"config": Registry()})
    assert inference.resolve_inference_config("external", openpi_root=Path("external")) == "existing upstream config"
    with pytest.raises(ValueError, match="unknown upstream"):
        inference.resolve_inference_config("absent", openpi_root=Path("external"))
