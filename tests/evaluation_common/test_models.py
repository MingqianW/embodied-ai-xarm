import json
from pathlib import Path

import pytest

from evaluation.common.models import load_model_spec, ModelSpec, validate_abc_comparison_specs


def _checkpoint(path: Path) -> None:
    (path / "params").mkdir(parents=True)
    (path / "params" / "manifest.ocdbt").write_bytes(b"fixture")
    asset = path / "assets" / "local" / "test"
    asset.mkdir(parents=True)
    (asset / "norm_stats.json").write_text('{"norm_stats": {}}', encoding="utf-8")


@pytest.mark.parametrize("step", [None, 15000])
def test_exported_root_and_numeric_manager_layouts(tmp_path: Path, step: int | None) -> None:
    root = tmp_path / "checkpoint"
    directory = root if step is None else root / str(step)
    _checkpoint(directory)
    spec_path = tmp_path / "model.json"
    spec_path.write_text(json.dumps({
        "model_id": "test_export",
        "training_config": "test_config",
        "checkpoint_root": str(root),
        "manager_step": step,
        "norm_asset_id": "local/test",
    }), encoding="utf-8")

    spec = load_model_spec(spec_path)
    assert spec.manager_step == step
    assert spec.manager_directory == directory
    assert spec.to_json()["manager_step"] == step
    assert spec.to_json()["resolved_manager_directory"] == str(directory)
    (directory / "params" / "manifest.ocdbt").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        load_model_spec(spec_path)


def test_abc_comparison_keeps_numeric_step_requirement(tmp_path: Path) -> None:
    specs = tuple(ModelSpec(name, "test", tmp_path / name, None, "test") for name in ("A", "B", "C"))
    with pytest.raises(ValueError, match="numeric manager steps"):
        validate_abc_comparison_specs(specs)
