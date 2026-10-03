import json
from pathlib import Path

import pytest

from evaluation.sim.config import load_protocol


@pytest.mark.parametrize("phase,base", [
    ("smoke", "formal_xarm_pi05_eval_smoke_v3.json"),
    ("formal", "formal_xarm_pi05_eval_v3.json"),
])
def test_deployment_protocol_changes_only_output(phase: str, base: str) -> None:
    directory = Path(__file__).resolve().parents[2] / "configs/evaluation/sim/protocols"
    original = load_protocol(directory / base).to_json()
    target = load_protocol(directory / f"hf_real_20260703_{phase}_v3.json").to_json()
    assert target.pop("output_root") != original.pop("output_root")
    assert target == original


def test_inherited_protocol_rejects_scientific_override(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"extends": "base.json", "outputs": {"formal_output_root": "/tmp/test"}, "seeds": {"count": 1}}))
    with pytest.raises(ValueError, match="only override"):
        load_protocol(path)


def test_inherited_protocol_rejects_escape(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"extends": "../base.json", "outputs": {"formal_output_root": "/tmp/test"}}))
    with pytest.raises(ValueError, match="same directory"):
        load_protocol(path)
