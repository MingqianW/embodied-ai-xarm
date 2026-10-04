import json
from pathlib import Path

from evaluation.common.models import ModelSpec
from evaluation.common.provenance import file_hash, json_hash
from evaluation.sim.config import load_protocol
from evaluation.sim.provenance import build_provenance, server_provenance


def test_provenance_resolves_reorganized_model_owner(tmp_path: Path) -> None:
    manager = tmp_path / "checkpoint" / "15000"
    manifest = manager / "params" / "manifest.ocdbt"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"test manifest")
    norm = manager / "assets" / "test_asset" / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text('{"norm_stats": {}}', encoding="utf-8")
    model = ModelSpec("test", "test_config", manager.parent, 15000, "test_asset")
    repository = Path(__file__).resolve().parents[2]

    provenance = build_provenance(
        protocol=load_protocol(),
        model=model,
        openpi_root=tmp_path / "external_openpi",
        embodied_ai_root=repository,
    )

    paths = provenance["paths"]
    assert paths["evaluation_common_code"]["models.py"] == file_hash(
        repository / "evaluation" / "common" / "models.py"
    )
    assert paths["checkpoint_params_manifest"]["sha256"] == file_hash(manifest)
    assert paths["checkpoint_norm_stats"]["sha256"] == file_hash(norm)
    assert paths["simulation_reset_code"]["simulation/scene/runtime.py"] == file_hash(
        repository / "simulation" / "scene" / "runtime.py"
    )
    identity = dict(provenance)
    digest = identity.pop("provenance_sha256")
    assert digest == json_hash(identity)
    assert server_provenance(provenance)["provenance_sha256"] == digest


def test_persisted_provenance_roundtrip_matches_live_resume_identity(tmp_path: Path) -> None:
    manager = tmp_path / "checkpoint"
    manifest = manager / "params/manifest.ocdbt"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"fixture manifest")
    norm = manager / "assets/own_asset/norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text('{"norm_stats": {}}')
    repository = Path(__file__).resolve().parents[2]
    requested = build_provenance(
        protocol=load_protocol(),
        model=ModelSpec("fixture", "fixture_config", manager, None, "own_asset"),
        openpi_root=tmp_path / "openpi",
        embodied_ai_root=repository,
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"provenance": requested}))
    recorded = json.loads(result_path.read_text())["provenance"]
    # This is the strict full-identity comparison used by the resume CLI,
    # after the real filesystem JSON boundary, not merely digest equality.
    assert recorded == requested
    recorded["protocol"]["tasks"][0]["prompt"] = "different prompt"
    assert recorded != requested
    assert json_hash(recorded) != json_hash(requested)
