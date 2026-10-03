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
    identity = dict(provenance)
    digest = identity.pop("provenance_sha256")
    assert digest == json_hash(identity)
    assert server_provenance(provenance)["provenance_sha256"] == digest
