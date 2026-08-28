"""Content-addressed provenance for formal evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.common.models import ModelSpec
from evaluation.common.provenance import canonical_json
from evaluation.common.provenance import file_hash
from evaluation.common.provenance import git_commit
from evaluation.common.provenance import json_hash
from evaluation.sim.config import FormalProtocol


def protocol_identity(protocol: FormalProtocol) -> dict[str, Any]:
    value = protocol.to_json()
    value["placement_max_center_distance_m"] = protocol.placement_max_center_distance_m
    return value


def build_provenance(
    *,
    protocol: FormalProtocol,
    model: ModelSpec,
    openpi_root: Path,
    embodied_ai_root: Path,
) -> dict[str, Any]:
    protocol_value = protocol_identity(protocol)
    model_value = model.to_json()
    evaluation_root = Path(__file__).resolve().parent
    common_root = evaluation_root.parent / "common"
    path_hashes = {
        "task_scene": {"path": str(protocol.task_scene_config_path), "sha256": file_hash(protocol.task_scene_config_path)},
        "camera": {"path": str(protocol.camera_config_path), "sha256": file_hash(protocol.camera_config_path)},
        "robot_xml": {"path": str(protocol.robot_xml_path), "sha256": file_hash(protocol.robot_xml_path)},
        "checkpoint_params_manifest": {
            "path": str(model.manager_directory / "params" / "manifest.ocdbt"),
            "sha256": file_hash(model.manager_directory / "params" / "manifest.ocdbt"),
        },
        "checkpoint_norm_stats": {
            "path": str(model.manager_directory / "assets" / model.norm_asset_id / "norm_stats.json"),
            "sha256": file_hash(model.manager_directory / "assets" / model.norm_asset_id / "norm_stats.json"),
        },
        "evaluation_code": {
            name: file_hash(evaluation_root / name)
            for name in (
                "config.py",
                "models.py",
                "rng.py",
                "success.py",
                "slip_trace.py",
                "episode_runner.py",
                "outputs.py",
                "summary.py",
            )
        },
        "evaluation_common_code": {
            name: file_hash(common_root / name)
            for name in (
                "contracts.py",
                "human_review.py",
                "models.py",
                "provenance.py",
            )
        },
    }
    static = {
        "backend": "sim",
        "evaluation_protocol_version": protocol.protocol_version,
        "protocol": protocol_value,
        "protocol_sha256": json_hash(protocol_value),
        "model": model_value,
        "model_spec_sha256": json_hash(model_value),
        "paths": path_hashes,
        "openpi_git_commit": git_commit(openpi_root),
        "embodied_ai_xarm_git_commit": git_commit(embodied_ai_root),
    }
    static["provenance_sha256"] = json_hash(static)
    return static


def server_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """Small identity object passed to the server and checked by the client."""

    return {
        "backend": provenance.get("backend", "sim"),
        "evaluation_protocol_version": provenance["evaluation_protocol_version"],
        "protocol_sha256": provenance["protocol_sha256"],
        "model_spec_sha256": provenance["model_spec_sha256"],
        "provenance_sha256": provenance["provenance_sha256"],
        "model": provenance["model"],
    }
