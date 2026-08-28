"""Strict formal-evaluation output layout and JSON schema validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sim_mujoco.formal_evaluation.failure_diagnosis import diagnose_episode_failure
from sim_mujoco.formal_evaluation.provenance import canonical_json
from sim_mujoco.formal_evaluation.provenance import json_hash

EPISODE_SCHEMA_VERSION = "xarm-formal-episode-v2"
LEGACY_EPISODE_SCHEMA_VERSION = "xarm-formal-episode-v1"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def model_output_root(output_root: Path, model_id: str) -> Path:
    return Path(output_root).expanduser().resolve() / "models" / model_id


def episode_output_root(output_root: Path, model_id: str, task_id: str, seed: int) -> Path:
    return model_output_root(output_root, model_id) / "tasks" / task_id / f"seed_{seed}"


def initialize_output(
    *,
    output_root: Path,
    model_id: str,
    provenance: dict[str, Any],
    resume: bool,
) -> Path:
    root = model_output_root(output_root, model_id)
    identity = {
        "evaluation_protocol_version": provenance["evaluation_protocol_version"],
        "protocol_sha256": provenance["protocol_sha256"],
        "model_spec_sha256": provenance["model_spec_sha256"],
        "provenance_sha256": provenance["provenance_sha256"],
    }
    if root.exists() and any(root.iterdir()):
        if not resume:
            raise FileExistsError(f"Formal model output already exists; use explicit --resume: {root}")
        recorded = read_json(root / "model_config.json")
        if recorded.get("identity") != identity:
            raise ValueError("Refusing resume: output provenance differs from requested model/protocol")
    elif root.exists() and not resume:
        # An empty directory is safe, but the root remains model-isolated.
        pass
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "model_config.json", {"identity": identity, "provenance": provenance})

    protocol_path = Path(output_root).expanduser().resolve() / "protocol.json"
    if protocol_path.exists():
        recorded_protocol = read_json(protocol_path)
        if recorded_protocol.get("protocol_sha256") != provenance["protocol_sha256"]:
            raise ValueError("Output root already belongs to a different formal evaluation protocol")
    else:
        write_json(protocol_path, {"protocol_sha256": provenance["protocol_sha256"], "protocol": provenance["protocol"]})
    return root


def validate_episode_result(result: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "evaluation_protocol_version",
        "timestamp_utc",
        "model",
        "episode",
        "metrics",
        "safety",
        "initial_state",
        "final_state",
        "provenance",
        "artifacts",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"Episode result is missing required fields: {missing}")
    schema_version = result["schema_version"]
    if schema_version not in {EPISODE_SCHEMA_VERSION, LEGACY_EPISODE_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported episode result schema: {result['schema_version']!r}")
    episode = result["episode"]
    for key in ("task", "prompt", "seed", "success", "valid", "termination_reason", "invalid_reason", "policy_steps", "executed_actions"):
        if key not in episode:
            raise ValueError(f"Episode result missing episode.{key}")
    if not isinstance(episode["valid"], bool) or not isinstance(episode["success"], bool):
        raise ValueError("Episode valid and success must be booleans")
    if episode["success"] and not episode["valid"]:
        raise ValueError("Invalid episodes cannot be counted as successes")
    if schema_version == EPISODE_SCHEMA_VERSION:
        for key in ("failure_category", "failure_reason", "failure_stage"):
            if key not in episode:
                raise ValueError(f"Episode result missing episode.{key}")
        if "failure_diagnostics" not in result:
            raise ValueError("Episode result missing failure_diagnostics")
        diagnosis_values = (
            episode["failure_category"],
            episode["failure_reason"],
            episode["failure_stage"],
            result["failure_diagnostics"],
        )
        if episode["valid"] and not episode["success"]:
            if not all(value is not None for value in diagnosis_values):
                raise ValueError("Valid unsuccessful episodes require a complete failure diagnosis")
        elif any(value is not None for value in diagnosis_values):
            raise ValueError("Successes and invalid episodes must not carry task-failure diagnoses")
    model = result["model"]
    for key in (
        "model_id",
        "training_config",
        "checkpoint_root",
        "manager_step",
        "resolved_manager_directory",
        "norm_asset_id",
    ):
        if key not in model:
            raise ValueError(f"Episode result missing model.{key}")
    provenance = result["provenance"]
    for key in (
        "protocol_sha256",
        "model_spec_sha256",
        "openpi_git_commit",
        "embodied_ai_xarm_git_commit",
        "paths",
    ):
        if key not in provenance:
            raise ValueError(f"Episode result missing provenance.{key}")


def result_fingerprint(result: dict[str, Any]) -> str:
    """Stable identity used only for diagnostics/tests, not result provenance."""

    return json_hash(json.loads(canonical_json(result)))


def upgrade_result_with_failure_diagnosis(result: dict[str, Any]) -> dict[str, Any]:
    """Return a v2 diagnosis copy of a legacy or current episode result.

    The caller is responsible for choosing a derived output location. This
    function never writes or mutates the historical source result.
    """

    validate_episode_result(result)
    upgraded = json.loads(json.dumps(result))
    episode = upgraded["episode"]
    diagnosis = diagnose_episode_failure(
        task_id=str(episode["task"]),
        success=bool(episode["success"]),
        valid=bool(episode["valid"]),
        metrics=dict(upgraded["metrics"]),
    )
    episode["failure_category"] = diagnosis.category
    episode["failure_reason"] = diagnosis.reason
    episode["failure_stage"] = diagnosis.stage
    upgraded["failure_diagnostics"] = diagnosis.diagnostics
    upgraded["schema_version"] = EPISODE_SCHEMA_VERSION
    validate_episode_result(upgraded)
    return upgraded
