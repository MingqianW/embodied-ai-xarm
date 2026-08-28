"""Compatibility adapter from formal simulation JSON to the common result view."""

from __future__ import annotations

from typing import Any

from evaluation.common.contracts import EpisodeIdentity
from evaluation.common.contracts import EvaluationBackend
from evaluation.common.contracts import EvaluationResult
from evaluation.common.contracts import EvaluationRunIdentity
from evaluation.common.contracts import EvaluationTask
from evaluation.common.contracts import outcome_from_flags
from evaluation.sim.outputs import validate_episode_result


def as_common_result(document: dict[str, Any]) -> EvaluationResult:
    """Normalize a v1/v2 formal result without changing its stored schema."""

    validate_episode_result(document)
    native_episode = document["episode"]
    provenance = document["provenance"]
    model_id = str(document["model"]["model_id"])
    task = EvaluationTask(
        task_id=str(native_episode["task"]),
        prompt=str(native_episode["prompt"]),
    )
    run_id = str(
        provenance.get("provenance_sha256")
        or f"legacy-sim:{document['evaluation_protocol_version']}:{model_id}"
    )
    result = EvaluationResult(
        run=EvaluationRunIdentity(
            run_id=run_id,
            backend=EvaluationBackend.SIM,
            model_id=model_id,
            protocol_version=str(document["evaluation_protocol_version"]),
        ),
        episode=EpisodeIdentity(
            run_id=run_id,
            task=task,
            trial_id=f"{task.task_id}:seed_{int(native_episode['seed'])}",
            seed=int(native_episode["seed"]),
        ),
        outcome=outcome_from_flags(
            success=bool(native_episode["success"]),
            valid=bool(native_episode["valid"]),
        ),
        failure_category=native_episode.get("failure_category"),
        execution_metadata={
            "policy_steps": int(native_episode["policy_steps"]),
            "executed_actions": int(native_episode["executed_actions"]),
            "termination_reason": native_episode["termination_reason"],
            "safety": document["safety"],
        },
        provenance=provenance,
        artifacts=document["artifacts"],
    )
    result.validate()
    return result

