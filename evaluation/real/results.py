"""Real-episode records with explicit human-review-only task outcomes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping

from evaluation.common.contracts import EpisodeIdentity
from evaluation.common.contracts import EpisodeOutcome
from evaluation.common.contracts import EvaluationBackend
from evaluation.common.contracts import EvaluationResult
from evaluation.common.contracts import EvaluationRunIdentity
from evaluation.common.contracts import EvaluationTask
from evaluation.common.human_review import HumanReviewRecord


REAL_EPISODE_SCHEMA_VERSION = "xarm-real-evaluation-episode-v1"


def build_real_episode_result(
    *,
    run_id: str,
    trial_id: str,
    model: Mapping[str, Any],
    task: EvaluationTask,
    execution_metadata: Mapping[str, Any],
    safety: Mapping[str, Any],
    provenance: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    human_review: HumanReviewRecord | None = None,
) -> dict[str, Any]:
    """Create an offline result; success is never inferred from telemetry."""

    task.validate()
    model_id = str(model.get("model_id") or "")
    if not model_id:
        raise ValueError("Real evaluation result requires model.model_id")
    review = None if human_review is None else human_review.to_json()
    document = {
        "schema_version": REAL_EPISODE_SCHEMA_VERSION,
        "backend": "real",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "model": dict(model),
        "episode": {
            "trial_id": trial_id,
            "task": task.task_id,
            "prompt": task.prompt,
            "automatic_success": None,
            "outcome_source": "pending_human_review" if review is None else "human_review",
        },
        "execution_metadata": dict(execution_metadata),
        "safety": dict(safety),
        "success_measurement": {
            "automatic_detector_available": False,
            "required_source": "human_review_or_external_validated_perception",
        },
        "human_review": review,
        "provenance": {**dict(provenance), "backend": "real"},
        "artifacts": dict(artifacts),
    }
    as_common_result(document)
    return document


def as_common_result(document: Mapping[str, Any]) -> EvaluationResult:
    if document.get("schema_version") != REAL_EPISODE_SCHEMA_VERSION:
        raise ValueError("Unsupported real evaluation result schema")
    if document.get("backend") != "real":
        raise ValueError("Real evaluation result backend must be 'real'")
    if document.get("success_measurement", {}).get("automatic_detector_available") is not False:
        raise ValueError("Repository real evaluation has no automatic success detector")
    episode = document["episode"]
    if episode.get("automatic_success") is not None:
        raise ValueError("Real task success cannot be populated automatically")
    task = EvaluationTask(str(episode["task"]), str(episode["prompt"]))
    review = document.get("human_review")
    if review is None:
        outcome = EpisodeOutcome.UNREVIEWED
        failure_category = None
    else:
        review_record = HumanReviewRecord(
            review_id=str(review["review_id"]),
            human_label=str(review["human_label"]),
            human_failure_reason=str(review.get("human_failure_reason") or ""),
            notes=str(review.get("notes") or ""),
            review_timestamp=str(review.get("review_timestamp") or ""),
        )
        review_record.validate()
        label = str(review["human_label"])
        outcome = {
            "SUCCESS": EpisodeOutcome.SUCCESS,
            "FAILURE": EpisodeOutcome.FAILURE,
            "UNCERTAIN": EpisodeOutcome.UNCERTAIN,
        }[label]
        failure_category = review.get("human_failure_reason") or None
    run_id = str(document["run_id"])
    model_id = str(document["model"]["model_id"])
    result = EvaluationResult(
        run=EvaluationRunIdentity(run_id, EvaluationBackend.REAL, model_id),
        episode=EpisodeIdentity(run_id, task, str(episode["trial_id"])),
        outcome=outcome,
        failure_category=failure_category,
        execution_metadata={
            **dict(document["execution_metadata"]),
            "safety": document["safety"],
        },
        provenance=document["provenance"],
        human_review=review,
        artifacts=document["artifacts"],
    )
    result.validate()
    return result
