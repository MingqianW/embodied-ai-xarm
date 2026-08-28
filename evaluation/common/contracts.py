"""Backend-neutral identities and the normalized evaluation-result view.

This module intentionally describes evaluation records, not how either backend
runs an episode or measures task success.  Canonical task names and prompts
remain owned by :mod:`data.common.task_identity`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from data.common.task_identity import TASK_BY_ID


class EvaluationBackend(StrEnum):
    SIM = "sim"
    REAL = "real"


class EpisodeOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    INVALID = "invalid"
    UNREVIEWED = "unreviewed"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class EvaluationTask:
    """The task identity carried by an evaluation record."""

    task_id: str
    prompt: str

    def validate(self) -> None:
        definition = TASK_BY_ID.get(self.task_id)
        if definition is None or definition.prompt != self.prompt:
            raise ValueError(
                "Evaluation task ID/prompt must match data.common.task_identity"
            )


@dataclass(frozen=True)
class EvaluationRunIdentity:
    run_id: str
    backend: EvaluationBackend
    model_id: str
    protocol_version: str | None = None

    def validate(self) -> None:
        if not self.run_id or not self.model_id:
            raise ValueError("Evaluation run_id and model_id must be non-empty")


@dataclass(frozen=True)
class EpisodeIdentity:
    run_id: str
    task: EvaluationTask
    trial_id: str
    seed: int | None = None

    def validate(self) -> None:
        if not self.run_id or not self.trial_id:
            raise ValueError("Episode run_id and trial_id must be non-empty")
        if self.seed is not None and self.seed < 0:
            raise ValueError("Episode seed must be non-negative")
        self.task.validate()


@dataclass(frozen=True)
class EvaluationResult:
    """Normalized view shared by compatible sim and real result documents.

    Backend-native documents remain authoritative.  This view prevents the
    architecture from forcing MuJoCo metrics or real operator evidence into a
    synthetic universal runner/schema.
    """

    run: EvaluationRunIdentity
    episode: EpisodeIdentity
    outcome: EpisodeOutcome
    failure_category: str | None
    execution_metadata: Mapping[str, Any]
    provenance: Mapping[str, Any]
    human_review: Mapping[str, Any] | None = None
    artifacts: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.run.validate()
        self.episode.validate()
        if self.run.run_id != self.episode.run_id:
            raise ValueError("Run and episode identities disagree")
        provenance_backend = self.provenance.get("backend")
        if provenance_backend is not None and provenance_backend != self.run.backend:
            raise ValueError("Result and provenance backends disagree")
        if self.outcome == EpisodeOutcome.SUCCESS and self.failure_category is not None:
            raise ValueError("Successful episodes cannot carry a failure category")


def outcome_from_flags(*, success: bool, valid: bool) -> EpisodeOutcome:
    if not valid:
        return EpisodeOutcome.INVALID
    return EpisodeOutcome.SUCCESS if success else EpisodeOutcome.FAILURE
