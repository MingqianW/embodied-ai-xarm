"""Backend-neutral evaluation identities, results, and review records."""

from evaluation.common.contracts import EpisodeIdentity
from evaluation.common.contracts import EpisodeOutcome
from evaluation.common.contracts import EvaluationBackend
from evaluation.common.contracts import EvaluationResult
from evaluation.common.contracts import EvaluationRunIdentity
from evaluation.common.contracts import EvaluationTask
from evaluation.common.models import ModelSpec

__all__ = [
    "EpisodeIdentity",
    "EpisodeOutcome",
    "EvaluationBackend",
    "EvaluationResult",
    "EvaluationRunIdentity",
    "EvaluationTask",
    "ModelSpec",
]
