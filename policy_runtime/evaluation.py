"""Compatibility imports for legacy evaluation records.

New evaluation code must import :mod:`evaluation.common.legacy_policy_results`.
The policy runtime package otherwise owns policy I/O and safety, not evaluation
outcomes or reporting.
"""

from evaluation.common.legacy_policy_results import EpisodeEvaluation
from evaluation.common.legacy_policy_results import LABELS
from evaluation.common.legacy_policy_results import summarize_episode_rows
from evaluation.common.legacy_policy_results import validate_label
from evaluation.common.legacy_policy_results import write_evaluation_outputs

__all__ = [
    "EpisodeEvaluation",
    "LABELS",
    "summarize_episode_rows",
    "validate_label",
    "write_evaluation_outputs",
]

