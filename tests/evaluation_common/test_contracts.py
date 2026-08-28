from __future__ import annotations

import pytest

from evaluation.common.contracts import EvaluationTask
from evaluation.common.human_review import HumanReviewRecord


def test_evaluation_task_reuses_canonical_data_registry() -> None:
    EvaluationTask("red_block", "pick up the red block").validate()
    with pytest.raises(ValueError, match="data.common.task_identity"):
        EvaluationTask("red_block", "lift a crimson cube").validate()


def test_human_review_contract_is_backend_neutral_and_strict() -> None:
    row = HumanReviewRecord("review-1", "FAILURE", "PARTIAL_LIFT").to_json()
    assert row["human_label"] == "FAILURE"
    with pytest.raises(ValueError, match="Only FAILURE"):
        HumanReviewRecord("review-2", "SUCCESS", "OTHER").validate()

