"""Backend-neutral human-review decision records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


PRIMARY_LABELS = ("SUCCESS", "FAILURE", "UNCERTAIN")
FAILURE_REASONS = (
    "NO_MEANINGFUL_GRASP_OR_LIFT",
    "PARTIAL_LIFT",
    "DROPPED_AFTER_LIFT",
    "WRONG_OBJECT",
    "NOT_RELEASED",
    "OUTSIDE_TARGET",
    "UNSTABLE_PLACEMENT",
    "OTHER",
)


def validate_decision(*, label: str, failure_reason: str) -> None:
    if label not in PRIMARY_LABELS:
        raise ValueError(f"Unsupported human label: {label!r}")
    if failure_reason and failure_reason not in FAILURE_REASONS:
        raise ValueError(f"Unsupported human failure reason: {failure_reason!r}")
    if label != "FAILURE" and failure_reason:
        raise ValueError("Only FAILURE decisions may include a human failure reason")


@dataclass(frozen=True)
class HumanReviewRecord:
    review_id: str
    human_label: str
    human_failure_reason: str = ""
    notes: str = ""
    review_timestamp: str = ""

    def validate(self) -> None:
        if not self.review_id:
            raise ValueError("Human review_id must be non-empty")
        validate_decision(
            label=self.human_label,
            failure_reason=self.human_failure_reason,
        )

    def to_json(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        if not value["review_timestamp"]:
            value["review_timestamp"] = datetime.now(UTC).isoformat()
        return value

