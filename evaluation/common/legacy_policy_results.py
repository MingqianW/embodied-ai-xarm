from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from policy_runtime.episode_logging import write_json


LABELS = ("success", "failure", "invalid")
# Frozen persisted format for the pre-formal evaluation documents. This is
# intentionally evaluation-owned and must not track policy transport versions.
LEGACY_EVALUATION_SCHEMA_VERSION = "1.0"


@dataclass
class EpisodeEvaluation:
    simulator: str
    task: str
    prompt: str
    seed: int | None
    checkpoint: str | None
    policy_server: str
    start_time: str
    duration_s: float
    success: bool | None
    score: float | None = None
    failure_reason: str | None = None
    notes: str = ""
    video_path: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    label: str | None = None
    schema_version: str = LEGACY_EVALUATION_SCHEMA_VERSION

    def validate(self) -> None:
        if self.label is not None and self.label not in LABELS:
            raise ValueError(f"Label must be one of {LABELS}, got {self.label!r}")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("Evaluation score must be between 0 and 1")

    def to_json(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def validate_label(label: str) -> str:
    # Windows PowerShell may prefix the first piped line with a UTF BOM.
    value = label.lstrip("\ufeff").strip().lower()
    if value not in LABELS:
        raise ValueError(f"Label must be one of {LABELS}, got {label!r}")
    return value


def summarize_episode_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = len(rows)
    labeled = [row for row in rows if row.get("label") in LABELS]
    successes = sum(row.get("label") == "success" for row in labeled)
    failures = sum(row.get("label") == "failure" for row in labeled)
    invalid = sum(row.get("label") == "invalid" for row in labeled)
    denominator = successes + failures
    scores = [
        float(row["score"])
        for row in labeled
        if row.get("score") not in (None, "")
    ]
    label_counts = {label: 0 for label in LABELS}
    termination_counts: dict[str, int] = {}
    for row in labeled:
        label_counts[str(row["label"])] += 1
        reason = str(row.get("termination_reason") or "unknown")
        termination_counts[reason] = termination_counts.get(reason, 0) + 1

    def mean_float(key: str) -> float | None:
        values = [float(row[key]) for row in labeled if row.get(key) not in (None, "")]
        return None if not values else float(np.mean(values))

    return {
        "schema_version": LEGACY_EVALUATION_SCHEMA_VERSION,
        "attempted_episodes": attempted,
        "labeled_episodes": len(labeled),
        "successes": successes,
        "failures": failures,
        "invalid_episodes": invalid,
        "human_rated_task_success_rate": None if denominator == 0 else successes / denominator,
        "end_to_end_success_rate": None if attempted == 0 else successes / attempted,
        "mean_score": None if not scores else float(np.mean(scores)),
        "label_counts": label_counts,
        "termination_reason_counts": termination_counts,
        "mean_policy_steps": mean_float("policy_steps"),
        "mean_simulation_time": mean_float("sim_time"),
        "mean_wall_time": mean_float("wall_time"),
    }


def write_evaluation_outputs(run_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_episode_rows(rows)
    write_json(run_dir / "summary.json", summary)
    fieldnames = sorted({key for row in rows for key in row})
    if fieldnames:
        with (run_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return summary
