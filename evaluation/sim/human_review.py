"""Blinded human-review manifests, decisions, and unblinded analysis.

Automated result artifacts are read-only inputs. Human decisions are stored in
separate review directories and are joined back only by review ID after review.
"""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
import csv
from datetime import UTC
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from evaluation.common.human_review import FAILURE_REASONS
from evaluation.common.human_review import PRIMARY_LABELS
from evaluation.common.human_review import validate_decision
from evaluation.sim.outputs import read_json
from evaluation.sim.outputs import validate_episode_result
from evaluation.sim.outputs import write_json
from evaluation.sim.representative_videos import load_representative_index

HUMAN_REVIEW_MANIFEST_VERSION = "xarm-human-review-manifest-v1"
HUMAN_REVIEW_SUMMARY_VERSION = "xarm-human-review-summary-v1"
FORMAL_MODEL_IDS = ("A", "B", "C")
DECISION_FIELDS = (
    "review_id",
    "human_label",
    "human_failure_reason",
    "notes",
    "review_timestamp",
)


def _video_path(result: dict[str, Any]) -> Path | None:
    artifacts = result.get("artifacts") or {}
    value = artifacts.get("combined_video_path")
    if value is None:
        value = (artifacts.get("video_paths") or {}).get("combined")
    return None if value is None else Path(str(value)).expanduser().resolve()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _review_rank(*, review_seed: int, model: str, task: str, seed: int) -> str:
    payload = f"xarm-human-review-v1\0{review_seed}\0{model}\0{task}\0{seed}".encode()
    return hashlib.blake2s(payload, digest_size=16).hexdigest()


def _category(result: dict[str, Any]) -> str:
    episode = result["episode"]
    if bool(episode["success"]):
        return "SUCCESS"
    if not bool(episode["valid"]):
        return "INVALID"
    return str(episode.get("failure_category") or "UNCLASSIFIED_LEGACY")


def _result_rows(evaluation_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((evaluation_root / "models").glob("*/tasks/*/seed_*/result.json")):
        result = read_json(path)
        validate_episode_result(result)
        episode = result["episode"]
        video = _video_path(result)
        rows.append(
            {
                "model": str(result["model"]["model_id"]),
                "task": str(episode["task"]),
                "prompt": str(episode["prompt"]),
                "seed": int(episode["seed"]),
                "result_json_path": str(path.resolve()),
                "video_path": None if video is None else str(video),
                "video_exists": bool(video is not None and video.is_file()),
                "automated_success": bool(episode["success"]),
                "automated_valid": bool(episode["valid"]),
                "automated_failure_category": episode.get("failure_category"),
                "automated_category": _category(result),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No formal result.json files found below: {evaluation_root}")
    return rows


def _expected_episode_rows(evaluation_root: Path, result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    protocol_path = evaluation_root / "protocol.json"
    if not protocol_path.is_file():
        return [
            {"model": row["model"], "task": row["task"], "seed": row["seed"]}
            for row in result_rows
        ]
    protocol_document = read_json(protocol_path)
    protocol = dict(protocol_document.get("protocol") or protocol_document)
    tasks = [str(task["task_id"]) for task in protocol["tasks"]]
    seed_start = int(protocol["seed_start"])
    seed_count = int(protocol["seed_count"])
    models = sorted(set(FORMAL_MODEL_IDS).union(str(row["model"]) for row in result_rows))
    return [
        {"model": model, "task": task, "seed": seed}
        for model in models
        for task in tasks
        for seed in range(seed_start, seed_start + seed_count)
    ]


def _coverage(evaluation_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = _expected_episode_rows(evaluation_root, rows)
    by_identity = {(row["model"], row["task"], row["seed"]): row for row in rows}
    missing_video = []
    for identity in expected:
        row = by_identity.get((identity["model"], identity["task"], identity["seed"]))
        if row is None:
            missing_video.append({**identity, "reason": "missing_result"})
        elif not row["video_exists"]:
            missing_video.append(
                {
                    **identity,
                    "reason": "missing_video",
                    "result_json_path": row["result_json_path"],
                    "video_path": row["video_path"],
                }
            )
    with_video = sum(bool(row["video_exists"]) for row in rows)
    return {
        "expected_episode_count": len(expected),
        "available_result_count": len(rows),
        "episodes_with_videos": with_video,
        "episodes_without_videos": len(missing_video),
        "missing_video_episodes": missing_video,
        "coverage_complete": len(missing_video) == 0,
    }


def _representative_rows(evaluation_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model_root in sorted((evaluation_root / "models").glob("*")):
        if not model_root.is_dir():
            continue
        model_id = model_root.name
        index = load_representative_index(model_root=model_root, model_id=model_id)
        for record in index["records"]:
            result_path = Path(str(record["result_json_path"])).resolve()
            video_path = Path(str(record["video_path"])).resolve()
            if not result_path.is_file() or not video_path.exists():
                continue
            result = read_json(result_path)
            validate_episode_result(result)
            episode = result["episode"]
            if str(result["model"]["model_id"]) != model_id:
                raise ValueError(f"Representative index/model mismatch: {result_path}")
            if str(record["category"]) != _category(result):
                raise ValueError(f"Representative category differs from result: {result_path}")
            rows.append(
                {
                    "model": model_id,
                    "task": str(episode["task"]),
                    "prompt": str(episode["prompt"]),
                    "seed": int(episode["seed"]),
                    "result_json_path": str(result_path),
                    "video_path": str(video_path),
                    "video_exists": True,
                    "automated_success": bool(episode["success"]),
                    "automated_valid": bool(episode["valid"]),
                    "automated_failure_category": episode.get("failure_category"),
                    "automated_category": str(record["category"]),
                }
            )
    if not rows:
        raise FileNotFoundError(
            "No indexed representative videos found; run category-aware evaluation and coverage validation first"
        )
    return rows


def build_manifest(
    *,
    evaluation_root: Path,
    review_seed: int,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build private/reviewer manifests without writing any result artifacts."""

    if mode not in {"full", "representative"}:
        raise ValueError(f"Unsupported review mode: {mode}")
    root = Path(evaluation_root).expanduser().resolve()
    rows = _result_rows(root)
    coverage = _coverage(root, rows)
    candidates = [row for row in rows if row["video_exists"]]
    if mode == "representative":
        candidates = _representative_rows(root)
    candidates.sort(
        key=lambda row: (
            _review_rank(
                review_seed=review_seed,
                model=str(row["model"]),
                task=str(row["task"]),
                seed=int(row["seed"]),
            ),
            str(row["model"]),
            str(row["task"]),
            int(row["seed"]),
        )
    )
    width = max(4, len(str(len(candidates))))
    private_items = []
    reviewer_items = []
    for index, row in enumerate(candidates, start=1):
        review_id = f"review_{index:0{width}d}"
        private_item = {"review_id": review_id, **row}
        private_items.append(private_item)
        reviewer_items.append(
            {
                "review_id": review_id,
                "task": row["task"],
                "prompt": row["prompt"],
            }
        )
    private_body = {
        "manifest_version": HUMAN_REVIEW_MANIFEST_VERSION,
        "evaluation_root": str(root),
        "review_seed": int(review_seed),
        "shuffle_scheme": "blake2s-sort-v1",
        "mode": mode,
        "coverage": coverage,
        "items": private_items,
    }
    private_manifest = {**private_body, "manifest_sha256": _sha256(private_body)}
    reviewer_manifest = {
        "manifest_version": HUMAN_REVIEW_MANIFEST_VERSION,
        "review_seed": int(review_seed),
        "mode": mode,
        "review_item_count": len(reviewer_items),
        "items": reviewer_items,
    }
    return private_manifest, reviewer_manifest, coverage


def write_manifest(
    *,
    output_root: Path,
    private_manifest: dict[str, Any],
    reviewer_manifest: dict[str, Any],
    coverage: dict[str, Any],
) -> None:
    root = Path(output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Review directory already exists and is non-empty: {root}")
    write_json(root / "manifest_private.json", private_manifest)
    write_json(root / "manifest_reviewer.json", reviewer_manifest)
    write_json(root / "video_coverage.json", coverage)
    missing_path = root / "missing_video_episodes.csv"
    with missing_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ("model", "task", "seed", "reason", "result_json_path", "video_path")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(coverage["missing_video_episodes"])


def reviewer_items(private_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["review_id"]): {
            "review_id": str(item["review_id"]),
            "task": str(item["task"]),
            "prompt": str(item["prompt"]),
            "video_path": str(item["video_path"]),
        }
        for item in private_manifest["items"]
    }


def load_decisions(path: Path) -> dict[str, dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != DECISION_FIELDS:
            raise ValueError(f"Unexpected human-review CSV schema: {csv_path}")
        rows = [{key: str(row[key]) for key in DECISION_FIELDS} for row in reader]
    decisions = {row["review_id"]: row for row in rows}
    if len(decisions) != len(rows):
        raise ValueError(f"Duplicate review IDs are not permitted: {csv_path}")
    return decisions


def save_decision(
    *,
    csv_path: Path,
    review_ids: set[str],
    review_id: str,
    label: str,
    failure_reason: str,
    notes: str,
    allow_overwrite: bool,
) -> dict[str, str]:
    if review_id not in review_ids:
        raise KeyError(f"Unknown review ID: {review_id}")
    validate_decision(label=label, failure_reason=failure_reason)
    decisions = load_decisions(csv_path)
    if review_id in decisions and not allow_overwrite:
        raise FileExistsError(f"Decision already exists for {review_id}; explicit overwrite is required")
    row = {
        "review_id": review_id,
        "human_label": label,
        "human_failure_reason": failure_reason,
        "notes": notes,
        "review_timestamp": datetime.now(UTC).isoformat(),
    }
    decisions[review_id] = row
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECISION_FIELDS)
        writer.writeheader()
        for decision_id in sorted(decisions):
            writer.writerow(decisions[decision_id])
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(csv_path)
    return row


def _human_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(row["human_label"]) for row in rows)
    determinate = labels["SUCCESS"] + labels["FAILURE"]
    return {
        "reviewed": len(rows),
        "successes": labels["SUCCESS"],
        "failures": labels["FAILURE"],
        "uncertain": labels["UNCERTAIN"],
        "success_rate_determinate": None if determinate == 0 else labels["SUCCESS"] / determinate,
    }


def _cohen_kappa(rows: list[dict[str, Any]]) -> float | None:
    determinate = [
        row
        for row in rows
        if bool(row["automated_valid"]) and row["human_label"] in {"SUCCESS", "FAILURE"}
    ]
    if not determinate:
        return None
    observed = sum(
        bool(row["automated_success"]) == (row["human_label"] == "SUCCESS") for row in determinate
    ) / len(determinate)
    auto_success = sum(bool(row["automated_success"]) for row in determinate) / len(determinate)
    human_success = sum(row["human_label"] == "SUCCESS" for row in determinate) / len(determinate)
    expected = auto_success * human_success + (1 - auto_success) * (1 - human_success)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def summarize_review_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)
        by_model_task[(str(row["model"]), str(row["task"]))].append(row)
    agreement = Counter(
        {
            "both_success": 0,
            "both_failure": 0,
            "automatic_success_human_failure": 0,
            "automatic_failure_human_success": 0,
            "uncertain": 0,
            "automated_invalid": 0,
        }
    )
    disagreements = []
    for row in rows:
        label = str(row["human_label"])
        if not bool(row["automated_valid"]):
            agreement["automated_invalid"] += 1
            continue
        auto_success = bool(row["automated_success"])
        if label == "UNCERTAIN":
            agreement["uncertain"] += 1
        elif auto_success and label == "SUCCESS":
            agreement["both_success"] += 1
        elif not auto_success and label == "FAILURE":
            agreement["both_failure"] += 1
        elif auto_success:
            agreement["automatic_success_human_failure"] += 1
            disagreements.append(row)
        else:
            agreement["automatic_failure_human_success"] += 1
            disagreements.append(row)
    determinate = len(rows) - agreement["uncertain"] - agreement["automated_invalid"]
    paired = paired_review_rows(rows)
    return {
        "summary_version": HUMAN_REVIEW_SUMMARY_VERSION,
        "reviewed_episode_count": len(rows),
        "human_success_by_model": {model: _human_rate(values) for model, values in sorted(by_model.items())},
        "human_success_by_model_task": {
            f"{model}:{task}": _human_rate(values)
            for (model, task), values in sorted(by_model_task.items())
        },
        "automatic_vs_human": {
            **dict(agreement),
            "agreement_rate_determinate": None
            if determinate == 0
            else (agreement["both_success"] + agreement["both_failure"]) / determinate,
            "cohen_kappa_binary_determinate": _cohen_kappa(rows),
            "disagreement_episodes": disagreements,
        },
        "paired_model_analysis": paired,
    }


def paired_review_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task_seed: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_task_seed[(str(row["task"]), int(row["seed"]))][str(row["model"])] = row
    pair_rows = []
    all_models = ("A", "B", "C")
    for (task, seed), values in sorted(by_task_seed.items()):
        pair_rows.append(
            {
                "task": task,
                "seed": seed,
                **{f"{model}_human_label": values.get(model, {}).get("human_label") for model in all_models},
                **{f"{model}_review_id": values.get(model, {}).get("review_id") for model in all_models},
            }
        )
    only_success = {
        model: sum(
            row[f"{model}_human_label"] == "SUCCESS"
            and all(row[f"{other}_human_label"] != "SUCCESS" for other in all_models if other != model)
            for row in pair_rows
        )
        for model in all_models
    }
    pairwise = {}
    for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
        counts = Counter()
        for row in pair_rows:
            left_label, right_label = row[f"{left}_human_label"], row[f"{right}_human_label"]
            if {left_label, right_label}.issubset({"SUCCESS", "FAILURE"}):
                if left_label == "SUCCESS" and right_label == "FAILURE":
                    counts[f"{left}_win"] += 1
                elif right_label == "SUCCESS" and left_label == "FAILURE":
                    counts[f"{right}_win"] += 1
                elif left_label == "SUCCESS":
                    counts["both_success"] += 1
                else:
                    counts["both_failure"] += 1
            else:
                counts["incomplete_or_uncertain"] += 1
        pairwise[f"{left}_vs_{right}"] = dict(counts)
    return {"task_seed_rows": pair_rows, "only_success_counts": only_success, "pairwise": pairwise}


def unblind_manifest(
    *,
    private_manifest: dict[str, Any],
    decisions: dict[str, dict[str, str]],
    allow_incomplete: bool,
) -> list[dict[str, Any]]:
    items = list(private_manifest["items"])
    missing = [str(item["review_id"]) for item in items if str(item["review_id"]) not in decisions]
    if missing and not allow_incomplete:
        raise ValueError(f"Human review is incomplete: {len(missing)} undecided items")
    rows = []
    for item in items:
        decision = decisions.get(str(item["review_id"]))
        if decision is None:
            continue
        rows.append(
            {
                **{key: decision[key] for key in DECISION_FIELDS},
                "model": item["model"],
                "task": item["task"],
                "prompt": item["prompt"],
                "seed": item["seed"],
                "result_json_path": item["result_json_path"],
                "video_path": item["video_path"],
                "automated_success": item["automated_success"],
                "automated_valid": item["automated_valid"],
                "automated_failure_category": item["automated_failure_category"],
            }
        )
    return sorted(rows, key=lambda row: (row["model"], row["task"], row["seed"]))


def write_unblinded_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        *DECISION_FIELDS,
        "model",
        "task",
        "prompt",
        "seed",
        "result_json_path",
        "video_path",
        "automated_success",
        "automated_valid",
        "automated_failure_category",
    )
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = ["# Human review summary", "", "## Human success by model", ""]
    lines.append("| Model | Reviewed | Success | Failure | Uncertain | Success rate (determinate) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model, row in summary["human_success_by_model"].items():
        rate = row["success_rate_determinate"]
        lines.append(
            f"| {model} | {row['reviewed']} | {row['successes']} | {row['failures']} | "
            f"{row['uncertain']} | {'n/a' if rate is None else f'{rate:.3f}'} |"
        )
    agreement = summary["automatic_vs_human"]
    lines.extend(
        [
            "",
            "## Automated vs human",
            "",
            f"- Both success: {agreement.get('both_success', 0)}",
            f"- Both failure: {agreement.get('both_failure', 0)}",
            f"- Automated success / human failure: {agreement.get('automatic_success_human_failure', 0)}",
            f"- Automated failure / human success: {agreement.get('automatic_failure_human_success', 0)}",
            f"- Uncertain: {agreement.get('uncertain', 0)}",
            f"- Automated invalid (excluded from agreement): {agreement.get('automated_invalid', 0)}",
            f"- Agreement rate (determinate): {agreement['agreement_rate_determinate']}",
            f"- Cohen's kappa (secondary): {agreement['cohen_kappa_binary_determinate']}",
        ]
    )
    disagreements = agreement["disagreement_episodes"]
    lines.extend(["", "## Disagreement episodes", ""])
    if not disagreements:
        lines.append("No determinate automated/human disagreements.")
    else:
        lines.append("| Review ID | Model | Task | Seed | Automated | Human | Video |")
        lines.append("|---|---|---|---:|---|---|---|")
        for row in disagreements:
            automated = "SUCCESS" if row["automated_success"] else "FAILURE"
            lines.append(
                f"| {row['review_id']} | {row['model']} | {row['task']} | {row['seed']} | "
                f"{automated} | {row['human_label']} | {row['video_path']} |"
            )
    return "\n".join(lines) + "\n"
