"""Schema-driven aggregates, failure-mode tables, and video indexes."""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
from collections.abc import Iterable
import csv
from pathlib import Path
from statistics import mean
from statistics import median
from typing import Any

from evaluation.sim.outputs import read_json
from evaluation.sim.outputs import validate_episode_result
from evaluation.sim.outputs import write_json


def _numeric(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _failure_category(result: dict[str, Any]) -> str:
    """Use an explicit marker only when a legacy result was not reclassified."""

    return str(result["episode"].get("failure_category") or "UNCLASSIFIED_LEGACY")


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    for result in results:
        validate_episode_result(result)
    attempted = len(results)
    valid = [result for result in results if result["episode"]["valid"]]
    invalid = [result for result in results if not result["episode"]["valid"]]
    successes = [result for result in valid if result["episode"]["success"]]
    failures = [result for result in valid if not result["episode"]["success"]]
    failure_counts = Counter(_failure_category(result) for result in failures)
    invalid_counts = Counter(
        str(result["episode"]["invalid_reason"])
        for result in invalid
        if result["episode"]["invalid_reason"]
    )
    summary: dict[str, Any] = {
        "attempted_episodes": attempted,
        "valid_episodes": len(valid),
        "invalid_episodes": len(invalid),
        "successes": len(successes),
        "failures": len(failures),
        "success_rate_valid": _rate(len(successes), len(valid)),
        "success_rate_all": _rate(len(successes), attempted),
        "invalid_rate": _rate(len(invalid), attempted),
        "invalid_reason_counts": dict(sorted(invalid_counts.items())),
        "failure_category_counts": dict(sorted(failure_counts.items())),
        "failure_category_rates_among_valid_failures": {
            category: _rate(count, len(failures))
            for category, count in sorted(failure_counts.items())
        },
    }
    metrics = [result["metrics"] for result in valid]
    lifts = _numeric(metric.get("max_lift_m") for metric in metrics)
    if lifts:
        summary["pick_metrics"] = {
            "mean_max_lift_m": mean(lifts),
            "median_max_lift_m": median(lifts),
            "max_max_lift_m": max(lifts),
            "failure_seeds": [int(result["episode"]["seed"]) for result in failures],
        }
    distances = _numeric(metric.get("pepper_ring_xy_distance_m") for metric in metrics)
    if distances:
        release_count = sum(bool(metric.get("release_confirmed")) for metric in metrics)
        stable_count = sum(bool(metric.get("stability_confirmed")) for metric in metrics)
        summary["placement_metrics"] = {
            "mean_final_xy_distance_m": mean(distances),
            "median_final_xy_distance_m": median(distances),
            "min_final_xy_distance_m": min(distances),
            "mean_containment_margin_m": mean(
                _numeric(metric.get("containment_margin_m") for metric in metrics)
            ),
            "release_confirmed_rate": _rate(release_count, len(valid)),
            "stability_confirmed_rate": _rate(stable_count, len(valid)),
        }
    return summary


def load_model_results(model_root: Path) -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted(model_root.glob("tasks/*/seed_*/result.json"))]


def _video_path(result: dict[str, Any]) -> str | None:
    artifacts = result.get("artifacts") or {}
    if artifacts.get("combined_video_path"):
        return str(artifacts["combined_video_path"])
    paths = artifacts.get("video_paths") or {}
    return None if not paths.get("combined") else str(paths["combined"])


def build_video_index(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Create deterministic, non-copying video records and representatives."""

    rows = []
    for result in results:
        episode = result["episode"]
        rows.append(
            {
                "model": result["model"]["model_id"],
                "task": episode["task"],
                "seed": int(episode["seed"]),
                "success": bool(episode["success"]),
                "valid": bool(episode["valid"]),
                "failure_category": episode.get("failure_category"),
                "failure_reason": episode.get("failure_reason"),
                "video_path": _video_path(result),
            }
        )
    rows.sort(key=lambda row: (row["model"], row["task"], row["seed"]))
    representatives: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        outcome = "SUCCESS" if row["success"] else (
            "INVALID" if not row["valid"] else str(row["failure_category"] or "UNCLASSIFIED_LEGACY")
        )
        representatives.setdefault((row["model"], row["task"], outcome), row)
    return {
        "selection_policy": "lowest_seed_per_model_task_category",
        "episodes": rows,
        "representatives": list(representatives.values()),
    }


def write_video_index(root: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    index = build_video_index(results)
    root = Path(root)
    write_json(root / "video_index.json", index)
    csv_path = root / "video_index.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "task",
        "seed",
        "success",
        "valid",
        "failure_category",
        "failure_reason",
        "video_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index["episodes"])
    return index


def write_model_summary(model_root: Path) -> dict[str, Any]:
    results = load_model_results(model_root)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_task[str(result["episode"]["task"])].append(result)
    task_reports = {task: summarize_results(rows) for task, rows in sorted(by_task.items())}
    task_rates = [report["success_rate_valid"] for report in task_reports.values()]
    macro_rate = None if any(rate is None for rate in task_rates) else mean(task_rates)
    report = {
        "model": results[0]["model"] if results else None,
        "provenance": results[0]["provenance"] if results else None,
        "overall": summarize_results(results),
        "tasks": task_reports,
        "macro_success_rate_valid": macro_rate,
        "macro_denominator": "unweighted mean of six per-task valid-episode success rates",
    }
    write_json(model_root / "summary.json", report)
    write_video_index(model_root, results)
    return report


def write_comparison(output_root: Path) -> dict[str, Any]:
    reports = {}
    all_results: list[dict[str, Any]] = []
    for model_root in sorted((Path(output_root) / "models").glob("*")):
        if model_root.is_dir():
            reports[model_root.name] = write_model_summary(model_root)
            all_results.extend(load_model_results(model_root))
    failure_mode_rows = []
    for model_id, report in sorted(reports.items()):
        for task, task_report in sorted(report["tasks"].items()):
            failure_mode_rows.append(
                {
                    "model": model_id,
                    "task": task,
                    "successes": task_report["successes"],
                    "valid_episodes": task_report["valid_episodes"],
                    "failures": task_report["failures"],
                    "invalid_episodes": task_report["invalid_episodes"],
                    "success_rate_valid": task_report["success_rate_valid"],
                    "failure_category_counts": task_report["failure_category_counts"],
                    "failure_category_rates_among_valid_failures": task_report[
                        "failure_category_rates_among_valid_failures"
                    ],
                }
            )
    comparison = {
        "models": {
            model_id: {
                "overall": report["overall"],
                "macro_success_rate_valid": report["macro_success_rate_valid"],
                "tasks": report["tasks"],
            }
            for model_id, report in reports.items()
        },
        "failure_mode_rows": failure_mode_rows,
        "macro_denominator": "unweighted mean of six per-task valid-episode success rates",
    }
    summary_root = Path(output_root) / "summaries"
    write_json(summary_root / "comparison.json", comparison)
    write_video_index(summary_root, all_results)
    return comparison
