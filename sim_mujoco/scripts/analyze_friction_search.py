#!/usr/bin/env python3
"""Select the best tested split-pad sliding friction under realism gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_mujoco.scripts.analyze_friction_ablation import (  # noqa: E402
    _hold_metrics,
    _pushing_metrics,
    _release_metrics,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser


def _setting(result: dict[str, Any]) -> str:
    return str(result["setting"]["name"])


def _mu(result: dict[str, Any]) -> float:
    return float(result["setting"]["sliding_friction"])


def _metrics(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result["protocol"] == "suspended_grasp":
            row = _hold_metrics(result)
        elif result["protocol"] == "pushing":
            row = _pushing_metrics(result)
        elif result["protocol"] == "placing_release":
            row = _release_metrics(result)
        else:
            raise RuntimeError(f"Unknown protocol: {result['protocol']}")
        row["setting"] = _setting(result)
        row["sliding_friction"] = _mu(result)
        rows.append(row)
    return rows


def _index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(row["setting"], row["protocol"], row["task"]): row for row in rows}


def _candidate_summary(
    setting: str,
    rows: list[dict[str, Any]],
    indexed: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    candidate = [row for row in rows if row["setting"] == setting]
    mu = float(candidate[0]["sliding_friction"])
    holds = [row for row in candidate if row["protocol"] == "suspended_grasp"]
    pushes = [row for row in candidate if row["protocol"] == "pushing"]
    release = next(row for row in candidate if row["protocol"] == "placing_release")
    baseline_release = indexed[("mu_2p00", "placing_release", release["task"])]
    push_ratios = {
        row["task"]: (
            row["object_displacement_xy_m"]
            / indexed[("mu_2p00", "pushing", row["task"])]["object_displacement_xy_m"]
        )
        for row in pushes
    }
    penetration_ratios: dict[str, float | None] = {}
    penetration_ok = True
    for row in candidate:
        baseline = indexed[("mu_2p00", row["protocol"], row["task"])]
        value = float(row["maximum_penetration_m"] or 0.0)
        origin = float(baseline["maximum_penetration_m"] or 0.0)
        penetration_ratios[f"{row['protocol']}:{row['task']}"] = (
            None if origin <= 1e-12 else value / origin
        )
        penetration_ok = penetration_ok and value <= max(
            0.001, 1.25 * origin, origin + 0.00025
        )
    release_latency_ok = bool(
        release["release_success"]
        and release["release_latency_s"] is not None
        and baseline_release["release_latency_s"] is not None
        and release["release_latency_s"]
        <= max(
            baseline_release["release_latency_s"] + 0.05,
            1.5 * baseline_release["release_latency_s"],
        )
    )
    pushing_ok = all(0.75 <= value <= 1.25 for value in push_ratios.values())
    warning_count = max(int(row["maximum_warning_count"]) for row in candidate)
    strict_holds = sum(bool(row["strict_stable_hold"]) for row in holds)
    retained_holds = sum(bool(row["retained"]) for row in holds)
    drop_count = sum(bool(row["drop"]) for row in holds)
    bilateral_ok = all(row["bilateral_contact_fraction"] >= 0.95 for row in holds)
    accepted = bool(
        strict_holds == 3
        and retained_holds == 3
        and drop_count == 0
        and bilateral_ok
        and pushing_ok
        and penetration_ok
        and release_latency_ok
        and warning_count == 0
    )
    slip_sum_mm = 1000.0 * sum(float(row["maximum_downward_slip_m"]) for row in holds)
    push_log_penalty = sum(abs(math.log(value)) for value in push_ratios.values())
    score = (
        1000.0 * drop_count
        + 100.0 * (3 - strict_holds)
        + slip_sum_mm
        + 10.0 * push_log_penalty
        + (0.0 if release_latency_ok else 100.0)
        + (0.0 if penetration_ok else 100.0)
        + 1000.0 * warning_count
    )
    return {
        "setting": setting,
        "sliding_friction": mu,
        "accepted": accepted,
        "score": score,
        "strict_stable_holds_of_3": strict_holds,
        "retained_holds_of_3": retained_holds,
        "drop_count": drop_count,
        "maximum_slip_by_task_m": {
            row["task"]: row["maximum_downward_slip_m"] for row in holds
        },
        "bilateral_fraction_by_task": {
            row["task"]: row["bilateral_contact_fraction"] for row in holds
        },
        "mean_normal_force_by_task_n": {
            row["task"]: row["mean_bilateral_normal_force_n"] for row in holds
        },
        "pushing_B_over_baseline_displacement_ratio": push_ratios,
        "pushing_ok": pushing_ok,
        "penetration_ratio_to_baseline": penetration_ratios,
        "penetration_ok": penetration_ok,
        "release_success": release["release_success"],
        "release_latency_s": release["release_latency_s"],
        "release_latency_ok": release_latency_ok,
        "maximum_warning_count": warning_count,
    }


def main() -> None:
    args = _parser().parse_args()
    root = args.run_root.expanduser().resolve()
    source = json.loads((root / "results.json").read_text(encoding="utf-8"))
    if source.get("status") != "complete":
        raise RuntimeError("Friction search is incomplete")
    rows = _metrics(source["validation_results"])
    indexed = _index(rows)
    settings = list(dict.fromkeys(row["setting"] for row in rows))
    if "mu_2p00" not in settings:
        raise RuntimeError("Validation lacks the mu=2.00 baseline")
    summaries = [_candidate_summary(setting, rows, indexed) for setting in settings]
    accepted = [row for row in summaries if row["accepted"]]
    selected = min(accepted or summaries, key=lambda row: row["score"])
    selection = {
        "schema_version": "xarm_split_pad_friction_selection_v1",
        "selection_scope": "best among tested candidates; not a continuous optimum",
        "accepted_candidate_count": len(accepted),
        "selected_setting": selected["setting"],
        "selected_sliding_friction": selected["sliding_friction"],
        "selected_passes_all_acceptance_gates": selected["accepted"],
        "selected_score": selected["score"],
        "selected_summary": selected,
        "parameter_results": sorted(summaries, key=lambda row: row["sliding_friction"]),
        "screen_ranking": source["screen_selection"]["screen_ranking"],
    }
    output = root / "analysis"
    output.mkdir(exist_ok=False)
    (output / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "validation_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "parameter_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = (
            "sliding_friction",
            "accepted",
            "score",
            "strict_stable_holds_of_3",
            "retained_holds_of_3",
            "drop_count",
            "pushing_ok",
            "penetration_ok",
            "release_success",
            "release_latency_s",
            "maximum_warning_count",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(summaries, key=lambda row: row["sliding_friction"]))
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
