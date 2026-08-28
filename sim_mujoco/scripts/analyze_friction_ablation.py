#!/usr/bin/env python3
"""Analyze the fixed legacy split-pad fingertip-friction A/B diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _condition(result: dict[str, Any]) -> str:
    return str(result["setting"]["condition"])


def _hold_metrics(result: dict[str, Any]) -> dict[str, Any]:
    rows = _rows(Path(result["artifacts"]["trace"]))
    hold = [row for row in rows if row["command"]["source"] == "scripted_hold"]
    events = set(result["event_names"])
    slips = [
        float(row["relative"]["downward_slip_m"])
        for row in hold
        if row["relative"]["downward_slip_m"] is not None
    ]
    velocities = [float(row["relative"]["vertical_slip_velocity_mps"]) for row in hold]
    normals = [float(row["contacts"]["target_gripper_normal_sum_n"]) for row in hold]
    penetrations = [
        float(row["contacts"]["maximum_target_penetration_m"] or 0.0) for row in hold
    ]
    warnings = [int(row["simulation"]["warning_count"]) for row in rows]
    bilateral_fraction = mean(bool(row["contacts"]["bilateral"]) for row in hold)
    final_lift = float(rows[-1]["object"]["lift_height_m"])
    maximum_slip = max(slips, default=math.nan)
    retained = bool("object_drop" not in events and final_lift >= 0.04)
    strict_stable = bool(
        retained
        and maximum_slip < 0.002
        and bilateral_fraction >= 0.95
        and max(warnings, default=0) == 0
    )
    return {
        "protocol": "suspended_grasp",
        "task": result["task"],
        "seed": result["seed"],
        "condition": _condition(result),
        "retained": retained,
        "strict_stable_hold": strict_stable,
        "drop": "object_drop" in events,
        "contact_loss": "contact_loss" in events,
        "final_lift_height_m": final_lift,
        "maximum_downward_slip_m": maximum_slip,
        "final_downward_slip_m": slips[-1] if slips else None,
        "maximum_downward_slip_velocity_mps": max(velocities, default=None),
        "mean_bilateral_normal_force_n": mean(normals) if normals else None,
        "maximum_bilateral_normal_force_n": max(normals, default=None),
        "bilateral_contact_fraction": bilateral_fraction,
        "maximum_penetration_m": max(penetrations, default=None),
        "maximum_warning_count": max(warnings, default=0),
    }


def _pushing_metrics(result: dict[str, Any]) -> dict[str, Any]:
    rows = _rows(Path(result["artifacts"]["trace"]))
    push = [row for row in rows if row["command"]["stage"] == "PUSH"]
    initial = rows[0]["trial"]["initial_target_position_m"]
    final = rows[-1]["object"]["position_m"]
    displacement = [float(final[index] - initial[index]) for index in range(3)]
    penetrations = [
        float(row["contacts"]["maximum_target_penetration_m"] or 0.0) for row in push
    ]
    normals = [float(row["contacts"]["target_gripper_normal_sum_n"]) for row in push]
    tangentials = [
        float(row["contacts"]["target_gripper_tangential_sum_n"]) for row in push
    ]
    warnings = [int(row["simulation"]["warning_count"]) for row in rows]
    return {
        "protocol": "pushing",
        "task": result["task"],
        "seed": result["seed"],
        "condition": _condition(result),
        "object_displacement_x_m": displacement[0],
        "object_displacement_xy_m": math.hypot(displacement[0], displacement[1]),
        "mean_normal_force_n": mean(normals) if normals else None,
        "mean_tangential_force_n": mean(tangentials) if tangentials else None,
        "maximum_penetration_m": max(penetrations, default=None),
        "maximum_warning_count": max(warnings, default=0),
    }


def _release_metrics(result: dict[str, Any]) -> dict[str, Any]:
    rows = _rows(Path(result["artifacts"]["trace"]))
    release_index = next(
        (
            index
            for index, row in enumerate(rows)
            if float(row["command"].get("gripper_returned_raw") or -math.inf) >= 650.0
        ),
        None,
    )
    loss_index = None
    if release_index is not None:
        loss_index = next(
            (
                index
                for index in range(release_index, len(rows))
                if int(rows[index]["contacts"]["target_gripper_contact_count"]) == 0
            ),
            None,
        )
    latency = None
    if release_index is not None and loss_index is not None:
        latency = float(
            rows[loss_index]["sim_time_s"] - rows[release_index]["sim_time_s"]
        )
    post_release = rows[release_index:] if release_index is not None else []
    residual_fraction = (
        mean(
            int(row["contacts"]["target_gripper_contact_count"]) > 0
            for row in post_release
        )
        if post_release
        else None
    )
    penetrations = [
        float(row["contacts"]["maximum_target_penetration_m"] or 0.0)
        for row in post_release
    ]
    warnings = [int(row["simulation"]["warning_count"]) for row in rows]
    place = result.get("place_stability") or {}
    return {
        "protocol": "placing_release",
        "task": result["task"],
        "seed": result["seed"],
        "condition": _condition(result),
        "release_command_seen": release_index is not None,
        "contact_loss_seen": loss_index is not None,
        "release_latency_s": latency,
        "post_release_residual_contact_fraction": residual_fraction,
        "release_success": bool(place.get("stable_place_success", False)),
        "maximum_penetration_m": max(penetrations, default=None),
        "maximum_warning_count": max(warnings, default=0),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if abs(denominator) < 1e-12 else numerator / denominator


def _paired(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["protocol"], row["task"]), {})[row["condition"]] = row
    pairs = []
    for (protocol, task), conditions in sorted(grouped.items()):
        if set(conditions) != {"A", "B"}:
            raise RuntimeError(f"Incomplete pair: {(protocol, task)}={set(conditions)}")
        a, b = conditions["A"], conditions["B"]
        pair: dict[str, Any] = {"protocol": protocol, "task": task, "A": a, "B": b}
        if protocol == "suspended_grasp":
            pair["B_over_A_slip_ratio"] = _ratio(
                b["maximum_downward_slip_m"], a["maximum_downward_slip_m"]
            )
            pair["B_minus_A_normal_force_n"] = (
                b["mean_bilateral_normal_force_n"] - a["mean_bilateral_normal_force_n"]
            )
        elif protocol == "pushing":
            pair["B_over_A_pushing_displacement_ratio"] = _ratio(
                b["object_displacement_xy_m"], a["object_displacement_xy_m"]
            )
        elif protocol == "placing_release":
            pair["B_minus_A_release_latency_s"] = (
                None
                if a["release_latency_s"] is None or b["release_latency_s"] is None
                else b["release_latency_s"] - a["release_latency_s"]
            )
        pairs.append(pair)
    return pairs


def _gate(rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    holds = [row for row in rows if row["protocol"] == "suspended_grasp"]
    a_holds = [row for row in holds if row["condition"] == "A"]
    b_holds = [row for row in holds if row["condition"] == "B"]
    strict_a = sum(bool(row["strict_stable_hold"]) for row in a_holds)
    strict_b = sum(bool(row["strict_stable_hold"]) for row in b_holds)
    slip_a = sum(float(row["maximum_downward_slip_m"]) for row in a_holds)
    slip_b = sum(float(row["maximum_downward_slip_m"]) for row in b_holds)
    clear_hold_improvement = bool(
        strict_b > strict_a
        or (
            sum(bool(row["retained"]) for row in b_holds)
            >= sum(bool(row["retained"]) for row in a_holds)
            and slip_b <= 0.75 * slip_a
        )
    )
    push_ratios = [
        pair["B_over_A_pushing_displacement_ratio"]
        for pair in pairs
        if pair["protocol"] == "pushing"
    ]
    pushing_ok = all(value is not None and value >= 0.75 for value in push_ratios)
    release_b = next(
        row
        for row in rows
        if row["protocol"] == "placing_release" and row["condition"] == "B"
    )
    release_a = next(
        row
        for row in rows
        if row["protocol"] == "placing_release" and row["condition"] == "A"
    )
    numerical_ok = all(row["maximum_warning_count"] == 0 for row in rows)
    penetration_regression_ok = all(
        (pair["B"]["maximum_penetration_m"] or 0.0)
        <= max(0.001, pair["A"]["maximum_penetration_m"] or 0.0)
        for pair in pairs
    )
    release_ok = bool(
        release_b["release_success"]
        and release_a["release_success"]
        and release_a["release_latency_s"] is not None
        and release_b["release_latency_s"] is not None
        and release_b["release_latency_s"]
        <= max(
            release_a["release_latency_s"] + 0.05,
            1.5 * release_a["release_latency_s"],
        )
    )
    passed = bool(
        clear_hold_improvement
        and pushing_ok
        and release_ok
        and numerical_ok
        and penetration_regression_ok
    )
    return {
        "phase2_policy_video_recommended": passed,
        "clear_hold_improvement": clear_hold_improvement,
        "strict_stable_holds_A_of_3": strict_a,
        "strict_stable_holds_B_of_3": strict_b,
        "aggregate_B_over_A_slip_ratio": _ratio(slip_b, slip_a),
        "pushing_ok": pushing_ok,
        "pushing_B_over_A_displacement_ratios": push_ratios,
        "release_ok": release_ok,
        "numerical_and_penetration_ok": numerical_ok,
        "penetration_regression_ok": penetration_regression_ok,
        "decision": "PASS_PHASE1" if passed else "FAIL_PHASE1_NO_POLICY_VIDEO",
    }


def main() -> None:
    args = _parser().parse_args()
    root = args.run_root.expanduser().resolve()
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    if results.get("status") != "complete" or int(results.get("trial_count", -1)) != 14:
        raise RuntimeError(f"Expected a complete 14-trial run: {root / 'results.json'}")
    rows = []
    for result in results["trials"]:
        protocol = str(result["protocol"])
        if protocol == "suspended_grasp":
            rows.append(_hold_metrics(result))
        elif protocol == "pushing":
            rows.append(_pushing_metrics(result))
        elif protocol == "placing_release":
            rows.append(_release_metrics(result))
        else:
            raise RuntimeError(f"Unexpected protocol: {protocol}")
    pairs = _paired(rows)
    gate = _gate(rows, pairs)
    analysis = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "analysis"
    )
    analysis.mkdir(exist_ok=False)
    (analysis / "trial_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (analysis / "paired_results.json").write_text(
        json.dumps(pairs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (analysis / "phase2_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = sorted({key for row in rows for key in row})
    with (analysis / "trial_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
