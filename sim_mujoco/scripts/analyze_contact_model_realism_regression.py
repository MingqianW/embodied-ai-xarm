#!/usr/bin/env python3
"""Analyze and structurally validate the paired contact-model realism suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_mujoco.gripper_slip_diagnostics import load_jsonl  # noqa: E402
from sim_mujoco.scripts.run_contact_model_realism_regression import (  # noqa: E402
    PLACE_TASK,
    PROTOCOLS,
    SEEDS,
    TASKS,
    contact_conditions,
)


PENETRATION_REFERENCE_M = 0.00544
RELEASE_SUSTAIN_S = 0.1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sample_dt(rows: list[dict[str, Any]]) -> float:
    differences = [
        float(right["sim_time_s"]) - float(left["sim_time_s"])
        for left, right in zip(rows, rows[1:], strict=False)
    ]
    positive = [value for value in differences if value > 0.0]
    return statistics.median(positive) if positive else 0.0


def _maximum(rows: list[dict[str, Any]], path: tuple[str, str]) -> float | None:
    values = [
        float(row[path[0]][path[1]])
        for row in rows
        if row[path[0]].get(path[1]) is not None
    ]
    return max(values, default=None)


def _active_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(row["contacts"][field])
        for row in rows
        if float(row["contacts"].get(field, 0.0)) > 0.0
    ]
    return statistics.mean(values) if values else None


def _trace_is_finite(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        values = [
            *row["object"]["position_m"],
            *row["object"]["linear_velocity_world_mps"],
            *row["tcp"]["position_m"],
            *row["tcp"]["linear_velocity_world_mps"],
            row["simulation"]["maximum_abs_qvel"],
            row["simulation"]["maximum_abs_qacc"],
            row["contacts"]["all_target_normal_sum_n"],
            row["contacts"]["all_target_tangential_sum_n"],
        ]
        if not all(math.isfinite(float(value)) for value in values):
            return False
    return True


def _displacement(
    rows: list[dict[str, Any]], start: list[float]
) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    previous = start
    path = 0.0
    for row in rows:
        position = row["object"]["position_m"]
        path += math.hypot(
            float(position[0]) - previous[0], float(position[1]) - previous[1]
        )
        previous = position
    final = rows[-1]["object"]["position_m"]
    net = math.hypot(float(final[0]) - start[0], float(final[1]) - start[1])
    return net, path


def _release_latency(
    rows: list[dict[str, Any]], dt: float
) -> tuple[float | None, float | None]:
    release_indices = [
        index for index, row in enumerate(rows) if row["command"]["stage"] == "RELEASE"
    ]
    if not release_indices:
        return None, None
    first_release = release_indices[0]
    baseline_target = (
        rows[first_release - 1]["command"].get("gripper_returned_raw")
        if first_release > 0
        else None
    )
    onset = next(
        (
            index
            for index in release_indices
            if rows[index]["command"].get("gripper_returned_raw") is not None
            and (
                baseline_target is None
                or float(rows[index]["command"]["gripper_returned_raw"])
                > float(baseline_target) + 1e-9
            )
        ),
        None,
    )
    if onset is None:
        return None, None
    sustain = max(1, int(math.ceil(RELEASE_SUSTAIN_S / dt))) if dt > 0.0 else 1
    release = next(
        (
            index
            for index in range(onset, len(rows) - sustain + 1)
            if all(
                int(row["contacts"]["target_gripper_contact_count"]) == 0
                for row in rows[index : index + sustain]
            )
        ),
        None,
    )
    onset_time = float(rows[onset]["sim_time_s"])
    return onset_time, None if release is None else float(
        rows[release]["sim_time_s"]
    ) - onset_time


def _manifold_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the left/right target-contact manifold without discarding points."""

    active = [
        row
        for row in rows
        if int(row["contacts"].get("target_gripper_contact_count", 0)) > 0
    ]
    bilateral = [row for row in active if bool(row["contacts"].get("bilateral"))]
    symmetry = [
        float(row["contacts"]["left_right_contact_count_symmetry"])
        if row["contacts"].get("left_right_contact_count_symmetry") is not None
        else 1.0
        - abs(
            int(row["contacts"].get("left_target_count", 0))
            - int(row["contacts"].get("right_target_count", 0))
        )
        / max(
            1,
            int(row["contacts"].get("left_target_count", 0))
            + int(row["contacts"].get("right_target_count", 0)),
        )
        for row in bilateral
    ]
    first = bilateral[0] if bilateral else None
    first_contacts = {} if first is None else first["contacts"]
    return {
        "first_bilateral_left_contact_count": (
            None if first is None else int(first_contacts.get("left_target_count", 0))
        ),
        "first_bilateral_right_contact_count": (
            None if first is None else int(first_contacts.get("right_target_count", 0))
        ),
        "first_bilateral_left_contact_positions_world_m": (
            None
            if first is None
            else first_contacts.get("left_target_contact_positions_world_m", [])
        ),
        "first_bilateral_right_contact_positions_world_m": (
            None
            if first is None
            else first_contacts.get("right_target_contact_positions_world_m", [])
        ),
        "mean_bilateral_contact_count_asymmetry": (
            statistics.mean(
                abs(
                    int(row["contacts"].get("left_target_count", 0))
                    - int(row["contacts"].get("right_target_count", 0))
                )
                for row in bilateral
            )
            if bilateral
            else None
        ),
        "mean_bilateral_contact_count_symmetry": (
            statistics.mean(symmetry) if symmetry else None
        ),
        "bilateral_exact_count_symmetry_fraction": (
            statistics.mean(
                int(row["contacts"].get("left_target_count", 0))
                == int(row["contacts"].get("right_target_count", 0))
                for row in bilateral
            )
            if bilateral
            else None
        ),
    }


def analyze_trial(result: dict[str, Any]) -> dict[str, Any]:
    rows = list(load_jsonl(Path(result["artifacts"]["trace"])))
    if not rows:
        raise ValueError(f"Empty trace: {result['artifacts']['trace']}")
    trial = rows[0]["trial"]
    protocol = str(result["protocol"])
    dt = _sample_dt(rows)
    penetrations = [
        float(row["contacts"].get("maximum_all_target_penetration_m") or 0.0)
        for row in rows
    ]
    gripper_penetrations = [
        float(row["contacts"].get("maximum_target_penetration_m") or 0.0)
        for row in rows
    ]
    table_penetrations = [
        float(row["contacts"].get("maximum_target_table_penetration_m") or 0.0)
        for row in rows
    ]
    hold_rows = [row for row in rows if row["command"]["source"] == "scripted_hold"]
    slip = [
        float(row["relative"]["drift_m"])
        for row in hold_rows
        if row["relative"]["drift_m"] is not None
    ]
    onset, release_latency = _release_latency(rows, dt)
    manifold = _manifold_metrics(hold_rows if hold_rows else rows)
    downward_slip = [
        float(row["relative"]["downward_slip_m"])
        for row in hold_rows
        if row["relative"].get("downward_slip_m") is not None
    ]
    contact_loss = "contact_loss" in result.get("event_names", [])
    stable_hold = (
        protocol == "suspended_grasp"
        and bool(hold_rows)
        and not contact_loss
        and sum(bool(row["contacts"].get("bilateral")) for row in hold_rows)
        / len(hold_rows)
        >= 0.95
        and max(downward_slip, default=math.inf) <= 0.002
    )

    push_rows = [
        row for row in rows if row["command"]["stage"] in {"PUSH", "PUSH_SETTLE"}
    ]
    if push_rows:
        first_push_index = int(push_rows[0]["sample_index"])
        push_start = (
            rows[first_push_index - 1]["object"]["position_m"]
            if first_push_index > 0
            else trial["initial_target_position_m"]
        )
        push_net, push_path = _displacement(push_rows, push_start)
        pushing_x = float(push_rows[-1]["object"]["position_m"][0]) - float(
            push_start[0]
        )
    else:
        push_net = push_path = pushing_x = None

    if protocol == "tabletop_sliding":
        slide_net, slide_path = _displacement(rows, trial["initial_target_position_m"])
        sliding_x = float(rows[-1]["object"]["position_m"][0]) - float(
            trial["initial_target_position_m"][0]
        )
    else:
        slide_net = slide_path = sliding_x = None

    maximum_penetration = max(penetrations)
    return {
        "protocol": protocol,
        "task": result["task"],
        "seed": int(result["seed"]),
        "condition": result["setting"]["condition"],
        "setting": result["setting"]["name"],
        "sample_count": len(rows),
        "sample_dt_s": dt,
        "maximum_target_penetration_m": maximum_penetration,
        "penetration_duration_s": sum(value > 0.0 for value in penetrations) * dt,
        "maximum_target_gripper_penetration_m": max(gripper_penetrations),
        "target_gripper_penetration_duration_s": (
            sum(value > 0.0 for value in gripper_penetrations) * dt
        ),
        "maximum_target_table_penetration_m": max(table_penetrations),
        "target_table_penetration_duration_s": (
            sum(value > 0.0 for value in table_penetrations) * dt
        ),
        "target_gripper_penetration_over_prior_5_44mm": (
            max(gripper_penetrations) > PENETRATION_REFERENCE_M
        ),
        "any_target_penetration_over_prior_5_44mm": (
            maximum_penetration > PENETRATION_REFERENCE_M
        ),
        "maximum_normal_contact_force_n": _maximum(
            rows, ("contacts", "all_target_normal_sum_n")
        ),
        "maximum_tangential_contact_force_n": _maximum(
            rows, ("contacts", "all_target_tangential_sum_n")
        ),
        "mean_active_normal_contact_force_n": _active_mean(
            rows, "all_target_normal_sum_n"
        ),
        "mean_active_tangential_contact_force_n": _active_mean(
            rows, "all_target_tangential_sum_n"
        ),
        "maximum_target_gripper_normal_force_n": _maximum(
            rows, ("contacts", "target_gripper_normal_sum_n")
        ),
        "maximum_target_gripper_tangential_force_n": _maximum(
            rows, ("contacts", "target_gripper_tangential_sum_n")
        ),
        "maximum_target_table_normal_force_n": _maximum(
            rows, ("contacts", "target_table_normal_sum_n")
        ),
        "maximum_target_table_tangential_force_n": _maximum(
            rows, ("contacts", "target_table_tangential_sum_n")
        ),
        "maximum_relative_grasp_slip_m": max(slip, default=None),
        "maximum_downward_grasp_slip_m": max(downward_slip, default=None),
        "stable_hold": stable_hold if protocol == "suspended_grasp" else None,
        "contact_loss": contact_loss,
        "hold_bilateral_contact_fraction": (
            sum(bool(row["contacts"]["bilateral"]) for row in hold_rows)
            / len(hold_rows)
            if hold_rows
            else None
        ),
        "place_stable_success": (
            bool(result.get("place_stability", {}).get("stable_place_success"))
            if protocol == "placing_release"
            else None
        ),
        "opening_target_onset_s": onset,
        "release_latency_s": release_latency,
        "pushing_x_displacement_m": pushing_x,
        "pushing_planar_displacement_m": push_net,
        "pushing_planar_path_m": push_path,
        "sliding_x_displacement_m": sliding_x,
        "sliding_planar_displacement_m": slide_net,
        "sliding_planar_path_m": slide_path,
        "simulation_warning_count": max(
            int(row["simulation"]["warning_count"]) for row in rows
        ),
        "simulation_finite": _trace_is_finite(rows),
        **manifold,
    }


def _manifest_hash(result: dict[str, Any], trial: dict[str, Any]) -> str:
    manifest = trial.get("oracle_action_manifest") or trial.get("action_manifest")
    if manifest is None:
        raise ValueError(
            f"Missing action manifest for {result['protocol']}/{result['task']}"
        )
    return str(manifest["sha256"])


def validate_pairs(results: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    expected_conditions = {row["condition"] for row in contact_conditions()}
    pairs: dict[tuple[str, str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for result in results:
        trial = json.loads(
            Path(result["artifacts"]["trial"]).read_text(encoding="utf-8")
        )
        protocol = str(result["protocol"])
        task = str(result["task"])
        key = (protocol, task, int(result["seed"]))
        pairs.setdefault(key, []).append((result, trial))
        allowed_tasks = {PLACE_TASK} if protocol == "placing_release" else set(TASKS)
        if protocol not in PROTOCOLS:
            errors.append(f"{key}: unexpected protocol")
        elif task not in allowed_tasks:
            errors.append(f"{key}: unexpected task for protocol")
        if trial.get("protocol") != protocol:
            errors.append(f"{key}: result/trial protocol mismatch")
        if result.get("setting") != trial.get("setting"):
            errors.append(f"{key}: result/trial setting mismatch")
        changed = trial["overrides"].get("changed_invariant_hashes")
        if changed:
            errors.append(f"{key}: forbidden invariant changes {changed}")
        effective = trial["overrides"]["effective"]["simulation"]
        expected = next(
            x
            for x in contact_conditions()
            if x["condition"] == result["setting"]["condition"]
        )
        if effective["cone"] != expected["cone"] or not math.isclose(
            float(effective["impratio"]), float(expected["impratio"])
        ):
            errors.append(f"{key}: effective contact condition mismatch")
    for key, values in sorted(pairs.items()):
        conditions = {item[0]["setting"]["condition"] for item in values}
        if conditions != expected_conditions or len(values) != 2:
            errors.append(f"{key}: expected exactly A/B, got {sorted(conditions)}")
            continue
        by_condition = {item[0]["setting"]["condition"]: item for item in values}
        a_result, a_trial = by_condition["A"]
        b_result, b_trial = by_condition["B"]
        if (
            a_trial["paired_initial_state"]["state_sha256"]
            != b_trial["paired_initial_state"]["state_sha256"]
        ):
            errors.append(f"{key}: initial-state hash mismatch")
        if _manifest_hash(a_result, a_trial) != _manifest_hash(b_result, b_trial):
            errors.append(f"{key}: action-manifest hash mismatch")
        if a_trial.get("model_sha256") != b_trial.get("model_sha256"):
            errors.append(f"{key}: model-file hash mismatch")
        if a_trial["overrides"].get("invariant_hashes_after") != b_trial[
            "overrides"
        ].get("invariant_hashes_after"):
            errors.append(f"{key}: A/B fixed-model invariant hash mismatch")
    expected_pair_count = 30
    if len(pairs) != expected_pair_count:
        errors.append(f"expected {expected_pair_count} pairs, got {len(pairs)}")
    if len(results) != 60:
        errors.append(f"expected 60 trials, got {len(results)}")
    expected_pair_keys = {
        (protocol, task, seed)
        for protocol in PROTOCOLS
        for task in ((PLACE_TASK,) if protocol == "placing_release" else TASKS)
        for seed in SEEDS
    }
    if set(pairs) != expected_pair_keys:
        errors.append("pair keys differ from the fixed protocol/task/seed matrix")
    protocol_trial_counts = {
        protocol: sum(str(result["protocol"]) == protocol for result in results)
        for protocol in PROTOCOLS
    }
    expected_protocol_trial_counts = {
        "suspended_grasp": 18,
        "pushing": 18,
        "placing_release": 6,
        "tabletop_sliding": 18,
    }
    if protocol_trial_counts != expected_protocol_trial_counts:
        errors.append(
            "protocol trial counts differ: "
            f"expected={expected_protocol_trial_counts}, actual={protocol_trial_counts}"
        )
    return {
        "status": "passed" if not errors else "failed",
        "trial_count": len(results),
        "pair_count": len(pairs),
        "protocol_trial_counts": protocol_trial_counts,
        "errors": errors,
    }


def _aggregate(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "maximum_target_penetration_m",
        "penetration_duration_s",
        "maximum_normal_contact_force_n",
        "maximum_tangential_contact_force_n",
        "maximum_relative_grasp_slip_m",
        "release_latency_s",
        "pushing_x_displacement_m",
        "sliding_x_displacement_m",
    )
    output: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        for condition in ("A", "B"):
            subset = [
                row
                for row in summaries
                if row["protocol"] == protocol and row["condition"] == condition
            ]
            summary: dict[str, Any] = {
                "protocol": protocol,
                "condition": condition,
                "trial_count": len(subset),
            }
            for metric in metrics:
                values = [
                    float(row[metric]) for row in subset if row[metric] is not None
                ]
                summary[f"mean_{metric}"] = statistics.mean(values) if values else None
                summary[f"max_{metric}"] = max(values, default=None)
            output.append(summary)
    return output


def _paired_deltas(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "maximum_target_penetration_m",
        "penetration_duration_s",
        "maximum_normal_contact_force_n",
        "maximum_tangential_contact_force_n",
        "maximum_relative_grasp_slip_m",
        "release_latency_s",
        "pushing_x_displacement_m",
        "sliding_x_displacement_m",
    )
    groups: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in summaries:
        key = (str(row["protocol"]), str(row["task"]), int(row["seed"]))
        groups.setdefault(key, {})[str(row["condition"])] = row
    output: list[dict[str, Any]] = []
    for (protocol, task, seed), conditions in sorted(groups.items()):
        a = conditions["A"]
        b = conditions["B"]
        result: dict[str, Any] = {
            "protocol": protocol,
            "task": task,
            "seed": seed,
            "delta_definition": "B_minus_A",
        }
        for metric in metrics:
            result[metric] = (
                None
                if a[metric] is None or b[metric] is None
                else float(b[metric]) - float(a[metric])
            )
        output.append(result)
    return output


def main() -> None:
    args = _parser().parse_args()
    suite_root = args.suite_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing analysis directory: {output}")
    results_payload = json.loads(
        (suite_root / "results.json").read_text(encoding="utf-8")
    )
    if (
        results_payload.get("status") != "complete"
        or results_payload.get("suite") != "realism"
    ):
        raise ValueError("Suite results are not complete realism results")
    results = results_payload["trials"]
    output.mkdir(parents=True, exist_ok=False)
    validation = validate_pairs(results)
    _write_json(output / "validation.json", validation)
    if validation["status"] != "passed":
        raise RuntimeError(f"Paired-design validation failed: {validation['errors']}")
    summaries = [analyze_trial(result) for result in results]
    warning_count = max(row["simulation_warning_count"] for row in summaries)
    non_finite_count = sum(not row["simulation_finite"] for row in summaries)
    if warning_count != 0 or non_finite_count != 0:
        validation["status"] = "failed"
        if warning_count != 0:
            validation["errors"].append(
                f"simulation warnings present; maximum count={warning_count}"
            )
        if non_finite_count != 0:
            validation["errors"].append(
                f"non-finite simulation traces present; count={non_finite_count}"
            )
        _write_json(output / "validation.json", validation)
        raise RuntimeError(f"Runtime validation failed: {validation['errors']}")
    aggregate = _aggregate(summaries)
    paired_deltas = _paired_deltas(summaries)
    _write_json(output / "trial_metrics.json", summaries)
    _write_json(output / "condition_protocol_summary.json", aggregate)
    _write_json(output / "paired_deltas.json", paired_deltas)
    _write_json(
        output / "metric_definitions.json",
        {
            "penetration": "maximum and positive-duration of any contact involving the target body",
            "contact_force": "sum across all simultaneous contacts involving the target; active means positive force",
            "relative_grasp_slip": "maximum TCP-minus-object drift from first bilateral contact during scripted hold",
            "release_latency": f"first increase in effective simulated opening target to {RELEASE_SUSTAIN_S:g} s sustained zero target-gripper contacts",
            "pushing_displacement": "target planar displacement from the state immediately before PUSH through settle",
            "sliding_displacement": "target planar displacement from the paired imposed-velocity initial state",
            "prior_target_gripper_penetration_reference_m": PENETRATION_REFERENCE_M,
        },
    )
    fields = list(summaries[0])
    with (output / "trial_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(
        json.dumps(
            {"validation": validation, "summary": aggregate}, indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
