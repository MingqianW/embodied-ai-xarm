#!/usr/bin/env python3
"""Analyze and validate the paired split-pad geometry experiment."""

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

from sim_mujoco.scripts.analyze_contact_model_realism_regression import (  # noqa: E402
    _manifest_hash,
    _write_json,
    analyze_trial,
)
from sim_mujoco.scripts.run_split_pad_geometry_experiment import (  # noqa: E402
    PAD_HALF_SIZE_M,
    PAD_SPECS,
    PAD_Z_CENTERS_M,
    PLACE_TASK,
    PROTOCOLS,
    SEEDS,
    TASKS,
    geometry_conditions,
)


NUMERIC_METRICS = (
    "maximum_target_penetration_m",
    "penetration_duration_s",
    "maximum_target_gripper_penetration_m",
    "target_gripper_penetration_duration_s",
    "maximum_normal_contact_force_n",
    "maximum_tangential_contact_force_n",
    "maximum_target_gripper_normal_force_n",
    "maximum_target_gripper_tangential_force_n",
    "maximum_relative_grasp_slip_m",
    "maximum_downward_grasp_slip_m",
    "mean_bilateral_contact_count_asymmetry",
    "mean_bilateral_contact_count_symmetry",
    "bilateral_exact_count_symmetry_fraction",
    "release_latency_s",
    "pushing_x_displacement_m",
    "pushing_planar_displacement_m",
    "pushing_planar_path_m",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _without_geom_ids(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {"id", "body_id"}}


def _target_signature(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "body": target["body"],
        "mass_kg": target["mass_kg"],
        "inertia_kg_m2": target["inertia_kg_m2"],
        "geoms": sorted(
            (_without_geom_ids(value) for value in target["geoms"]),
            key=lambda value: str(value["name"]),
        ),
    }


def _fixed_mechanics_signature(effective: dict[str, Any]) -> dict[str, Any]:
    return {
        "simulation": effective["simulation"],
        "actuator": effective["actuator"],
        "finger_joints": effective["finger_joints"],
        "gripper_equality": effective["gripper_equality"],
        "target": _target_signature(effective["target"]),
    }


def _pad_groups(effective: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = {"left": {}, "right": {}}
    for pad in effective["finger_pads"]:
        name = str(pad["name"])
        side = "left" if name.startswith("left_fingertip_pad") else "right"
        if side not in groups:
            raise ValueError(f"Unexpected pad name: {name}")
        groups[side][name] = _without_geom_ids(pad)
    return groups


def _validate_pad_topology(
    a_effective: dict[str, Any], b_effective: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    a_groups = _pad_groups(a_effective)
    b_groups = _pad_groups(b_effective)
    allowed = ("name", "size_m", "pos_m")
    for side, base_name in (
        ("left", "left_fingertip_pad"),
        ("right", "right_fingertip_pad"),
    ):
        if set(a_groups[side]) != {base_name}:
            errors.append(
                f"A has unexpected {side} pad names: {sorted(a_groups[side])}"
            )
            continue
        expected_b = {base_name, f"{base_name}_upper"}
        if set(b_groups[side]) != expected_b:
            errors.append(
                f"B has unexpected {side} pad names: {sorted(b_groups[side])}"
            )
            continue
        baseline = a_groups[side][base_name]
        for suffix, z_value in (
            ("", PAD_Z_CENTERS_M[0]),
            ("_upper", PAD_Z_CENTERS_M[1]),
        ):
            candidate = b_groups[side][f"{base_name}{suffix}"]
            for field in candidate:
                if field in allowed:
                    continue
                if candidate[field] != baseline[field]:
                    errors.append(f"{side}{suffix}: changed fixed pad field {field}")
            if not math.isclose(
                candidate["size_m"][0], PAD_HALF_SIZE_M[0], abs_tol=1e-12
            ):
                errors.append(f"{side}{suffix}: x half-size mismatch")
            if not math.isclose(
                candidate["size_m"][1], PAD_HALF_SIZE_M[1], abs_tol=1e-12
            ):
                errors.append(f"{side}{suffix}: y half-size mismatch")
            if not math.isclose(
                candidate["size_m"][2], PAD_HALF_SIZE_M[2], abs_tol=1e-12
            ):
                errors.append(f"{side}{suffix}: z half-size mismatch")
            expected_y = PAD_SPECS[base_name]["y"]
            position = candidate.get("pos_m")
            if position is not None:
                if not math.isclose(position[1], expected_y, abs_tol=1e-12):
                    errors.append(f"{side}{suffix}: y position mismatch")
                if not math.isclose(position[2], z_value, abs_tol=1e-12):
                    errors.append(f"{side}{suffix}: z position mismatch")
    return errors


def _validate_compiled_geometry(model_validation: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if model_validation.get("passed") is not True:
        errors.append("compiled topology-only model validation did not pass")
        return errors
    if model_validation.get("added_geom_count") != 2:
        errors.append("compiled split-pad model did not add exactly two geoms")
    if model_validation.get("invariants_identical") is not True:
        errors.append("compiled fixed mechanics are not identical")
    if model_validation.get("finger_body_mass_and_inertia_identical") is not True:
        errors.append("compiled finger mass/inertia are not identical")
    pads = model_validation.get("diagnostic_pads", {})
    for base_name, expected in PAD_SPECS.items():
        values = pads.get(base_name, {})
        for suffix, z_value in (
            ("", PAD_Z_CENTERS_M[0]),
            ("_upper", PAD_Z_CENTERS_M[1]),
        ):
            value = values.get(f"{base_name}{suffix}")
            if value is None:
                errors.append(f"compiled diagnostic pad is absent: {base_name}{suffix}")
                continue
            if not math.isclose(value["pos_m"][1], expected["y"], abs_tol=1e-12):
                errors.append(f"compiled {base_name}{suffix}: y position mismatch")
            if not math.isclose(value["pos_m"][2], z_value, abs_tol=1e-12):
                errors.append(f"compiled {base_name}{suffix}: z position mismatch")
    return errors


def validate_pairs(suite_root: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    manifest = json.loads((suite_root / "manifest.json").read_text(encoding="utf-8"))
    model_validation = json.loads(
        (suite_root / "geometry_model_validation.json").read_text(encoding="utf-8")
    )
    if manifest.get("suite") != "split_pad_geometry":
        errors.append("manifest suite is not split_pad_geometry")
    errors.extend(_validate_compiled_geometry(model_validation))

    expected_conditions = {value["condition"] for value in geometry_conditions()}
    expected_pairs = {
        (protocol, task, seed)
        for protocol in PROTOCOLS
        for task in ((PLACE_TASK,) if protocol == "placing_release" else TASKS)
        for seed in SEEDS
    }
    pairs: dict[
        tuple[str, str, int], dict[str, tuple[dict[str, Any], dict[str, Any]]]
    ] = {}
    for result in results:
        protocol = str(result["protocol"])
        task = str(result["task"])
        seed = int(result["seed"])
        key = (protocol, task, seed)
        trial = json.loads(
            Path(result["artifacts"]["trial"]).read_text(encoding="utf-8")
        )
        pairs.setdefault(key, {})[str(result["setting"]["condition"])] = (result, trial)
        allowed_tasks = {PLACE_TASK} if protocol == "placing_release" else set(TASKS)
        if protocol not in PROTOCOLS or task not in allowed_tasks:
            errors.append(f"{key}: unexpected protocol/task")
        if result.get("setting") != trial.get("setting"):
            errors.append(f"{key}: result/trial setting mismatch")
        effective = trial["overrides"]["effective"]
        simulation = effective["simulation"]
        if simulation["cone"] != "pyramidal" or not math.isclose(
            float(simulation["impratio"]), 1.0
        ):
            errors.append(f"{key}: cone or impratio changed")
        if trial["overrides"].get("changed_invariant_hashes"):
            errors.append(f"{key}: runtime override changed an invariant")

    for key, conditions in sorted(pairs.items()):
        if set(conditions) != expected_conditions:
            errors.append(f"{key}: expected A/B, got {sorted(conditions)}")
            continue
        a_result, a_trial = conditions["A"]
        b_result, b_trial = conditions["B"]
        runtime_model = manifest["runtime_model"]
        if (
            a_trial["paired_initial_state"]["state_sha256"]
            != b_trial["paired_initial_state"]["state_sha256"]
        ):
            errors.append(f"{key}: initial-state hash mismatch")
        if _manifest_hash(a_result, a_trial) != _manifest_hash(b_result, b_trial):
            errors.append(f"{key}: action-manifest hash mismatch")
        if a_trial["model_sha256"] == b_trial["model_sha256"]:
            errors.append(f"{key}: A/B model hashes unexpectedly match")
        if a_trial["model_sha256"] != runtime_model["production_model_sha256"]:
            errors.append(f"{key}: A hash differs from production model")
        if b_trial["model_sha256"] != runtime_model["diagnostic_model_sha256"]:
            errors.append(f"{key}: B hash differs from diagnostic model")
        if (
            a_trial["production_model_file_modified"]
            or b_trial["production_model_file_modified"]
        ):
            errors.append(f"{key}: production model was marked modified")
        if _fixed_mechanics_signature(
            a_trial["overrides"]["effective"]
        ) != _fixed_mechanics_signature(b_trial["overrides"]["effective"]):
            errors.append(f"{key}: fixed mechanics differ")
        errors.extend(
            f"{key}: {value}"
            for value in _validate_pad_topology(
                a_trial["overrides"]["effective"], b_trial["overrides"]["effective"]
            )
        )
        if (
            b_trial["setting"].get("geometry_variant")
            != "split_pad_two_zone_same_envelope"
        ):
            errors.append(f"{key}: B geometry variant label mismatch")

    if set(pairs) != expected_pairs:
        errors.append("pair keys differ from the fixed protocol/task/seed matrix")
    expected_protocol_counts = {
        "suspended_grasp": 18,
        "pushing": 18,
        "placing_release": 6,
    }
    protocol_counts = {
        protocol: sum(str(result["protocol"]) == protocol for result in results)
        for protocol in PROTOCOLS
    }
    if protocol_counts != expected_protocol_counts:
        errors.append(
            f"protocol trial counts differ: expected={expected_protocol_counts}, actual={protocol_counts}"
        )
    if len(results) != 42:
        errors.append(f"expected 42 trials, got {len(results)}")
    return {
        "status": "passed" if not errors else "failed",
        "trial_count": len(results),
        "pair_count": len(pairs),
        "protocol_trial_counts": protocol_counts,
        "errors": errors,
    }


def _condition_summary(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        for condition in ("A", "B"):
            subset = [
                row
                for row in summaries
                if row["protocol"] == protocol and row["condition"] == condition
            ]
            value: dict[str, Any] = {
                "protocol": protocol,
                "condition": condition,
                "trial_count": len(subset),
                "stable_hold_rate": (
                    statistics.mean(bool(row["stable_hold"]) for row in subset)
                    if protocol == "suspended_grasp"
                    else None
                ),
                "contact_loss_count": sum(bool(row["contact_loss"]) for row in subset),
            }
            for metric in NUMERIC_METRICS:
                values = [
                    float(row[metric]) for row in subset if row.get(metric) is not None
                ]
                value[f"mean_{metric}"] = statistics.mean(values) if values else None
                value[f"max_{metric}"] = max(values, default=None)
            output.append(value)
    return output


def _paired_deltas(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in summaries:
        key = (str(row["protocol"]), str(row["task"]), int(row["seed"]))
        grouped.setdefault(key, {})[str(row["condition"])] = row
    output: list[dict[str, Any]] = []
    for (protocol, task, seed), pair in sorted(grouped.items()):
        a = pair["A"]
        b = pair["B"]
        value: dict[str, Any] = {
            "protocol": protocol,
            "task": task,
            "seed": seed,
            "delta_definition": "B_minus_A",
            "stable_hold": (
                None
                if a["stable_hold"] is None or b["stable_hold"] is None
                else int(bool(b["stable_hold"])) - int(bool(a["stable_hold"]))
            ),
            "contact_loss": int(bool(b["contact_loss"])) - int(bool(a["contact_loss"])),
        }
        for metric in NUMERIC_METRICS:
            value[metric] = (
                None
                if a.get(metric) is None or b.get(metric) is None
                else float(b[metric]) - float(a[metric])
            )
        output.append(value)
    return output


def main() -> None:
    args = _parser().parse_args()
    suite_root = args.suite_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing analysis directory: {output}")
    payload = json.loads((suite_root / "results.json").read_text(encoding="utf-8"))
    if (
        payload.get("status") != "complete"
        or payload.get("suite") != "split_pad_geometry"
    ):
        raise ValueError("Suite results are not complete split-pad geometry results")

    output.mkdir(parents=True, exist_ok=False)
    results = payload["trials"]
    validation = validate_pairs(suite_root, results)
    _write_json(output / "validation.json", validation)
    if validation["status"] != "passed":
        raise RuntimeError(f"Paired-design validation failed: {validation['errors']}")

    summaries = [analyze_trial(result) for result in results]
    warnings = max(row["simulation_warning_count"] for row in summaries)
    non_finite = sum(not row["simulation_finite"] for row in summaries)
    if warnings or non_finite:
        validation["status"] = "failed"
        validation["errors"].append(
            f"runtime health failure: warnings={warnings}, non_finite={non_finite}"
        )
        _write_json(output / "validation.json", validation)
        raise RuntimeError(f"Runtime validation failed: {validation['errors']}")

    summary = _condition_summary(summaries)
    paired = _paired_deltas(summaries)
    _write_json(output / "trial_metrics.json", summaries)
    _write_json(output / "condition_protocol_summary.json", summary)
    _write_json(output / "paired_deltas.json", paired)
    _write_json(
        output / "metric_definitions.json",
        {
            "relative_grasp_slip": "maximum TCP-minus-object drift during suspended scripted hold",
            "stable_hold_rate": "fraction with at least 95% bilateral hold contact, no contact-loss event, and downward slip at most 0.002 m",
            "contact_loss": "PhysicsTraceRecorder contact_loss event after bilateral target contact",
            "manifold_symmetry": "per-sample 1 - abs(left_count-right_count)/(left_count+right_count), summarized only during bilateral target contact",
            "contact_positions": "first bilateral left/right world contact positions; all per-sample positions remain in physics_trace.jsonl",
            "normal_tangential_force": "sum across target contacts",
            "penetration": "maximum and positive-duration of target contacts",
            "pushing_displacement": "target displacement from immediately before PUSH through settle",
            "release_latency": "opening-target onset to 0.1 s sustained zero target-gripper contact",
        },
    )
    with (output / "trial_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(
        json.dumps(
            {"validation": validation, "summary": summary}, indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
