#!/usr/bin/env python3
"""Aggregate the isolated Menagerie actuator-force-range tuning experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


FORCE_LIMITS = (1.0, 1.5, 2.0, 3.0, 5.0)
TASKS = (
    "smallest_block",
    "largest_block",
    "red_block",
    "blue_block",
    "red_pepper",
)
MAX_PENETRATION_M = 0.001
MIN_BILATERAL_FRACTION = 0.95
MIN_COUNT_SYMMETRY_FRACTION = 0.95
MAX_FORCE_ASYMMETRY_FRACTION = 0.10
DIAGNOSTIC_MAX_BILATERAL_NORMAL_N = 50.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--grasp-analysis-name",
        default="analysis",
        help="Analysis directory under grasp_slip (recovery runs may use analysis_v2).",
    )
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _force_label(value: float) -> str:
    return f"pm{value:g}"


def _setting_name(value: float) -> str:
    return f"menagerie_forcerange_pm{value:g}"


def _mean(row: dict[str, Any], metric: str) -> float:
    return float(row["metrics"][metric]["mean"])


def _maximum(row: dict[str, Any], metric: str) -> float:
    return float(row["metrics"][metric]["max"])


def _finite(values: list[float | None]) -> list[float]:
    return [
        float(value) for value in values if value is not None and math.isfinite(value)
    ]


def analyze(
    run_root: Path,
    *,
    grasp_analysis_name: str = "analysis",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not grasp_analysis_name or Path(grasp_analysis_name).name != grasp_analysis_name:
        raise ValueError("grasp_analysis_name must be one directory name")
    grasp_rows = _read_json(
        run_root / "grasp_slip" / grasp_analysis_name / "summary.json"
    )
    expected_settings = {_setting_name(value) for value in FORCE_LIMITS}
    if {row["setting"] for row in grasp_rows} != expected_settings:
        raise ValueError("Grasp summary does not contain the exact force-range matrix")
    if len(grasp_rows) != len(FORCE_LIMITS) * len(TASKS):
        raise ValueError(f"Expected 25 grasp rows, found {len(grasp_rows)}")

    candidates: list[dict[str, Any]] = []
    width_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    for limit in FORCE_LIMITS:
        label = _force_label(limit)
        force_root = run_root / "force_width" / label
        force_summary = _read_json(
            force_root / "analysis" / "steady_state_summary.json"
        )
        force_acceptance = _read_json(force_root / "analysis" / "acceptance.json")
        if [int(row["width_mm"]) for row in force_summary] != [
            25,
            30,
            35,
            40,
            45,
            50,
            55,
        ]:
            raise ValueError(f"Incomplete width matrix for force limit {limit:g}")
        selected = [row for row in grasp_rows if row["setting"] == _setting_name(limit)]
        if {row["task"] for row in selected} != set(TASKS):
            raise ValueError(f"Incomplete object matrix for force limit {limit:g}")

        for row in force_summary:
            left_normal = _mean(row, "normal_force_left_n")
            right_normal = _mean(row, "normal_force_right_n")
            total_normal = _mean(row, "normal_force_total_n")
            width_rows.append(
                {
                    "force_limit_actuator_space": limit,
                    "width_mm": int(row["width_mm"]),
                    "bilateral_contact_fraction": _mean(
                        row, "bilateral_contact_fraction"
                    ),
                    "exact_count_symmetry_fraction": _mean(
                        row, "exact_count_symmetry_fraction"
                    ),
                    "actuator_force_actuator_space": _mean(
                        row, "actuator_force_magnitude"
                    ),
                    "normal_force_left_n": left_normal,
                    "normal_force_right_n": right_normal,
                    "normal_force_total_n": total_normal,
                    "force_asymmetry_fraction": (
                        abs(left_normal - right_normal) / total_normal
                        if total_normal > 0.0
                        else None
                    ),
                    "tangential_force_total_n": _mean(row, "tangential_force_total_n"),
                    "penetration_mean_mm": 1000.0 * _mean(row, "penetration_m"),
                    "penetration_max_mm": 1000.0 * _maximum(row, "penetration_m"),
                    "realized_opening_mm": 1000.0 * _mean(row, "realized_opening_m"),
                }
            )

        for row in selected:
            object_rows.append(
                {
                    "force_limit_actuator_space": limit,
                    "task": row["task"],
                    "stable_2mm": bool(row["mechanically_stable_2mm"]),
                    "failure": row["diagnostic_failure_label"],
                    "maximum_slip_mm": (
                        None
                        if row["maximum_downward_slip_m"] is None
                        else 1000.0 * float(row["maximum_downward_slip_m"])
                    ),
                    "bilateral_fraction": float(row["hold_bilateral_contact_fraction"]),
                    "count_symmetry_fraction": float(
                        row["hold_exact_contact_count_symmetry_fraction"]
                    ),
                    "force_asymmetry_fraction": row[
                        "mean_bilateral_force_asymmetry_fraction"
                    ],
                    "mean_bilateral_normal_n": row["mean_bilateral_normal_force_sum_n"],
                    "max_bilateral_normal_n": row[
                        "maximum_bilateral_normal_force_sum_n"
                    ],
                    "max_penetration_mm": (
                        None
                        if row["maximum_target_penetration_m"] is None
                        else 1000.0 * float(row["maximum_target_penetration_m"])
                    ),
                    "max_actuator_force_actuator_space": float(
                        row["maximum_abs_actuator_force_actuator_space"]
                    ),
                    "force_saturation_fraction": float(
                        row["force_saturation_fraction"]
                    ),
                    "warning_count": int(row["simulation_warning_count"]),
                    "finite": bool(row["simulation_finite"]),
                }
            )

        candidate_objects = [
            row for row in object_rows if row["force_limit_actuator_space"] == limit
        ]
        candidate_widths = [
            row for row in width_rows if row["force_limit_actuator_space"] == limit
        ]
        max_fixture_force = max(row["normal_force_total_n"] for row in candidate_widths)
        max_grasp_force = max(
            _finite([row["max_bilateral_normal_n"] for row in candidate_objects]),
            default=0.0,
        )
        max_penetration_mm = max(
            max(row["penetration_max_mm"] for row in candidate_widths),
            max(
                _finite([row["max_penetration_mm"] for row in candidate_objects]),
                default=0.0,
            ),
        )
        min_bilateral = min(row["bilateral_fraction"] for row in candidate_objects)
        min_symmetry = min(
            min(row["exact_count_symmetry_fraction"] for row in candidate_widths),
            min(row["count_symmetry_fraction"] for row in candidate_objects),
        )
        max_asymmetry_values = _finite(
            [row["force_asymmetry_fraction"] for row in candidate_widths]
            + [row["force_asymmetry_fraction"] for row in candidate_objects]
        )
        max_asymmetry = max(max_asymmetry_values) if max_asymmetry_values else None
        slip_values = _finite([row["maximum_slip_mm"] for row in candidate_objects])
        max_slip_mm = max(slip_values) if slip_values else None
        rejection_reasons: list[str] = []
        if not force_acceptance["valid_contact_precondition"]:
            rejection_reasons.append("invalid_force_width_contact")
        if any(not row["finite"] or row["warning_count"] for row in candidate_objects):
            rejection_reasons.append("numerical_instability_or_warning")
        if max_penetration_mm > 1000.0 * MAX_PENETRATION_M:
            rejection_reasons.append("penetration_above_1mm")
        if max(max_fixture_force, max_grasp_force) > DIAGNOSTIC_MAX_BILATERAL_NORMAL_N:
            rejection_reasons.append("bilateral_normal_above_50N_diagnostic_ceiling")
        if min_symmetry < MIN_COUNT_SYMMETRY_FRACTION or (
            max_asymmetry is not None and max_asymmetry > MAX_FORCE_ASYMMETRY_FRACTION
        ):
            rejection_reasons.append("asymmetric_contact")
        if any(row["failure"] == "CONTACT_LOSS" for row in candidate_objects):
            rejection_reasons.append("contact_loss_or_ejection")
        if sum(row["stable_2mm"] for row in candidate_objects) < len(TASKS):
            rejection_reasons.append("not_all_objects_stable_below_2mm")

        candidates.append(
            {
                "force_limit_actuator_space": limit,
                "width_contact_valid": bool(
                    force_acceptance["valid_contact_precondition"]
                ),
                "fixture_max_bilateral_normal_n": max_fixture_force,
                "grasp_max_bilateral_normal_n": max_grasp_force,
                "stable_object_count": sum(
                    row["stable_2mm"] for row in candidate_objects
                ),
                "contact_loss_count": sum(
                    row["failure"] == "CONTACT_LOSS" for row in candidate_objects
                ),
                "maximum_slip_mm": max_slip_mm,
                "minimum_bilateral_fraction": min_bilateral,
                "minimum_count_symmetry_fraction": min_symmetry,
                "maximum_force_asymmetry_fraction": max_asymmetry,
                "maximum_penetration_mm": max_penetration_mm,
                "maximum_actuator_force_actuator_space": max(
                    row["max_actuator_force_actuator_space"]
                    for row in candidate_objects
                ),
                "maximum_force_saturation_fraction": max(
                    row["force_saturation_fraction"] for row in candidate_objects
                ),
                "warnings": sum(row["warning_count"] for row in candidate_objects),
                "all_finite": all(row["finite"] for row in candidate_objects),
                "accepted": not rejection_reasons,
                "rejection_reasons": rejection_reasons,
            }
        )
    return candidates, width_rows, object_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(
    candidates: list[dict[str, Any]],
    width_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
) -> str:
    def number(value: float | None, digits: int = 3) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Menagerie gripper force-range tuning: phase 1",
        "",
        "Only the symmetric actuator-space `forcerange` differs across candidates. "
        "The 50 N bilateral-normal ceiling is a conservative diagnostic rejection "
        "gate, not a claimed real-gripper calibration.",
        "",
        "## Candidate summary",
        "",
        "| ±limit | width valid | stable objects | losses | max slip mm | fixture/grasp max normal N | max pen mm | accepted |",
        "|---:|:---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in candidates:
        lines.append(
            f"| {row['force_limit_actuator_space']:g} | {row['width_contact_valid']} | "
            f"{row['stable_object_count']}/5 | {row['contact_loss_count']} | "
            f"{number(row['maximum_slip_mm'])} | "
            f"{row['fixture_max_bilateral_normal_n']:.3f}/"
            f"{row['grasp_max_bilateral_normal_n']:.3f} | "
            f"{row['maximum_penetration_mm']:.3f} | {row['accepted']} |"
        )
    lines.extend(
        [
            "",
            "## Force versus width",
            "",
            "| ±limit | width mm | bilateral | symmetry | actuator-space force | L/R/total normal N | opening mm | pen max mm |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in width_rows:
        lines.append(
            f"| {row['force_limit_actuator_space']:g} | {row['width_mm']} | "
            f"{row['bilateral_contact_fraction']:.3f} | "
            f"{row['exact_count_symmetry_fraction']:.3f} | "
            f"{row['actuator_force_actuator_space']:.3f} | "
            f"{row['normal_force_left_n']:.3f}/{row['normal_force_right_n']:.3f}/"
            f"{row['normal_force_total_n']:.3f} | {row['realized_opening_mm']:.3f} | "
            f"{row['penetration_max_mm']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Grasp/lift/two-second hold",
            "",
            "| ±limit | object | result | max slip mm | bilateral | max normal N | max pen mm |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in object_rows:
        lines.append(
            f"| {row['force_limit_actuator_space']:g} | {row['task']} | "
            f"{row['failure']} | "
            f"{row['maximum_slip_mm'] if row['maximum_slip_mm'] is not None else 'n/a'} | "
            f"{row['bilateral_fraction']:.3f} | "
            f"{row['max_bilateral_normal_n'] if row['max_bilateral_normal_n'] is not None else 'n/a'} | "
            f"{row['max_penetration_mm'] if row['max_penetration_mm'] is not None else 'n/a'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parser().parse_args()
    run_root = args.run_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing output directory: {output}")
    candidates, width_rows, object_rows = analyze(
        run_root,
        grasp_analysis_name=args.grasp_analysis_name,
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "parameter_results.json").write_text(
        json.dumps(
            {
                "schema_version": "xarm_menagerie_forcerange_tuning_v1",
                "force_limits_actuator_space": list(FORCE_LIMITS),
                "fixed_policy_rates_raw_per_s": {"closing": 244.0, "opening": 220.0},
                "diagnostic_bilateral_normal_ceiling_n": DIAGNOSTIC_MAX_BILATERAL_NORMAL_N,
                "candidates": candidates,
                "force_width": width_rows,
                "grasp_hold": object_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "candidate_summary.csv", candidates)
    _write_csv(output / "force_width.csv", width_rows)
    _write_csv(output / "grasp_hold.csv", object_rows)
    (output / "report.md").write_text(
        _markdown(candidates, width_rows, object_rows), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "candidate_count": len(candidates),
                "accepted_count": sum(row["accepted"] for row in candidates),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
