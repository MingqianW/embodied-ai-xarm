#!/usr/bin/env python3
"""Validate and summarize Menagerie grip force versus object width."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_mujoco.scripts.run_grip_force_vs_width import (  # noqa: E402
    MAX_PENETRATION_M,
    MIN_BILATERAL_FRACTION,
    MIN_EXACT_COUNT_SYMMETRY_FRACTION,
    MIN_NORMAL_ALIGNMENT,
    WIDTHS_MM,
    evaluate_trial_validity,
)


SCALAR_FIELDS = {
    "commanded_gripper_raw": ("command_raw",),
    "menagerie_ctrl": ("menagerie_ctrl",),
    "actuator_force": ("actuator", "force_actuator_space"),
    "actuator_force_magnitude": ("actuator", "force_actuator_space"),
    "actuator_length_rad": ("actuator", "tendon_length_rad"),
    "actuator_velocity_radps": ("actuator", "tendon_velocity_radps"),
    "equilibrium_length_error_rad": ("actuator", "equilibrium_length_error_rad"),
    "affine_formula_force": ("actuator", "affine_formula_force_actuator_space"),
    "actuator_moment_left": ("actuator", "moment_left"),
    "actuator_moment_right": ("actuator", "moment_right"),
    "actuator_qfrc_left": ("actuator", "qfrc_left"),
    "actuator_qfrc_right": ("actuator", "qfrc_right"),
    "finger_left_qpos_rad": ("fingers", "left_driver_qpos_rad"),
    "finger_right_qpos_rad": ("fingers", "right_driver_qpos_rad"),
    "finger_left_qvel_radps": ("fingers", "left_driver_qvel_radps"),
    "finger_right_qvel_radps": ("fingers", "right_driver_qvel_radps"),
    "realized_opening_m": ("fingers", "realized_opening_m"),
    "normal_force_left_n": ("contacts", "normal_left_n"),
    "normal_force_right_n": ("contacts", "normal_right_n"),
    "normal_force_total_n": ("contacts", "normal_total_n"),
    "normal_force_symmetry": ("contacts", "normal_force_symmetry"),
    "contact_normal_alignment_min": ("contacts", "normal_alignment_min"),
    "tangential_force_left_n": ("contacts", "tangential_left_n"),
    "tangential_force_right_n": ("contacts", "tangential_right_n"),
    "tangential_force_total_n": ("contacts", "tangential_total_n"),
    "contact_count_left": ("contacts", "left_count"),
    "contact_count_right": ("contacts", "right_count"),
    "contact_count_total": ("contacts", "total_count"),
    "bilateral_contact_fraction": ("contacts", "bilateral"),
    "exact_count_symmetry_fraction": ("contacts", "exact_count_symmetry"),
    "unintended_fixture_contact_count": (
        "contacts",
        "unintended_fixture_contact_count",
    ),
    "penetration_left_m": ("contacts", "penetration_left_m"),
    "penetration_right_m": ("contacts", "penetration_right_m"),
    "penetration_m": ("contacts", "penetration_max_m"),
    "equality_qfrc_left": ("constraints", "equality_qfrc_left"),
    "equality_qfrc_right": ("constraints", "equality_qfrc_right"),
    "total_constraint_qfrc_left": (
        "constraints",
        "total_qfrc_constraint_left",
    ),
    "total_constraint_qfrc_right": (
        "constraints",
        "total_qfrc_constraint_right",
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--revalidate-existing-traces",
        action="store_true",
        help=(
            "Permit analysis of a complete failed-run trace matrix only after "
            "independently reapplying the current validity gates."
        ),
    )
    parser.add_argument(
        "--report-invalid-contact-outcomes",
        action="store_true",
        help=(
            "Write an explicitly INVALID_CONTACT_PRECONDITION baseline report "
            "when a complete matrix safely records a no-contact outcome."
        ),
    )
    return parser


def _nested(row: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = row
    for key in path:
        value = value[key]
    return float(value)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = fraction * (len(ordered) - 1)
    low = int(math.floor(index))
    high = int(math.ceil(index))
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_values(
    values: Iterable[float], *, magnitude: bool = False
) -> dict[str, float]:
    finite = [abs(float(value)) if magnitude else float(value) for value in values]
    if not finite or not all(math.isfinite(value) for value in finite):
        raise ValueError("Steady-state metric is empty or non-finite")
    return {
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite),
        "min": min(finite),
        "p05": _percentile(finite, 0.05),
        "median": statistics.median(finite),
        "p95": _percentile(finite, 0.95),
        "max": max(finite),
    }


def summarize_rows(width_mm: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"No samples for width {width_mm} mm")
    if {int(row["width_mm"]) for row in rows} != {width_mm}:
        raise ValueError(f"Mixed width labels in {width_mm} mm trial")
    summary: dict[str, Any] = {
        "width_mm": width_mm,
        "sample_count": len(rows),
        "window_start_s": float(rows[0]["sim_time_s"]),
        "window_end_s": float(rows[-1]["sim_time_s"]),
        "metrics": {},
    }
    for name, path in SCALAR_FIELDS.items():
        summary["metrics"][name] = summarize_values(
            (_nested(row, path) for row in rows),
            magnitude=name == "actuator_force_magnitude",
        )
    equality_rows = [
        abs(float(value))
        for row in rows
        for value in row["constraints"]["equality_row_force"]
    ]
    summary["metrics"]["equality_row_force_magnitude"] = summarize_values(equality_rows)
    return summary


def _read_trial_rows(suite_root: Path, trial: dict[str, Any]) -> list[dict[str, Any]]:
    trace_path = Path(trial["trace"]).resolve()
    if trace_path != suite_root and suite_root not in trace_path.parents:
        raise ValueError(f"Trace is outside suite root: {trace_path}")
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 500:
        raise ValueError(
            f"Expected 500 steady-state samples for width {trial['width_mm']}; "
            f"found {len(rows)}"
        )
    return rows


def _authorize_analysis(
    suite_root: Path,
    results: dict[str, Any],
    *,
    revalidate_existing_traces: bool,
    report_invalid_contact_outcomes: bool = False,
) -> tuple[list[tuple[dict[str, Any], list[dict[str, Any]]]], dict[str, Any]]:
    trials = results.get("trials", [])
    if tuple(int(row["width_mm"]) for row in trials) != WIDTHS_MM:
        raise ValueError("Width diagnostic trial order/matrix is incomplete")
    trial_rows = [(trial, _read_trial_rows(suite_root, trial)) for trial in trials]
    runtime_authorized = (
        results.get("status") == "complete"
        and results.get("force_metrics_authorized") is True
        and results.get("all_widths_valid") is True
        and all(trial["validity"]["passed"] for trial in trials)
    )
    if runtime_authorized:
        return trial_rows, {
            "authorization": "runtime_results",
            "current_maximum_penetration_m": MAX_PENETRATION_M,
        }
    if report_invalid_contact_outcomes:
        if (
            results.get("status") != "failed"
            or results.get("force_metrics_authorized") is not False
        ):
            raise ValueError("Invalid-outcome reporting requires a fail-closed run")
        allowed_outcome_gates = {
            "bilateral_contact_fraction",
            "minimum_contact_normal_axis_alignment",
        }
        failed_gates = {
            gate
            for trial in trials
            for gate in trial["validity"].get("failed_gates", [])
        }
        if not failed_gates or failed_gates - allowed_outcome_gates:
            raise ValueError(
                "Invalid-outcome report only permits no-contact gates; found "
                f"{sorted(failed_gates)}"
            )
        return trial_rows, {
            "authorization": "invalid_contact_outcome_report_only",
            "force_width_conclusion_authorized": False,
            "original_status": results.get("status"),
            "original_error": results.get("error"),
            "failed_gates": sorted(failed_gates),
            "invalid_widths": [
                int(trial["width_mm"])
                for trial in trials
                if not trial["validity"]["passed"]
            ],
            "current_maximum_penetration_m": MAX_PENETRATION_M,
        }
    if not revalidate_existing_traces:
        raise ValueError("Width diagnostic did not pass the fail-closed gates")
    if (
        results.get("status") != "failed"
        or results.get("force_metrics_authorized") is not False
    ):
        raise ValueError("Corrected-gate revalidation requires a fail-closed run")
    old_failed_gates = {
        gate for trial in trials for gate in trial["validity"].get("failed_gates", [])
    }
    if not old_failed_gates or old_failed_gates - {"maximum_penetration_m"}:
        raise ValueError(
            "Original run failed gates other than maximum_penetration_m: "
            f"{sorted(old_failed_gates)}"
        )
    corrected = [
        {
            "width_mm": int(trial["width_mm"]),
            "validity": evaluate_trial_validity(rows, trial["fixture_placement"]),
        }
        for trial, rows in trial_rows
    ]
    rejected = [row["width_mm"] for row in corrected if not row["validity"]["passed"]]
    if rejected:
        raise ValueError(f"Current validity gates still reject widths: {rejected}")
    return trial_rows, {
        "authorization": "corrected_gate_revalidation",
        "original_status": results.get("status"),
        "original_error": results.get("error"),
        "original_failed_gates": sorted(old_failed_gates),
        "current_maximum_penetration_m": MAX_PENETRATION_M,
        "revalidated_trials": corrected,
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


def acceptance(summary: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(summary, key=lambda row: row["width_mm"])
    widths = [float(row["width_mm"]) for row in ordered]
    force = [row["metrics"]["normal_force_total_n"]["mean"] for row in ordered]
    actuator_force = [
        row["metrics"]["actuator_force_magnitude"]["mean"] for row in ordered
    ]
    moments = [row["metrics"]["actuator_moment_left"]["mean"] for row in ordered]
    nondecreasing_steps = sum(
        later >= earlier for earlier, later in zip(force, force[1:])
    )
    width_force_r = _pearson(widths, force)
    actuator_normal_r = _pearson(actuator_force, force)
    moment_span = max(moments) - min(moments)
    moment_scale = max(abs(statistics.fmean(moments)), 1e-12)
    narrow_to_wide_ratio = force[0] / force[-1] if force[-1] > 0.0 else math.inf
    force_mean = statistics.fmean(force)
    force_cv = statistics.pstdev(force) / force_mean if force_mean else math.inf
    bilateral_min = min(
        row["metrics"]["bilateral_contact_fraction"]["mean"] for row in ordered
    )
    count_symmetry_min = min(
        row["metrics"]["exact_count_symmetry_fraction"]["mean"] for row in ordered
    )
    maximum_penetration = max(row["metrics"]["penetration_m"]["max"] for row in ordered)
    maximum_unintended_contacts = max(
        row["metrics"]["unintended_fixture_contact_count"]["max"] for row in ordered
    )
    minimum_normal_alignment = min(
        row["metrics"]["contact_normal_alignment_min"]["min"] for row in ordered
    )
    valid_contact_precondition = (
        bilateral_min >= MIN_BILATERAL_FRACTION
        and count_symmetry_min >= MIN_EXACT_COUNT_SYMMETRY_FRACTION
        and maximum_penetration <= MAX_PENETRATION_M
        and maximum_unintended_contacts == 0
        and minimum_normal_alignment >= MIN_NORMAL_ALIGNMENT
    )
    return {
        "criterion": "report untuned Menagerie force-width behavior only after every contact gate passes",
        "classification": (
            "INVALID_CONTACT_PRECONDITION"
            if not valid_contact_precondition
            else "VALID_MENAGERIE_FORCE_WIDTH_RESULT"
        ),
        "valid_contact_precondition": valid_contact_precondition,
        "observed": {
            "nondecreasing_adjacent_steps": nondecreasing_steps,
            "pearson_width_total_normal": width_force_r,
            "normal_force_25_to_55_ratio": narrow_to_wide_ratio,
            "normal_force_max_to_min_ratio": (
                max(force) / min(force) if min(force) > 0.0 else math.inf
            ),
            "normal_force_coefficient_of_variation": force_cv,
            "minimum_bilateral_contact_fraction": bilateral_min,
            "minimum_exact_count_symmetry_fraction": count_symmetry_min,
            "maximum_penetration_m": maximum_penetration,
            "maximum_unintended_fixture_contact_count": maximum_unintended_contacts,
            "minimum_contact_normal_axis_alignment": minimum_normal_alignment,
            "pearson_actuator_force_total_normal": actuator_normal_r,
            "actuator_moment_relative_span": moment_span / moment_scale,
        },
    }


def _flat_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in summary:
        row: dict[str, Any] = {
            "width_mm": value["width_mm"],
            "sample_count": value["sample_count"],
        }
        for metric, stats in value["metrics"].items():
            for statistic, number in stats.items():
                row[f"{metric}_{statistic}"] = number
        row["penetration_mm_mean"] = 1000.0 * value["metrics"]["penetration_m"]["mean"]
        row["penetration_mm_max"] = 1000.0 * value["metrics"]["penetration_m"]["max"]
        rows.append(row)
    return rows


def _force_chain_rows(summary: list[dict[str, Any]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for value in sorted(summary, key=lambda item: item["width_mm"]):
        metric = value["metrics"]
        actuator = metric["actuator_force_magnitude"]["mean"]
        equality = metric["equality_row_force_magnitude"]["mean"]
        right_normal = metric["normal_force_right_n"]["mean"]
        total_normal = metric["normal_force_total_n"]["mean"]
        rows.append(
            {
                "width_mm": float(value["width_mm"]),
                "unloaded_equilibrium_minus_tendon_length_rad": metric[
                    "equilibrium_length_error_rad"
                ]["mean"],
                "actuator_force_magnitude": actuator,
                "equality_force_magnitude": equality,
                "normal_force_left_n": metric["normal_force_left_n"]["mean"],
                "normal_force_right_n": right_normal,
                "normal_force_total_n": total_normal,
                "total_normal_to_actuator_ratio": (
                    total_normal / actuator if actuator > 0.0 else math.nan
                ),
                "right_normal_to_equality_ratio": (
                    right_normal / equality if equality > 0.0 else math.nan
                ),
            }
        )
    return rows


def _write_force_chain_plot(rows: list[dict[str, float]], target: Path) -> None:
    import matplotlib.pyplot as plt

    width = [row["width_mm"] for row in rows]
    series = (
        (
            "equilibrium - tendon length",
            "rad",
            "unloaded_equilibrium_minus_tendon_length_rad",
        ),
        (
            "actuator-space force magnitude",
            "actuator space",
            "actuator_force_magnitude",
        ),
        ("equality force magnitude", "constraint space", "equality_force_magnitude"),
        ("bilateral normal force", "N", "normal_force_total_n"),
    )
    figure, axes = plt.subplots(4, 1, figsize=(7.0, 10.0), sharex=True)
    for axis, (title, unit, field) in zip(axes, series, strict=True):
        axis.plot(width, [row[field] for row in rows], marker="o", linewidth=1.8)
        axis.set_ylabel(unit)
        axis.set_title(title)
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel("fixture face separation / grasp width (mm)")
    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)


def _markdown(summary: list[dict[str, Any]], result: dict[str, Any]) -> str:
    precondition = (
        "All rows passed the fail-closed runtime contact gates before force metrics were computed."
        if result["valid_contact_precondition"]
        else (
            "This is an outcome report, not an authorized force-width conclusion: "
            "at least one width did not establish bilateral contact."
        )
    )
    lines = [
        "# Grip force versus width: computed steady-state summary",
        "",
        precondition,
        "",
        "| width (mm) | left N | right N | total N | actuator-space |q| | equality |q| | driver L/R (rad) | eq-length error (rad) | penetration mean/max (mm) | bilateral | count symmetry |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda item: item["width_mm"]):
        metric = row["metrics"]
        lines.append(
            "| {width} | {left:.6f} | {right:.6f} | {total:.6f} | {actuator:.6f} | "
            "{equality:.6f} | {ql:.3f}/{qr:.3f} | {error:.3f} | "
            "{pen_mean:.4f}/{pen_max:.4f} | {bilateral:.3f} | {symmetry:.3f} |".format(
                width=row["width_mm"],
                left=metric["normal_force_left_n"]["mean"],
                right=metric["normal_force_right_n"]["mean"],
                total=metric["normal_force_total_n"]["mean"],
                actuator=metric["actuator_force_magnitude"]["mean"],
                equality=metric["equality_row_force_magnitude"]["mean"],
                ql=metric["finger_left_qpos_rad"]["mean"],
                qr=metric["finger_right_qpos_rad"]["mean"],
                error=metric["equilibrium_length_error_rad"]["mean"],
                pen_mean=1000.0 * metric["penetration_m"]["mean"],
                pen_max=1000.0 * metric["penetration_m"]["max"],
                bilateral=metric["bilateral_contact_fraction"]["mean"],
                symmetry=metric["exact_count_symmetry_fraction"]["mean"],
            )
        )
    observed = result["observed"]
    lines.extend(
        [
            "",
            "## Pre-registered acceptance result",
            "",
            f"- classification: `{result['classification']}`",
            f"- valid contact precondition: `{result['valid_contact_precondition']}`",
            f"- adjacent nondecreasing force steps: {observed['nondecreasing_adjacent_steps']}/6",
            f"- Pearson(width, total normal): {observed['pearson_width_total_normal']:.6f}",
            f"- 25/55 mm force ratio: {observed['normal_force_25_to_55_ratio']:.6f}",
            f"- minimum bilateral fraction: {observed['minimum_bilateral_contact_fraction']:.6f}",
            f"- minimum exact-count symmetry: {observed['minimum_exact_count_symmetry_fraction']:.6f}",
            f"- maximum penetration: {1000.0 * observed['maximum_penetration_m']:.6f} mm",
            f"- maximum unintended fixture contacts: {observed['maximum_unintended_fixture_contact_count']:.0f}",
            f"- minimum contact-normal axis alignment: {observed['minimum_contact_normal_axis_alignment']:.6f}",
            f"- force max/min ratio: {observed['normal_force_max_to_min_ratio']:.6f}",
            f"- force coefficient of variation: {observed['normal_force_coefficient_of_variation']:.6f}",
            f"- Pearson(actuator-space force, total normal): {observed['pearson_actuator_force_total_normal']:.6f}",
            f"- actuator-moment relative span: {observed['actuator_moment_relative_span']:.6g}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parser().parse_args()
    suite_root = args.suite_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if suite_root not in output.parents:
        raise ValueError("--output-dir must be inside --suite-root")
    results = json.loads((suite_root / "results.json").read_text(encoding="utf-8"))
    trial_rows, provenance = _authorize_analysis(
        suite_root,
        results,
        revalidate_existing_traces=args.revalidate_existing_traces,
        report_invalid_contact_outcomes=args.report_invalid_contact_outcomes,
    )

    summaries: list[dict[str, Any]] = []
    for trial, rows in trial_rows:
        summaries.append(summarize_rows(int(trial["width_mm"]), rows))
    result = acceptance(summaries)
    if (
        not result["valid_contact_precondition"]
        and not args.report_invalid_contact_outcomes
    ):
        raise RuntimeError("Independent analysis rejected the runtime validity result")

    output.mkdir(parents=True, exist_ok=False)
    (output / "steady_state_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "acceptance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "analysis_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    flat = _flat_rows(summaries)
    with (output / "steady_state_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    force_chain = _force_chain_rows(summaries)
    with (output / "force_chain.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(force_chain[0]))
        writer.writeheader()
        writer.writerows(force_chain)
    (output / "summary.md").write_text(_markdown(summaries, result), encoding="utf-8")
    _write_force_chain_plot(force_chain, output / "force_chain_vs_width.png")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
