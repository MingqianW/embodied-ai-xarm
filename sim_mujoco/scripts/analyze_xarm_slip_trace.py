#!/usr/bin/env python3
"""Summarize and plot one or more xArm physics-cadence slip traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="Named slip trace; repeat for c5/c2/c1 comparisons.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _parse_trace_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Trace must use LABEL=CSV syntax: {value!r}")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label) or not path.is_file():
        raise ValueError(f"Invalid trace argument: {value!r}")
    return label, path


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _load(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Trace has no physics samples: {path}")
    return rows


def _first(rows: list[dict[str, str]], predicate) -> dict[str, str] | None:
    return next((row for row in rows if predicate(row)), None)


def _relative_offset(row: dict[str, str]) -> tuple[float, float, float]:
    return (
        float(row["relative_x_m"]),
        float(row["relative_y_m"]),
        float(row["relative_z_m"]),
    )


def _reference_summary(
    rows: list[dict[str, str]], reference: dict[str, str] | None
) -> dict[str, Any] | None:
    """Measure relative motion using an explicit, auditable reference row."""

    if reference is None:
        return None
    reference_offset = _relative_offset(reference)
    downward_slips: list[float] = []
    drifts: list[float] = []
    first_slip_1mm_time_s: float | None = None
    for row in rows[rows.index(reference) :]:
        offset = _relative_offset(row)
        delta = tuple(value - origin for value, origin in zip(offset, reference_offset, strict=True))
        downward_slip = max(0.0, delta[2])
        drift = math.sqrt(sum(value * value for value in delta))
        downward_slips.append(downward_slip)
        drifts.append(drift)
        if first_slip_1mm_time_s is None and downward_slip >= 0.001:
            first_slip_1mm_time_s = float(row["sim_time_s"])
    return {
        "reference_time_s": float(reference["sim_time_s"]),
        "reference_policy_step": int(reference["policy_step"]),
        "reference_executed_action_index": int(reference["executed_action_index"]),
        "reference_relative_offset_m": list(reference_offset),
        "maximum_relative_downward_slip_m": max(downward_slips),
        "final_relative_downward_slip_m": downward_slips[-1],
        "maximum_relative_3d_drift_m": max(drifts),
        "final_relative_3d_drift_m": drifts[-1],
        "first_relative_downward_slip_1mm_time_s": first_slip_1mm_time_s,
    }


def _contact_intervals(
    rows: list[dict[str, str]], predicate
) -> list[list[dict[str, str]]]:
    """Return contiguous sampled intervals for a Boolean contact predicate."""

    intervals: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    for row in rows:
        if predicate(row):
            current.append(row)
        elif current:
            intervals.append(current)
            current = []
    if current:
        intervals.append(current)
    return intervals


def _table_contact_interval_summaries(
    rows: list[dict[str, str]], first_target_contact: dict[str, str] | None
) -> list[dict[str, Any]]:
    first_target_time = (
        float(first_target_contact["sim_time_s"]) if first_target_contact else None
    )
    summaries = []
    for index, interval in enumerate(
        _contact_intervals(rows, lambda row: _truth(row["fingertip_table_contact"])),
        start=1,
    ):
        distances = [
            value
            for row in interval
            if (value := _optional_float(row["fingertip_table_min_distance_m"])) is not None
        ]
        start_time = float(interval[0]["sim_time_s"])
        end_time = float(interval[-1]["sim_time_s"])
        summaries.append(
            {
                "event_index": index,
                "start_time_s": start_time,
                "end_time_s": end_time,
                "duration_s": end_time - start_time,
                "physics_samples": len(interval),
                "maximum_normal_force_n": max(
                    float(row["fingertip_table_max_normal_force_n"]) for row in interval
                ),
                "minimum_contact_distance_m": min(distances, default=None),
                "left_finger_contact": any(
                    _truth(row["left_finger_table_contact"]) for row in interval
                ),
                "right_finger_contact": any(
                    _truth(row["right_finger_table_contact"]) for row in interval
                ),
                "before_first_target_contact": bool(
                    first_target_time is not None and end_time < first_target_time
                ),
                "overlaps_target_contact": any(
                    int(row["target_gripper_contact_count"]) > 0 for row in interval
                ),
            }
        )
    return summaries


def _summary(label: str, path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    times = [float(row["sim_time_s"]) for row in rows]
    slips = [_optional_float(row["relative_downward_slip_m"]) for row in rows]
    drifts = [_optional_float(row["relative_3d_drift_m"]) for row in rows]
    finite_slips = [value for value in slips if value is not None]
    finite_drifts = [value for value in drifts if value is not None]
    first_contact = _first(rows, lambda row: int(row["target_gripper_contact_count"]) > 0)
    first_both_finger_contact = _first(
        rows,
        lambda row: int(row["left_finger_target_contact_count"]) > 0
        and int(row["right_finger_target_contact_count"]) > 0,
    )
    first_table = _first(rows, lambda row: _truth(row["fingertip_table_contact"]))
    first_slip_1mm = _first(
        rows,
        lambda row: (_optional_float(row["relative_downward_slip_m"]) or 0.0) >= 0.001,
    )
    first_success = _first(rows, lambda row: _truth(row["original_v1_success_reached"]))
    initial_object_z = float(rows[0]["object_z_m"])
    first_lift_5mm = _first(
        rows, lambda row: float(row["object_z_m"]) - initial_object_z >= 0.005
    )
    first_lift_5cm = _first(
        rows, lambda row: float(row["object_z_m"]) - initial_object_z >= 0.05
    )
    maximum_slip_row = max(
        rows,
        key=lambda row: _optional_float(row["relative_downward_slip_m"]) or 0.0,
    )
    contact_rows = [row for row in rows if int(row["target_gripper_contact_count"]) > 0]
    post_reference_rows = [row for row in rows if _truth(row["relative_reference_established"])]
    table_forces = [float(row["fingertip_table_max_normal_force_n"]) for row in rows]
    table_distances = [
        value
        for row in rows
        if (value := _optional_float(row["fingertip_table_min_distance_m"])) is not None
    ]
    raw_commands = [float(row["gripper_raw_command_clamped"]) for row in post_reference_rows]
    actual_gripper = [float(row["actual_gripper_state"]) for row in post_reference_rows]
    table_slip = (
        _optional_float(first_table["relative_downward_slip_m"]) if first_table else None
    )
    table_plus_100ms = None
    if first_table is not None:
        target_time = float(first_table["sim_time_s"]) + 0.1
        table_plus_100ms = next(
            (row for row in rows if float(row["sim_time_s"]) + 1e-12 >= target_time),
            rows[-1],
        )
    table_plus_100ms_slip = (
        _optional_float(table_plus_100ms["relative_downward_slip_m"])
        if table_plus_100ms
        else None
    )
    object_heights = [float(row["object_z_m"]) for row in rows]
    world_drop_from_peak = max(object_heights) - object_heights[-1]
    pre_success_rows = (
        [row for row in rows if float(row["sim_time_s"]) < float(first_success["sim_time_s"])]
        if first_success
        else rows
    )
    pre_success_slips = [
        value
        for row in pre_success_rows
        if (value := _optional_float(row["relative_downward_slip_m"])) is not None
    ]
    first_contact_loss_after_success = None
    if first_success is not None:
        success_index = rows.index(first_success)
        first_contact_loss_after_success = _first(
            rows[success_index:],
            lambda row: int(row["target_gripper_contact_count"]) == 0,
        )
    references = {
        "first_any_finger_target_contact": _reference_summary(rows, first_contact),
        "first_both_fingers_target_contact": _reference_summary(
            rows, first_both_finger_contact
        ),
        "first_5mm_world_lift": _reference_summary(rows, first_lift_5mm),
        "first_5cm_world_lift": _reference_summary(rows, first_lift_5cm),
        "original_v1_success": _reference_summary(rows, first_success),
    }
    return {
        "label": label,
        "path": str(path),
        "physics_samples": len(rows),
        "start_time_s": times[0],
        "end_time_s": times[-1],
        "duration_s": times[-1] - times[0],
        "first_target_gripper_contact_time_s": (
            float(first_contact["sim_time_s"]) if first_contact else None
        ),
        "first_both_finger_target_contact_time_s": (
            float(first_both_finger_contact["sim_time_s"])
            if first_both_finger_contact
            else None
        ),
        "first_fingertip_table_contact_time_s": (
            float(first_table["sim_time_s"]) if first_table else None
        ),
        "table_contact_before_or_at_first_target_contact": bool(
            first_table
            and first_contact
            and float(first_table["sim_time_s"]) <= float(first_contact["sim_time_s"])
        ),
        "first_relative_downward_slip_1mm_time_s": (
            float(first_slip_1mm["sim_time_s"]) if first_slip_1mm else None
        ),
        "original_v1_success_trace_start_time_s": (
            float(first_success["sim_time_s"]) if first_success else None
        ),
        "first_target_contact_loss_after_success_time_s": (
            float(first_contact_loss_after_success["sim_time_s"])
            if first_contact_loss_after_success
            else None
        ),
        "maximum_relative_downward_slip_m": max(finite_slips, default=None),
        "maximum_relative_downward_slip_before_original_success_m": max(
            pre_success_slips, default=None
        ),
        "maximum_relative_3d_drift_m": max(finite_drifts, default=None),
        "maximum_slip_time_s": float(maximum_slip_row["sim_time_s"]),
        "relative_downward_slip_at_first_table_contact_m": table_slip,
        "relative_downward_slip_100ms_after_first_table_contact_m": table_plus_100ms_slip,
        "relative_slip_change_first_100ms_after_table_contact_m": (
            table_plus_100ms_slip - table_slip
            if table_plus_100ms_slip is not None and table_slip is not None
            else None
        ),
        "target_contact_sample_fraction_after_reference": (
            len(contact_rows) / len(post_reference_rows) if post_reference_rows else None
        ),
        "maximum_fingertip_table_normal_force_n": max(table_forces, default=0.0),
        "minimum_fingertip_table_contact_distance_m": min(table_distances, default=None),
        "gripper_raw_command_clamped_min_after_reference": min(raw_commands, default=None),
        "gripper_raw_command_clamped_max_after_reference": max(raw_commands, default=None),
        "actual_gripper_state_min_after_reference": min(actual_gripper, default=None),
        "actual_gripper_state_max_after_reference": max(actual_gripper, default=None),
        "actual_gripper_state_median_after_reference": (
            statistics.median(actual_gripper) if actual_gripper else None
        ),
        "gripper_raw_command_change_first_contact_to_max_slip": (
            float(maximum_slip_row["gripper_raw_command_clamped"])
            - float(first_contact["gripper_raw_command_clamped"])
            if first_contact
            else None
        ),
        "actual_gripper_state_change_first_contact_to_max_slip": (
            float(maximum_slip_row["actual_gripper_state"])
            - float(first_contact["actual_gripper_state"])
            if first_contact
            else None
        ),
        "world_frame_object_drop_from_episode_peak_to_final_m": world_drop_from_peak,
        "final_object_z_m": float(rows[-1]["object_z_m"]),
        "final_tcp_z_m": float(rows[-1]["tcp_z_m"]),
        "final_relative_z_m": float(rows[-1]["relative_z_m"]),
        "reference_sensitivity": references,
        "fingertip_table_contact_events": _table_contact_interval_summaries(
            rows, first_contact
        ),
    }


def _plot(label: str, rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_values = [float(row["sim_time_s"]) for row in rows]

    def values(field: str) -> list[float]:
        return [float("nan") if (value := _optional_float(row[field])) is None else value for row in rows]

    figure, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    axes[0].plot(time_values, values("object_z_m"), label="object_z")
    axes[0].plot(time_values, values("tcp_z_m"), label="tcp_z")
    axes[0].set_ylabel("world z (m)")
    axes[0].legend()
    axes[1].plot(time_values, values("relative_z_m"), label="tcp_z - object_z")
    axes[1].set_ylabel("relative z (m)")
    axes[1].legend()
    axes[2].plot(time_values, values("relative_downward_slip_m"), label="downward slip")
    axes[2].plot(time_values, values("relative_3d_drift_m"), label="3D drift", alpha=0.8)
    axes[2].set_ylabel("relative displacement (m)")
    axes[2].legend()
    axes[3].plot(time_values, values("gripper_raw_command_clamped"), label="command raw")
    axes[3].plot(time_values, values("actual_gripper_state"), label="actual raw", alpha=0.8)
    axes[3].set_ylabel("gripper raw")
    axes[3].legend()
    axes[4].plot(time_values, values("fingertip_table_max_normal_force_n"), label="table normal force")
    axes[4].step(
        time_values,
        [1.0 if _truth(row["fingertip_table_contact"]) else 0.0 for row in rows],
        where="post",
        label="table contact",
        alpha=0.7,
    )
    axes[4].set_ylabel("contact / force")
    axes[4].set_xlabel("simulation time (s)")
    axes[4].legend()
    figure.suptitle(label)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Analysis output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for raw in args.trace:
        label, path = _parse_trace_argument(raw)
        rows = _load(path)
        summaries.append(_summary(label, path, rows))
        _plot(label, rows, output / f"{label}_slip_diagnostics.png")
    document = {"schema_version": "xarm-slip-analysis-v2", "traces": summaries}
    (output / "slip_analysis.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# xArm slip diagnostic comparison",
        "",
        "| Trace | Max downward slip (m) | Max 3D drift (m) | First table contact (s) | Max table force (N) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['label']} | {row['maximum_relative_downward_slip_m']} | "
            f"{row['maximum_relative_3d_drift_m']} | "
            f"{row['first_fingertip_table_contact_time_s']} | "
            f"{row['maximum_fingertip_table_normal_force_n']} |"
        )
    (output / "slip_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
