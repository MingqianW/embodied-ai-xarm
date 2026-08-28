#!/usr/bin/env python3
"""Analyze controlled xArm gripper-slip suites and generate event plots."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from sim_mujoco.gripper_slip_diagnostics import load_jsonl  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--comparison-root",
        type=Path,
        action="append",
        default=[],
        help="Completed suite root whose analysis/summary.json is included in comparison plots.",
    )
    return parser


def _first_sustained(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    *,
    samples: int = 10,
    start: int = 0,
) -> dict[str, Any] | None:
    run = 0
    beginning = start
    for index in range(start, len(rows)):
        if predicate(rows[index]):
            if run == 0:
                beginning = index
            run += 1
            if run >= samples:
                return rows[beginning]
        else:
            run = 0
    return None


def _event_times(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    first_contact = _first_sustained(
        rows,
        lambda row: (
            row["contacts"]["left_target_count"] + row["contacts"]["right_target_count"]
        )
        > 0,
        samples=2,
    )
    bilateral = _first_sustained(
        rows, lambda row: bool(row["contacts"]["bilateral"]), samples=5
    )
    lift = _first_sustained(
        rows, lambda row: float(row["object"]["lift_height_m"]) >= 0.005, samples=5
    )
    leaves_table = _first_sustained(
        rows, lambda row: float(row["object"]["lift_height_m"]) >= 0.001, samples=5
    )
    bilateral_index = 0 if bilateral is None else int(bilateral["sample_index"])
    slip = _first_sustained(
        rows,
        lambda row: (
            row["relative"]["downward_slip_m"] is not None
            and float(row["relative"]["downward_slip_m"]) >= 0.002
        ),
        samples=10,
        start=bilateral_index,
    )
    detectable_slip = _first_sustained(
        rows,
        lambda row: (
            row["relative"].get("vertical_slip_m") is not None
            and float(row["relative"]["vertical_slip_m"]) >= 0.0001
        ),
        samples=10,
        start=bilateral_index,
    )
    loss = _first_sustained(
        rows,
        lambda row: (
            row["contacts"]["left_target_count"] + row["contacts"]["right_target_count"]
        )
        == 0,
        samples=10,
        start=bilateral_index + 1,
    )
    impact = (
        None
        if lift is None
        else _first_sustained(
            rows,
            lambda row: int(row["contacts"]["target_table_count"]) > 0,
            samples=2,
            start=int(lift["sample_index"]),
        )
    )
    saturation = _first_sustained(
        rows,
        lambda row: (
            row["actuator"]["force_fraction"] is not None
            and float(row["actuator"]["force_fraction"]) >= 0.99
        ),
        samples=10,
    )

    def time_of(row: dict[str, Any] | None) -> float | None:
        return None if row is None else float(row["sim_time_s"])

    return {
        "grasp_contact_onset_s": time_of(first_contact),
        "bilateral_grasp_s": time_of(bilateral),
        "grasp_established_s": time_of(bilateral),
        "object_leaves_table_s": time_of(leaves_table),
        "lift_onset_s": time_of(lift),
        "slip_onset_s": time_of(slip),
        "first_detectable_relative_slip_s": time_of(detectable_slip),
        "sustained_contact_loss_s": time_of(loss),
        "table_impact_s": time_of(impact),
        "object_drop_s": time_of(impact),
        "force_saturation_s": time_of(saturation),
    }


def _maximum(
    rows: list[dict[str, Any]], path: tuple[str, ...], *, absolute: bool = False
) -> float | None:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        if value is not None:
            values.append(abs(float(value)) if absolute else float(value))
    return max(values, default=None)


def _maximum_finger_speed_mps(rows: list[dict[str, Any]]) -> float:
    """Return physical finger speed for legacy slides or Menagerie linkage."""
    if not rows:
        return 0.0
    fingers = rows[0]["fingers"]
    if "left_qvel_mps" in fingers and "right_qvel_mps" in fingers:
        return max(
            max(
                abs(float(row["fingers"]["left_qvel_mps"])),
                abs(float(row["fingers"]["right_qvel_mps"])),
            )
            for row in rows
        )
    speeds = []
    for previous, current in zip(rows, rows[1:]):
        dt = float(current["sim_time_s"]) - float(previous["sim_time_s"])
        if dt <= 0.0:
            continue
        aperture_delta = float(current["fingers"]["opening_width_m"]) - float(
            previous["fingers"]["opening_width_m"]
        )
        speeds.append(0.5 * abs(aperture_delta / dt))
    return max(speeds, default=0.0)


def _analyze_trial(
    result: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    hold = [row for row in rows if row["command"]["source"] == "scripted_hold"]
    dynamic = [row for row in rows if row["command"]["source"] == "scripted_dynamic"]
    if not hold:
        raise ValueError("Trace has no scripted-hold samples")
    trial = rows[0]["trial"]
    events = _event_times(rows)
    hold_contact_loss = _first_sustained(
        hold,
        lambda row: (
            int(row["contacts"]["left_target_count"])
            + int(row["contacts"]["right_target_count"])
        )
        == 0,
        samples=10,
    )
    hold_table_impact = (
        None
        if result["hold_kind"] == "static"
        else _first_sustained(
            hold,
            lambda row: int(row["contacts"]["target_table_count"]) > 0,
            samples=2,
        )
    )
    events["sustained_contact_loss_s"] = (
        None if hold_contact_loss is None else float(hold_contact_loss["sim_time_s"])
    )
    events["table_impact_s"] = (
        None if hold_table_impact is None else float(hold_table_impact["sim_time_s"])
    )
    hold_start_time = float(hold[0]["sim_time_s"])
    hold_complete_time = float(hold[-1]["sim_time_s"])
    hold_detectable_slip = _first_sustained(
        hold,
        lambda row: (
            row["relative"].get("vertical_slip_m") is not None
            and float(row["relative"]["vertical_slip_m"]) >= 0.0001
        ),
        samples=10,
    )
    events["stationary_hold_start_s"] = hold_start_time
    events["hold_complete_s"] = hold_complete_time
    events["first_detectable_relative_slip_s"] = (
        None
        if hold_detectable_slip is None
        else float(hold_detectable_slip["sim_time_s"])
    )
    normal = [float(row["contacts"]["target_gripper_normal_sum_n"]) for row in hold]
    tangential = [
        float(row["contacts"]["target_gripper_tangential_sum_n"]) for row in hold
    ]
    left_normal = [float(row["contacts"]["left_target_normal_sum_n"]) for row in hold]
    right_normal = [float(row["contacts"]["right_target_normal_sum_n"]) for row in hold]
    force_fraction = [
        float(row["actuator"]["force_fraction"])
        for row in hold
        if row["actuator"]["force_fraction"] is not None
    ]
    slips = [
        float(row["relative"]["downward_slip_m"])
        for row in hold
        if row["relative"]["downward_slip_m"] is not None
    ]
    signed_vertical_slips = [
        float(
            row["relative"].get("vertical_slip_m", row["relative"]["downward_slip_m"])
        )
        for row in hold
        if row["relative"].get("vertical_slip_m", row["relative"]["downward_slip_m"])
        is not None
    ]
    bilateral_hold = [row for row in hold if bool(row["contacts"]["bilateral"])]
    bilateral_normal = [
        float(row["contacts"]["target_gripper_normal_sum_n"]) for row in bilateral_hold
    ]
    bilateral_slips = [
        float(row["relative"]["downward_slip_m"])
        for row in bilateral_hold
        if row["relative"]["downward_slip_m"] is not None
    ]
    bilateral_slip_velocities = [
        float(row["relative"]["vertical_slip_velocity_mps"])
        for row in bilateral_hold
        if row["relative"].get("vertical_slip_velocity_mps") is not None
    ]
    mass = float(trial["target_mass_kg"])
    weight = mass * 9.81
    hold_accel_z = [
        float(row["object"]["linear_acceleration_world_mps2"][2]) for row in hold
    ]
    peak_downward_load = mass * (
        9.81 + max((max(0.0, -value) for value in hold_accel_z), default=0.0)
    )
    bilateral_fraction = sum(bool(row["contacts"]["bilateral"]) for row in hold) / len(
        hold
    )
    exact_count_symmetry_fraction = sum(
        int(row["contacts"]["left_target_count"])
        == int(row["contacts"]["right_target_count"])
        for row in hold
    ) / len(hold)
    bilateral_force_asymmetry = [
        abs(left - right) / (left + right)
        for left, right in zip(left_normal, right_normal, strict=True)
        if left + right > 0.0
    ]
    saturation_fraction = sum(value >= 0.99 for value in force_fraction) / len(
        force_fraction
    )
    maximum_slip = max(slips, default=None)
    contact_lost = events["sustained_contact_loss_s"] is not None
    table_impact = events["table_impact_s"] is not None
    failure_onset_candidates = [
        float(value)
        for value in (
            events["sustained_contact_loss_s"],
            events["table_impact_s"],
        )
        if value is not None
    ]
    failure_onset_time = min(failure_onset_candidates, default=None)
    drop_time = events["table_impact_s"]
    hold_timestep = (
        float(hold[1]["sim_time_s"] - hold[0]["sim_time_s"]) if len(hold) > 1 else 0.0
    )
    pre_drop = [
        row
        for row in hold
        if failure_onset_time is None or float(row["sim_time_s"]) < failure_onset_time
    ]
    pre_drop_slips = [
        float(row["relative"]["downward_slip_m"])
        for row in pre_drop
        if row["relative"]["downward_slip_m"] is not None
    ]
    mechanically_stable = bool(
        bilateral_fraction >= 0.95
        and not contact_lost
        and not table_impact
        and maximum_slip is not None
        and math.isfinite(maximum_slip)
        and maximum_slip <= 0.002
    )
    if mechanically_stable:
        failure_label = "STABLE"
    elif contact_lost:
        failure_label = "CONTACT_LOSS"
    elif (
        bilateral_fraction >= 0.95
        and maximum_slip is not None
        and math.isfinite(maximum_slip)
        and maximum_slip > 0.002
    ):
        failure_label = "STATIC_CONTACT_SLIP"
    else:
        failure_label = "UNKNOWN"
    dynamic_slips = [
        float(row["relative"]["downward_slip_m"])
        for row in dynamic
        if row["relative"]["downward_slip_m"] is not None
    ]
    return {
        "task": result["task"],
        "seed": int(result["seed"]),
        "hold_kind": result["hold_kind"],
        "setting": result["setting"]["name"],
        "condition": result["setting"].get("condition"),
        "command_variant": result["setting"].get(
            "command_variant",
            (
                "max_closed_raw50"
                if result["setting"].get("closed_gripper_raw_override") == 50.0
                else "oracle_command"
            ),
        ),
        "effective_cone": trial["overrides"]["effective"]["simulation"]["cone"],
        "effective_impratio": float(
            trial["overrides"]["effective"]["simulation"]["impratio"]
        ),
        "force_multiplier": float(result["setting"]["force_multiplier"]),
        "force_limit_actuator_space": result["setting"].get(
            "force_limit_actuator_space"
        ),
        "effective_actuator_forcerange": trial["overrides"]["effective"]["actuator"][
            "forcerange_actuator_space"
        ],
        "kp_multiplier": float(result["setting"].get("kp_multiplier", 1.0)),
        "friction_multiplier": float(result["setting"]["friction_multiplier"]),
        "sample_count": len(rows),
        "hold_duration_s": float(hold[-1]["sim_time_s"] - hold[0]["sim_time_s"]),
        "object_mass_kg": mass,
        "object_weight_n": weight,
        "peak_estimated_vertical_load_n": peak_downward_load,
        "hold_bilateral_contact_fraction": bilateral_fraction,
        "hold_exact_contact_count_symmetry_fraction": (exact_count_symmetry_fraction),
        "mean_bilateral_force_asymmetry_fraction": (
            float(np.mean(bilateral_force_asymmetry))
            if bilateral_force_asymmetry
            else None
        ),
        "hold_bilateral_contact_duration_s": float(len(bilateral_hold) * hold_timestep),
        "force_saturation_fraction": saturation_fraction,
        "mean_abs_actuator_force_actuator_space": float(
            np.mean(
                [abs(float(row["actuator"]["force_actuator_space"])) for row in hold]
            )
        ),
        "mean_normal_force_sum_n": float(np.mean(normal)),
        "mean_bilateral_normal_force_sum_n": (
            float(np.mean(bilateral_normal)) if bilateral_normal else None
        ),
        "maximum_bilateral_normal_force_sum_n": max(bilateral_normal, default=None),
        "bilateral_normal_force_peak_to_mean": (
            max(bilateral_normal) / float(np.mean(bilateral_normal))
            if bilateral_normal and float(np.mean(bilateral_normal)) > 0.0
            else None
        ),
        "median_normal_force_sum_n": statistics.median(normal),
        "maximum_normal_force_sum_n": max(normal),
        "median_tangential_force_sum_n": statistics.median(tangential),
        "maximum_tangential_force_sum_n": max(tangential),
        "maximum_abs_actuator_force_actuator_space": _maximum(
            hold, ("actuator", "force_actuator_space"), absolute=True
        ),
        "maximum_tcp_linear_acceleration_mps2": max(
            float(np.linalg.norm(row["tcp"]["linear_acceleration_world_mps2"]))
            for row in hold
        ),
        "maximum_object_angular_speed_radps": max(
            float(np.linalg.norm(row["object"]["angular_velocity_world_radps"]))
            for row in hold
        ),
        "maximum_downward_slip_m": maximum_slip,
        "mean_bilateral_downward_slip_m": (
            float(np.mean(bilateral_slips)) if bilateral_slips else None
        ),
        "maximum_bilateral_downward_slip_m": max(bilateral_slips, default=None),
        "mean_downward_slip_m": (float(np.mean(slips)) if slips else None),
        "mean_signed_vertical_slip_m": (
            float(np.mean(signed_vertical_slips)) if signed_vertical_slips else None
        ),
        "total_relative_slip_before_drop_m": max(pre_drop_slips, default=None),
        "mean_stationary_slip_velocity_mps": (
            float(np.mean(bilateral_slip_velocities))
            if bilateral_slip_velocities
            else None
        ),
        "peak_stationary_slip_velocity_mps": (
            max(bilateral_slip_velocities) if bilateral_slip_velocities else None
        ),
        "final_downward_slip_m": slips[-1] if slips else None,
        "final_lift_height_m": float(hold[-1]["object"]["lift_height_m"]),
        "mechanically_stable_2mm": mechanically_stable,
        "hold_success": mechanically_stable,
        "drop_time_s": drop_time,
        "failure_onset_time_s": failure_onset_time,
        "failure_onset_time_from_hold_start_s": (
            None
            if failure_onset_time is None
            else float(failure_onset_time - hold_start_time)
        ),
        "drop_time_from_hold_start_s": (
            None if drop_time is None else float(drop_time - hold_start_time)
        ),
        "first_detectable_slip_from_hold_start_s": (
            None
            if events["first_detectable_relative_slip_s"] is None
            else float(events["first_detectable_relative_slip_s"] - hold_start_time)
        ),
        "contact_loss_from_hold_start_s": (
            None
            if events["sustained_contact_loss_s"] is None
            else float(events["sustained_contact_loss_s"] - hold_start_time)
        ),
        "drop_time_from_bilateral_grasp_s": (
            None
            if drop_time is None or events["bilateral_grasp_s"] is None
            else float(drop_time - float(events["bilateral_grasp_s"]))
        ),
        "diagnostic_failure_label": failure_label,
        "mean_tcp_linear_speed_during_hold_mps": float(
            np.mean(
                [
                    np.linalg.norm(row["tcp"]["linear_velocity_world_mps"])
                    for row in hold
                ]
            )
        ),
        "maximum_tcp_linear_speed_during_hold_mps": max(
            float(np.linalg.norm(row["tcp"]["linear_velocity_world_mps"]))
            for row in hold
        ),
        "maximum_finger_speed_during_hold_mps": _maximum_finger_speed_mps(hold),
        "maximum_target_penetration_m": max(
            (
                float(row["contacts"]["maximum_target_penetration_m"])
                for row in hold
                if row["contacts"].get("maximum_target_penetration_m") is not None
            ),
            default=None,
        ),
        "maximum_solver_iterations": max(
            (
                int(row.get("simulation", {}).get("solver_iterations", 0))
                for row in hold
            ),
            default=0,
        ),
        "maximum_abs_solver_fwdinv": max(
            (
                float(
                    np.max(
                        np.abs(
                            np.asarray(
                                row.get("simulation", {}).get(
                                    "solver_fwdinv", [0.0, 0.0]
                                ),
                                dtype=np.float64,
                            )
                        )
                    )
                )
                for row in hold
            ),
            default=0.0,
        ),
        "maximum_abs_qvel": max(
            (
                float(row.get("simulation", {}).get("maximum_abs_qvel", 0.0))
                for row in hold
            ),
            default=0.0,
        ),
        "maximum_abs_qacc": max(
            (
                float(row.get("simulation", {}).get("maximum_abs_qacc", 0.0))
                for row in hold
            ),
            default=0.0,
        ),
        "simulation_warning_count": max(
            (int(row.get("simulation", {}).get("warning_count", 0)) for row in hold),
            default=0,
        ),
        "simulation_finite": all(
            all(
                np.isfinite(np.asarray(value, dtype=np.float64)).all()
                for value in (
                    row["object"]["position_m"],
                    row["object"]["linear_velocity_world_mps"],
                    row["tcp"]["position_m"],
                    row["tcp"]["linear_velocity_world_mps"],
                    row["actuator"]["force_actuator_space"],
                )
            )
            for row in hold
        ),
        "dynamic_sample_count": len(dynamic),
        "dynamic_duration_s": (
            float(dynamic[-1]["sim_time_s"] - dynamic[0]["sim_time_s"])
            if dynamic
            else 0.0
        ),
        "dynamic_maximum_downward_slip_m": max(dynamic_slips, default=None),
        "dynamic_maximum_tcp_linear_speed_mps": max(
            (
                float(np.linalg.norm(row["tcp"]["linear_velocity_world_mps"]))
                for row in dynamic
            ),
            default=None,
        ),
        "dynamic_maximum_tcp_linear_acceleration_mps2": max(
            (
                float(np.linalg.norm(row["tcp"]["linear_acceleration_world_mps2"]))
                for row in dynamic
            ),
            default=None,
        ),
        **events,
    }


def _plot_trial(
    rows: list[dict[str, Any]], summary: dict[str, Any], output: Path
) -> None:
    import matplotlib.pyplot as plt

    time = np.asarray([float(row["sim_time_s"]) for row in rows])
    time -= time[0]
    lift = np.asarray([float(row["object"]["lift_height_m"]) for row in rows])
    slip = np.asarray(
        [
            np.nan
            if row["relative"]["downward_slip_m"] is None
            else float(row["relative"]["downward_slip_m"])
            for row in rows
        ]
    )
    raw = np.asarray(
        [
            np.nan
            if row["command"]["gripper_clamped_raw"] is None
            else float(row["command"]["gripper_clamped_raw"])
            for row in rows
        ]
    )
    actual = np.asarray([float(row["fingers"]["left_raw_equivalent"]) for row in rows])
    actuator = np.asarray(
        [float(row["actuator"]["force_actuator_space"]) for row in rows]
    )
    normal = np.asarray(
        [float(row["contacts"]["target_gripper_normal_sum_n"]) for row in rows]
    )
    left_normal = np.asarray(
        [float(row["contacts"]["left_target_normal_sum_n"]) for row in rows]
    )
    right_normal = np.asarray(
        [float(row["contacts"]["right_target_normal_sum_n"]) for row in rows]
    )
    bilateral = np.asarray([int(bool(row["contacts"]["bilateral"])) for row in rows])
    target_table = np.asarray(
        [int(row["contacts"]["target_table_count"] > 0) for row in rows]
    )

    figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(time, 1000 * lift, label="object lift")
    axes[0].plot(time, 1000 * slip, label="downward TCP-relative slip")
    axes[0].set_ylabel("mm")
    axes[0].legend(loc="best")
    axes[1].plot(time, raw, label="commanded raw")
    axes[1].plot(time, actual, label="actual raw equivalent")
    axes[1].invert_yaxis()
    axes[1].set_ylabel("raw (closed downward)")
    axes[1].legend(loc="best")
    axes[2].plot(time, actuator, label="actuator force")
    axes[2].plot(time, normal, label="total target normal force")
    axes[2].plot(time, left_normal, label="left normal force", alpha=0.8)
    axes[2].plot(time, right_normal, label="right normal force", alpha=0.8)
    axes[2].set_ylabel("N")
    axes[2].legend(loc="best")
    axes[3].step(time, bilateral, where="post", label="bilateral contact")
    axes[3].step(time, target_table, where="post", label="target/table contact")
    axes[3].set_ylabel("contact")
    axes[3].set_xlabel("simulation time since trace start (s)")
    axes[3].legend(loc="best")
    figure.suptitle(
        f"{summary['task']} seed {summary['seed']} | {summary['hold_kind']} | {summary['setting']}"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)

    dynamic_mask = np.asarray(
        [row["command"]["source"] == "scripted_dynamic" for row in rows]
    )
    if np.any(dynamic_mask):
        speed = np.asarray(
            [
                float(np.linalg.norm(row["tcp"]["linear_velocity_world_mps"]))
                for row in rows
            ]
        )
        acceleration = np.asarray(
            [
                float(np.linalg.norm(row["tcp"]["linear_acceleration_world_mps2"]))
                for row in rows
            ]
        )
        dynamic_figure, dynamic_axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        dynamic_axes[0].plot(time, speed, label="TCP speed")
        dynamic_axes[0].plot(time, acceleration, label="TCP acceleration")
        dynamic_axes[0].set_ylabel("m/s or m/s²")
        dynamic_axes[0].legend(loc="best")
        dynamic_axes[1].plot(time, 1000 * slip, label="downward relative slip")
        dynamic_axes[1].set_ylabel("mm")
        dynamic_axes[1].legend(loc="best")
        dynamic_axes[2].plot(time, actuator, label="actuator force")
        dynamic_axes[2].plot(time, normal, label="target normal-force sum")
        dynamic_axes[2].fill_between(
            time,
            0.0,
            1.0,
            where=dynamic_mask,
            transform=dynamic_axes[2].get_xaxis_transform(),
            alpha=0.15,
            label="scripted dynamic interval",
        )
        dynamic_axes[2].set_ylabel("N")
        dynamic_axes[2].set_xlabel("simulation time since trace start (s)")
        dynamic_axes[2].legend(loc="best")
        dynamic_figure.suptitle(
            f"Dynamics | {summary['task']} seed {summary['seed']} | {summary['setting']}"
        )
        dynamic_figure.tight_layout()
        dynamic_path = output.with_name(output.stem + "_dynamics.png")
        dynamic_figure.savefig(dynamic_path, dpi=160)
        plt.close(dynamic_figure)


def _plot_contact_map(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    points: dict[str, list[np.ndarray]] = {"left_finger": [], "right_finger": []}
    for row in rows:
        if row["command"]["source"] not in {"scripted_hold", "scripted_dynamic"}:
            continue
        object_position = np.asarray(row["object"]["position_m"], dtype=np.float64)
        target = str(row["trial"]["target_body"])
        for contact in row["contacts"]["all"]:
            bodies = {str(contact["body1"]), str(contact["body2"])}
            if target not in bodies:
                continue
            for finger in points:
                if finger in bodies:
                    points[finger].append(
                        np.asarray(contact["position_world_m"], dtype=np.float64)
                        - object_position
                    )
    figure = plt.figure(figsize=(7, 6))
    axis = figure.add_subplot(111, projection="3d")
    for finger, color in (("left_finger", "tab:blue"), ("right_finger", "tab:orange")):
        values = np.asarray(points[finger], dtype=np.float64)
        if values.size:
            # Physics-cadence contacts repeat heavily; stride only for display.
            stride = max(1, len(values) // 2000)
            shown = values[::stride]
            axis.scatter(
                1000.0 * shown[:, 0],
                1000.0 * shown[:, 1],
                1000.0 * shown[:, 2],
                s=4,
                alpha=0.25,
                label=finger,
                color=color,
            )
    axis.set_xlabel("contact x - object x (mm)")
    axis.set_ylabel("contact y - object y (mm)")
    axis.set_zlabel("contact z - object z (mm)")
    axis.set_title(
        f"Measured target contact locations | {summary['task']} | {summary['setting']}"
    )
    if any(points.values()):
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=165)
    plt.close(figure)


def _plot_suite(summaries: list[dict[str, Any]], output: Path) -> Path:
    import matplotlib.pyplot as plt

    settings = sorted({str(row["setting"]) for row in summaries})
    tasks = sorted({str(row["task"]) for row in summaries})
    holds = sorted({str(row["hold_kind"]) for row in summaries})
    x = np.arange(len(settings), dtype=np.float64)
    width = 0.8 / max(1, len(tasks) * len(holds))
    figure, axes = plt.subplots(
        2, 1, figsize=(max(10, 1.8 * len(settings)), 8), sharex=True
    )
    offset_index = 0
    for task in tasks:
        for hold in holds:
            stability: list[float] = []
            slip_mm: list[float] = []
            for setting in settings:
                selected = [
                    row
                    for row in summaries
                    if row["task"] == task
                    and row["hold_kind"] == hold
                    and row["setting"] == setting
                ]
                stability.append(
                    float(np.mean([row["mechanically_stable_2mm"] for row in selected]))
                    if selected
                    else math.nan
                )
                finite_slips = [
                    float(row["maximum_downward_slip_m"])
                    for row in selected
                    if row["maximum_downward_slip_m"] is not None
                    and math.isfinite(float(row["maximum_downward_slip_m"]))
                ]
                slip_mm.append(
                    1000.0 * float(np.mean(finite_slips)) if finite_slips else math.nan
                )
            offset = -0.4 + width / 2 + offset_index * width
            label = f"{task} / {hold}"
            axes[0].bar(x + offset, stability, width=width, label=label)
            axes[1].bar(x + offset, slip_mm, width=width, label=label)
            offset_index += 1
    axes[0].set_ylabel("stable-hold fraction")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(loc="best", fontsize="small")
    axes[1].set_ylabel("mean maximum slip (mm)")
    axes[1].set_xticks(x, settings, rotation=20, ha="right")
    axes[1].set_xlabel("controlled setting")
    axes[1].legend(loc="best", fontsize="small")
    figure.tight_layout()
    path = output / "suite_outcomes.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


CONTACT_SETTING_ORDER = (
    "pyramidal_impratio1",
    "elliptic_impratio1",
    "elliptic_impratio10",
)
CONTACT_SETTING_LABELS = {
    "pyramidal_impratio1": "A: pyramidal / 1",
    "elliptic_impratio1": "B: elliptic / 1",
    "elliptic_impratio10": "C: elliptic / 10",
}


def _contact_hold_series(result: dict[str, Any]) -> dict[str, np.ndarray]:
    rows = [
        row
        for row in load_jsonl(Path(result["artifacts"]["trace"]))
        if row["command"]["source"] == "scripted_hold"
    ]
    if not rows:
        raise ValueError(f"Contact trial has no hold samples: {result}")
    time = np.asarray([float(row["sim_time_s"]) for row in rows])
    time -= time[0]
    return {
        "time_s": time,
        "vertical_slip_m": np.asarray(
            [
                math.nan
                if row["relative"].get("vertical_slip_m") is None
                else float(row["relative"]["vertical_slip_m"])
                for row in rows
            ]
        ),
        "vertical_slip_velocity_mps": np.asarray(
            [float(row["relative"]["vertical_slip_velocity_mps"]) for row in rows]
        ),
        "actuator_force_actuator_space": np.asarray(
            [float(row["actuator"]["force_actuator_space"]) for row in rows]
        ),
        "left_contact": np.asarray(
            [int(row["contacts"]["left_target_count"] > 0) for row in rows]
        ),
        "right_contact": np.asarray(
            [int(row["contacts"]["right_target_count"] > 0) for row in rows]
        ),
        "left_normal_force_n": np.asarray(
            [float(row["contacts"]["left_target_normal_sum_n"]) for row in rows]
        ),
        "right_normal_force_n": np.asarray(
            [float(row["contacts"]["right_target_normal_sum_n"]) for row in rows]
        ),
    }


def _representative_contact_results(
    results: dict[str, Any],
) -> list[dict[str, Any]]:
    manifest_path = (
        Path(results["trials"][0]["artifacts"]["trial"]).parents[2] / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = manifest["representative_video_selection"]
    selected = [
        row
        for row in results["trials"]
        if row["task"] == selection["task"]
        and int(row["seed"]) == int(selection["seed"])
        and row["hold_kind"] == selection["hold_kind"]
        and row["setting"].get("command_variant", "oracle_command")
        == selection.get("command_variant", "oracle_command")
    ]
    by_setting = {row["setting"]["name"]: row for row in selected}
    if set(by_setting) != set(CONTACT_SETTING_ORDER):
        raise ValueError(
            "Representative contact set is incomplete: "
            f"expected={CONTACT_SETTING_ORDER}, actual={sorted(by_setting)}"
        )
    return [by_setting[value] for value in CONTACT_SETTING_ORDER]


def _plot_contact_comparison(
    results: dict[str, Any],
    summaries: list[dict[str, Any]],
    output: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    representative = _representative_contact_results(results)
    series = {
        row["setting"]["name"]: _contact_hold_series(row) for row in representative
    }
    task = representative[0]["task"]
    seed = int(representative[0]["seed"])
    paths: dict[str, Path] = {}

    figure, axis = plt.subplots(figsize=(9, 5))
    for setting in CONTACT_SETTING_ORDER:
        values = series[setting]
        axis.plot(
            values["time_s"],
            1000.0 * values["vertical_slip_m"],
            label=CONTACT_SETTING_LABELS[setting],
        )
    axis.axhline(
        2.0, color="black", linestyle="--", linewidth=1, label="2 mm criterion"
    )
    axis.set_xlabel("stationary-hold time (s)")
    axis.set_ylabel("relative downward displacement (mm; positive downward)")
    axis.set_title(f"Object–gripper vertical displacement | {task} seed {seed}")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    paths["relative_vertical_displacement"] = (
        output / "relative_vertical_displacement_overlay.png"
    )
    figure.savefig(paths["relative_vertical_displacement"], dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for setting in CONTACT_SETTING_ORDER:
        values = series[setting]
        axis.plot(
            values["time_s"],
            1000.0 * values["vertical_slip_velocity_mps"],
            label=CONTACT_SETTING_LABELS[setting],
        )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("stationary-hold time (s)")
    axis.set_ylabel("relative vertical slip velocity (mm/s; positive downward)")
    axis.set_title(f"Object–gripper slip velocity | {task} seed {seed}")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    paths["relative_slip_velocity"] = output / "relative_slip_velocity_overlay.png"
    figure.savefig(paths["relative_slip_velocity"], dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for setting in CONTACT_SETTING_ORDER:
        values = series[setting]
        axis.plot(
            values["time_s"],
            values["actuator_force_actuator_space"],
            label=CONTACT_SETTING_LABELS[setting],
        )
    axis.set_xlabel("stationary-hold time (s)")
    axis.set_ylabel("gripper actuator-space force")
    axis.set_title(f"Actuator-space force | {task} seed {seed}")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    paths["actuator_force"] = output / "actuator_force_overlay.png"
    figure.savefig(paths["actuator_force"], dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for setting in CONTACT_SETTING_ORDER:
        values = series[setting]
        label = CONTACT_SETTING_LABELS[setting]
        axes[0].plot(
            values["time_s"],
            values["left_normal_force_n"],
            label=f"{label}, left",
        )
        axes[0].plot(
            values["time_s"],
            values["right_normal_force_n"],
            linestyle="--",
            label=f"{label}, right",
        )
        axes[1].step(
            values["time_s"],
            values["left_contact"] + values["right_contact"],
            where="post",
            label=label,
        )
    axes[0].set_ylabel("normal force (N)")
    axes[0].set_title(f"Finger contact force/state | {task} seed {seed}")
    axes[0].legend(loc="best", fontsize="small", ncol=2)
    axes[0].grid(alpha=0.25)
    axes[1].set_ylabel("contacting fingers (0–2)")
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_xlabel("stationary-hold time (s)")
    axes[1].legend(loc="best", fontsize="small")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    paths["contact_state_force"] = output / "contact_state_force_overlay.png"
    figure.savefig(paths["contact_state_force"], dpi=180)
    plt.close(figure)

    tasks = sorted({str(row["task"]) for row in summaries})
    x = np.arange(len(CONTACT_SETTING_ORDER), dtype=np.float64)
    width = 0.8 / len(tasks)
    figure, axis = plt.subplots(figsize=(9, 5))
    for task_index, task_name in enumerate(tasks):
        values = []
        for setting in CONTACT_SETTING_ORDER:
            selected = [
                row
                for row in summaries
                if row["task"] == task_name and row["setting"] == setting
            ]
            values.append(
                1000.0
                * float(
                    np.mean(
                        _finite_metric(selected, "total_relative_slip_before_drop_m")
                    )
                )
            )
        offset = -0.4 + width / 2 + task_index * width
        axis.bar(x + offset, values, width=width, label=task_name)
    axis.set_xticks(
        x, [CONTACT_SETTING_LABELS[value] for value in CONTACT_SETTING_ORDER]
    )
    axis.set_ylabel("mean relative slip before contact loss/drop (mm)")
    axis.set_xlabel("contact-model condition")
    axis.set_title("Condition versus total relative slip")
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    paths["condition_total_relative_slip"] = (
        output / "condition_total_relative_slip.png"
    )
    figure.savefig(paths["condition_total_relative_slip"], dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for task_index, task_name in enumerate(tasks):
        values = []
        stable_counts = []
        for setting in CONTACT_SETTING_ORDER:
            selected = [
                row
                for row in summaries
                if row["task"] == task_name and row["setting"] == setting
            ]
            values.append(
                float(
                    np.mean(
                        [
                            row["hold_duration_s"]
                            if row["drop_time_from_hold_start_s"] is None
                            else row["drop_time_from_hold_start_s"]
                            for row in selected
                        ]
                    )
                )
            )
            stable_counts.append(sum(row["hold_success"] for row in selected))
        offset = -0.4 + width / 2 + task_index * width
        bars = axis.bar(x + offset, values, width=width, label=task_name)
        for bar, stable_count in zip(bars, stable_counts, strict=True):
            if stable_count:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{stable_count}/{len(selected)} stable",
                    ha="center",
                    va="bottom",
                    fontsize="x-small",
                    rotation=90,
                )
    axis.set_xticks(
        x, [CONTACT_SETTING_LABELS[value] for value in CONTACT_SETTING_ORDER]
    )
    axis.set_ylabel("mean observed hold before drop or completion (s)")
    axis.set_xlabel("contact-model condition")
    axis.set_title("Condition versus time-to-drop / stable hold")
    axis.set_ylim(0.0, 5.35)
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    paths["condition_time_to_drop"] = output / "condition_time_to_drop.png"
    figure.savefig(paths["condition_time_to_drop"], dpi=180)
    plt.close(figure)
    return paths


def _finite_metric(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]


def _contact_condition_summary(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for setting in CONTACT_SETTING_ORDER:
        selected = [row for row in summaries if row["setting"] == setting]
        if not selected:
            raise ValueError(f"Contact condition is absent from summaries: {setting}")
        mean_slips = _finite_metric(selected, "mean_bilateral_downward_slip_m")
        maximum_slips = _finite_metric(selected, "maximum_bilateral_downward_slip_m")
        mean_velocities = _finite_metric(selected, "mean_stationary_slip_velocity_mps")
        peak_velocities = _finite_metric(selected, "peak_stationary_slip_velocity_mps")
        mean_actuator = _finite_metric(
            selected, "mean_abs_actuator_force_actuator_space"
        )
        maximum_actuator = _finite_metric(
            selected, "maximum_abs_actuator_force_actuator_space"
        )
        mean_normal = _finite_metric(selected, "mean_bilateral_normal_force_sum_n")
        maximum_normal = _finite_metric(
            selected, "maximum_bilateral_normal_force_sum_n"
        )
        bilateral_duration = _finite_metric(
            selected, "hold_bilateral_contact_duration_s"
        )
        result.append(
            {
                "condition": selected[0]["condition"],
                "setting": setting,
                "cone": selected[0]["effective_cone"],
                "impratio": selected[0]["effective_impratio"],
                "trial_count": len(selected),
                "stable_hold_count": sum(row["hold_success"] for row in selected),
                "bilateral_slip_count": sum(
                    row["diagnostic_failure_label"] == "STATIC_CONTACT_SLIP"
                    for row in selected
                ),
                "contact_loss_drop_count": sum(
                    row["diagnostic_failure_label"] == "CONTACT_LOSS"
                    for row in selected
                ),
                "unknown_count": sum(
                    row["diagnostic_failure_label"] == "UNKNOWN" for row in selected
                ),
                "mean_relative_slip_m": float(np.mean(mean_slips)),
                "maximum_relative_slip_m": max(maximum_slips),
                "maximum_full_hold_downward_displacement_m": max(
                    _finite_metric(selected, "maximum_downward_slip_m")
                ),
                "mean_total_relative_slip_before_drop_m": float(
                    np.mean(
                        _finite_metric(selected, "total_relative_slip_before_drop_m")
                    )
                ),
                "mean_stationary_slip_velocity_mps": float(np.mean(mean_velocities)),
                "peak_stationary_slip_velocity_mps": max(peak_velocities),
                "mean_bilateral_contact_duration_s": float(np.mean(bilateral_duration)),
                "mean_abs_actuator_force_actuator_space": float(np.mean(mean_actuator)),
                "maximum_abs_actuator_force_actuator_space": max(maximum_actuator),
                "mean_contact_normal_force_n": float(np.mean(mean_normal)),
                "maximum_contact_normal_force_n": max(maximum_normal),
                "maximum_solver_iterations": max(
                    int(row["maximum_solver_iterations"]) for row in selected
                ),
                "maximum_abs_solver_fwdinv": max(
                    _finite_metric(selected, "maximum_abs_solver_fwdinv")
                ),
                "maximum_bilateral_normal_force_peak_to_mean": max(
                    _finite_metric(selected, "bilateral_normal_force_peak_to_mean"),
                    default=None,
                ),
                "maximum_target_penetration_m": max(
                    _finite_metric(selected, "maximum_target_penetration_m"),
                    default=None,
                ),
                "maximum_abs_qvel": max(_finite_metric(selected, "maximum_abs_qvel")),
                "maximum_abs_qacc": max(_finite_metric(selected, "maximum_abs_qacc")),
                "simulation_warning_count": max(
                    int(row["simulation_warning_count"]) for row in selected
                ),
                "all_simulation_finite": all(
                    bool(row["simulation_finite"]) for row in selected
                ),
            }
        )
    return result


def _without_contact_intervention(configuration: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(configuration)
    simulation = value["simulation"]
    for key in ("cone", "cone_enum", "impratio"):
        simulation.pop(key)
    return value


def _close_optional(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _contact_validation(
    results: dict[str, Any],
    summaries: list[dict[str, Any]],
    comparison_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    trial_metadata = {
        (
            str(row["task"]),
            int(row["seed"]),
            str(row["hold_kind"]),
            str(row["setting"]["name"]),
            str(row["setting"].get("command_variant", "oracle_command")),
        ): json.loads(Path(row["artifacts"]["trial"]).read_text(encoding="utf-8"))
        for row in results["trials"]
    }
    expected_trial_count = (
        len({str(row["task"]) for row in results["trials"]})
        * len({int(row["seed"]) for row in results["trials"]})
        * len(CONTACT_SETTING_ORDER)
        * 2
    )
    if len(results["trials"]) != expected_trial_count:
        errors.append(
            f"Expected {expected_trial_count} paired contact trials, "
            f"found {len(results['trials'])}"
        )
    if {str(row["hold_kind"]) for row in results["trials"]} != {"suspended"}:
        errors.append("Contact suite contains a non-suspended trial")
    command_variants = {
        str(row["setting"].get("command_variant", "oracle_command"))
        for row in results["trials"]
    }
    if command_variants != {"oracle_command", "max_closed_raw50"}:
        errors.append(
            f"Contact suite command variants are incomplete: {sorted(command_variants)}"
        )

    expected_contact = {
        "pyramidal_impratio1": ("A", "pyramidal", 1.0),
        "elliptic_impratio1": ("B", "elliptic", 1.0),
        "elliptic_impratio10": ("C", "elliptic", 10.0),
    }
    effective_settings: dict[str, Any] = {}
    for key, trial in trial_metadata.items():
        setting = key[3]
        command_variant = key[4]
        expected = expected_contact.get(setting)
        if expected is None:
            errors.append(f"Unexpected contact setting: {setting}")
            continue
        condition, cone, impratio = expected
        if trial["setting"].get("condition") != condition:
            errors.append(f"Condition label mismatch for {key}")
        overrides = trial["overrides"]
        baseline_simulation = overrides["baseline"]["simulation"]
        effective_simulation = overrides["effective"]["simulation"]
        if baseline_simulation["cone"] != "elliptic" or not math.isclose(
            float(baseline_simulation["impratio"]), 10.0
        ):
            errors.append(
                f"Compiled baseline mismatch for {key}: {baseline_simulation}"
            )
        if effective_simulation["cone"] != cone or not math.isclose(
            float(effective_simulation["impratio"]), impratio
        ):
            errors.append(f"Effective contact setting mismatch for {key}")
        if overrides["changed_invariant_hashes"]:
            errors.append(
                f"Forbidden model arrays changed for {key}: "
                f"{overrides['changed_invariant_hashes']}"
            )
        if overrides["invariant_hashes_before"] != overrides["invariant_hashes_after"]:
            errors.append(f"Invariant hash mismatch for {key}")
        target = str(trial["target_body"])
        effective_settings.setdefault(setting, {}).setdefault(command_variant, {})[
            target
        ] = overrides["effective"]

    pairing_records: list[dict[str, Any]] = []
    grouping_keys = sorted({(*key[:3], key[4]) for key in trial_metadata})
    for group_key in grouping_keys:
        group = {
            key[3]: trial
            for key, trial in trial_metadata.items()
            if (*key[:3], key[4]) == group_key
        }
        group_errors: list[str] = []
        if set(group) != set(CONTACT_SETTING_ORDER):
            group_errors.append(
                f"settings={sorted(group)}, expected={list(CONTACT_SETTING_ORDER)}"
            )
        else:
            reference = group[CONTACT_SETTING_ORDER[0]]
            if reference.get("oracle_plan_source_condition") != "A":
                group_errors.append("condition A did not originate the oracle plan")
            reference_initial_state = reference.get("paired_initial_state")
            if not reference_initial_state:
                group_errors.append("condition A lacks paired_initial_state")
            for setting in CONTACT_SETTING_ORDER[1:]:
                candidate = group[setting]
                if candidate.get("oracle_plan_source_condition") != "A-reused":
                    group_errors.append(f"{setting} did not reuse condition A plan")
                for field in (
                    "model_path",
                    "model_sha256",
                    "oracle_config",
                    "oracle_plan",
                    "oracle_action_manifest",
                    "target_body",
                    "target_mass_kg",
                    "initial_target_z_m",
                ):
                    if candidate[field] != reference[field]:
                        group_errors.append(f"{setting} differs in {field}")
                if _without_contact_intervention(
                    candidate["overrides"]["effective"]
                ) != _without_contact_intervention(reference["overrides"]["effective"]):
                    group_errors.append(f"{setting} differs outside cone/impratio")
                candidate_initial_state = candidate.get("paired_initial_state")
                if not candidate_initial_state:
                    group_errors.append(f"{setting} lacks paired_initial_state")
                elif reference_initial_state and (
                    candidate_initial_state["state_spec"]
                    != reference_initial_state["state_spec"]
                    or candidate_initial_state["state_size"]
                    != reference_initial_state["state_size"]
                    or candidate_initial_state["state_sha256"]
                    != reference_initial_state["state_sha256"]
                    or candidate_initial_state["initial_target_z_m"]
                    != reference_initial_state["initial_target_z_m"]
                    or candidate_initial_state["initial_conditions"]
                    != reference_initial_state["initial_conditions"]
                ):
                    group_errors.append(f"{setting} paired initial state differs")
        if group_errors:
            errors.extend(f"{group_key}: {value}" for value in group_errors)
        pairing_records.append(
            {
                "task": group_key[0],
                "seed": group_key[1],
                "hold_kind": group_key[2],
                "command_variant": group_key[3],
                "passed": not group_errors,
                "errors": group_errors,
            }
        )

    baseline_reproduction: list[dict[str, Any]] = []
    condition_a = [row for row in summaries if row["setting"] == "pyramidal_impratio1"]
    for actual in condition_a:
        baseline_setting = (
            "baseline_max_closed_raw50"
            if actual["command_variant"] == "max_closed_raw50"
            else "baseline_oracle_command"
        )
        baseline = next(
            (
                row
                for row in comparison_summaries
                if row is not actual
                and row["task"] == actual["task"]
                and int(row["seed"]) == int(actual["seed"])
                and row["hold_kind"] == "suspended"
                and row["setting"] == baseline_setting
            ),
            None,
        )
        if baseline is None:
            errors.append(
                f"Missing validated baseline pair for {actual['task']} seed {actual['seed']}"
            )
            continue
        metrics = {
            key: {
                "validated_baseline": baseline[key],
                "condition_a": actual[key],
                "close": _close_optional(baseline[key], actual[key]),
            }
            for key in (
                "maximum_downward_slip_m",
                "hold_bilateral_contact_fraction",
                "maximum_abs_actuator_force_actuator_space",
            )
        }
        metrics["contact_loss_from_hold_start_s"] = {
            # The validated v1 analyzer called first contact loss or table
            # impact "drop". The v2 analyzer separates contact loss from the
            # later table-impact/drop event, so compare like with like here.
            "validated_baseline": baseline["drop_time_from_hold_start_s"],
            "condition_a": actual["contact_loss_from_hold_start_s"],
            "close": _close_optional(
                baseline["drop_time_from_hold_start_s"],
                actual["contact_loss_from_hold_start_s"],
            ),
        }
        label_match = (
            baseline["diagnostic_failure_label"] == actual["diagnostic_failure_label"]
        )
        success_match = bool(baseline["hold_success"]) == bool(actual["hold_success"])
        passed = (
            all(value["close"] for value in metrics.values())
            and label_match
            and success_match
        )
        if not passed:
            errors.append(
                f"Condition A failed baseline reproduction for "
                f"{actual['task']} seed {actual['seed']}"
            )
        baseline_reproduction.append(
            {
                "task": actual["task"],
                "seed": actual["seed"],
                "command_variant": actual["command_variant"],
                "passed": passed,
                "failure_label_match": label_match,
                "hold_success_match": success_match,
                "metrics": metrics,
            }
        )
    if len(baseline_reproduction) != expected_trial_count // 3:
        errors.append(
            "Condition A baseline reproduction coverage is incomplete: "
            f"{len(baseline_reproduction)}"
        )

    return {
        "passed": not errors,
        "errors": errors,
        "expected_trial_count": expected_trial_count,
        "actual_trial_count": len(results["trials"]),
        "pairing": pairing_records,
        "baseline_reproduction": baseline_reproduction,
        "effective_settings": effective_settings,
        "only_cone_and_impratio_changed": not any(
            "outside cone/impratio" in value
            or "Forbidden model arrays" in value
            or "Invariant hash mismatch" in value
            for value in errors
        ),
    }


def _setting_summary(
    summaries: list[dict[str, Any]],
    *,
    split_command_variant: bool = False,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    keys = sorted(
        {
            (
                str(row["task"]),
                str(row["hold_kind"]),
                str(row["setting"]),
                (
                    str(row.get("command_variant", "oracle_command"))
                    if split_command_variant
                    else "all_paired_commands"
                ),
            )
            for row in summaries
        }
    )
    for task, hold_kind, setting, command_variant in keys:
        selected = [
            row
            for row in summaries
            if row["task"] == task
            and row["hold_kind"] == hold_kind
            and row["setting"] == setting
            and (
                not split_command_variant
                or row.get("command_variant", "oracle_command") == command_variant
            )
        ]
        drop_times = [
            float(row["drop_time_from_bilateral_grasp_s"])
            for row in selected
            if row["drop_time_from_bilateral_grasp_s"] is not None
        ]
        result.append(
            {
                "task": task,
                "hold_kind": hold_kind,
                "setting": setting,
                "command_variant": command_variant,
                "condition": selected[0].get("condition"),
                "force_limit_actuator_space": selected[0].get(
                    "force_limit_actuator_space"
                ),
                "cone": selected[0].get("effective_cone"),
                "impratio": selected[0].get("effective_impratio"),
                "trial_count": len(selected),
                "hold_success_count": sum(row["hold_success"] for row in selected),
                "five_second_stable_hold_count": sum(
                    row["hold_success"] for row in selected
                ),
                "bilateral_slip_count": sum(
                    row["diagnostic_failure_label"] == "STATIC_CONTACT_SLIP"
                    for row in selected
                ),
                "contact_loss_drop_count": sum(
                    row["diagnostic_failure_label"] == "CONTACT_LOSS"
                    for row in selected
                ),
                "hold_success_rate": float(
                    np.mean([row["hold_success"] for row in selected])
                ),
                "drop_count": sum(row["drop_time_s"] is not None for row in selected),
                "mean_drop_time_from_bilateral_grasp_s": (
                    float(np.mean(drop_times)) if drop_times else None
                ),
                "mean_maximum_downward_slip_m": (
                    float(np.mean(finite_slips))
                    if (
                        finite_slips := [
                            float(row["maximum_downward_slip_m"])
                            for row in selected
                            if row["maximum_downward_slip_m"] is not None
                            and math.isfinite(float(row["maximum_downward_slip_m"]))
                        ]
                    )
                    else None
                ),
                "mean_relative_slip_m": (
                    float(np.mean(values))
                    if (
                        values := _finite_metric(
                            selected, "mean_bilateral_downward_slip_m"
                        )
                    )
                    else None
                ),
                "maximum_relative_slip_m": max(
                    _finite_metric(selected, "maximum_bilateral_downward_slip_m"),
                    default=None,
                ),
                "maximum_full_hold_downward_displacement_m": max(
                    _finite_metric(selected, "maximum_downward_slip_m"),
                    default=None,
                ),
                "mean_total_relative_slip_before_drop_m": (
                    float(np.mean(values))
                    if (
                        values := _finite_metric(
                            selected, "total_relative_slip_before_drop_m"
                        )
                    )
                    else None
                ),
                "mean_stationary_slip_velocity_mps": (
                    float(np.mean(values))
                    if (
                        values := _finite_metric(
                            selected, "mean_stationary_slip_velocity_mps"
                        )
                    )
                    else None
                ),
                "peak_stationary_slip_velocity_mps": max(
                    _finite_metric(selected, "peak_stationary_slip_velocity_mps"),
                    default=None,
                ),
                "mean_time_to_first_detectable_slip_s": (
                    float(np.mean(values))
                    if (
                        values := _finite_metric(
                            selected, "first_detectable_slip_from_hold_start_s"
                        )
                    )
                    else None
                ),
                "mean_time_to_contact_loss_s": (
                    float(np.mean(values))
                    if (
                        values := _finite_metric(
                            selected, "contact_loss_from_hold_start_s"
                        )
                    )
                    else None
                ),
                "mean_time_to_drop_s": (
                    float(np.mean(values))
                    if (
                        values := _finite_metric(
                            selected, "drop_time_from_hold_start_s"
                        )
                    )
                    else None
                ),
                "mean_bilateral_contact_duration_s": float(
                    np.mean(
                        _finite_metric(selected, "hold_bilateral_contact_duration_s")
                    )
                ),
                "minimum_exact_contact_count_symmetry_fraction": min(
                    _finite_metric(
                        selected, "hold_exact_contact_count_symmetry_fraction"
                    ),
                    default=None,
                ),
                "maximum_mean_bilateral_force_asymmetry_fraction": max(
                    _finite_metric(selected, "mean_bilateral_force_asymmetry_fraction"),
                    default=None,
                ),
                "mean_abs_actuator_force_actuator_space": float(
                    np.mean(
                        _finite_metric(
                            selected, "mean_abs_actuator_force_actuator_space"
                        )
                    )
                ),
                "mean_maximum_abs_actuator_force_actuator_space": (
                    float(np.mean(values))
                    if (
                        values := _finite_metric(
                            selected, "maximum_abs_actuator_force_actuator_space"
                        )
                    )
                    else None
                ),
                "maximum_abs_actuator_force_actuator_space": max(
                    _finite_metric(
                        selected, "maximum_abs_actuator_force_actuator_space"
                    )
                ),
                "mean_contact_normal_force_n": (
                    float(np.mean(values))
                    if (
                        values := _finite_metric(
                            selected, "mean_bilateral_normal_force_sum_n"
                        )
                    )
                    else None
                ),
                "maximum_contact_normal_force_n": max(
                    _finite_metric(selected, "maximum_bilateral_normal_force_sum_n"),
                    default=None,
                ),
                "failure_label_counts": dict(
                    sorted(
                        Counter(
                            row["diagnostic_failure_label"] for row in selected
                        ).items()
                    )
                ),
            }
        )
    return result


def _refine_dynamic_labels(
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    for row in summaries:
        if row["hold_success"]:
            continue
        baseline = next(
            (
                candidate
                for candidate in comparisons
                if candidate is not row
                and candidate["task"] == row["task"]
                and candidate["seed"] == row["seed"]
                and candidate["hold_kind"] == "suspended"
                and candidate["setting"] == "baseline_oracle_command"
            ),
            None,
        )
        if baseline is not None and baseline["hold_success"]:
            row["diagnostic_failure_label"] = "DYNAMIC_SHAKEOUT"


def _paired_interventions(
    suite: str,
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for intervention in summaries:
        command_variant = intervention.get("command_variant", "oracle_command")
        baseline_setting = (
            "baseline_max_closed_raw50"
            if command_variant == "max_closed_raw50"
            else "baseline_oracle_command"
        )
        baseline = next(
            (
                row
                for row in comparisons
                if row is not intervention
                and row["task"] == intervention["task"]
                and row["seed"] == intervention["seed"]
                and row["hold_kind"] == intervention["hold_kind"]
                and row["setting"] == baseline_setting
            ),
            None,
        )
        if baseline is None:
            continue
        baseline_slip = (
            None
            if baseline["maximum_downward_slip_m"] is None
            else float(baseline["maximum_downward_slip_m"])
        )
        intervention_slip = (
            None
            if intervention["maximum_downward_slip_m"] is None
            else float(intervention["maximum_downward_slip_m"])
        )
        slip_reduction_fraction = (
            None
            if baseline_slip is None
            or intervention_slip is None
            or not math.isfinite(baseline_slip)
            or not math.isfinite(intervention_slip)
            or baseline_slip <= 0.0
            else 1.0 - intervention_slip / baseline_slip
        )
        pairs.append(
            {
                "suite": suite,
                "task": intervention["task"],
                "seed": intervention["seed"],
                "hold_kind": intervention["hold_kind"],
                "command_variant": command_variant,
                "baseline_setting": baseline["setting"],
                "intervention_setting": intervention["setting"],
                "baseline_hold_success": baseline["hold_success"],
                "intervention_hold_success": intervention["hold_success"],
                "baseline_maximum_downward_slip_m": baseline_slip,
                "intervention_maximum_downward_slip_m": intervention_slip,
                "slip_reduction_fraction": slip_reduction_fraction,
                "material_improvement": bool(
                    (not baseline["hold_success"] and intervention["hold_success"])
                    or (
                        slip_reduction_fraction is not None
                        and slip_reduction_fraction >= 0.5
                    )
                ),
            }
        )
    return pairs


def main() -> None:
    args = _parser().parse_args()
    suite = args.suite_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing an existing analysis directory: {output}")
    results_path = suite / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("status") != "complete":
        raise ValueError(f"Suite is not complete: {results_path}")
    output.mkdir(parents=True, exist_ok=False)
    plots = output / "plots"
    plots.mkdir()

    summaries: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    is_contact_suite = results.get("suite") == "contact"
    for result in results["trials"]:
        trace = Path(result["artifacts"]["trace"])
        rows = list(load_jsonl(trace))
        summary = _analyze_trial(result, rows)
        summaries.append(summary)
        event_records.append(
            {
                "trial": {
                    key: summary[key]
                    for key in ("task", "seed", "hold_kind", "setting")
                },
                "events": {
                    key: value for key, value in summary.items() if key.endswith("_s")
                },
            }
        )
        if not is_contact_suite:
            name = f"{summary['task']}_seed{summary['seed']}_{summary['hold_kind']}_{summary['setting']}.png"
            _plot_trial(rows, summary, plots / name)
            _plot_contact_map(
                rows,
                summary,
                plots / f"{Path(name).stem}_contact_map.png",
            )

    if not summaries:
        raise ValueError("Suite contains no trials")
    comparison_summaries = list(summaries)
    comparison_sources: list[str] = []
    for root in args.comparison_root:
        resolved = root.expanduser().resolve()
        summary_path = resolved / "analysis" / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Comparison analysis is absent: {summary_path}")
        values = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(values, list) or not all(
            isinstance(value, dict) for value in values
        ):
            raise ValueError(f"Invalid comparison summary: {summary_path}")
        comparison_summaries.extend(values)
        comparison_sources.append(str(summary_path))
    if results.get("suite") == "dynamics":
        _refine_dynamic_labels(summaries, comparison_summaries)
    fields = list(summaries[0])
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    (output / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "events.json").write_text(
        json.dumps(event_records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    setting_summary = _setting_summary(
        summaries if is_contact_suite else comparison_summaries
    )
    (output / "setting_summary.json").write_text(
        json.dumps(setting_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if is_contact_suite:
        contact_object_command_summary = _setting_summary(
            summaries, split_command_variant=True
        )
        (output / "contact_model_by_object_command.json").write_text(
            json.dumps(contact_object_command_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (output / "contact_model_by_object_command.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(contact_object_command_summary[0])
            )
            writer.writeheader()
            writer.writerows(contact_object_command_summary)
    paired = _paired_interventions(
        str(results.get("suite")), summaries, comparison_summaries
    )
    (output / "paired_interventions.json").write_text(
        json.dumps(paired, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation: dict[str, Any] | None = None
    contact_summary: list[dict[str, Any]] | None = None
    contact_plots: dict[str, Path] = {}
    if is_contact_suite:
        validation = _contact_validation(results, summaries, comparison_summaries)
        (output / "validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not validation["passed"]:
            raise RuntimeError(
                "Contact-model experiment failed validation; inspect "
                f"{output / 'validation.json'}"
            )
        contact_summary = _contact_condition_summary(summaries)
        (output / "contact_model_summary.json").write_text(
            json.dumps(contact_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (output / "contact_model_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(contact_summary[0]))
            writer.writeheader()
            writer.writerows(contact_summary)
        (output / "metric_definitions.json").write_text(
            json.dumps(
                {
                    "relative_vertical_slip": (
                        "(TCP position - object position)_z minus its value at "
                        "first bilateral grasp; positive is object motion downward "
                        "relative to the gripper"
                    ),
                    "first_detectable_slip": (
                        "relative vertical slip >= 0.0001 m for 10 consecutive "
                        "physics samples (0.020 s at the effective 0.002 s timestep)"
                    ),
                    "mechanically_stable": (
                        ">=95% bilateral hold contact, no sustained contact loss or "
                        "table impact, and maximum downward slip <=0.002 m"
                    ),
                    "stationary_slip_velocity": (
                        "instantaneous TCP_z velocity minus object_z velocity, "
                        "summarized only while bilateral contact is present during hold"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        contact_plots = _plot_contact_comparison(results, summaries, plots)
        suite_plot = None
    else:
        suite_plot = _plot_suite(comparison_summaries, plots)
    (output / "plot_manifest.json").write_text(
        json.dumps(
            {
                "suite_outcomes": None if suite_plot is None else str(suite_plot),
                "required_contact_plots": {
                    key: str(value) for key, value in contact_plots.items()
                },
                "per_trial_plot_count": 0 if is_contact_suite else len(summaries),
                "dynamic_plot_count": sum(
                    int(row["dynamic_sample_count"] > 0) for row in summaries
                ),
                "comparison_summary_sources": comparison_sources,
                "comparison_trial_count": len(comparison_summaries),
                "contact_validation": (
                    None if validation is None else str(output / "validation.json")
                ),
                "contact_condition_summary": (
                    None
                    if contact_summary is None
                    else str(output / "contact_model_summary.json")
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps({"trial_count": len(summaries), "output_dir": str(output)}, indent=2)
    )


if __name__ == "__main__":
    main()
