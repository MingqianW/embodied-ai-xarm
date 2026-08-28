#!/usr/bin/env python3
"""Analyze detailed c1/c2/c5 normal-versus-latch gripper-slip traces."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
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
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default="red_block")
    parser.add_argument("--model-id", default="B")
    parser.add_argument("--seed", action="append", type=int, dest="seeds", required=True)
    parser.add_argument("--latch-raw", type=float, default=50.0)
    return parser


def _first_sustained(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    *,
    count: int,
    start: int = 0,
) -> dict[str, Any] | None:
    run = 0
    beginning = start
    for index in range(start, len(rows)):
        if predicate(rows[index]):
            if run == 0:
                beginning = index
            run += 1
            if run >= count:
                return rows[beginning]
        else:
            run = 0
    return None


def _action_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous: tuple[int, int, int] | None = None
    for row in rows:
        command = row["command"]
        key = (
            int(command["inference_index"]),
            int(command["action_index_in_chunk"]),
            int(command["action_step"]),
        )
        if key != previous:
            result.append(row)
            previous = key
    return result


def _event_time(row: dict[str, Any] | None) -> float | None:
    return None if row is None else float(row["sim_time_s"])


def _summarize_trace(
    *,
    chunk_steps: int,
    seed: int,
    intervention: str,
    trace_path: Path,
    result_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(load_jsonl(trace_path))
    if not rows:
        raise ValueError(f"Empty trace: {trace_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    bilateral = _first_sustained(
        rows, lambda row: bool(row["contacts"]["bilateral"]), count=5
    )
    bilateral_index = 0 if bilateral is None else int(bilateral["sample_index"])
    lift_5mm = _first_sustained(
        rows,
        lambda row: float(row["object"]["lift_height_m"]) >= 0.005,
        count=5,
        start=bilateral_index,
    )
    lift_5cm = _first_sustained(
        rows,
        lambda row: float(row["object"]["lift_height_m"]) >= 0.05,
        count=5,
        start=bilateral_index,
    )
    slip = _first_sustained(
        rows,
        lambda row: (
            row["relative"]["downward_slip_m"] is not None
            and float(row["relative"]["downward_slip_m"]) >= 0.002
        ),
        count=10,
        start=bilateral_index,
    )
    loss_start = (
        bilateral_index + 1
        if lift_5mm is None
        else int(lift_5mm["sample_index"])
    )
    loss = _first_sustained(
        rows,
        lambda row: (
            int(row["contacts"]["left_target_count"])
            + int(row["contacts"]["right_target_count"])
        )
        == 0,
        count=10,
        start=loss_start,
    )
    impact = _first_sustained(
        rows,
        lambda row: int(row["contacts"]["target_table_count"]) > 0,
        count=2,
        start=(
            bilateral_index
            if lift_5mm is None
            else int(lift_5mm["sample_index"])
        ),
    )
    actions = _action_rows(rows)
    reopen = None
    transitions: list[dict[str, Any]] = []
    for previous, current in zip(actions, actions[1:], strict=False):
        previous_command = previous["command"]
        command = current["command"]
        returned_previous = previous_command["returned_action"]
        returned = command["returned_action"]
        if returned_previous is None or returned is None:
            continue
        returned_previous_array = np.asarray(returned_previous, dtype=np.float64)
        returned_array = np.asarray(returned, dtype=np.float64)
        gripper_delta = float(returned_array[6] - returned_previous_array[6])
        cross_chunk = int(command["inference_index"]) != int(
            previous_command["inference_index"]
        )
        transition = {
            "chunk_steps": chunk_steps,
            "seed": seed,
            "intervention": intervention,
            "time_s": float(current["sim_time_s"]),
            "cross_chunk": cross_chunk,
            "gripper_delta_raw": gripper_delta,
            "arm_action_delta_norm_rad": float(
                np.linalg.norm(returned_array[:6] - returned_previous_array[:6])
            ),
            "tcp_position_delta_m": float(
                np.linalg.norm(
                    np.asarray(current["tcp"]["position_m"], dtype=np.float64)
                    - np.asarray(previous["tcp"]["position_m"], dtype=np.float64)
                )
            ),
        }
        transitions.append(transition)
        if (
            reopen is None
            and bilateral is not None
            and float(current["sim_time_s"]) >= float(bilateral["sim_time_s"])
            and gripper_delta >= 10.0
        ):
            reopen = current

    loss_time = _event_time(loss)
    preceding_boundaries = [
        row
        for row in transitions
        if row["cross_chunk"]
        and loss_time is not None
        and float(row["time_s"]) <= loss_time
    ]
    last_boundary = preceding_boundaries[-1] if preceding_boundaries else None
    boundary_to_loss = (
        None if last_boundary is None else loss_time - float(last_boundary["time_s"])
    )
    post_grasp = rows[bilateral_index:] if bilateral is not None else []
    slip_values = [
        float(row["relative"]["downward_slip_m"])
        for row in post_grasp
        if row["relative"]["downward_slip_m"] is not None
    ]
    saturation_values = [
        float(row["actuator"]["force_fraction"])
        for row in post_grasp
        if row["actuator"]["force_fraction"] is not None
    ]
    dropped = bool(
        bilateral is not None
        and lift_5mm is not None
        and (loss is not None or impact is not None)
    )
    reopen_time = _event_time(reopen)
    if not dropped:
        failure_label = "NONE"
    elif (
        reopen_time is not None
        and loss_time is not None
        and 0.0 <= loss_time - reopen_time <= 0.5
    ):
        failure_label = "POLICY_RELEASE"
    elif (
        boundary_to_loss is not None
        and 0.0 <= boundary_to_loss <= 0.1
        and last_boundary is not None
        and abs(float(last_boundary["gripper_delta_raw"])) >= 10.0
    ):
        failure_label = "CHUNK_BOUNDARY_RELEASE"
    elif slip is not None and bool(slip["contacts"]["bilateral"]):
        failure_label = "STATIC_CONTACT_SLIP"
    elif saturation_values and statistics.mean(value >= 0.99 for value in saturation_values) >= 0.5:
        failure_label = "GRIP_FORCE_FAILURE"
    elif loss is not None:
        failure_label = "CONTACT_LOSS"
    else:
        failure_label = "UNKNOWN"

    summary = {
        "chunk_steps": chunk_steps,
        "seed": seed,
        "intervention": intervention,
        "trace_path": str(trace_path),
        "result_path": str(result_path),
        "physics_sample_count": len(rows),
        "action_count": len(actions),
        "bilateral_grasp": bilateral is not None,
        "bilateral_grasp_time_s": _event_time(bilateral),
        "lift_5mm": lift_5mm is not None,
        "lift_5mm_time_s": _event_time(lift_5mm),
        "lift_5cm": lift_5cm is not None,
        "lift_5cm_time_s": _event_time(lift_5cm),
        "slip_2mm_time_s": _event_time(slip),
        "contact_loss_time_s": loss_time,
        "table_impact_time_s": _event_time(impact),
        "policy_reopen_time_s": reopen_time,
        "last_chunk_boundary_before_loss_s": (
            None if last_boundary is None else float(last_boundary["time_s"])
        ),
        "chunk_boundary_to_loss_s": boundary_to_loss,
        "dropped_after_grasp": dropped,
        "maximum_downward_slip_m": max(slip_values, default=None),
        "post_grasp_force_saturation_fraction": (
            statistics.mean(value >= 0.99 for value in saturation_values)
            if saturation_values
            else None
        ),
        "formal_success": bool(result["episode"]["success"]),
        "formal_valid": bool(result["episode"]["valid"]),
        "diagnostic_failure_label": failure_label,
        "final_returned_gripper_raw": (
            None
            if not actions
            else actions[-1]["command"]["gripper_returned_raw"]
        ),
        "final_effective_gripper_raw": (
            None
            if not actions
            else actions[-1]["command"]["gripper_clamped_raw"]
        ),
        "final_sim_ctrl_m": float(rows[-1]["actuator"]["ctrl_m"]),
        "final_actual_gripper_raw": float(rows[-1]["fingers"]["left_raw_equivalent"]),
    }
    return summary, rows, transitions


def _plot_trace(rows: list[dict[str, Any]], title: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    time = np.asarray([float(row["sim_time_s"]) for row in rows])
    actions = _action_rows(rows)
    action_time = np.asarray([float(row["sim_time_s"]) for row in actions])
    network = np.asarray(
        [
            math.nan
            if row["command"]["gripper_network_normalized"] is None
            else float(row["command"]["gripper_network_normalized"])
            for row in actions
        ]
    )
    returned = np.asarray(
        [float(row["command"]["gripper_returned_raw"]) for row in actions]
    )
    effective = np.asarray(
        [float(row["command"]["gripper_clamped_raw"]) for row in actions]
    )
    actual = np.asarray([float(row["fingers"]["left_raw_equivalent"]) for row in rows])
    ctrl = np.asarray([float(row["actuator"]["ctrl_m"]) for row in rows])
    qpos = np.asarray([float(row["fingers"]["left_qpos_m"]) for row in rows])
    slip = np.asarray(
        [
            math.nan
            if row["relative"]["downward_slip_m"] is None
            else float(row["relative"]["downward_slip_m"])
            for row in rows
        ]
    )
    lift = np.asarray([float(row["object"]["lift_height_m"]) for row in rows])
    actuator = np.asarray([float(row["actuator"]["force_n"]) for row in rows])
    normal = np.asarray(
        [float(row["contacts"]["target_gripper_normal_sum_n"]) for row in rows]
    )
    left_normal = np.asarray(
        [float(row["contacts"]["left_target_normal_sum_n"]) for row in rows]
    )
    right_normal = np.asarray(
        [float(row["contacts"]["right_target_normal_sum_n"]) for row in rows]
    )
    left = np.asarray([int(row["contacts"]["left_target_count"] > 0) for row in rows])
    right = np.asarray([int(row["contacts"]["right_target_count"] > 0) for row in rows])

    figure, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    axes[0].step(action_time, network, where="post", label="reconstructed network gripper")
    axes[0].set_ylabel("network value")
    axes[0].legend(loc="best")
    axes[1].step(action_time, returned, where="post", label="returned raw")
    axes[1].step(action_time, effective, where="post", label="effective raw")
    axes[1].plot(time, actual, label="actual raw equivalent", alpha=0.75)
    axes[1].invert_yaxis()
    axes[1].set_ylabel("raw (closed down)")
    axes[1].legend(loc="best")
    axes[2].plot(time, ctrl, label="final MuJoCo ctrl")
    axes[2].plot(time, qpos, label="finger qpos")
    axes[2].set_ylabel("slide position (m)")
    axes[2].legend(loc="best")
    axes[3].plot(time, 1000.0 * lift, label="object lift")
    axes[3].plot(time, 1000.0 * slip, label="downward relative slip")
    axes[3].set_ylabel("mm")
    axes[3].legend(loc="best")
    axes[4].plot(time, actuator, label="actuator force")
    axes[4].plot(time, normal, label="normal-force total")
    axes[4].plot(time, left_normal, label="left normal force", alpha=0.8)
    axes[4].plot(time, right_normal, label="right normal force", alpha=0.8)
    axes[4].step(time, left, where="post", label="left contact")
    axes[4].step(time, right, where="post", label="right contact")
    axes[4].set_ylabel("N / contact")
    axes[4].set_xlabel("simulation time (s)")
    axes[4].legend(loc="best")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=165)
    plt.close(figure)


def _aggregate(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for chunk_steps in (1, 2, 5):
        for intervention in ("normal", "latch"):
            selected = [
                row
                for row in summaries
                if row["chunk_steps"] == chunk_steps
                and row["intervention"] == intervention
            ]
            result.append(
                {
                    "chunk_steps": chunk_steps,
                    "intervention": intervention,
                    "episode_count": len(selected),
                    "bilateral_grasp_count": sum(row["bilateral_grasp"] for row in selected),
                    "lift_5cm_count": sum(row["lift_5cm"] for row in selected),
                    "dropped_after_grasp_count": sum(row["dropped_after_grasp"] for row in selected),
                    "formal_success_count": sum(row["formal_success"] for row in selected),
                    "drop_after_grasp_rate": (
                        statistics.mean(row["dropped_after_grasp"] for row in selected)
                        if selected
                        else None
                    ),
                    "formal_success_rate": (
                        statistics.mean(row["formal_success"] for row in selected)
                        if selected
                        else None
                    ),
                }
            )
    return result


def _plot_aggregate(
    aggregates: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    output: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    x = np.arange(3, dtype=np.float64)
    figure, axis = plt.subplots(figsize=(9, 5))
    for offset, intervention in ((-0.18, "normal"), (0.18, "latch")):
        rows = [row for row in aggregates if row["intervention"] == intervention]
        axis.bar(
            x + offset,
            [row["drop_after_grasp_rate"] for row in rows],
            width=0.36,
            label=intervention,
        )
    axis.set_xticks(x, ["c1", "c2", "c5"])
    axis.set_ylabel("drop-after-grasp rate")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Normal policy versus diagnostic gripper latch")
    axis.legend()
    figure.tight_layout()
    latch_path = output / "normal_vs_latch_drop_rate.png"
    figure.savefig(latch_path, dpi=175)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    normal_transitions = [
        row for row in transitions if row["intervention"] == "normal"
    ]
    for cross, label, color in (
        (False, "within chunk", "tab:blue"),
        (True, "cross chunk", "tab:orange"),
    ):
        selected = [
            row
            for row in normal_transitions
            if bool(row["cross_chunk"]) == cross
        ]
        axes[0].hist(
            [row["gripper_delta_raw"] for row in selected],
            bins=50,
            alpha=0.5,
            label=label,
            color=color,
        )
        axes[1].hist(
            [row["arm_action_delta_norm_rad"] for row in selected],
            bins=50,
            alpha=0.5,
            label=label,
            color=color,
        )
    axes[0].set_xlabel("consecutive gripper delta (raw; positive=open)")
    axes[1].set_xlabel("consecutive arm-action delta norm (rad)")
    for axis in axes:
        axis.set_ylabel("transition count")
        axis.legend()
    figure.suptitle("Within-chunk versus cross-chunk discontinuities")
    figure.tight_layout()
    chunk_path = output / "chunk_boundary_discontinuities.png"
    figure.savefig(chunk_path, dpi=175)
    plt.close(figure)
    return {"normal_vs_latch": str(latch_path), "chunk_boundaries": str(chunk_path)}


def _transition_summary(
    transitions: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for intervention in ("normal", "latch"):
        result[intervention] = {}
        for cross, label in ((False, "within_chunk"), (True, "cross_chunk")):
            selected = [
                row
                for row in transitions
                if row["intervention"] == intervention
                and bool(row["cross_chunk"]) == cross
            ]
            gripper = np.asarray(
                [abs(float(row["gripper_delta_raw"])) for row in selected],
                dtype=np.float64,
            )
            arm = np.asarray(
                [float(row["arm_action_delta_norm_rad"]) for row in selected],
                dtype=np.float64,
            )
            result[intervention][label] = {
                "count": len(selected),
                "absolute_gripper_delta_raw_median": (
                    float(np.median(gripper)) if len(gripper) else None
                ),
                "absolute_gripper_delta_raw_q95": (
                    float(np.quantile(gripper, 0.95)) if len(gripper) else None
                ),
                "arm_action_delta_norm_rad_median": (
                    float(np.median(arm)) if len(arm) else None
                ),
                "arm_action_delta_norm_rad_q95": (
                    float(np.quantile(arm, 0.95)) if len(arm) else None
                ),
            }
    return result


def _drop_boundary_summary(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for intervention in ("normal", "latch"):
        dropped = [
            row
            for row in summaries
            if row["intervention"] == intervention
            and row["dropped_after_grasp"]
        ]
        delays = [
            float(row["chunk_boundary_to_loss_s"])
            for row in dropped
            if row["chunk_boundary_to_loss_s"] is not None
            and float(row["chunk_boundary_to_loss_s"]) >= 0.0
        ]
        result[intervention] = {
            "drop_count": len(dropped),
            "drop_count_with_boundary_delay": len(delays),
            "within_0p1s_of_boundary_count": sum(value <= 0.1 for value in delays),
            "within_0p2s_of_boundary_count": sum(value <= 0.2 for value in delays),
            "boundary_to_loss_delay_s_median": (
                float(np.median(delays)) if delays else None
            ),
        }
    return result


def main() -> None:
    args = _parser().parse_args()
    base = args.output_base.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing analysis output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    plots = output / "plots"
    plots.mkdir()
    summaries: list[dict[str, Any]] = []
    all_transitions: list[dict[str, Any]] = []
    representative_seed = min(args.seeds)
    for chunk_steps in (1, 2, 5):
        for seed in args.seeds:
            for intervention, suffix in (
                ("normal", "normal"),
                ("latch", f"latch_raw{args.latch_raw:g}"),
            ):
                root = base / f"c{chunk_steps}_seed{seed}_{suffix}"
                episode = root / "models" / args.model_id / "tasks" / args.task / f"seed_{seed}"
                summary, rows, transitions = _summarize_trace(
                    chunk_steps=chunk_steps,
                    seed=seed,
                    intervention=intervention,
                    trace_path=episode / "physics_trace.jsonl",
                    result_path=episode / "result.json",
                )
                summaries.append(summary)
                all_transitions.extend(transitions)
                if seed == representative_seed:
                    _plot_trace(
                        rows,
                        f"c{chunk_steps} seed {seed} {intervention}",
                        plots / f"c{chunk_steps}_seed{seed}_{intervention}.png",
                    )

    aggregates = _aggregate(summaries)
    plot_manifest = _plot_aggregate(aggregates, all_transitions, plots)
    taxonomy = Counter(
        row["diagnostic_failure_label"] for row in summaries if row["intervention"] == "normal"
    )
    taxonomy_total = sum(taxonomy.values())
    document = {
        "schema_version": "xarm_policy_gripper_slip_matrix_v1",
        "output_base": str(base),
        "task": args.task,
        "seeds": args.seeds,
        "latch_raw": args.latch_raw,
        "traces": summaries,
        "aggregates": aggregates,
        "failure_taxonomy_normal_policy": [
            {
                "failure_type": key,
                "count": count,
                "percentage": 100.0 * count / taxonomy_total if taxonomy_total else 0.0,
            }
            for key, count in sorted(taxonomy.items())
        ],
        "transition_counts": {
            intervention: {
                "within_chunk": sum(
                    row["intervention"] == intervention
                    and not row["cross_chunk"]
                    for row in all_transitions
                ),
                "cross_chunk": sum(
                    row["intervention"] == intervention and row["cross_chunk"]
                    for row in all_transitions
                ),
            }
            for intervention in ("normal", "latch")
        },
        "transition_summary": _transition_summary(all_transitions),
        "drop_boundary_summary": _drop_boundary_summary(summaries),
        "plots": plot_manifest,
        "classification_limit": (
            "GRIP_FORCE_FAILURE remains provisional until paired scripted force "
            "interventions are joined to this evidence."
        ),
    }
    (output / "results.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "trace_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output / "transitions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "chunk_steps",
                "seed",
                "intervention",
                "time_s",
                "cross_chunk",
                "gripper_delta_raw",
                "arm_action_delta_norm_rad",
                "tcp_position_delta_m",
            ],
        )
        writer.writeheader()
        writer.writerows(all_transitions)
    print(json.dumps({"trace_count": len(summaries), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
