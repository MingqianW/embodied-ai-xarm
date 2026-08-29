#!/usr/bin/env python3
"""Read-only behavioral audit of real and MuJoCo xArm gripper trajectories.

The LeRobot ``actions`` column is treated only as the next-state imitation
label.  It is never interpreted as an independently recorded low-level
command.  Source image columns are not read.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np


from simulation.resources import repository_root


PROJECT_ROOT = repository_root()

PICK_PREFIX = "pick up "
GRIPPER_INDEX = 6
MOTION_THRESHOLD_RAW = 1.0
MIN_SEGMENT_DISPLACEMENT_RAW = 10.0
MAX_ZERO_GAP_FRAMES = 1
HOLD_STABLE_RANGE_RAW = 10.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=PROJECT_ROOT / "simulation/assets/xarm6/xarm6_pick_scene.xml",
    )
    return parser


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _fixed_list(table: Any, key: str) -> np.ndarray:
    array = table[key].combine_chunks()
    if not hasattr(array, "values") or int(array.type.list_size) != 7:
        raise ValueError(f"Expected a fixed-size 7D list column: {key}")
    return np.asarray(
        array.values.to_numpy(zero_copy_only=False), dtype=np.float64
    ).reshape(-1, 7)


def _quantiles(values: Any) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "count": 0,
            "min": None,
            "q01": None,
            "q05": None,
            "q25": None,
            "median": None,
            "mean": None,
            "q75": None,
            "q95": None,
            "q99": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def next_state_label_metrics(
    states_by_episode: list[np.ndarray],
    labels_by_episode: list[np.ndarray],
) -> dict[str, Any]:
    same_state: list[np.ndarray] = []
    same_label: list[np.ndarray] = []
    next_state: list[np.ndarray] = []
    shifted_label: list[np.ndarray] = []
    for states, labels in zip(states_by_episode, labels_by_episode, strict=True):
        same_state.append(states[:, GRIPPER_INDEX])
        same_label.append(labels[:, GRIPPER_INDEX])
        if len(states) > 1:
            next_state.append(states[1:, GRIPPER_INDEX])
            shifted_label.append(labels[:-1, GRIPPER_INDEX])
    state = np.concatenate(same_state)
    label = np.concatenate(same_label)
    shifted_state = np.concatenate(next_state)
    shifted = np.concatenate(shifted_label)
    same_difference = label - state
    shifted_difference = shifted - shifted_state
    return {
        "same_frame": {
            "correlation": _safe_corr(state, label),
            "mean_absolute_difference_raw": float(np.mean(np.abs(same_difference))),
            "median_absolute_difference_raw": float(np.median(np.abs(same_difference))),
            "fraction_exactly_equal": float(np.mean(same_difference == 0.0)),
            "fraction_within_1_raw": float(np.mean(np.abs(same_difference) <= 1.0)),
        },
        "label_t_vs_state_t_plus_1": {
            "pairs": int(len(shifted)),
            "correlation": _safe_corr(shifted, shifted_state),
            "maximum_absolute_difference_raw": float(
                np.max(np.abs(shifted_difference))
            ),
            "mean_absolute_difference_raw": float(np.mean(np.abs(shifted_difference))),
            "fraction_exactly_equal": float(np.mean(shifted_difference == 0.0)),
            "fraction_within_1e_minus_6_raw": float(
                np.mean(np.abs(shifted_difference) <= 1e-6)
            ),
        },
        "interpretation": (
            "actions[6]_t is a copied next-state imitation label, not an "
            "independent gripper command"
        ),
    }


def _motion_sign(delta: np.ndarray) -> np.ndarray:
    sign = np.zeros(len(delta), dtype=np.int8)
    sign[delta > MOTION_THRESHOLD_RAW] = 1
    sign[delta < -MOTION_THRESHOLD_RAW] = -1
    return sign


def motion_segments(
    values: np.ndarray,
    times: np.ndarray,
    *,
    episode_index: int,
    task: str,
    source: str,
) -> list[dict[str, Any]]:
    """Extract robust measured-state motion bouts; positive means opening."""

    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if len(values) < 2:
        return []
    signs = _motion_sign(np.diff(values))
    segments: list[dict[str, Any]] = []
    index = 0
    while index < len(signs):
        direction = int(signs[index])
        if direction == 0:
            index += 1
            continue
        start = index
        last_motion = index
        cursor = index + 1
        while cursor < len(signs):
            if int(signs[cursor]) == direction:
                last_motion = cursor
                cursor += 1
                continue
            if int(signs[cursor]) == -direction:
                break
            gap_end = cursor
            while gap_end < len(signs) and int(signs[gap_end]) == 0:
                gap_end += 1
            gap = gap_end - cursor
            if (
                gap <= MAX_ZERO_GAP_FRAMES
                and gap_end < len(signs)
                and int(signs[gap_end]) == direction
            ):
                last_motion = gap_end
                cursor = gap_end + 1
                continue
            break
        end = last_motion + 1
        displacement = float(values[end] - values[start])
        duration = float(times[end] - times[start])
        if abs(displacement) >= MIN_SEGMENT_DISPLACEMENT_RAW and duration > 0.0:
            step_velocity = np.diff(values[start : end + 1]) / np.diff(
                times[start : end + 1]
            )
            directed_velocity = direction * step_velocity
            directed_velocity = directed_velocity[directed_velocity > 0.0]
            segments.append(
                {
                    "source": source,
                    "episode_index": episode_index,
                    "task": task,
                    "direction": "opening" if direction > 0 else "closing",
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "start_time_s": float(times[start]),
                    "end_time_s": float(times[end]),
                    "duration_s": duration,
                    "start_state_raw": float(values[start]),
                    "end_state_raw": float(values[end]),
                    "state_displacement_raw": displacement,
                    "absolute_displacement_raw": abs(displacement),
                    "effective_speed_raw_per_s": abs(displacement) / duration,
                    "median_moving_speed_raw_per_s": (
                        float(np.median(directed_velocity))
                        if directed_velocity.size
                        else None
                    ),
                    "true_command_available": False,
                }
            )
        index = max(cursor, end)
    return segments


def _phase_boundaries(gripper: np.ndarray, tcp_z: np.ndarray) -> dict[str, int] | None:
    """Infer closure/lift/hold using only feedback and arm forward kinematics."""

    length = len(gripper)
    if length < 20:
        return None
    search_end = max(2, int(0.8 * length))
    open_index = int(np.argmax(gripper[:search_end]))
    q10, q90 = np.quantile(gripper, [0.1, 0.9])
    if q90 - q10 < 50.0:
        return None
    threshold = 0.5 * (q10 + q90)
    candidates = np.flatnonzero(gripper[open_index:] <= threshold)
    if not candidates.size:
        return None
    closure = open_index + int(candidates[0])
    if closure >= length - 5:
        return None
    grasp_z = float(np.min(tcp_z[closure : min(length, closure + 15)]))
    lift_candidates = np.flatnonzero(tcp_z[closure:] >= grasp_z + 0.01)
    if not lift_candidates.size:
        return None
    lift = closure + int(lift_candidates[0])
    peak = lift + int(np.argmax(tcp_z[lift:]))
    rise = float(tcp_z[peak] - grasp_z)
    if rise < 0.02:
        return None
    hold_candidates = np.flatnonzero(tcp_z[lift : peak + 1] >= grasp_z + 0.8 * rise)
    hold = lift + int(hold_candidates[0]) if hold_candidates.size else peak
    end = length
    descent = np.flatnonzero(tcp_z[peak:] <= float(tcp_z[peak]) - 0.01)
    if descent.size:
        end = peak + int(descent[0])
    if end <= hold + 1:
        end = length
    return {
        "closure_start": closure,
        "lift_start": lift,
        "hold_start": hold,
        "hold_end": end,
    }


def _tcp_z_trajectory(
    model: Any,
    data: Any,
    states: np.ndarray,
    joint_qpos_addresses: np.ndarray,
    tcp_site_id: int,
) -> np.ndarray:
    import mujoco

    from simulation.robot.joint_mapping import raw_arm_state_to_mujoco_qpos

    result = np.empty(len(states), dtype=np.float64)
    for index, state in enumerate(states):
        data.qpos[joint_qpos_addresses] = raw_arm_state_to_mujoco_qpos(state[:6])
        mujoco.mj_kinematics(model, data)
        result[index] = float(data.site_xpos[tcp_site_id, 2])
    return result


def _summarize_segments(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for direction in ("opening", "closing"):
        selected = [row for row in rows if row["direction"] == direction]
        result[direction] = {
            "count": len(selected),
            "duration_s": _quantiles([row["duration_s"] for row in selected]),
            "absolute_displacement_raw": _quantiles(
                [row["absolute_displacement_raw"] for row in selected]
            ),
            "effective_speed_raw_per_s": _quantiles(
                [row["effective_speed_raw_per_s"] for row in selected]
            ),
            "near_full_range_count": int(
                sum(
                    row["absolute_displacement_raw"] >= 0.8 * (845.0 - 50.0)
                    for row in selected
                )
            ),
        }
    return result


def _by_task_hold_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in sorted({str(row["task"]) for row in rows}):
        selected = [row for row in rows if row["task"] == task]
        result[task] = {
            "episode_count": len(selected),
            "state_at_lift_raw": _quantiles(
                [row["state_at_lift_raw"] for row in selected]
            ),
            "hold_state_median_raw": _quantiles(
                [row["hold_state_median_raw"] for row in selected]
            ),
            "hold_state_range_raw": _quantiles(
                [row["hold_state_range_raw"] for row in selected]
            ),
            "hold_abs_net_change_raw": _quantiles(
                [row["hold_abs_net_change_raw"] for row in selected]
            ),
            "hold_duration_s": _quantiles([row["hold_duration_s"] for row in selected]),
            "fraction_stable_within_10_raw": float(
                np.mean([row["stable_within_10_raw"] for row in selected])
            ),
        }
    return result


def _directional_speed_by_state(
    states_by_episode: list[np.ndarray], fps: float
) -> dict[str, Any]:
    """Summarize apparent state speed by direction and raw-state region."""

    edges = np.arange(0.0, 901.0, 100.0)
    samples: dict[str, list[list[float]]] = {
        "opening": [[] for _ in range(len(edges) - 1)],
        "closing": [[] for _ in range(len(edges) - 1)],
    }
    for states in states_by_episode:
        values = states[:, GRIPPER_INDEX]
        if len(values) < 2:
            continue
        delta = np.diff(values)
        midpoint = 0.5 * (values[:-1] + values[1:])
        bin_index = np.clip(
            np.searchsorted(edges, midpoint, side="right") - 1,
            0,
            len(edges) - 2,
        )
        for index, change in zip(bin_index, delta, strict=True):
            if change > MOTION_THRESHOLD_RAW:
                samples["opening"][int(index)].append(float(change * fps))
            elif change < -MOTION_THRESHOLD_RAW:
                samples["closing"][int(index)].append(float(-change * fps))
    return {
        direction: {
            f"{int(edges[index])}-{int(edges[index + 1])}": _quantiles(values)
            for index, values in enumerate(direction_samples)
        }
        for direction, direction_samples in samples.items()
    }


def _analyze_dataset(
    root: Path,
    *,
    label: str,
    model: Any,
    data: Any,
    joint_qpos_addresses: np.ndarray,
    tcp_site_id: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    import pyarrow.parquet as pq

    root = root.expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    episodes = {
        int(row["episode_index"]): row for row in _jsonl(root / "meta/episodes.jsonl")
    }
    tasks = {
        int(row["task_index"]): str(row["task"])
        for row in _jsonl(root / "meta/tasks.jsonl")
    }
    expected = int(info["total_episodes"])
    if set(episodes) != set(range(expected)):
        raise ValueError(f"Incomplete episode metadata: {root}")
    fps = float(info["fps"])
    states_by_episode: list[np.ndarray] = []
    labels_by_episode: list[np.ndarray] = []
    state_values: list[np.ndarray] = []
    all_dt: list[np.ndarray] = []
    timestamp_nominal_error: list[np.ndarray] = []
    segments: list[dict[str, Any]] = []
    holds: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    missing_files: list[str] = []

    for episode_index in range(expected):
        meta = episodes[episode_index]
        task = str(meta["tasks"][0])
        path = root / str(info["data_path"]).format(
            episode_chunk=episode_index // int(info["chunks_size"]),
            episode_index=episode_index,
        )
        if not path.is_file():
            missing_files.append(str(path))
            continue
        table = pq.read_table(
            path,
            columns=["state", "actions", "timestamp", "frame_index", "task_index"],
        )
        states = _fixed_list(table, "state")
        labels = _fixed_list(table, "actions")
        timestamps = np.asarray(
            table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        frame_indices = np.asarray(
            table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        task_indices = np.asarray(
            table["task_index"].combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        if len(states) != int(meta["length"]) or states.shape != labels.shape:
            raise ValueError(f"Shape/metadata mismatch: {path}")
        if not np.isfinite(states).all() or not np.isfinite(labels).all():
            raise ValueError(f"NaN/Inf in {path}")
        if (
            np.any(task_indices != task_indices[0])
            or tasks[int(task_indices[0])] != task
        ):
            raise ValueError(f"Task metadata mismatch: {path}")
        states_by_episode.append(states)
        labels_by_episode.append(labels)
        state_values.append(states[:, GRIPPER_INDEX])
        if len(timestamps) > 1:
            all_dt.append(np.diff(timestamps))
        timestamp_nominal_error.append(timestamps - frame_indices / fps)
        segments.extend(
            motion_segments(
                states[:, GRIPPER_INDEX],
                timestamps,
                episode_index=episode_index,
                task=task,
                source=label,
            )
        )
        for frame_index, timestamp, state, next_label in zip(
            frame_indices,
            timestamps,
            states[:, GRIPPER_INDEX],
            labels[:, GRIPPER_INDEX],
            strict=True,
        ):
            frame_rows.append(
                {
                    "source": label,
                    "episode_index": episode_index,
                    "task": task,
                    "frame_index": int(frame_index),
                    "nominal_time_s": float(timestamp),
                    "gripper_state_raw": float(state),
                    "dataset_action6_next_state_label_raw": float(next_label),
                    "independent_gripper_command": "",
                }
            )
        if task.startswith(PICK_PREFIX):
            tcp_z = _tcp_z_trajectory(
                model,
                data,
                states,
                joint_qpos_addresses,
                tcp_site_id,
            )
            boundaries = _phase_boundaries(states[:, GRIPPER_INDEX], tcp_z)
            if boundaries is not None:
                selection = slice(boundaries["hold_start"], boundaries["hold_end"])
                hold_state = states[selection, GRIPPER_INDEX]
                if len(hold_state) >= 2:
                    hold_range = float(np.max(hold_state) - np.min(hold_state))
                    holds.append(
                        {
                            "source": label,
                            "episode_index": episode_index,
                            "task": task,
                            **boundaries,
                            "state_at_lift_raw": float(
                                states[boundaries["lift_start"], GRIPPER_INDEX]
                            ),
                            "hold_state_median_raw": float(np.median(hold_state)),
                            "hold_state_min_raw": float(np.min(hold_state)),
                            "hold_state_max_raw": float(np.max(hold_state)),
                            "hold_state_range_raw": hold_range,
                            "hold_abs_net_change_raw": float(
                                abs(hold_state[-1] - hold_state[0])
                            ),
                            "hold_duration_s": float((len(hold_state) - 1) / fps),
                            "stable_within_10_raw": hold_range <= HOLD_STABLE_RANGE_RAW,
                            "phase_source": (
                                "gripper feedback closure + arm FK TCP lift; no image/contact ground truth"
                            ),
                        }
                    )

    if missing_files:
        raise FileNotFoundError(f"Missing {len(missing_files)} episode files")
    all_state = np.concatenate(state_values)
    dt = np.concatenate(all_dt)
    nominal_error = np.concatenate(timestamp_nominal_error)
    summary = {
        "label": label,
        "root": str(root),
        "codebase_version": info.get("codebase_version"),
        "episode_count": expected,
        "frame_count": int(sum(len(values) for values in state_values)),
        "fps": fps,
        "task_count": len(tasks),
        "task_episode_counts": {
            task: int(sum(str(row["tasks"][0]) == task for row in episodes.values()))
            for task in sorted(tasks.values())
        },
        "timestamp": {
            "source": "LeRobot frame_index/fps; original wall-clock timestamps are not present",
            "dt_s": _quantiles(dt),
            "maximum_abs_timestamp_minus_frame_over_fps_s": float(
                np.max(np.abs(nominal_error))
            ),
        },
        "gripper_state_raw": _quantiles(all_state),
        "state_outside_project_50_845_fraction": float(
            np.mean((all_state < 50.0) | (all_state > 845.0))
        ),
        "motion_segments": _summarize_segments(segments),
        "directional_speed_by_state_bin_raw_per_s": _directional_speed_by_state(
            states_by_episode, fps
        ),
        "pick_phase_episode_count": len(holds),
        "pick_phase_detection_fraction": float(
            len(holds)
            / max(
                1,
                sum(
                    task.startswith(PICK_PREFIX)
                    for task in (str(row["tasks"][0]) for row in episodes.values())
                ),
            )
        ),
        "hold_by_task": _by_task_hold_summary(holds),
        "next_state_label_check": next_state_label_metrics(
            states_by_episode, labels_by_episode
        ),
    }
    return summary, segments, holds, frame_rows


def _comparison(real: dict[str, Any], sim: dict[str, Any]) -> dict[str, Any]:
    def median(section: dict[str, Any], *keys: str) -> float | None:
        value: Any = section
        for key in keys:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        return float(value) if isinstance(value, (int, float)) else None

    result: dict[str, Any] = {
        "scope_warning": (
            "The converted simulation dataset was generated on 2026-08-06 with the prior "
            "simplified slide gripper; it is the authoritative simulation training-data "
            "distribution, not a fresh Menagerie rollout."
        ),
        "global": {},
        "hold_by_task": {},
    }
    metrics = {
        "state_median_raw": (
            median(real, "gripper_state_raw", "median"),
            median(sim, "gripper_state_raw", "median"),
        ),
        "opening_speed_median_raw_per_s": (
            median(
                real,
                "motion_segments",
                "opening",
                "effective_speed_raw_per_s",
                "median",
            ),
            median(
                sim, "motion_segments", "opening", "effective_speed_raw_per_s", "median"
            ),
        ),
        "closing_speed_median_raw_per_s": (
            median(
                real,
                "motion_segments",
                "closing",
                "effective_speed_raw_per_s",
                "median",
            ),
            median(
                sim, "motion_segments", "closing", "effective_speed_raw_per_s", "median"
            ),
        ),
        "closure_duration_median_s": (
            median(real, "motion_segments", "closing", "duration_s", "median"),
            median(sim, "motion_segments", "closing", "duration_s", "median"),
        ),
    }
    for name, (real_value, sim_value) in metrics.items():
        result["global"][name] = {
            "real": real_value,
            "sim": sim_value,
            "sim_minus_real": (
                None
                if real_value is None or sim_value is None
                else sim_value - real_value
            ),
            "sim_over_real": (
                None
                if real_value in (None, 0.0) or sim_value is None
                else sim_value / real_value
            ),
        }
    tasks = sorted(set(real["hold_by_task"]) & set(sim["hold_by_task"]))
    for task in tasks:
        real_task = real["hold_by_task"][task]
        sim_task = sim["hold_by_task"][task]
        result["hold_by_task"][task] = {}
        for key in (
            "state_at_lift_raw",
            "hold_state_median_raw",
            "hold_state_range_raw",
            "hold_abs_net_change_raw",
        ):
            real_value = real_task[key]["median"]
            sim_value = sim_task[key]["median"]
            result["hold_by_task"][task][key] = {
                "real": real_value,
                "sim": sim_value,
                "sim_minus_real": sim_value - real_value,
            }
        result["hold_by_task"][task]["fraction_stable_within_10_raw"] = {
            "real": real_task["fraction_stable_within_10_raw"],
            "sim": sim_task["fraction_stable_within_10_raw"],
            "sim_minus_real": (
                sim_task["fraction_stable_within_10_raw"]
                - real_task["fraction_stable_within_10_raw"]
            ),
        }
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plots(
    real: dict[str, Any],
    sim: dict[str, Any],
    segments: list[dict[str, Any]],
    holds: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    output: Path,
) -> list[str]:
    import matplotlib.pyplot as plt

    paths: list[str] = []
    colors = {"real": "tab:blue", "sim_training_simplified": "tab:orange"}
    figure, axis = plt.subplots(figsize=(9, 5))
    for source in colors:
        values = [row["gripper_state_raw"] for row in frames if row["source"] == source]
        axis.hist(
            values,
            bins=np.linspace(0, 850, 86),
            density=True,
            histtype="step",
            linewidth=1.8,
            color=colors[source],
            label=source,
        )
    axis.set_xlabel("measured/reconstructed gripper state (raw; larger=open)")
    axis.set_ylabel("density")
    axis.set_title("Policy-facing gripper-state coverage")
    axis.legend()
    figure.tight_layout()
    path = output / "gripper_state_coverage.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, direction in zip(axes, ("opening", "closing"), strict=True):
        for source in colors:
            values = [
                row["effective_speed_raw_per_s"]
                for row in segments
                if row["source"] == source and row["direction"] == direction
            ]
            if values:
                upper = float(np.quantile(values, 0.99))
                clipped = [value for value in values if value <= upper]
                axis.hist(
                    clipped,
                    bins=50,
                    density=True,
                    alpha=0.45,
                    color=colors[source],
                    label=source,
                )
        axis.set_title(direction)
        axis.set_xlabel("segment speed (raw units/s; clipped at source q99)")
        axis.legend()
    axes[0].set_ylabel("density")
    figure.tight_layout()
    path = output / "opening_closing_speed.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))

    tasks = sorted({row["task"] for row in holds})
    figure, axis = plt.subplots(figsize=(max(10, len(tasks) * 1.6), 5))
    positions = np.arange(len(tasks), dtype=np.float64)
    for offset, source in ((-0.18, "real"), (0.18, "sim_training_simplified")):
        data = [
            [
                row["hold_state_median_raw"]
                for row in holds
                if row["source"] == source and row["task"] == task
            ]
            for task in tasks
        ]
        valid = [(index, values) for index, values in enumerate(data) if values]
        if valid:
            axis.boxplot(
                [values for _, values in valid],
                positions=[positions[index] + offset for index, _ in valid],
                widths=0.30,
                patch_artist=True,
                boxprops={"facecolor": colors[source], "alpha": 0.45},
                medianprops={"color": "black"},
                manage_ticks=False,
            )
    axis.set_xticks(
        positions,
        [task.removeprefix(PICK_PREFIX) for task in tasks],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel("hold gripper state (raw; larger=open)")
    axis.set_title("Lift/hold state by task (phase inferred without images)")
    figure.tight_layout()
    path = output / "hold_state_by_task.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))
    return paths


def main() -> None:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.robot_xml.expanduser().resolve()))
    data = mujoco.MjData(model)
    addresses = np.asarray(
        [
            int(
                model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index}")
                ]
            )
            for index in range(1, 7)
        ],
        dtype=np.int64,
    )
    tcp_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "tool_center_point"
    )
    if tcp_site_id < 0:
        raise RuntimeError("tool_center_point site is absent")

    real, real_segments, real_holds, real_frames = _analyze_dataset(
        args.real_root,
        label="real",
        model=model,
        data=data,
        joint_qpos_addresses=addresses,
        tcp_site_id=tcp_site_id,
    )
    sim, sim_segments, sim_holds, sim_frames = _analyze_dataset(
        args.sim_root,
        label="sim_training_simplified",
        model=model,
        data=data,
        joint_qpos_addresses=addresses,
        tcp_site_id=tcp_site_id,
    )
    segments = [*real_segments, *sim_segments]
    holds = [*real_holds, *sim_holds]
    frames = [*real_frames, *sim_frames]
    plot_paths = _plots(real, sim, segments, holds, frames, output)
    result = {
        "schema_version": "xarm_real_sim_gripper_behavior_audit_v1",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "semantics_guardrail": (
            "For both converted datasets, actions[6]_t is tested only as the "
            "next-state IL label state[6]_(t+1); it is never treated as actuator ctrl or command."
        ),
        "real": real,
        "simulation_training_dataset": sim,
        "real_vs_simulation_training": _comparison(real, sim),
        "identifiability": {
            "directly_observable": [
                "policy-facing gripper feedback/state distribution",
                "state-only opening/closing trajectories at nominal 10 Hz",
                "state-only segment duration and apparent velocity",
                "accepted-demonstration lift/hold state distributions inferred from arm FK",
                "real/sim next-state IL label consistency",
            ],
            "weakly_inferable": [
                "direction-dependent apparent state speed versus state region",
                "successful-demonstration behavioral hold-state envelopes",
                "contact/stall candidates only as ambiguous state plateaus during inferred manipulation",
            ],
            "not_identifiable": [
                "independent low-level command-to-state mapping or delay",
                "command tracking error, deadband, or command-conditioned hysteresis",
                "gainprm or biasprm",
                "armature, damping, or frictionloss",
                "actuator force limit or fingertip normal force",
                "pad/object friction",
                "contact stiffness, solref, or solimp",
            ],
        },
        "plots": plot_paths,
        "limitations": [
            "Canonical LeRobot timestamps are frame_index/fps, not preserved real wall-clock timestamps.",
            "The original external teleoperation collector source and its gripper event log are absent locally.",
            "Lift/hold phases are inferred from feedback closure and FK TCP height; images are intentionally not processed.",
            "The compared simulation dataset may predate the current compiled model and is selection-filtered for successful oracle episodes.",
            "This audit characterizes the supplied datasets; it does not validate the current compiled contact model.",
        ],
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output / "motion_segments.csv", segments)
    _write_csv(output / "pick_hold_metrics.csv", holds)
    _write_csv(output / "frame_trajectories.csv", frames)
    print(
        json.dumps(
            {
                "output": str(output),
                "real_episodes": real["episode_count"],
                "simulation_episodes": sim["episode_count"],
                "real_pick_phases": real["pick_phase_episode_count"],
                "simulation_pick_phases": sim["pick_phase_episode_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
