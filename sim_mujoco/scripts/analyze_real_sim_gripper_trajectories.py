#!/usr/bin/env python3
"""Compare real and stable-v4 simulation gripper trajectories without images."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from simulation.robot.joint_mapping import raw_arm_state_to_mujoco_qpos  # noqa: E402


PICK_TASK_PREFIX = "pick up "
PHASES = ("pregrasp", "closure", "lift", "post_lift_hold")


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    root: Path


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
    parser.add_argument("--gripper-q01", type=float, required=True)
    parser.add_argument("--gripper-q99", type=float, required=True)
    return parser


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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
    return np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1, 7)


def _tcp_z_trajectory(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    states: np.ndarray,
    *,
    joint_qpos_addresses: np.ndarray,
    tcp_site_id: int,
) -> np.ndarray:
    result = np.empty(len(states), dtype=np.float64)
    for index, state in enumerate(states):
        data.qpos[joint_qpos_addresses] = raw_arm_state_to_mujoco_qpos(state[:6])
        mujoco.mj_kinematics(model, data)
        result[index] = float(data.site_xpos[tcp_site_id, 2])
    return result


def _phase_boundaries(gripper: np.ndarray, tcp_z: np.ndarray) -> dict[str, int] | None:
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
    if candidates.size == 0:
        return None
    closure = open_index + int(candidates[0])
    if closure >= length - 5:
        return None
    grasp_z = float(np.min(tcp_z[closure : min(length, closure + 15)]))
    lift_candidates = np.flatnonzero(tcp_z[closure:] >= grasp_z + 0.01)
    if lift_candidates.size == 0:
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
        "pregrasp_start": 0,
        "closure_start": closure,
        "lift_start": lift,
        "hold_start": hold,
        "hold_end": end,
    }


def _resample(values: np.ndarray, count: int = 101) -> np.ndarray:
    if len(values) == 1:
        return np.repeat(values, count)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, count)
    return np.interp(target, source, values)


def _quantiles(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "q01": None, "q10": None, "median": None, "q90": None, "q99": None, "mean": None}
    return {
        "count": int(array.size),
        "q01": float(np.quantile(array, 0.01)),
        "q10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
        "q99": float(np.quantile(array, 0.99)),
        "mean": float(np.mean(array)),
    }


def _aligned_window(
    values: np.ndarray,
    center: int,
    *,
    before: int = 20,
    after: int = 30,
) -> np.ndarray:
    result = np.full(before + after + 1, np.nan, dtype=np.float64)
    source_start = max(0, center - before)
    source_end = min(len(values), center + after + 1)
    target_start = before - (center - source_start)
    result[target_start : target_start + source_end - source_start] = values[
        source_start:source_end
    ]
    return result


def _analyze_dataset(
    spec: DatasetSpec,
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_qpos_addresses: np.ndarray,
    tcp_site_id: int,
    gripper_q01: float,
    gripper_q99: float,
) -> dict[str, Any]:
    info = json.loads((spec.root / "meta/info.json").read_text(encoding="utf-8"))
    episodes = {int(row["episode_index"]): row for row in _jsonl(spec.root / "meta/episodes.jsonl")}
    expected = int(info["total_episodes"])
    if set(episodes) != set(range(expected)):
        raise ValueError(f"Episode metadata is incomplete for {spec.root}")
    fps = float(info["fps"])
    phases: dict[str, dict[str, list[float]]] = {
        phase: {"state": [], "action": [], "velocity": [], "normalized_action": []}
        for phase in PHASES
    }
    task_phase_actions: dict[str, dict[str, list[float]]] = {}
    episode_metrics: list[dict[str, Any]] = []
    trajectories: list[np.ndarray] = []
    lift_aligned_states: list[np.ndarray] = []
    lift_aligned_actions: list[np.ndarray] = []
    analyzed_frames = 0
    skipped_place = 0
    skipped_no_phase = 0

    for episode_index in range(expected):
        task = str(episodes[episode_index]["tasks"][0])
        if not task.startswith(PICK_TASK_PREFIX):
            skipped_place += 1
            continue
        path = spec.root / info["data_path"].format(
            episode_chunk=episode_index // int(info["chunks_size"]),
            episode_index=episode_index,
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        table = pq.read_table(path, columns=["state", "actions"])
        states = _fixed_list(table, "state")
        actions = _fixed_list(table, "actions")
        if states.shape != actions.shape or len(states) != int(episodes[episode_index]["length"]):
            raise ValueError(f"Episode shape/metadata mismatch: {path}")
        tcp_z = _tcp_z_trajectory(
            model,
            data,
            states,
            joint_qpos_addresses=joint_qpos_addresses,
            tcp_site_id=tcp_site_id,
        )
        boundaries = _phase_boundaries(states[:, 6], tcp_z)
        if boundaries is None:
            skipped_no_phase += 1
            continue
        slices = {
            "pregrasp": slice(0, boundaries["closure_start"]),
            "closure": slice(boundaries["closure_start"], boundaries["lift_start"]),
            "lift": slice(boundaries["lift_start"], boundaries["hold_start"]),
            "post_lift_hold": slice(boundaries["hold_start"], boundaries["hold_end"]),
        }
        task_phase_actions.setdefault(task, {phase: [] for phase in PHASES})
        for phase, selection in slices.items():
            state_values = states[selection, 6]
            action_values = actions[selection, 6]
            phases[phase]["state"].extend(state_values.tolist())
            phases[phase]["action"].extend(action_values.tolist())
            phases[phase]["normalized_action"].extend(
                (
                    2.0 * (action_values - gripper_q01)
                    / (gripper_q99 - gripper_q01 + 1e-6)
                    - 1.0
                ).tolist()
            )
            task_phase_actions[task][phase].extend(action_values.tolist())
            if len(state_values) > 1:
                phases[phase]["velocity"].extend((np.diff(state_values) * fps).tolist())

        post = slices["post_lift_hold"]
        post_state = states[post, 6]
        post_action = actions[post, 6]
        post_delta = np.diff(post_action)
        trajectories.append(_resample(states[boundaries["closure_start"] : boundaries["hold_end"], 6]))
        lift_aligned_states.append(
            _aligned_window(states[:, 6], boundaries["lift_start"])
        )
        lift_aligned_actions.append(
            _aligned_window(actions[:, 6], boundaries["lift_start"])
        )
        analyzed_frames += len(states)
        episode_metrics.append(
            {
                "source": spec.label,
                "episode_index": episode_index,
                "task": task,
                "length": len(states),
                **boundaries,
                "tcp_lift_m": float(np.max(tcp_z[boundaries["lift_start"] : boundaries["hold_end"]]) - tcp_z[boundaries["closure_start"]]),
                "post_lift_state_start": float(post_state[0]),
                "post_lift_state_end": float(post_state[-1]),
                "post_lift_action_min": float(np.min(post_action)),
                "post_lift_action_max": float(np.max(post_action)),
                "post_lift_net_opening_raw": float(post_action[-1] - post_action[0]),
                "post_lift_max_opening_jump_raw": max((float(np.max(post_delta)),), default=0.0) if post_delta.size else 0.0,
                "post_lift_max_closing_jump_raw": min((float(np.min(post_delta)),), default=0.0) if post_delta.size else 0.0,
                "post_lift_opening_jump_gt10_count": int(np.sum(post_delta > 10.0)),
                "post_lift_closing_jump_lt_minus10_count": int(np.sum(post_delta < -10.0)),
                "post_lift_duration_s": float(len(post_action) / fps),
                "post_lift_state_median": float(np.median(post_state)),
                "post_lift_action_median": float(np.median(post_action)),
                "post_lift_action_std": float(np.std(post_action)),
            }
        )

    phase_summary = {
        phase: {
            **{measure: _quantiles(values) for measure, values in measures.items()},
            "normalized_action_outside_minus1_plus1_fraction": (
                float(
                    np.mean(
                        np.abs(
                            np.asarray(measures["normalized_action"], dtype=np.float64)
                        )
                        > 1.0
                    )
                )
                if measures["normalized_action"]
                else None
            ),
        }
        for phase, measures in phases.items()
    }
    trajectory_array = np.asarray(trajectories, dtype=np.float64)
    trajectory_summary = {
        "normalized_time": np.linspace(0.0, 1.0, 101).tolist(),
        "q10": np.quantile(trajectory_array, 0.1, axis=0).tolist() if len(trajectory_array) else [],
        "median": np.median(trajectory_array, axis=0).tolist() if len(trajectory_array) else [],
        "q90": np.quantile(trajectory_array, 0.9, axis=0).tolist() if len(trajectory_array) else [],
    }
    aligned_state_array = np.asarray(lift_aligned_states, dtype=np.float64)
    aligned_action_array = np.asarray(lift_aligned_actions, dtype=np.float64)

    def aligned_summary(values: np.ndarray) -> dict[str, list[float]]:
        if not len(values):
            return {"q10": [], "median": [], "q90": []}
        return {
            "q10": np.nanquantile(values, 0.1, axis=0).tolist(),
            "median": np.nanmedian(values, axis=0).tolist(),
            "q90": np.nanquantile(values, 0.9, axis=0).tolist(),
        }

    by_task: dict[str, Any] = {}
    for task in sorted(task_phase_actions):
        task_episodes = [row for row in episode_metrics if row["task"] == task]
        by_task[task] = {
            "episode_count": len(task_episodes),
            "phase_action": {
                phase: _quantiles(task_phase_actions[task][phase])
                for phase in PHASES
            },
            "post_lift_duration_s": _quantiles(
                [row["post_lift_duration_s"] for row in task_episodes]
            ),
            "post_lift_action_median": _quantiles(
                [row["post_lift_action_median"] for row in task_episodes]
            ),
            "post_lift_action_std": _quantiles(
                [row["post_lift_action_std"] for row in task_episodes]
            ),
            "episode_fraction_with_opening_jump_gt10": (
                float(
                    np.mean(
                        [
                            row["post_lift_opening_jump_gt10_count"] > 0
                            for row in task_episodes
                        ]
                    )
                )
                if task_episodes
                else None
            ),
        }
    return {
        "label": spec.label,
        "root": str(spec.root),
        "info": info,
        "episode_count_analyzed": len(episode_metrics),
        "frame_count_analyzed": analyzed_frames,
        "episodes_skipped_place": skipped_place,
        "episodes_skipped_no_detectable_lift_phase": skipped_no_phase,
        "phase_detection": {
            "closure": "first crossing below midpoint of episode gripper q10/q90 after maximum opening",
            "lift": "first FK TCP rise >= 10 mm above post-closure minimum",
            "post_lift_hold": "from 80% of peak TCP rise until a 10 mm descent or episode end",
        },
        "phases": phase_summary,
        "trajectory": trajectory_summary,
        "lift_aligned": {
            "time_s": (np.arange(-20, 31, dtype=np.float64) / fps).tolist(),
            "state": aligned_summary(aligned_state_array),
            "action": aligned_summary(aligned_action_array),
        },
        "by_task": by_task,
        "episode_metrics": episode_metrics,
        "raw_phase_values": phases,
    }


def _plots(real: dict[str, Any], sim: dict[str, Any], output: Path) -> list[str]:
    import matplotlib.pyplot as plt

    paths: list[str] = []
    colors = {"real": "tab:blue", "sim": "tab:orange"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, phase in zip(axes.flat, PHASES, strict=True):
        for result in (real, sim):
            values = result["raw_phase_values"][phase]["action"]
            axis.hist(values, bins=60, density=True, alpha=0.45, color=colors[result["label"]], label=result["label"])
        axis.set_title(phase.replace("_", " "))
        axis.set_xlabel("absolute gripper action (raw; larger=open)")
        axis.set_ylabel("density")
        axis.legend()
    figure.tight_layout()
    path = output / "gripper_action_by_phase.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for result in (real, sim):
        x = np.asarray(result["lift_aligned"]["time_s"])
        for axis, measure in zip(axes, ("state", "action"), strict=True):
            median = np.asarray(result["lift_aligned"][measure]["median"])
            q10 = np.asarray(result["lift_aligned"][measure]["q10"])
            q90 = np.asarray(result["lift_aligned"][measure]["q90"])
            axis.plot(
                x,
                median,
                color=colors[result["label"]],
                label=f"{result['label']} median",
            )
            axis.fill_between(
                x,
                q10,
                q90,
                color=colors[result["label"]],
                alpha=0.2,
            )
    axes[0].set_ylabel("gripper state raw")
    axes[1].set_ylabel("gripper action raw")
    axes[1].set_xlabel("time relative to detected lift onset (s)")
    for axis in axes:
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.legend()
    figure.suptitle("Aggregate gripper behavior aligned at first 10 mm TCP lift")
    figure.tight_layout()
    path = output / "gripper_state_action_aligned_lift.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))

    figure, axis = plt.subplots(figsize=(10, 5))
    for result in (real, sim):
        x = np.asarray(result["trajectory"]["normalized_time"])
        median = np.asarray(result["trajectory"]["median"])
        q10 = np.asarray(result["trajectory"]["q10"])
        q90 = np.asarray(result["trajectory"]["q90"])
        axis.plot(x, median, color=colors[result["label"]], label=f"{result['label']} median")
        axis.fill_between(x, q10, q90, color=colors[result["label"]], alpha=0.2)
    axis.set_xlabel("normalized closure-to-post-lift trajectory time")
    axis.set_ylabel("gripper state (raw; larger=open)")
    axis.set_title("Real versus simulation gripper trajectory")
    axis.legend()
    figure.tight_layout()
    path = output / "gripper_trajectory_real_vs_sim.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))

    figure, axis = plt.subplots(figsize=(10, 5))
    for result in (real, sim):
        values = np.asarray(result["raw_phase_values"]["post_lift_hold"]["velocity"])
        values = values[np.abs(values) <= np.quantile(np.abs(values), 0.995)] if len(values) else values
        axis.hist(values, bins=80, density=True, alpha=0.45, color=colors[result["label"]], label=result["label"])
    axis.set_xlabel("post-lift gripper-state velocity (raw units/s; positive=open)")
    axis.set_ylabel("density")
    axis.set_title("Post-lift gripper continuity")
    axis.legend()
    figure.tight_layout()
    path = output / "post_lift_gripper_velocity.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))

    tasks = sorted(set(real["by_task"]) | set(sim["by_task"]))
    x = np.arange(len(tasks), dtype=np.float64)
    figure, axis = plt.subplots(figsize=(max(11, 1.6 * len(tasks)), 5))
    for offset, result in ((-0.18, real), (0.18, sim)):
        medians = [
            result["by_task"].get(task, {})
            .get("post_lift_action_median", {})
            .get("median", math.nan)
            for task in tasks
        ]
        axis.bar(
            x + offset,
            medians,
            width=0.36,
            color=colors[result["label"]],
            label=result["label"],
        )
    axis.set_xticks(x, [task.removeprefix(PICK_TASK_PREFIX) for task in tasks], rotation=20, ha="right")
    axis.set_ylabel("median post-lift gripper action (raw)")
    axis.set_title("Task-specific post-lift closure")
    axis.legend()
    figure.tight_layout()
    path = output / "post_lift_gripper_by_task.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))
    return paths


def _strip_raw(result: dict[str, Any]) -> dict[str, Any]:
    value = dict(result)
    value.pop("raw_phase_values", None)
    return value


def _comparison(real: dict[str, Any], sim: dict[str, Any]) -> dict[str, Any]:
    real_post = real["phases"]["post_lift_hold"]
    sim_post = sim["phases"]["post_lift_hold"]

    def episode_fraction(result: dict[str, Any], key: str, predicate) -> float | None:
        rows = result["episode_metrics"]
        return float(np.mean([predicate(row[key]) for row in rows])) if rows else None

    tasks = sorted(set(real["by_task"]) | set(sim["by_task"]))
    task_offsets = {}
    for task in tasks:
        real_value = (
            real["by_task"].get(task, {})
            .get("post_lift_action_median", {})
            .get("median")
        )
        sim_value = (
            sim["by_task"].get(task, {})
            .get("post_lift_action_median", {})
            .get("median")
        )
        task_offsets[task] = {
            "real_median_raw": real_value,
            "sim_median_raw": sim_value,
            "sim_minus_real_raw": (
                None
                if real_value is None or sim_value is None
                else float(sim_value - real_value)
            ),
        }
    real_action_median = real_post["action"]["median"]
    sim_action_median = sim_post["action"]["median"]
    return {
        "post_lift_action_median_raw": {
            "real": real_action_median,
            "sim": sim_action_median,
            "sim_minus_real": float(sim_action_median - real_action_median),
        },
        "post_lift_state_median_raw": {
            "real": real_post["state"]["median"],
            "sim": sim_post["state"]["median"],
            "sim_minus_real": float(
                sim_post["state"]["median"] - real_post["state"]["median"]
            ),
        },
        "post_lift_velocity_q10_q90_raw_per_s": {
            "real": [real_post["velocity"]["q10"], real_post["velocity"]["q90"]],
            "sim": [sim_post["velocity"]["q10"], sim_post["velocity"]["q90"]],
        },
        "episode_fraction_with_post_lift_opening_jump_gt10": {
            "real": episode_fraction(
                real, "post_lift_opening_jump_gt10_count", lambda value: value > 0
            ),
            "sim": episode_fraction(
                sim, "post_lift_opening_jump_gt10_count", lambda value: value > 0
            ),
        },
        "post_lift_duration_s": {
            "real": _quantiles(
                [row["post_lift_duration_s"] for row in real["episode_metrics"]]
            ),
            "sim": _quantiles(
                [row["post_lift_duration_s"] for row in sim["episode_metrics"]]
            ),
        },
        "normalized_post_lift_action_outside_minus1_plus1_fraction": {
            "real": real_post["normalized_action_outside_minus1_plus1_fraction"],
            "sim": sim_post["normalized_action_outside_minus1_plus1_fraction"],
        },
        "per_task_post_lift_action_offset": task_offsets,
        "normalization_interpretation": (
            "Checkpoint quantile normalization is affine and does not compress "
            "closed/partial/open distinctions; values outside q01/q99 map outside [-1,1]."
        ),
    }


def main() -> None:
    args = _parser().parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Full dataset comparison must run inside a Slurm job")
    if not math.isfinite(args.gripper_q01) or not math.isfinite(args.gripper_q99) or args.gripper_q99 <= args.gripper_q01:
        raise ValueError("Invalid gripper quantiles")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing an existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(args.robot_xml.expanduser().resolve()))
    data = mujoco.MjData(model)
    addresses = np.asarray([
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index}")])
        for index in range(1, 7)
    ])
    tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_center_point")
    if tcp_site_id < 0:
        raise RuntimeError("tool_center_point site is absent")
    real = _analyze_dataset(
        DatasetSpec("real", args.real_root.expanduser().resolve()),
        model=model,
        data=data,
        joint_qpos_addresses=addresses,
        tcp_site_id=tcp_site_id,
        gripper_q01=args.gripper_q01,
        gripper_q99=args.gripper_q99,
    )
    sim = _analyze_dataset(
        DatasetSpec("sim", args.sim_root.expanduser().resolve()),
        model=model,
        data=data,
        joint_qpos_addresses=addresses,
        tcp_site_id=tcp_site_id,
        gripper_q01=args.gripper_q01,
        gripper_q99=args.gripper_q99,
    )
    plot_paths = _plots(real, sim, output)
    summary = {
        "schema_version": "xarm_real_sim_gripper_comparison_v1",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "action_semantics": {
            "shape": [7],
            "order": ["joint_1_rad", "joint_2_rad", "joint_3_rad", "joint_4_rad", "joint_5_rad", "joint_6_rad", "gripper_driver_raw"],
            "stored": "next-frame absolute target",
            "openpi_training": "arm dimensions converted to delta; gripper remains absolute",
            "gripper_direction": "larger=open",
            "quantile_normalization": {"q01": args.gripper_q01, "q99": args.gripper_q99},
        },
        "real": _strip_raw(real),
        "sim": _strip_raw(sim),
        "real_sim_comparison": _comparison(real, sim),
        "plots": plot_paths,
        "limitations": [
            "Neither LeRobot dataset stores object pose or contact forces.",
            "Lift phases are inferred from forward-kinematic TCP height and gripper closure.",
            "Post-lift dataset stability is a selection property for simulation and cannot prove policy-grasp stability.",
        ],
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    episode_rows = [*real["episode_metrics"], *sim["episode_metrics"]]
    with (output / "episode_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    print(json.dumps({"output": str(output), "real_episodes": real["episode_count_analyzed"], "sim_episodes": sim["episode_count_analyzed"]}, indent=2))


if __name__ == "__main__":
    main()
