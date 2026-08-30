from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from policy_runtime.episode_logging import (
    json_default as _shared_json_default,
    write_json as _shared_write_json,
)
from evaluation.common.legacy_policy_results import (
    LABELS,
    summarize_episode_rows as _shared_summarize_episode_rows,
    validate_label as _shared_validate_label,
)
from simulation.robot.model import ARM_JOINT_NAMES
from simulation.robot.model import arm_joint_limits
from simulation.robot.model import joint_position


def json_default(value: Any) -> Any:
    try:
        return _shared_json_default(value)
    except TypeError:
        return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    _shared_write_json(path, payload)


def validate_label(label: str) -> str:
    return _shared_validate_label(label)


def summarize_episode_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _shared_summarize_episode_rows(rows)


def write_episodes_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode_index",
        "seed",
        "task",
        "prompt",
        "label",
        "valid",
        "automatic_task_success",
        "comment",
        "termination_reason",
        "policy_steps",
        "sim_time",
        "wall_time",
        "initial_object_x",
        "initial_object_y",
        "initial_object_yaw",
        "video_frames",
        "video_fps",
        "combined_video_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_episodes_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def write_summary(run_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_episode_rows(rows)
    tasks = sorted({str(row.get("task") or "") for row in rows if row.get("task")})
    summary["task_breakdown"] = {
        task: summarize_episode_rows(
            [row for row in rows if str(row.get("task") or "") == task]
        )
        for task in tasks
    }
    write_json(run_dir / "summary.json", summary)
    rate = summary["human_rated_task_success_rate"]
    e2e = summary["end_to_end_success_rate"]
    lines = [
        f"attempted episodes: {summary['attempted_episodes']}",
        f"labeled episodes: {summary['labeled_episodes']}",
        f"successes: {summary['successes']}",
        f"failures: {summary['failures']}",
        f"invalid episodes: {summary['invalid_episodes']}",
        f"human-rated task success rate: {'n/a' if rate is None else f'{rate:.3f}'}",
        f"end-to-end success rate: {'n/a' if e2e is None else f'{e2e:.3f}'}",
        f"label counts: {summary['label_counts']}",
        f"termination reason counts: {summary['termination_reason_counts']}",
        f"mean policy steps: {summary['mean_policy_steps']}",
        f"mean simulation time: {summary['mean_simulation_time']}",
        f"mean wall time: {summary['mean_wall_time']}",
        f"task breakdown: {summary['task_breakdown']}",
    ]
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_episodes_csv(run_dir / "episodes.csv", rows)
    return summary


def quaternion_from_yaw(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float64)


def yaw_from_quaternion(quat: np.ndarray) -> float:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def object_qpos_address(model: mujoco.MjModel) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
    if joint_id < 0:
        raise RuntimeError("Object freejoint not found: object_freejoint")
    return int(model.jnt_qposadr[joint_id])


def apply_initial_randomization(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    seed: int,
    object_xy_range: float,
    object_yaw_range_deg: float,
    joint_noise: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    object_addr = object_qpos_address(model)
    nominal_object_xy = np.asarray(data.qpos[object_addr : object_addr + 2], dtype=np.float64).copy()
    nominal_object_z = float(data.qpos[object_addr + 2])
    xy_delta = rng.uniform(-float(object_xy_range), float(object_xy_range), size=2)
    yaw = math.radians(float(rng.uniform(-float(object_yaw_range_deg), float(object_yaw_range_deg))))

    data.qpos[object_addr : object_addr + 2] = nominal_object_xy + xy_delta
    data.qpos[object_addr + 2] = nominal_object_z
    data.qpos[object_addr + 3 : object_addr + 7] = quaternion_from_yaw(yaw)

    limits = arm_joint_limits(model)
    joint_values = []
    for index, joint_name in enumerate(ARM_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_addr = int(model.jnt_qposadr[joint_id])
        noisy = float(data.qpos[qpos_addr] + rng.normal(0.0, float(joint_noise)))
        clamped = float(np.clip(noisy, limits[index, 0], limits[index, 1]))
        data.qpos[qpos_addr] = clamped
        if index < model.nu:
            data.ctrl[index] = clamped
        joint_values.append(clamped)

    mujoco.mj_forward(model, data)
    return {
        "seed": int(seed),
        "initial_object_x": float(data.qpos[object_addr]),
        "initial_object_y": float(data.qpos[object_addr + 1]),
        "initial_object_z": float(data.qpos[object_addr + 2]),
        "initial_object_yaw": yaw,
        "initial_joint_positions": joint_values,
        "object_xy_delta": xy_delta.tolist(),
    }
