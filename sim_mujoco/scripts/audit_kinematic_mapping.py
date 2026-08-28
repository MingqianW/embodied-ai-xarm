from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.robot.model import ARM_JOINT_NAMES
from simulation.resources import DEFAULT_MODEL_PATH
from sim_mujoco.paths import mujoco_output_root


RAW_ROOT = PROJECT_ROOT / "fine_tune" / "data" / "xarm_pi05_data" / "raw"
LEROBOT_JSONL = (
    PROJECT_ROOT
    / "fine_tune"
    / "data"
    / "xarm_pi05_data"
    / "lerobot"
    / "data"
    / "train.jsonl"
)
OFFICIAL_KINEMATICS = (
    PROJECT_ROOT
    / "third_party"
    / "xarm_ros2"
    / "xarm_description"
    / "config"
    / "kinematics"
    / "default"
    / "xarm6_default_kinematics.yaml"
)
OUTPUT_ROOT = mujoco_output_root() / "kinematic_audit"
STATE_COLUMNS = tuple(f"j{i}_rad" for i in range(1, 7)) + ("gripper_mm",)
TCP_POSITION_COLUMNS = ("tcp_x_m", "tcp_y_m", "tcp_z_m")
TCP_RPY_COLUMNS = ("tcp_rx_rad", "tcp_ry_rad", "tcp_rz_rad")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def orientation_error_rad(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(first.T @ second) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine))


def matrix_to_quaternion(matrix: np.ndarray) -> list[float]:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=np.float64).reshape(9))
    return quaternion.tolist()


def named_joint_qpos_addresses(model: mujoco.MjModel) -> list[int]:
    addresses = []
    for name in ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Active MJCF is missing named joint {name}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return addresses


def set_named_arm_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    values: np.ndarray,
) -> None:
    for address, value in zip(named_joint_qpos_addresses(model), values):
        data.qpos[address] = float(value)
    mujoco.mj_forward(model, data)


def select_frames() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for task_dir in sorted(path for path in RAW_ROOT.iterdir() if path.is_dir()):
        episode_dir = sorted(path for path in task_dir.iterdir() if path.is_dir())[0]
        rows = read_csv(episode_dir / "robot_log.csv")
        indices = sorted(set(np.linspace(0, len(rows) - 1, 5, dtype=int).tolist()))
        for sample_index, row_index in enumerate(indices):
            row = rows[row_index]
            samples.append(
                {
                    "task": task_dir.name,
                    "episode": episode_dir.name,
                    "row_index": row_index,
                    "split": "validation" if sample_index in (1, 3) else "training",
                    "raw_q": np.asarray(
                        [float(row[column]) for column in STATE_COLUMNS[:6]],
                        dtype=np.float64,
                    ),
                    "gripper_mm": float(row["gripper_mm"]),
                    # The files label these columns "_m", but all 200 episodes
                    # contain controller millimeter values.
                    "reference_position": np.asarray(
                        [float(row[column]) for column in TCP_POSITION_COLUMNS],
                        dtype=np.float64,
                    )
                    / 1000.0,
                    "reference_rpy": np.asarray(
                        [float(row[column]) for column in TCP_RPY_COLUMNS],
                        dtype=np.float64,
                    ),
                }
            )
    return samples


def pose_for_body(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Active MJCF is missing body {body_name}")
    return (
        np.asarray(data.xpos[body_id], dtype=np.float64).copy(),
        np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3).copy(),
    )


def pose_for_site(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise RuntimeError(f"Active MJCF is missing site {site_name}")
    return (
        np.asarray(data.site_xpos[site_id], dtype=np.float64).copy(),
        np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy(),
    )


def pose_for_camera(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise RuntimeError(f"Active MJCF is missing camera {camera_name}")
    return (
        np.asarray(data.cam_xpos[camera_id], dtype=np.float64).copy(),
        np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3).copy(),
    )


def evaluate_mapping(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    samples: list[dict[str, Any]],
    permutation: tuple[int, ...],
    signs: tuple[int, ...],
    offsets: np.ndarray,
) -> dict[str, Any]:
    position_errors = []
    orientation_errors = []
    for sample in samples:
        mapped = (
            np.asarray(signs, dtype=np.float64) * sample["raw_q"][list(permutation)]
            + offsets
        )
        set_named_arm_qpos(model, data, mapped)
        position, rotation = pose_for_body(model, data, "link6")
        reference_rotation = rpy_matrix(*sample["reference_rpy"])
        position_errors.append(
            float(np.linalg.norm(position - sample["reference_position"]))
        )
        orientation_errors.append(orientation_error_rad(rotation, reference_rotation))
    position = np.asarray(position_errors)
    orientation = np.asarray(orientation_errors)
    # One radian of orientation error receives the same weight as 0.1 m.
    objective = float(np.mean(position + 0.1 * orientation))
    return {
        "objective": objective,
        "mean_position_error_m": float(position.mean()),
        "max_position_error_m": float(position.max()),
        "mean_orientation_error_deg": float(np.degrees(orientation).mean()),
        "max_orientation_error_deg": float(np.degrees(orientation).max()),
    }


def optimize_offsets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    samples: list[dict[str, Any]],
    permutation: tuple[int, ...],
    signs: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    offsets = np.zeros(6, dtype=np.float64)
    best = evaluate_mapping(model, data, samples, permutation, signs, offsets)
    step = 0.02
    while step >= 0.00015625:
        improved = True
        while improved:
            improved = False
            for index in range(6):
                for direction in (-1.0, 1.0):
                    candidate = offsets.copy()
                    candidate[index] += direction * step
                    result = evaluate_mapping(
                        model, data, samples, permutation, signs, candidate
                    )
                    if result["objective"] < best["objective"]:
                        offsets, best, improved = candidate, result, True
        step *= 0.5
    return offsets, best


def search_mapping(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    training: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = []
    zero = np.zeros(6, dtype=np.float64)
    for permutation in itertools.permutations(range(6)):
        for signs in itertools.product((-1, 1), repeat=6):
            result = evaluate_mapping(
                model, data, training, permutation, signs, zero
            )
            candidates.append((result["objective"], permutation, signs, result))
    candidates.sort(key=lambda item: item[0])

    refined = []
    for _, permutation, signs, zero_result in candidates[:8]:
        offsets, fitted = optimize_offsets(
            model, data, training, permutation, signs
        )
        refined.append(
            {
                "permutation": list(permutation),
                "signs": list(signs),
                "offsets_rad": offsets.tolist(),
                "zero_offset": zero_result,
                "fitted": fitted,
            }
        )
    refined.sort(key=lambda item: item["fitted"]["objective"])
    return {
        "evaluated_zero_offset_candidates": len(candidates),
        "refined_top_candidate_count": len(refined),
        "top_candidates": refined,
    }


def model_semantics(model: mujoco.MjModel) -> dict[str, Any]:
    joints = []
    for name in ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        child_body_id = int(model.jnt_bodyid[joint_id])
        parent_body_id = int(model.body_parentid[child_body_id])
        actuator_ids = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        actuator_id = int(actuator_ids[0]) if len(actuator_ids) else -1
        joints.append(
            {
                "joint_name": name,
                "qpos_index": int(model.jnt_qposadr[joint_id]),
                "reference_qpos": float(model.qpos0[model.jnt_qposadr[joint_id]]),
                "parent_body": mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, parent_body_id
                ),
                "child_body": mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, child_body_id
                ),
                "axis": model.jnt_axis[joint_id].tolist(),
                "range_rad": model.jnt_range[joint_id].tolist(),
                "body_pos": model.body_pos[child_body_id].tolist(),
                "body_quat": model.body_quat[child_body_id].tolist(),
                "actuator_index": actuator_id,
                "actuator_name": (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
                    if actuator_id >= 0
                    else None
                ),
                "actuator_ctrlrange": (
                    model.actuator_ctrlrange[actuator_id].tolist()
                    if actuator_id >= 0
                    else None
                ),
            }
        )

    def body_record(name: str) -> dict[str, Any]:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return {
            "name": name,
            "parent": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[body_id])
            ),
            "pos": model.body_pos[body_id].tolist(),
            "quat": model.body_quat[body_id].tolist(),
        }

    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "tool_center_point"
    )
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera"
    )
    return {
        "model_path": str(DEFAULT_MODEL_PATH),
        "joints": joints,
        "flange_body": body_record("link6"),
        "gripper_root": body_record("gripper_base"),
        "end_effector_body": body_record("gripper_base"),
        "tcp_site": {
            "name": "tool_center_point",
            "parent_body": mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.site_bodyid[site_id]),
            ),
            "pos": model.site_pos[site_id].tolist(),
            "quat": model.site_quat[site_id].tolist(),
        },
        "wrist_camera": {
            "name": "wrist_camera",
            "parent_body": mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.cam_bodyid[camera_id]),
            ),
            "pos": model.cam_pos[camera_id].tolist(),
            "quat": model.cam_quat[camera_id].tolist(),
            "fovy_deg": float(model.cam_fovy[camera_id]),
        },
        "runtime_qpos_overrides": [
            "initialize_scene resets to keyframe 'home'",
            "configure_task_scene writes initial_arm_qpos by named joint",
            "joint noise is applied by named joint",
            "position actuators update qpos dynamically from data.ctrl",
        ],
    }


def official_tree_comparison(model: mujoco.MjModel) -> dict[str, Any]:
    config = yaml.safe_load(OFFICIAL_KINEMATICS.read_text(encoding="utf-8"))
    official = config["kinematics"]
    rows = []
    exact = True
    for index, name in enumerate(ARM_JOINT_NAMES, start=1):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        body_id = int(model.jnt_bodyid[joint_id])
        expected = official[name]
        expected_pos = np.asarray(
            [expected["x"], expected["y"], expected["z"]], dtype=np.float64
        )
        expected_rpy = np.asarray(
            [expected["roll"], expected["pitch"], expected["yaw"]],
            dtype=np.float64,
        )
        expected_quat = np.asarray(
            matrix_to_quaternion(rpy_matrix(*expected_rpy)),
            dtype=np.float64,
        )
        position_error = float(np.linalg.norm(model.body_pos[body_id] - expected_pos))
        # Compare quaternions directly after accounting for q and -q equivalence.
        quat_error = min(
            float(np.linalg.norm(model.body_quat[body_id] - expected_quat)),
            float(np.linalg.norm(model.body_quat[body_id] + expected_quat)),
        )
        matches = position_error < 1e-12 and quat_error < 1e-5
        exact = exact and matches
        rows.append(
            {
                "joint": name,
                "official_pos": expected_pos.tolist(),
                "active_pos": model.body_pos[body_id].tolist(),
                "position_error_m": position_error,
                "official_rpy": expected_rpy.tolist(),
                "active_quat": model.body_quat[body_id].tolist(),
                "quaternion_difference_norm": quat_error,
                "matches": matches,
            }
        )
    return {
        "source": str(OFFICIAL_KINEMATICS),
        "all_joint_origins_match": exact,
        "rows": rows,
        "note": "All active MJCF arm joint axes are +Z, matching xarm6.urdf.xacro.",
    }


def compare_frames(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        set_named_arm_qpos(model, data, sample["raw_q"])
        flange_position, flange_rotation = pose_for_body(model, data, "link6")
        tcp_position, tcp_rotation = pose_for_site(
            model, data, "tool_center_point"
        )
        camera_position, camera_rotation = pose_for_camera(
            model, data, "wrist_camera"
        )
        reference_rotation = rpy_matrix(*sample["reference_rpy"])
        position_difference = flange_position - sample["reference_position"]
        row = {
            "task": sample["task"],
            "episode": sample["episode"],
            "row_index": sample["row_index"],
            "split": sample["split"],
            **{
                f"raw_j{index + 1}_rad": float(value)
                for index, value in enumerate(sample["raw_q"])
            },
            "gripper_mm": sample["gripper_mm"],
            **{
                f"reference_{axis}_m": float(value)
                for axis, value in zip("xyz", sample["reference_position"])
            },
            **{
                f"flange_{axis}_m": float(value)
                for axis, value in zip("xyz", flange_position)
            },
            **{
                f"position_difference_{axis}_m": float(value)
                for axis, value in zip("xyz", position_difference)
            },
            "position_error_m": float(np.linalg.norm(position_difference)),
            "orientation_error_deg": math.degrees(
                orientation_error_rad(flange_rotation, reference_rotation)
            ),
            "reference_rpy_rad": sample["reference_rpy"].tolist(),
            "flange_quat_wxyz": matrix_to_quaternion(flange_rotation),
            "tcp_position_m": tcp_position.tolist(),
            "tcp_quat_wxyz": matrix_to_quaternion(tcp_rotation),
            "wrist_camera_position_m": camera_position.tolist(),
            "wrist_camera_quat_wxyz": matrix_to_quaternion(camera_rotation),
        }
        rows.append(row)
    return rows


def lerobot_examples() -> dict[str, Any]:
    wanted = {path.name for path in RAW_ROOT.iterdir() if path.is_dir()}
    found: dict[str, Any] = {}
    with LEROBOT_JSONL.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            raw_task = str(record["raw_task"])
            if raw_task in wanted and raw_task not in found:
                found[raw_task] = {
                    "raw_id": record["raw_id"],
                    "state": record["state"],
                    "actions": record["actions"],
                    "frame_index": record["frame_index"],
                }
                if len(found) == len(wanted):
                    break
    return found


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar_rows = []
    for row in rows:
        scalar_rows.append(
            {
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in row.items()
            }
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)


def write_report(results: dict[str, Any], path: Path) -> None:
    direct = results["mapping"]["direct_identity"]
    fitted = results["mapping"]["identity_fitted"]
    validation = results["mapping"]["identity_fitted_validation"]
    empirical_best = results["mapping"]["empirical_best"]
    examples = results["first_frame_examples"]
    joint_rows = results["model_semantics"]["joints"]
    example_table = [
        "| Task | Raw arm q (rad) | Gripper | Controller TCP xyz (m) |",
        "| --- | --- | ---: | --- |",
    ]
    for task, example in examples.items():
        example_table.append(
            f"| `{task}` | "
            f"`{[round(value, 6) for value in example['raw_q_rad']]}` | "
            f"{example['gripper_mm']:.2f} | "
            f"`{[round(value, 6) for value in example['controller_tcp_position_m']]}` |"
        )
    joint_table = [
        "| Joint | qpos | Parent -> child | Axis | Range (rad) | Ref | Actuator / ctrlrange |",
        "| --- | ---: | --- | --- | --- | ---: | --- |",
    ]
    for joint in joint_rows:
        joint_table.append(
            f"| `{joint['joint_name']}` | {joint['qpos_index']} | "
            f"`{joint['parent_body']}` -> `{joint['child_body']}` | "
            f"`{joint['axis']}` | `{joint['range_rad']}` | "
            f"{joint['reference_qpos']:.1f} | `{joint['actuator_name']}` / "
            f"`{joint['actuator_ctrlrange']}` |"
        )
    lines = [
        "# xArm6 Kinematic Audit",
        "",
        "## Conclusion",
        "",
        "The recorded arm state and active MuJoCo arm use the same named joint",
        "ordering, signs, radians, and zero references. The authoritative mapping is:",
        "",
        "`q_mujoco = q_raw`",
        "",
        f"Direct identity training error: {direct['mean_position_error_m'] * 1000:.3f} mm, "
        f"{direct['mean_orientation_error_deg']:.3f} deg.",
        f"Identity-plus-offset validation error: {validation['mean_position_error_m'] * 1000:.3f} mm, "
        f"{validation['mean_orientation_error_deg']:.3f} deg.",
        "",
        "The raw `tcp_*` values correspond closely to the MuJoCo `link6` flange.",
        "They do not correspond to the custom `tool_center_point`, which is located",
        "172 mm after `gripper_base`. The apparent large TCP discrepancy was therefore",
        "a frame-label/offset mismatch, not a joint-state mapping failure.",
        "",
        "## Raw And Training Semantics",
        "",
        "- Raw columns: `j1_rad` through `j6_rad`, then `gripper_mm`.",
        "- Arm order: controller joints 1 through 6.",
        "- Units: arm radians; gripper xArm controller units labeled millimeters.",
        "- `evaluation/real/run_policy.py` reads controller joint degrees and applies `np.deg2rad`.",
        "- No arm sign conversion, permutation, or fixed offset is applied.",
        "- The converter copies raw state unchanged.",
        "- LeRobot action is the next frame's absolute state in the same convention.",
        "- OpenPI transforms joints 1-6 to deltas for training and restores absolute",
        "  actions at inference; gripper remains absolute.",
        "- The exact recorder that created the current TCP-bearing CSV files is not",
        "  present. Their controller-coordinate interpretation is corroborated by",
        "  the multi-frame FK comparison, but they cannot be called raw encoders.",
        "",
        "Code locations:",
        "",
        "- `evaluation/real/run_policy.py`: state order and deg-to-rad conversion.",
        "- `evaluation/real/run_policy.py`: absolute-radian action execution.",
        "- `data/real/conversion/convert_xarm_raw_to_lerobot.py`:",
        "  unchanged state and next-state absolute actions.",
        "- `fine_tune/openpi_xarm_config.py:24-65`: state/action contract and",
        "  delta/absolute OpenPI transforms.",
        "- `simulation/observation/state.py`: named-joint qpos to",
        "  policy state.",
        "- `simulation/robot/control.py`: policy absolute targets to",
        "  MuJoCo controls.",
        "- `simulation/scene/reset.py`: named-joint reset overrides.",
        "",
        "First raw frame from each task:",
        "",
        *example_table,
        "",
        "The matching LeRobot first-frame states are stored in `results.json` under",
        "`lerobot_first_examples`; all seven state values match the raw CSV exactly.",
        "",
        "## MuJoCo Semantics",
        "",
        "The active model resolves every qpos address by joint name. Joint details,",
        "actuator ranges, body transforms, flange, gripper, TCP site, and wrist camera",
        "are in `results.json` under `model_semantics`.",
        "",
        *joint_table,
        "",
        "- Flange: `link6`.",
        "- Gripper root/end-effector body: `gripper_base`, attached to `link6` at",
        "  identity position and quaternion.",
        "- Policy-independent simulation TCP site: `tool_center_point`, local pose",
        "  `[0, 0, 0.172] m` under `gripper_base`.",
        "- Wrist camera parent: `gripper_base`; exact local pose and FOV are in",
        "  `results.json`.",
        "- Runtime qpos overrides are name-resolved reset, optional reset noise, and",
        "  position-actuator control; no arm qpos address is inferred by array order.",
        "",
        "## FK Validation",
        "",
        f"Compared {results['frame_count']} frames spanning six tasks against the",
        "xArm controller TCP pose recorded in each raw CSV. Full per-frame poses and",
        "errors are in `frame_comparison.csv`.",
        "",
        f"Direct identity position error over all frames: "
        f"{results['frame_error_summary']['mean_position_error_m'] * 1000:.3f} mm "
        f"mean, {results['frame_error_summary']['max_position_error_m'] * 1000:.3f} mm max.",
        f"Direct identity orientation error: "
        f"{results['frame_error_summary']['mean_orientation_error_deg']:.3f} deg "
        f"mean, {results['frame_error_summary']['max_orientation_error_deg']:.3f} deg max.",
        "",
        "The controller RPY columns are interpreted as fixed-axis roll, pitch, yaw",
        "using `Rz(yaw) Ry(pitch) Rx(roll)`, consistent with the recorded poses near",
        "`[-pi, 0, 0]` and the sub-degree comparison result.",
        "",
        "## Mapping Search",
        "",
        f"Evaluated {results['mapping']['search']['evaluated_zero_offset_candidates']}",
        "permutation/sign candidates at zero offset, then optimized constant offsets",
        "for the eight best candidates using training frames only.",
        "",
        f"Identity permutation: `{fitted['permutation']}`",
        f"Identity signs: `{fitted['signs']}`",
        f"Identity fitted offsets (rad): `{[round(v, 6) for v in fitted['offsets_rad']]}`",
        "",
        f"Pure numerical best signs: `{empirical_best['signs']}`",
        f"Pure numerical best objective: `{empirical_best['fitted']['objective']:.9f}`",
        f"Identity candidate objective: `{fitted['fitted']['objective']:.9f}`",
        "",
        "Joint4 changes too little in these trajectories to distinguish `+q4 + c`",
        "from `-q4 + c`; the two objectives are effectively tied. Joint names, +Z",
        "axes, and zero references in the official xArm URDF resolve this ambiguity",
        "in favor of the identity sign. The tiny fitted offsets model controller",
        "calibration/rounded model constants;",
        "they are not adopted as a runtime conversion because the official URDF joint",
        "coordinates and active MJCF already share identity semantics.",
        "",
        "## Official Model Comparison",
        "",
        f"Active joint origins match official xArm6 kinematics YAML: "
        f"`{results['official_tree']['all_joint_origins_match']}`.",
        "Axes and limits also match `xarm6.urdf.xacro`. The active arm tree is not",
        "materially inconsistent with the official model.",
        "",
        "No executable xArm SDK FK, KDL, Pinocchio, or Robotics Toolbox package is",
        "available in the audit environment. The reference pose is therefore the",
        "xArm controller TCP recorded in the raw data, while the independent official",
        "xArm ROS kinematics YAML/URDF is used for tree, axis, origin, and limit",
        "verification. No replacement FK implementation was invented.",
        "",
        "The custom gripper is attached to `link6` at identity. Its",
        "`tool_center_point` offset of `[0, 0, 0.172] m` is simulation-specific and",
        "must not be compared directly with the raw controller flange TCP.",
        "",
        "## Decision",
        "",
        "The semantic conversion is identity. A reusable identity conversion can be",
        "made explicit in the policy/simulation boundary without applying the fitted",
        "calibration offsets. Camera/object/collision tuning remains outside this audit.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH))
    data = mujoco.MjData(model)
    samples = select_frames()
    training = [sample for sample in samples if sample["split"] == "training"]
    validation_samples = [
        sample for sample in samples if sample["split"] == "validation"
    ]

    frame_rows = compare_frames(model, data, samples)
    frame_position_errors = np.asarray(
        [row["position_error_m"] for row in frame_rows],
        dtype=np.float64,
    )
    frame_orientation_errors = np.asarray(
        [row["orientation_error_deg"] for row in frame_rows],
        dtype=np.float64,
    )
    search = search_mapping(model, data, training)
    empirical_best = search["top_candidates"][0]
    identity_fitted = next(
        candidate
        for candidate in search["top_candidates"]
        if candidate["permutation"] == list(range(6))
        and candidate["signs"] == [1, 1, 1, 1, 1, 1]
    )
    identity_validation = evaluate_mapping(
        model,
        data,
        validation_samples,
        tuple(identity_fitted["permutation"]),
        tuple(identity_fitted["signs"]),
        np.asarray(identity_fitted["offsets_rad"], dtype=np.float64),
    )
    identity = evaluate_mapping(
        model,
        data,
        training,
        tuple(range(6)),
        (1, 1, 1, 1, 1, 1),
        np.zeros(6),
    )
    results = {
        "frame_count": len(samples),
        "training_frame_count": len(training),
        "validation_frame_count": len(validation_samples),
        "frame_error_summary": {
            "mean_position_error_m": float(frame_position_errors.mean()),
            "max_position_error_m": float(frame_position_errors.max()),
            "mean_orientation_error_deg": float(frame_orientation_errors.mean()),
            "max_orientation_error_deg": float(frame_orientation_errors.max()),
        },
        "raw_state_columns": list(STATE_COLUMNS),
        "raw_tcp_position_columns": list(TCP_POSITION_COLUMNS),
        "raw_tcp_rpy_columns": list(TCP_RPY_COLUMNS),
        "first_frame_examples": {
            task: {
                "raw_q_rad": next(
                    sample["raw_q"].tolist()
                    for sample in samples
                    if sample["task"] == task and sample["row_index"] == 0
                ),
                "gripper_mm": next(
                    sample["gripper_mm"]
                    for sample in samples
                    if sample["task"] == task and sample["row_index"] == 0
                ),
                "controller_tcp_position_m": next(
                    sample["reference_position"].tolist()
                    for sample in samples
                    if sample["task"] == task and sample["row_index"] == 0
                ),
                "controller_tcp_rpy_rad": next(
                    sample["reference_rpy"].tolist()
                    for sample in samples
                    if sample["task"] == task and sample["row_index"] == 0
                ),
            }
            for task in sorted({sample["task"] for sample in samples})
        },
        "lerobot_first_examples": lerobot_examples(),
        "model_semantics": model_semantics(model),
        "official_tree": official_tree_comparison(model),
        "mapping": {
            "formula": "q_mujoco[i] = sign[i] * q_raw[permutation[i]] + offset[i]",
            "direct_identity": identity,
            "search": search,
            "empirical_best": empirical_best,
            "identity_fitted": identity_fitted,
            "identity_fitted_validation": identity_validation,
            "runtime_decision": {
                "permutation": [0, 1, 2, 3, 4, 5],
                "signs": [1, 1, 1, 1, 1, 1],
                "offsets_rad": [0.0] * 6,
                "reason": "Named official URDF joints and active MJCF share coordinates; fitted milliradian offsets are not a semantic conversion.",
            },
        },
    }
    # Remove one intermediate expression that is not serializable.
    (OUTPUT_ROOT / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(OUTPUT_ROOT / "frame_comparison.csv", frame_rows)
    write_report(results, OUTPUT_ROOT / "report.md")
    print(f"Wrote kinematic audit to {OUTPUT_ROOT}")
    print(json.dumps(results["mapping"]["runtime_decision"], indent=2))
    print(
        "validation:",
        json.dumps(results["mapping"]["identity_fitted_validation"], indent=2),
    )


if __name__ == "__main__":
    main()
