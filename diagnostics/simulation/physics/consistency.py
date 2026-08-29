#!/usr/bin/env python3
"""Dump and exactly compare generation/evaluation compiled MuJoCo physics.

This diagnostic performs model compilation, reset-time model mutation, and
``mj_forward`` only.  It does not construct a renderer and never calls
``mj_step``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from data.sim.generation.config import load_pipeline_config  # noqa: E402
from evaluation.sim.config import load_protocol  # noqa: E402
from simulation.resources import model_path  # noqa: E402
from simulation.resources import repository_root  # noqa: E402
from simulation.observation.cameras import apply_camera_calibration  # noqa: E402
from simulation.runtime import initialize_scene  # noqa: E402
from simulation.configuration import load_simulation_config  # noqa: E402
from simulation.scene import configure_task_scene  # noqa: E402


PROJECT_ROOT = repository_root()
DEFAULT_GENERATION_CONFIG = (
    PROJECT_ROOT
    / "configs/data/sim/generation/clean_multitask_stable_v4_10x_real.yaml"
)
DEFAULT_EVALUATION_PROTOCOL = (
    PROJECT_ROOT / "configs/evaluation/sim/protocols/formal_xarm_pi05_eval_v2.json"
)
FINGER_BODIES = ("left_finger", "right_finger")
OBJECT_BODIES = ("object", "blue_block", "small_block", "large_block", "red_pepper")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-config", type=Path, default=DEFAULT_GENERATION_CONFIG)
    parser.add_argument("--evaluation-protocol", type=Path, default=DEFAULT_EVALUATION_PROTOCOL)
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Task to compare; repeat as needed. Defaults to every configured task.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new JSON path. Without this option, JSON is written to stdout.",
    )
    return parser


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, object_id: int) -> str:
    value = mujoco.mj_id2name(model, kind, int(object_id))
    return f"unnamed_{int(object_id)}" if value is None else str(value)


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = int(mujoco.mj_name2id(model, kind, name))
    if value < 0:
        raise RuntimeError(f"Required MuJoCo object not found: {name}")
    return value


def _array(value: Any) -> list[Any]:
    return np.asarray(value).tolist()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _enum(enum_type: Any, value: Any) -> str:
    return str(enum_type(int(value)).name)


def _geom(model: mujoco.MjModel, geom_id: int) -> dict[str, Any]:
    return {
        "name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
        "body": _name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
        ),
        "type": _enum(mujoco.mjtGeom, model.geom_type[geom_id]),
        "size": _array(model.geom_size[geom_id]),
        "position": _array(model.geom_pos[geom_id]),
        "quaternion_wxyz": _array(model.geom_quat[geom_id]),
        "friction": _array(model.geom_friction[geom_id]),
        "condim": int(model.geom_condim[geom_id]),
        "solref": _array(model.geom_solref[geom_id]),
        "solimp": _array(model.geom_solimp[geom_id]),
        "solmix": float(model.geom_solmix[geom_id]),
        "margin": float(model.geom_margin[geom_id]),
        "gap": float(model.geom_gap[geom_id]),
        "priority": int(model.geom_priority[geom_id]),
        "contype": int(model.geom_contype[geom_id]),
        "conaffinity": int(model.geom_conaffinity[geom_id]),
    }


def _body(model: mujoco.MjModel, body_name: str) -> dict[str, Any]:
    body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
    return {
        "name": body_name,
        "mass": float(model.body_mass[body_id]),
        "inertia_diagonal": _array(model.body_inertia[body_id]),
        "inertial_position": _array(model.body_ipos[body_id]),
        "inertial_quaternion_wxyz": _array(model.body_iquat[body_id]),
        "gravity_compensation": float(model.body_gravcomp[body_id]),
        "geoms": [_geom(model, int(geom_id)) for geom_id in geom_ids],
    }


def _joint(model: mujoco.MjModel, joint_name: str) -> dict[str, Any]:
    joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    dof_address = int(model.jnt_dofadr[joint_id])
    return {
        "name": joint_name,
        "type": _enum(mujoco.mjtJoint, model.jnt_type[joint_id]),
        "axis": _array(model.jnt_axis[joint_id]),
        "range": _array(model.jnt_range[joint_id]),
        "limited": int(model.jnt_limited[joint_id]),
        "damping": float(model.dof_damping[dof_address]),
        "frictionloss": float(model.dof_frictionloss[dof_address]),
        "armature": float(model.dof_armature[dof_address]),
        "stiffness": float(model.jnt_stiffness[joint_id]),
        "spring_reference": float(model.qpos_spring[int(model.jnt_qposadr[joint_id])]),
    }


def _actuator(model: mujoco.MjModel, actuator_name: str) -> dict[str, Any]:
    actuator_id = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    gainprm = np.asarray(model.actuator_gainprm[actuator_id], dtype=np.float64)
    biasprm = np.asarray(model.actuator_biasprm[actuator_id], dtype=np.float64)
    return {
        "name": actuator_name,
        "transmission_type": _enum(mujoco.mjtTrn, model.actuator_trntype[actuator_id]),
        "transmission_ids": _array(model.actuator_trnid[actuator_id]),
        "dynamics_type": _enum(mujoco.mjtDyn, model.actuator_dyntype[actuator_id]),
        "gain_type": _enum(mujoco.mjtGain, model.actuator_gaintype[actuator_id]),
        "bias_type": _enum(mujoco.mjtBias, model.actuator_biastype[actuator_id]),
        "gear": _array(model.actuator_gear[actuator_id]),
        "gainprm": gainprm.tolist(),
        "biasprm": biasprm.tolist(),
        "position_kp_derived": float(gainprm[0]),
        "velocity_kv_derived": float(-biasprm[2]),
        "ctrl_limited": int(model.actuator_ctrllimited[actuator_id]),
        "ctrlrange": _array(model.actuator_ctrlrange[actuator_id]),
        "force_limited": int(model.actuator_forcelimited[actuator_id]),
        "forcerange": _array(model.actuator_forcerange[actuator_id]),
    }


def _equalities(model: mujoco.MjModel) -> list[dict[str, Any]]:
    rows = []
    equality_active = (
        model.eq_active0 if hasattr(model, "eq_active0") else model.eq_active
    )
    for equality_id in range(int(model.neq)):
        equality_type = int(model.eq_type[equality_id])
        object_kind = (
            mujoco.mjtObj.mjOBJ_JOINT
            if equality_type == int(mujoco.mjtEq.mjEQ_JOINT)
            else mujoco.mjtObj.mjOBJ_BODY
        )
        rows.append(
            {
                "name": _name(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id),
                "type": _enum(mujoco.mjtEq, equality_type),
                "object1": _name(model, object_kind, int(model.eq_obj1id[equality_id])),
                "object2": _name(model, object_kind, int(model.eq_obj2id[equality_id])),
                "active": int(equality_active[equality_id]),
                "data": _array(model.eq_data[equality_id]),
                "solref": _array(model.eq_solref[equality_id]),
                "solimp": _array(model.eq_solimp[equality_id]),
            }
        )
    return rows


def _compiled_model(model: mujoco.MjModel) -> dict[str, Any]:
    option = model.opt
    return {
        "mujoco_version": mujoco.__version__,
        "global": {
            "timestep_s": float(option.timestep),
            "integrator": _enum(mujoco.mjtIntegrator, option.integrator),
            "solver": _enum(mujoco.mjtSolver, option.solver),
            "iterations": int(option.iterations),
            "ls_iterations": None,
            "ls_iterations_applicable": False,
            "cone": _enum(mujoco.mjtCone, option.cone),
            "impratio": float(option.impratio),
            "gravity": _array(option.gravity),
            "tolerance": float(option.tolerance),
            "noslip_iterations": int(option.noslip_iterations),
            "noslip_tolerance": float(option.noslip_tolerance),
        },
        "finger_joints": {
            name: _joint(model, name)
            for name in ("left_driver_joint", "right_driver_joint")
        },
        "finger_bodies": {name: _body(model, name) for name in FINGER_BODIES},
        "gripper_actuator": _actuator(model, "gripper_actuator"),
        "equalities": _equalities(model),
        "tendon_count": int(model.ntendon),
        "objects": {name: _body(model, name) for name in OBJECT_BODIES},
    }


def _diff(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            if key not in left:
                rows.append({"path": child, "generation": None, "evaluation": right[key]})
            elif key not in right:
                rows.append({"path": child, "generation": left[key], "evaluation": None})
            else:
                rows.extend(_diff(left[key], right[key], child))
        return rows
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [{"path": path, "generation": left, "evaluation": right}]
        rows = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            rows.extend(_diff(left_item, right_item, f"{path}[{index}]"))
        return rows
    return [] if left == right else [{"path": path, "generation": left, "evaluation": right}]


def _build(
    *,
    model_path: Path,
    camera_config_path: Path,
    task_config_path: Path,
    task: str,
    seed: int,
    object_xy_range: float,
    object_yaw_range_deg: float,
    joint_noise: float,
) -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, Any]]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    apply_camera_calibration(model, load_simulation_config(camera_config_path))
    data = mujoco.MjData(model)
    initialize_scene(model, data, settle_steps=0)
    _, initial = configure_task_scene(
        model,
        data,
        task=task,
        seed=seed,
        object_xy_range=object_xy_range,
        object_yaw_range_deg=object_yaw_range_deg,
        joint_noise=joint_noise,
        scene_variant="clean",
        settle_steps=0,
        config_path=task_config_path,
    )
    return model, data, initial


def audit(generation_config_path: Path, evaluation_protocol_path: Path, tasks: list[str] | None) -> dict[str, Any]:
    generation = load_pipeline_config(generation_config_path.expanduser().resolve())
    evaluation = load_protocol(evaluation_protocol_path.expanduser().resolve())
    generation_model_path = model_path()
    configured_tasks = [task.task_id for task in generation.tasks]
    selected_tasks = configured_tasks if not tasks else list(dict.fromkeys(tasks))
    unknown = sorted(set(selected_tasks) - set(configured_tasks))
    if unknown:
        raise ValueError(f"Tasks absent from generation config: {unknown}")

    task_rows: dict[str, Any] = {}
    all_physics_differences: list[dict[str, Any]] = []
    for task in selected_tasks:
        generation_task = next(row for row in generation.tasks if row.task_id == task)
        generation_seed = int(generation_task.base_seed)
        evaluation_seed = int(evaluation.seed_start)
        generation_model, generation_data, generation_initial = _build(
            model_path=generation_model_path,
            camera_config_path=generation.camera_config,
            task_config_path=generation.task_scene_config,
            task=task,
            seed=generation_seed,
            object_xy_range=generation.object_xy_range_m,
            object_yaw_range_deg=generation.object_yaw_range_deg,
            joint_noise=generation.joint_noise_rad,
        )
        evaluation_model, evaluation_data, evaluation_initial = _build(
            model_path=evaluation.robot_xml_path,
            camera_config_path=evaluation.camera_config_path,
            task_config_path=evaluation.task_scene_config_path,
            task=task,
            seed=evaluation_seed,
            object_xy_range=evaluation.object_xy_range_m,
            object_yaw_range_deg=evaluation.object_yaw_range_deg,
            joint_noise=evaluation.joint_noise_rad,
        )
        generation_physics = _compiled_model(generation_model)
        evaluation_physics = _compiled_model(evaluation_model)
        physics_differences = _diff(generation_physics, evaluation_physics)
        all_physics_differences.extend(
            [{"task": task, **difference} for difference in physics_differences]
        )
        task_rows[task] = {
            "generation": {
                "seed": generation_seed,
                "compiled_physics": generation_physics,
                "initial_conditions": generation_initial,
                "initial_ctrl": _array(generation_data.ctrl),
            },
            "evaluation": {
                "seed": evaluation_seed,
                "compiled_physics": evaluation_physics,
                "initial_conditions": evaluation_initial,
                "initial_ctrl": _array(evaluation_data.ctrl),
            },
            "compiled_physics_exact_match": not physics_differences,
            "compiled_physics_differences": physics_differences,
            "initialization_differences": _diff(generation_initial, evaluation_initial),
        }

    reference_task = task_rows[selected_tasks[0]]
    generation_timestep = float(
        reference_task["generation"]["compiled_physics"]["global"]["timestep_s"]
    )
    evaluation_timestep = float(
        reference_task["evaluation"]["compiled_physics"]["global"]["timestep_s"]
    )
    generation_model_sha = _sha256(generation_model_path)
    evaluation_model_sha = _sha256(evaluation.robot_xml_path)
    generation_steps_per_control = int(
        round((1.0 / generation.action_hz) / generation_timestep)
    )
    evaluation_steps_per_control = int(
        round(evaluation.control_duration_s / evaluation_timestep)
    )

    return {
        "schema_version": "xarm_generation_evaluation_physics_audit_v1",
        "execution_safety": {
            "renderer_constructed": False,
            "mj_step_calls": 0,
            "operations": ["MjModel compilation", "reset-time model mutation", "mj_forward"],
        },
        "sources": {
            "generation_config": str(generation.path),
            "evaluation_protocol": str(evaluation_protocol_path.expanduser().resolve()),
            "generation_model_path": str(generation_model_path),
            "evaluation_model_path": str(evaluation.robot_xml_path),
            "generation_model_sha256": generation_model_sha,
            "evaluation_model_sha256": evaluation_model_sha,
            "generation_camera_config": str(generation.camera_config),
            "evaluation_camera_config": str(evaluation.camera_config_path),
            "generation_task_config": str(generation.task_scene_config),
            "evaluation_task_config": str(evaluation.task_scene_config_path),
        },
        "control_cadence": {
            "generation": {
                "control_duration_s": 1.0 / generation.action_hz,
                "mj_step_calls_per_control": generation_steps_per_control,
            },
            "evaluation": {
                "control_duration_s": evaluation.control_duration_s,
                "mj_step_calls_per_control": evaluation_steps_per_control,
                "executed_chunk_steps": evaluation.execute_chunk_steps,
            },
        },
        "reset_randomization": {
            "generation": {
                "object_xy_range_m": generation.object_xy_range_m,
                "object_yaw_range_deg": generation.object_yaw_range_deg,
                "joint_noise_rad": generation.joint_noise_rad,
            },
            "evaluation": {
                "object_xy_range_m": evaluation.object_xy_range_m,
                "object_yaw_range_deg": evaluation.object_yaw_range_deg,
                "joint_noise_rad": evaluation.joint_noise_rad,
            },
        },
        "compiled_physics_exact_match_all_tasks": not all_physics_differences,
        "compiled_physics_difference_count": len(all_physics_differences),
        "compiled_physics_differences": all_physics_differences,
        "model_sha_equal": generation_model_sha == evaluation_model_sha,
        "generation_steps_per_control": generation_steps_per_control,
        "evaluation_steps_per_control": evaluation_steps_per_control,
        "tasks": task_rows,
    }


def main() -> None:
    args = _parser().parse_args()
    result = audit(args.generation_config, args.evaluation_protocol, args.tasks)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
        return
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
