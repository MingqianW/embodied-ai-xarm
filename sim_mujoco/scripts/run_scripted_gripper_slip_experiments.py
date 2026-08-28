#!/usr/bin/env python3
"""Run controlled scripted xArm grasp/hold experiments on a Slurm compute node."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from simulation.physics.collision import collision_diagnostics  # noqa: E402
from sim_mujoco.data_collection.oracle_controller import (  # noqa: E402
    OracleConfig,
    OracleStage,
    ScriptedOracleController,
)
from sim_mujoco.data_collection.ik_solver import solve_site_pose  # noqa: E402
from simulation.environment import MuJoCoEnvironment  # noqa: E402
from sim_mujoco.gripper_slip_diagnostics import (  # noqa: E402
    CommandContext,
    PhysicsTraceRecorder,
)
from sim_mujoco.remote_policy_evaluation import VideoRecorder  # noqa: E402
from simulation.scene import resolve_task  # noqa: E402


ALLOWED_OUTPUT_ROOT = Path("/work/nvme/bfmk/mw89")
CLOSED_RAW_BY_TASK = {
    "red_block": 200.0,
    "blue_block": 211.0,
    "smallest_block": 200.0,
    "largest_block": 211.0,
    "red_pepper": 250.0,
}
GRASP_OFFSET_BY_TASK = {"red_pepper": -0.020}
HOLD_KINDS = ("static", "suspended")
BASE_MODEL_PATH = PROJECT_ROOT / "simulation/assets/xarm6/xarm6_pick_scene.xml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--suite",
        choices=(
            "baseline",
            "contact",
            "force",
            "friction",
            "geometry",
            "object",
            "dynamics",
            "menagerie_forcerange",
        ),
        required=True,
    )
    parser.add_argument("--task", action="append", dest="tasks", required=True)
    parser.add_argument(
        "--seed", action="append", dest="seeds", type=int, required=True
    )
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument(
        "--prerequisite-root",
        type=Path,
        action="append",
        default=[],
        help="Completed earlier suite root; mandatory for every non-baseline suite.",
    )
    return parser


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *command], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_args(args: argparse.Namespace) -> Path:
    output = args.output_root.expanduser().resolve()
    if output == ALLOWED_OUTPUT_ROOT or ALLOWED_OUTPUT_ROOT not in output.parents:
        raise ValueError(f"--output-root must be a child of {ALLOWED_OUTPUT_ROOT}")
    if output.exists():
        raise FileExistsError(f"Refusing an existing output root: {output}")
    if not np.isfinite(args.hold_seconds) or args.hold_seconds < 2.0:
        raise ValueError("--hold-seconds must be finite and at least 2 seconds")
    unknown = sorted(set(args.tasks) - set(CLOSED_RAW_BY_TASK))
    if unknown:
        raise ValueError(f"Unsupported scripted pick tasks: {unknown}")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be non-negative")
    if args.suite == "baseline" and args.prerequisite_root:
        raise ValueError("Baseline must not depend on intervention results")
    if args.suite != "baseline" and not args.prerequisite_root:
        raise ValueError("Non-baseline suites require --prerequisite-root")
    for root in args.prerequisite_root:
        resolved = root.expanduser().resolve()
        if (
            resolved == ALLOWED_OUTPUT_ROOT
            or ALLOWED_OUTPUT_ROOT not in resolved.parents
        ):
            raise ValueError(
                f"Prerequisite must be under {ALLOWED_OUTPUT_ROOT}: {resolved}"
            )
        results = resolved / "results.json"
        if not results.is_file():
            raise FileNotFoundError(f"Prerequisite is incomplete: {results}")
        value = json.loads(results.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("status") != "complete":
            raise ValueError(f"Prerequisite did not complete cleanly: {results}")
    return output


def _settings(suite: str) -> list[dict[str, Any]]:
    if suite == "baseline":
        return [
            {
                "name": "baseline_oracle_command",
                "force_multiplier": 1.0,
                "friction_multiplier": 1.0,
            },
            {
                "name": "baseline_max_closed_raw50",
                "force_multiplier": 1.0,
                "friction_multiplier": 1.0,
                "closed_gripper_raw_override": 50.0,
            },
        ]
    if suite == "force":
        return [
            {
                "name": f"gripper_kp_{value:g}x",
                "force_multiplier": 1.0,
                "kp_multiplier": value,
                "friction_multiplier": 1.0,
            }
            for value in (2.0, 5.0)
        ]
    if suite == "menagerie_forcerange":
        return [
            {
                "name": f"menagerie_forcerange_pm{value:g}",
                "force_multiplier": 1.0,
                "force_limit_actuator_space": value,
                "friction_multiplier": 1.0,
                "gripper_closing_rate_raw_per_s": 244.0,
                "gripper_opening_rate_raw_per_s": 220.0,
            }
            for value in (1.0, 1.5, 2.0, 3.0, 5.0)
        ]
    if suite == "contact":
        return [
            {
                "name": "pyramidal_impratio1",
                "condition": "A",
                "force_multiplier": 1.0,
                "friction_multiplier": 1.0,
                "cone": "pyramidal",
                "impratio": 1.0,
            },
            {
                "name": "elliptic_impratio1",
                "condition": "B",
                "force_multiplier": 1.0,
                "friction_multiplier": 1.0,
                "cone": "elliptic",
                "impratio": 1.0,
            },
            {
                "name": "elliptic_impratio10",
                "condition": "C",
                "force_multiplier": 1.0,
                "friction_multiplier": 1.0,
                "cone": "elliptic",
                "impratio": 10.0,
            },
        ]
    if suite == "friction":
        return [
            {
                "name": f"friction_{value:g}x",
                "force_multiplier": 1.0,
                "friction_multiplier": value,
            }
            for value in (2.0, 5.0)
        ]
    if suite == "geometry":
        return [
            {
                "name": "three_patch_pad_same_envelope",
                "force_multiplier": 1.0,
                "friction_multiplier": 1.0,
                "geometry_variant": "three_patch_pad_same_envelope",
            },
            {
                "name": "pad_condim4",
                "force_multiplier": 1.0,
                "friction_multiplier": 1.0,
                "pad_condim": 4,
            },
            {
                "name": "pad_condim4_elliptic_impratio10",
                "force_multiplier": 1.0,
                "friction_multiplier": 1.0,
                "pad_condim": 4,
                "cone": "elliptic",
                "impratio": 10.0,
            },
        ]
    if suite == "object":
        return [
            {
                "name": f"object_mass_inertia_{value:g}x",
                "force_multiplier": 1.0,
                "friction_multiplier": 1.0,
                "object_mass_inertia_multiplier": value,
            }
            for value in (0.5, 2.0)
        ]
    return [
        {
            "name": "lift_slow_0.5x",
            "force_multiplier": 1.0,
            "friction_multiplier": 1.0,
            "lift_step_multiplier": 0.5,
        },
        {
            "name": "lift_fast_2x",
            "force_multiplier": 1.0,
            "friction_multiplier": 1.0,
            "lift_step_multiplier": 2.0,
        },
        {
            "name": "horizontal_transport_5cm",
            "force_multiplier": 1.0,
            "friction_multiplier": 1.0,
            "motion_profile": "horizontal_transport",
        },
        {
            "name": "horizontal_direction_change_4cm",
            "force_multiplier": 1.0,
            "friction_multiplier": 1.0,
            "motion_profile": "direction_change",
        },
        {
            "name": "wrist_yaw_20deg",
            "force_multiplier": 1.0,
            "friction_multiplier": 1.0,
            "motion_profile": "rotation",
        },
    ]


def _contact_command_variants() -> list[dict[str, Any]]:
    return [
        {"command_variant": "oracle_command"},
        {
            "command_variant": "max_closed_raw50",
            "closed_gripper_raw_override": 50.0,
        },
    ]


def _trial_settings(suite: str) -> list[dict[str, Any]]:
    settings = _settings(suite)
    if suite != "contact":
        return settings
    return [
        {**setting, **command_variant}
        for command_variant in _contact_command_variants()
        for setting in settings
    ]


def _geometry_variant_xml(setting: dict[str, Any]) -> str | None:
    variant = setting.get("geometry_variant")
    if variant is None:
        return None
    raise RuntimeError(
        f"Legacy simplified-pad geometry variant {variant!r} is incompatible "
        "with the Menagerie hand; define a separate Menagerie experiment first"
    )


def _write_geometry_variant(setting: dict[str, Any], directory: Path) -> Path:
    xml = _geometry_variant_xml(setting)
    if xml is None:
        return BASE_MODEL_PATH
    target = directory / f"{setting['name']}.xml"
    target.write_text(xml, encoding="utf-8")
    return target


def _interpolate_arm_targets(
    start: np.ndarray,
    target: np.ndarray,
    *,
    max_step_rad: float,
) -> list[np.ndarray]:
    start_value = np.asarray(start, dtype=np.float64)
    target_value = np.asarray(target, dtype=np.float64)
    count = max(
        1,
        int(
            np.ceil(
                float(np.max(np.abs(target_value - start_value))) / float(max_step_rad)
            )
        ),
    )
    return [
        start_value + alpha * (target_value - start_value)
        for alpha in np.linspace(0.0, 1.0, count + 1)[1:]
    ]


def _dynamic_arm_segments(
    environment: MuJoCoEnvironment,
    controller: ScriptedOracleController,
    setting: dict[str, Any],
) -> list[tuple[str, list[np.ndarray]]]:
    profile = setting.get("motion_profile")
    if profile is None:
        return []
    model = environment.context.model
    data = environment.context.data
    base_position = np.asarray(
        [
            controller.plan.object_position[0],
            controller.plan.object_position[1],
            controller.plan.object_position[2]
            + controller.config.lift_clearance_from_object_m,
        ],
        dtype=np.float64,
    )
    base_rotation = np.asarray(controller.plan.tcp_rotation, dtype=np.float64)
    waypoints: list[tuple[str, np.ndarray, np.ndarray]]
    if profile == "horizontal_transport":
        waypoints = [
            (
                "DYNAMIC_HORIZONTAL_TRANSPORT",
                base_position + np.asarray([0.05, 0.0, 0.0]),
                base_rotation,
            )
        ]
    elif profile == "direction_change":
        waypoints = [
            (
                "DYNAMIC_DIRECTION_OUTBOUND",
                base_position + np.asarray([0.04, 0.0, 0.0]),
                base_rotation,
            ),
            (
                "DYNAMIC_DIRECTION_REVERSAL",
                base_position + np.asarray([-0.04, 0.0, 0.0]),
                base_rotation,
            ),
        ]
    elif profile == "rotation":
        angle = np.deg2rad(20.0)
        world_yaw = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        waypoints = [
            (
                "DYNAMIC_WRIST_ROTATION",
                base_position,
                world_yaw @ base_rotation,
            )
        ]
    else:
        raise ValueError(f"Unknown motion profile: {profile}")

    segments: list[tuple[str, list[np.ndarray]]] = []
    previous = np.asarray(controller.plan.lift.joint_position, dtype=np.float64)
    for stage, position, rotation in waypoints:
        solution = solve_site_pose(
            model,
            data,
            site_name=controller.config.tcp_site,
            target_position=position,
            target_rotation=rotation,
            seed_joint_qpos=previous,
        )
        if not solution.success:
            raise RuntimeError(
                f"Dynamic waypoint IK failed for {profile}: "
                f"position_error={solution.position_error_m}, "
                f"orientation_error={solution.orientation_error_rad}"
            )
        targets = _interpolate_arm_targets(
            previous,
            solution.joint_position,
            max_step_rad=controller.config.lift_max_joint_step_rad,
        )
        segments.append((stage, targets))
        previous = np.asarray(solution.joint_position, dtype=np.float64)
    return segments


def _target_geom_ids(model: mujoco.MjModel, target_body_id: int) -> list[int]:
    def belongs_to_target(body_id: int) -> bool:
        current = int(body_id)
        while current > 0:
            if current == target_body_id:
                return True
            current = int(model.body_parentid[current])
        return current == target_body_id

    return [
        index
        for index in range(int(model.ngeom))
        if belongs_to_target(int(model.geom_bodyid[index]))
    ]


def _geom_configuration(model: mujoco.MjModel, geom_id: int) -> dict[str, Any]:
    return {
        "id": int(geom_id),
        "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
        "body_id": int(model.geom_bodyid[geom_id]),
        "type": int(model.geom_type[geom_id]),
        "size_m": model.geom_size[geom_id].tolist(),
        "pos_m": model.geom_pos[geom_id].tolist(),
        "friction": model.geom_friction[geom_id].tolist(),
        "condim": int(model.geom_condim[geom_id]),
        "solref": model.geom_solref[geom_id].tolist(),
        "solimp": model.geom_solimp[geom_id].tolist(),
        "margin_m": float(model.geom_margin[geom_id]),
        "gap_m": float(model.geom_gap[geom_id]),
    }


def _model_configuration(
    model: mujoco.MjModel,
    *,
    actuator_id: int,
    pad_ids: list[int],
    target_body_id: int,
    target_body: str,
) -> dict[str, Any]:
    menagerie = (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint") >= 0
    )
    left_joint_name = "left_driver_joint" if menagerie else "left_finger_slide"
    right_joint_name = "right_driver_joint" if menagerie else "right_finger_slide"
    equality_name = "symmetric_gripper"
    left_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, left_joint_name)
    right_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, right_joint_name
    )
    equality_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name)
    if min(left_joint_id, right_joint_id, equality_id) < 0:
        raise RuntimeError("Required gripper joints or equality are absent")
    target_geoms = _target_geom_ids(model, target_body_id)
    return {
        "simulation": {
            "mujoco_version": mujoco.__version__,
            "cone": (
                "pyramidal"
                if int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
                else "elliptic"
            ),
            "cone_enum": int(model.opt.cone),
            "impratio": float(model.opt.impratio),
            "solver": mujoco.mjtSolver(int(model.opt.solver)).name,
            "solver_enum": int(model.opt.solver),
            "iterations": int(model.opt.iterations),
            "tolerance": float(model.opt.tolerance),
            "timestep_s": float(model.opt.timestep),
            "integrator": mujoco.mjtIntegrator(int(model.opt.integrator)).name,
            "integrator_enum": int(model.opt.integrator),
            "noslip_iterations": int(model.opt.noslip_iterations),
            "noslip_tolerance": float(model.opt.noslip_tolerance),
        },
        "actuator": {
            "name": "gripper_actuator",
            "representation": (
                "local_position"
                if menagerie and float(model.actuator_ctrlrange[actuator_id, 1]) <= 1.0
                else "menagerie_affine"
                if menagerie
                else "legacy_position"
            ),
            "affine_ctrl_gain": float(model.actuator_gainprm[actuator_id, 0]),
            "length_bias": float(model.actuator_biasprm[actuator_id, 1]),
            "velocity_bias": float(model.actuator_biasprm[actuator_id, 2]),
            "gainprm": model.actuator_gainprm[actuator_id].tolist(),
            "biasprm": model.actuator_biasprm[actuator_id].tolist(),
            "gear": model.actuator_gear[actuator_id].tolist(),
            "ctrlrange": model.actuator_ctrlrange[actuator_id].tolist(),
            "forcerange_actuator_space": model.actuator_forcerange[
                actuator_id
            ].tolist(),
        },
        "finger_joints": {
            "left": {
                "name": left_joint_name,
                "range": model.jnt_range[left_joint_id].tolist(),
                "damping": float(
                    model.dof_damping[int(model.jnt_dofadr[left_joint_id])]
                ),
                "armature": float(
                    model.dof_armature[int(model.jnt_dofadr[left_joint_id])]
                ),
            },
            "right": {
                "name": right_joint_name,
                "range": model.jnt_range[right_joint_id].tolist(),
                "damping": float(
                    model.dof_damping[int(model.jnt_dofadr[right_joint_id])]
                ),
                "armature": float(
                    model.dof_armature[int(model.jnt_dofadr[right_joint_id])]
                ),
            },
        },
        "gripper_equality": {
            "name": equality_name,
            "solref": model.eq_solref[equality_id].tolist(),
            "solimp": model.eq_solimp[equality_id].tolist(),
        },
        "finger_pads": [_geom_configuration(model, value) for value in pad_ids],
        "target": {
            "body": target_body,
            "mass_kg": float(model.body_mass[target_body_id]),
            "inertia_kg_m2": model.body_inertia[target_body_id].tolist(),
            "geoms": [_geom_configuration(model, value) for value in target_geoms],
        },
    }


def _model_invariant_hashes(model: mujoco.MjModel) -> dict[str, str]:
    arrays = {
        "actuator_biasprm": model.actuator_biasprm,
        "actuator_ctrlrange": model.actuator_ctrlrange,
        "actuator_forcerange": model.actuator_forcerange,
        "actuator_gainprm": model.actuator_gainprm,
        "actuator_gear": model.actuator_gear,
        "body_ipos": model.body_ipos,
        "body_iquat": model.body_iquat,
        "body_inertia": model.body_inertia,
        "body_mass": model.body_mass,
        "body_pos": model.body_pos,
        "body_quat": model.body_quat,
        "eq_solimp": model.eq_solimp,
        "eq_solref": model.eq_solref,
        "geom_bodyid": model.geom_bodyid,
        "geom_condim": model.geom_condim,
        "geom_friction": model.geom_friction,
        "geom_gap": model.geom_gap,
        "geom_margin": model.geom_margin,
        "geom_pos": model.geom_pos,
        "geom_quat": model.geom_quat,
        "geom_size": model.geom_size,
        "geom_solimp": model.geom_solimp,
        "geom_solref": model.geom_solref,
        "geom_type": model.geom_type,
        "jnt_axis": model.jnt_axis,
        "jnt_pos": model.jnt_pos,
        "jnt_range": model.jnt_range,
        "jnt_type": model.jnt_type,
        "dof_armature": model.dof_armature,
        "dof_damping": model.dof_damping,
        "qpos0": model.qpos0,
    }
    hashes = {
        name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for name, value in arrays.items()
    }
    option_invariants = {
        "integrator": int(model.opt.integrator),
        "iterations": int(model.opt.iterations),
        "noslip_iterations": int(model.opt.noslip_iterations),
        "noslip_tolerance": float(model.opt.noslip_tolerance),
        "solver": int(model.opt.solver),
        "timestep": float(model.opt.timestep),
        "tolerance": float(model.opt.tolerance),
    }
    hashes["option_invariants"] = hashlib.sha256(
        json.dumps(option_invariants, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return hashes


def _apply_overrides(
    environment: MuJoCoEnvironment,
    setting: dict[str, Any],
    *,
    target_body: str,
) -> dict[str, Any]:
    model = environment.context.model
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_actuator"
    )
    pad_ids = [
        geom_id
        for geom_id in range(int(model.ngeom))
        if (
            (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
            is not None
            and str(name).startswith(("left_finger_pad_", "right_finger_pad_"))
            or (
                name is not None
                and str(name).startswith(("left_fingertip_pad", "right_fingertip_pad"))
            )
        )
    ]
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body)
    if actuator_id < 0 or len(pad_ids) < 2 or target_body_id < 0:
        raise RuntimeError("Required actuator, pad geom, or target body is absent")
    baseline = _model_configuration(
        model,
        actuator_id=actuator_id,
        pad_ids=pad_ids,
        target_body_id=target_body_id,
        target_body=target_body,
    )
    invariant_hashes_before = _model_invariant_hashes(model)
    force_limit = setting.get("force_limit_actuator_space")
    if force_limit is None:
        model.actuator_forcerange[actuator_id] *= float(setting["force_multiplier"])
    else:
        force_limit = float(force_limit)
        if not math.isfinite(force_limit) or force_limit <= 0.0:
            raise ValueError("force_limit_actuator_space must be finite and positive")
        if not np.allclose(
            model.actuator_forcerange[actuator_id], [-8.0, 8.0], atol=1e-12
        ):
            raise RuntimeError(
                "Force-range sweep requires untouched LOCAL [-8, 8] baseline"
            )
        model.actuator_forcerange[actuator_id] = [-force_limit, force_limit]
    kp_multiplier = float(setting.get("kp_multiplier", 1.0))
    # Scale both position-servo terms so the unloaded equilibrium is unchanged.
    model.actuator_gainprm[actuator_id, 0] *= kp_multiplier
    model.actuator_biasprm[actuator_id, 1] *= kp_multiplier
    sliding_by_name = setting.get("pad_sliding_friction_by_name", {})
    for pad_id in pad_ids:
        # Isolate sliding friction. Torsional and rolling coefficients stay at
        # baseline so this sweep is not silently a contact-patch experiment.
        pad_name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, pad_id))
        if pad_name in sliding_by_name:
            value = float(sliding_by_name[pad_name])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Invalid sliding friction for {pad_name}: {value}")
            model.geom_friction[pad_id, 0] = value
        else:
            model.geom_friction[pad_id, 0] *= float(setting["friction_multiplier"])
        if "pad_condim" in setting:
            model.geom_condim[pad_id] = int(setting["pad_condim"])
    cone = setting.get("cone")
    if cone is not None:
        if cone == "pyramidal":
            model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        elif cone == "elliptic":
            model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        else:
            raise ValueError(f"Unknown friction cone: {cone}")
    if "impratio" in setting:
        model.opt.impratio = float(setting["impratio"])
    mass_multiplier = float(setting.get("object_mass_inertia_multiplier", 1.0))
    model.body_mass[target_body_id] *= mass_multiplier
    model.body_inertia[target_body_id] *= mass_multiplier
    if mass_multiplier != 1.0:
        mujoco.mj_setConst(model, environment.context.data)
    effective = _model_configuration(
        model,
        actuator_id=actuator_id,
        pad_ids=pad_ids,
        target_body_id=target_body_id,
        target_body=target_body,
    )
    invariant_hashes_after = _model_invariant_hashes(model)
    changed_invariant_hashes = sorted(
        key
        for key, value in invariant_hashes_before.items()
        if invariant_hashes_after[key] != value
    )
    if force_limit is not None and changed_invariant_hashes != ["actuator_forcerange"]:
        raise RuntimeError(
            "Force-range isolation failed; changed compiled invariants: "
            f"{changed_invariant_hashes}"
        )
    allowed_changed = sorted(setting.get("allowed_changed_invariants", ()))
    if (
        setting.get("condition") is not None
        and changed_invariant_hashes != allowed_changed
    ):
        raise RuntimeError(
            "Contact-model condition changed forbidden model state: "
            f"actual={changed_invariant_hashes}, allowed={allowed_changed}"
        )
    if setting.get("condition") is not None:
        if baseline["simulation"]["cone"] != "elliptic" or not math.isclose(
            float(baseline["simulation"]["impratio"]), 10.0
        ):
            raise RuntimeError(
                "Validated baseline is not LOCAL elliptic/impratio=10: "
                f"{baseline['simulation']}"
            )
        if effective["simulation"]["cone"] != setting["cone"] or not math.isclose(
            float(effective["simulation"]["impratio"]),
            float(setting["impratio"]),
        ):
            raise RuntimeError(
                "Contact-model override did not produce the requested condition: "
                f"setting={setting}, effective={effective['simulation']}"
            )
    return {
        "baseline": baseline,
        "effective": effective,
        "invariant_hashes_before": invariant_hashes_before,
        "invariant_hashes_after": invariant_hashes_after,
        "changed_invariant_hashes": changed_invariant_hashes,
    }


def _step(
    environment: MuJoCoEnvironment,
    recorder: PhysicsTraceRecorder,
    command: CommandContext,
    duration_s: float,
    *,
    video_recorder: VideoRecorder | None = None,
) -> dict[str, Any]:
    model = environment.context.model
    data = environment.context.data
    steps = max(1, int(round(duration_s / model.opt.timestep)))
    collision: dict[str, Any] = {}
    for _ in range(steps):
        mujoco.mj_step(model, data)
        recorder.sample(command)
        if video_recorder is not None:
            video_recorder.maybe_record(environment.context)
        collision = collision_diagnostics(model, data)
    return collision


def _capture_initial_state(environment: MuJoCoEnvironment) -> dict[str, Any]:
    """Capture all non-warm-start state used to begin a paired intervention."""
    model = environment.context.model
    data = environment.context.data
    # Warm-start accelerations are outputs of the selected constraint formulation,
    # so copying them across A/B/C would itself confound the intervention. Preserve
    # time, qpos/qvel/act, controls/applied forces/mocap/user state, and any plugin
    # state; then let mj_forward derive contacts and accelerations under each cone.
    state_spec = (
        int(mujoco.mjtState.mjSTATE_TIME)
        | int(mujoco.mjtState.mjSTATE_PHYSICS)
        | int(mujoco.mjtState.mjSTATE_USER)
        | int(mujoco.mjtState.mjSTATE_PLUGIN)
    )
    state = np.empty(mujoco.mj_stateSize(model, state_spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, state_spec)
    runtime = environment.task_runtime
    assert runtime is not None
    return {
        "state_spec": int(state_spec),
        "state_size": int(state.size),
        "state_sha256": hashlib.sha256(
            np.ascontiguousarray(state).tobytes()
        ).hexdigest(),
        "state": state.tolist(),
        "initial_target_z_m": float(runtime.initial_target_z),
        "initial_conditions": deepcopy(environment.initial_conditions),
    }


def _restore_initial_state(
    environment: MuJoCoEnvironment,
    reference: dict[str, Any],
) -> None:
    """Restore condition A's post-reset state before planning a paired trial."""
    model = environment.context.model
    data = environment.context.data
    state_spec = int(reference["state_spec"])
    state = np.asarray(reference["state"], dtype=np.float64)
    expected_size = mujoco.mj_stateSize(model, state_spec)
    if state.shape != (expected_size,):
        raise ValueError(
            f"Paired state has shape {state.shape}; expected {(expected_size,)}"
        )
    actual_hash = hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()
    if actual_hash != reference["state_sha256"]:
        raise ValueError("Paired initial-state payload failed its SHA-256 check")
    mujoco.mj_setState(model, data, state, state_spec)
    mujoco.mj_forward(model, data)
    runtime = environment.task_runtime
    assert runtime is not None
    runtime.initial_target_z = float(reference["initial_target_z_m"])
    environment.initial_conditions = deepcopy(reference["initial_conditions"])
    environment._last_step_started_s = float(data.time)
    environment._last_step_duration_s = 0.0


def _reuse_paired_oracle_plan(
    controller: ScriptedOracleController,
    reference_plan: Any,
) -> None:
    """Make B/C execute condition A's exact scripted trajectory."""
    controller.stage = OracleStage.RESET
    controller.failure_reason = None
    controller.action_steps = 0
    controller.transitions = controller.transitions[:1]
    controller.plan = deepcopy(reference_plan)
    controller._stage_actions = controller._build_stage_actions()
    controller._stage_action_index = 0


def _oracle_action_manifest(
    controller: ScriptedOracleController,
) -> dict[str, Any]:
    """Fingerprint every scripted command before trial execution."""
    stages: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for stage in controller._SEQUENCE:
        actions = np.asarray(
            controller._stage_actions.get(stage, []),
            dtype=np.float64,
        ).reshape(-1, 7)
        stage_name = stage.value
        digest.update(stage_name.encode("utf-8"))
        digest.update(np.asarray(actions.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(actions).tobytes())
        stages.append(
            {
                "stage": stage_name,
                "action_count": int(actions.shape[0]),
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(actions).tobytes()
                ).hexdigest(),
            }
        )
    return {
        "action_dtype": "float64",
        "action_width": 7,
        "total_action_count": sum(row["action_count"] for row in stages),
        "sha256": digest.hexdigest(),
        "stages": stages,
    }


def _run_trial(
    *,
    output_dir: Path,
    task: str,
    seed: int,
    hold_kind: str,
    hold_seconds: float,
    setting: dict[str, Any],
    model_path: Path,
    record_video: bool,
    initial_state_reference: dict[str, Any] | None = None,
    oracle_plan_reference: Any | None = None,
    protocol: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    if hold_kind not in HOLD_KINDS:
        raise ValueError(f"Unknown hold kind: {hold_kind}")
    with ExitStack() as stack:
        environment = stack.enter_context(
            MuJoCoEnvironment(
                task=task,
                settle_steps=500,
                model_path=model_path,
            )
        )
        _, task_spec = resolve_task(task)
        force_range_pair = "force_limit_actuator_space" in setting
        if force_range_pair:
            # Establish every force-range candidate from the untouched
            # Menagerie reset. Applying the intervention before reset would
            # let the candidate alter its own initial linkage state.
            environment.reset(seed=seed)
            overrides = None
        else:
            overrides = _apply_overrides(
                environment,
                setting,
                target_body=str(task_spec["target_body"]),
            )
            environment.reset(seed=seed)
        if initial_state_reference is not None:
            _restore_initial_state(environment, initial_state_reference)
        paired_initial_state = _capture_initial_state(environment)
        if (
            initial_state_reference is not None
            and paired_initial_state["state_sha256"]
            != initial_state_reference["state_sha256"]
        ):
            raise RuntimeError("Restored paired state differs from condition A")
        if force_range_pair:
            overrides = _apply_overrides(
                environment,
                setting,
                target_body=str(task_spec["target_body"]),
            )
        assert overrides is not None
        base_config = OracleConfig(
            task=task,
            closed_gripper_raw=float(
                setting.get(
                    "closed_gripper_raw_override",
                    CLOSED_RAW_BY_TASK[task],
                )
            ),
            grasp_tcp_offset_from_object_m=GRASP_OFFSET_BY_TASK.get(task, -0.011),
            gripper_closing_rate_raw_per_s=float(
                setting.get("gripper_closing_rate_raw_per_s", 244.0)
            ),
            gripper_opening_rate_raw_per_s=float(
                setting.get("gripper_opening_rate_raw_per_s", 220.0)
            ),
        )
        lift_multiplier = float(setting.get("lift_step_multiplier", 1.0))
        config = replace(
            base_config,
            lift_max_joint_step_rad=(
                base_config.lift_max_joint_step_rad * lift_multiplier
            ),
        )
        controller = ScriptedOracleController(environment, config)
        if oracle_plan_reference is not None:
            _reuse_paired_oracle_plan(controller, oracle_plan_reference)
        if controller.failure_reason is not None:
            raise RuntimeError(f"Oracle planning failed: {controller.failure_reason}")
        oracle_action_manifest = _oracle_action_manifest(controller)
        runtime = environment.task_runtime
        assert runtime is not None
        target_body_id = mujoco.mj_name2id(
            environment.context.model,
            mujoco.mjtObj.mjOBJ_BODY,
            runtime.target_body,
        )
        if target_body_id < 0:
            raise RuntimeError(f"Target body is absent: {runtime.target_body}")
        trial = {
            "task": task,
            "seed": seed,
            "hold_kind": hold_kind,
            "setting": setting,
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "production_model_file_modified": False,
            "overrides": overrides,
            "oracle_config": asdict(config),
            "oracle_plan": controller.plan.to_json(),
            "oracle_plan_source_condition": (
                "A" if oracle_plan_reference is None else "A-reused"
            ),
            "oracle_action_manifest": oracle_action_manifest,
            "target_body": runtime.target_body,
            "target_mass_kg": float(
                environment.context.model.body_mass[target_body_id]
            ),
            "initial_target_z_m": runtime.initial_target_z,
            "paired_initial_state": {
                **paired_initial_state,
                "state": None,
                "source_condition": (
                    "A" if initial_state_reference is None else "A-restored"
                ),
            },
        }
        if protocol is not None:
            trial["protocol"] = protocol
        recorder = PhysicsTraceRecorder(
            model=environment.context.model,
            data=environment.context.data,
            target_body=runtime.target_body,
            camera_config=environment.context.config,
            initial_target_z_m=runtime.initial_target_z,
            trial=trial,
        )
        video_recorder = (
            VideoRecorder(output_dir / "video", fps=30) if record_video else None
        )
        if video_recorder is not None:
            stack.callback(video_recorder.close)
            video_recorder.maybe_record(environment.context)

        final_executed_stage = (
            OracleStage.HOLD if hold_kind == "static" else OracleStage.LIFT
        )
        while True:
            action = controller.next_action()
            if action is None:
                raise RuntimeError(
                    f"Oracle stopped before completing {final_executed_stage.value}: "
                    f"{controller.failure_reason}"
                )
            command = CommandContext(
                source="scripted_oracle",
                stage=controller.stage.value,
                action_step=controller.action_steps,
                gripper_returned_raw=float(action[6]),
                gripper_clamped_raw=float(action[6]),
                gripper_ctrl=float(environment.context.data.ctrl[6]),
            )
            environment.apply_action(action)
            command = replace(
                command,
                gripper_ctrl=float(environment.context.data.ctrl[6]),
            )
            collision = _step(
                environment,
                recorder,
                command,
                config.action_dt_s,
                video_recorder=video_recorder,
            )
            controller.notify_post_step(
                task_metrics=runtime.metrics(),
                collision=collision,
                simulation_finite=bool(
                    np.isfinite(environment.context.data.qpos).all()
                    and np.isfinite(environment.context.data.qvel).all()
                ),
            )
            if controller.terminal:
                raise RuntimeError(
                    f"Oracle failed during approach: {controller.failure_reason}"
                )
            if (
                controller.stage == final_executed_stage
                and controller.stage_actions_remaining == 0
            ):
                break

        hold_arm = np.asarray(
            controller.plan.grasp.joint_position
            if hold_kind == "static"
            else controller.plan.lift.joint_position,
            dtype=np.float64,
        )
        dynamic_action_count = 0
        if hold_kind == "suspended":
            for dynamic_stage, arm_targets in _dynamic_arm_segments(
                environment, controller, setting
            ):
                for arm_target in arm_targets:
                    dynamic_action = np.concatenate(
                        [
                            np.asarray(arm_target, dtype=np.float64),
                            [config.closed_gripper_raw],
                        ]
                    )
                    environment.apply_action(dynamic_action)
                    command = CommandContext(
                        source="scripted_dynamic",
                        stage=dynamic_stage,
                        action_step=controller.action_steps + dynamic_action_count,
                        gripper_returned_raw=config.closed_gripper_raw,
                        gripper_clamped_raw=config.closed_gripper_raw,
                        gripper_ctrl=float(environment.context.data.ctrl[6]),
                    )
                    _step(
                        environment,
                        recorder,
                        command,
                        config.action_dt_s,
                        video_recorder=video_recorder,
                    )
                    dynamic_action_count += 1
                    hold_arm = np.asarray(arm_target, dtype=np.float64)
        hold_action = np.concatenate([hold_arm, [config.closed_gripper_raw]])
        hold_intervals = int(round(hold_seconds / config.action_dt_s))
        for hold_index in range(hold_intervals):
            environment.apply_action(hold_action)
            command = CommandContext(
                source="scripted_hold",
                stage=f"DIAGNOSTIC_{hold_kind.upper()}_HOLD",
                action_step=(
                    controller.action_steps + dynamic_action_count + hold_index
                ),
                gripper_returned_raw=config.closed_gripper_raw,
                gripper_clamped_raw=config.closed_gripper_raw,
                gripper_ctrl=float(environment.context.data.ctrl[6]),
            )
            _step(
                environment,
                recorder,
                command,
                config.action_dt_s,
                video_recorder=video_recorder,
            )

        artifacts = recorder.write(output_dir)
        video_metadata: dict[str, Any] | None = None
        if video_recorder is not None:
            video_recorder.close()
            video_recorder.validate_outputs()
            video_metadata = video_recorder.metadata()
        rows = recorder.rows
        slips = [
            row["relative"]["downward_slip_m"]
            for row in rows
            if row["relative"]["downward_slip_m"] is not None
        ]
        hold_rows = [row for row in rows if row["command"]["source"] == "scripted_hold"]
        result = {
            "status": "complete",
            "task": task,
            "seed": seed,
            "hold_kind": hold_kind,
            "setting": setting,
            "sample_count": len(rows),
            "hold_sample_count": len(hold_rows),
            "final_lift_height_m": float(rows[-1]["object"]["lift_height_m"]),
            "maximum_downward_slip_m": max(slips, default=None),
            "final_downward_slip_m": slips[-1] if slips else None,
            "hold_bilateral_contact_fraction": (
                sum(bool(row["contacts"]["bilateral"]) for row in hold_rows)
                / len(hold_rows)
                if hold_rows
                else None
            ),
            "maximum_abs_actuator_force_actuator_space": max(
                (
                    abs(float(row["actuator"]["force_actuator_space"]))
                    for row in hold_rows
                ),
                default=None,
            ),
            "event_names": [event.event for event in recorder.events],
            "artifacts": {key: str(value) for key, value in artifacts.items()},
            "video": video_metadata,
        }
        if protocol is not None:
            result["protocol"] = protocol
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result, paired_initial_state, deepcopy(controller.plan)


def main() -> None:
    args = _parser().parse_args()
    output = _validate_args(args)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": "xarm_scripted_gripper_slip_suite_v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "suite": args.suite,
        "tasks": args.tasks,
        "seeds": args.seeds,
        "hold_seconds": args.hold_seconds,
        "settings": _settings(args.suite),
        "command_variants": (
            _contact_command_variants() if args.suite == "contact" else []
        ),
        "prerequisite_roots": [
            str(path.expanduser().resolve()) for path in args.prerequisite_root
        ],
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
        "repository": str(PROJECT_ROOT),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "git_status_short": _git(["status", "--short"]),
        "mujoco_version": mujoco.__version__,
        "python": sys.version,
        "representative_video_selection": {
            "suite": args.suite,
            "task": args.tasks[0],
            "seed": args.seeds[0],
            "hold_kind": "suspended",
            "command_variant": "oracle_command",
            "all_settings": args.suite in {"baseline", "contact"},
        },
        "paired_design": {
            "randomization_added": False,
            "pairing_keys": [
                "task",
                "seed",
                "hold_kind",
                "command_variant",
            ],
            "contact_suite_commands": [
                "task-specific validated oracle command",
                "validated maximum-closure raw=50 command",
            ],
            "contact_suite_changes_only": ["cone", "impratio"],
            "menagerie_forcerange_changes_only": ["actuator_forcerange"],
            "gripper_rates_raw_per_s": {"closing": 244.0, "opening": 220.0},
        },
    }
    if not manifest["slurm_job_id"]:
        raise RuntimeError(
            "Controlled MuJoCo experiments must run inside a Slurm allocation"
        )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    results: list[dict[str, Any]] = []
    paired_initial_states: dict[tuple[str, int, str], dict[str, Any]] = {}
    paired_oracle_plans: dict[tuple[str, int, str], Any] = {}
    try:
        model_variant_root = output / "model_variants"
        model_variant_root.mkdir(exist_ok=False)
        model_paths = {
            setting["name"]: _write_geometry_variant(setting, model_variant_root)
            for setting in manifest["settings"]
        }
        for task in args.tasks:
            for seed in args.seeds:
                for setting in _trial_settings(args.suite):
                    hold_kinds = (
                        ("suspended",)
                        if args.suite in {"contact", "dynamics", "menagerie_forcerange"}
                        else HOLD_KINDS
                    )
                    for hold_kind in hold_kinds:
                        command_suffix = (
                            f"_{setting['command_variant']}"
                            if args.suite == "contact"
                            else ""
                        )
                        trial_name = (
                            f"{task}_seed{seed}_{hold_kind}_{setting['name']}"
                            f"{command_suffix}"
                        )
                        trial_dir = output / "trials" / trial_name
                        trial_dir.mkdir(parents=True, exist_ok=False)
                        pairing_key = (
                            task,
                            seed,
                            str(setting.get("command_variant", "oracle_command")),
                        )
                        is_contact_reference = (
                            args.suite == "contact" and setting["condition"] == "A"
                        )
                        is_force_reference = (
                            args.suite == "menagerie_forcerange"
                            and setting["name"] == _settings(args.suite)[0]["name"]
                        )
                        paired_suite = args.suite in {
                            "contact",
                            "menagerie_forcerange",
                        }
                        is_pair_reference = is_contact_reference or is_force_reference
                        if paired_suite and not is_pair_reference:
                            if (
                                pairing_key not in paired_initial_states
                                or pairing_key not in paired_oracle_plans
                            ):
                                raise RuntimeError(
                                    f"Condition A has not established pair {pairing_key}"
                                )
                            initial_state_reference = paired_initial_states[pairing_key]
                            oracle_plan_reference = paired_oracle_plans[pairing_key]
                        else:
                            initial_state_reference = None
                            oracle_plan_reference = None
                        result, paired_initial_state, oracle_plan = _run_trial(
                            output_dir=trial_dir,
                            task=task,
                            seed=seed,
                            hold_kind=hold_kind,
                            hold_seconds=args.hold_seconds,
                            setting=setting,
                            model_path=model_paths[setting["name"]],
                            record_video=(
                                args.suite in {"baseline", "contact"}
                                and task == args.tasks[0]
                                and seed == args.seeds[0]
                                and hold_kind == "suspended"
                                and setting.get("command_variant", "oracle_command")
                                == "oracle_command"
                            ),
                            initial_state_reference=initial_state_reference,
                            oracle_plan_reference=oracle_plan_reference,
                        )
                        if is_pair_reference:
                            paired_initial_states[pairing_key] = paired_initial_state
                            paired_oracle_plans[pairing_key] = oracle_plan
                        results.append(result)
    except BaseException as exc:
        failure = {
            "status": "failed",
            "error": repr(exc),
            "completed_trial_count": len(results),
            "trials": results,
        }
        (output / "results.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise

    complete = {
        "status": "complete",
        "suite": args.suite,
        "trial_count": len(results),
        "trials": results,
    }
    (output / "results.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(complete, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
