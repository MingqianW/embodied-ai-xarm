#!/usr/bin/env python3
"""Measure Menagerie xArm grip force with separately backed contact faces."""

from __future__ import annotations

import argparse
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
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sim_mujoco.gripper_mapping import (  # noqa: E402
    measure_fingertip_aperture_m,
    raw_gripper_to_menagerie_ctrl,
)
from sim_mujoco.remote_policy_observation import (  # noqa: E402
    DEFAULT_CAMERA_CONFIG_PATH,
    load_camera_config,
)
from sim_mujoco.remote_policy_control import (  # noqa: E402
    DEFAULT_GRIPPER_CLOSING_RATE_RAW_PER_S,
    DEFAULT_GRIPPER_OPENING_RATE_RAW_PER_S,
)
from sim_mujoco.scripts.run_scripted_gripper_slip_experiments import (  # noqa: E402
    BASE_MODEL_PATH,
    _sha256,
)

ALLOWED_OUTPUT_ROOT = Path("/work/nvme/bfmk/mw89")
WIDTHS_MM = (25, 30, 35, 40, 45, 50, 55)
OPEN_RAW = 845.0
CLOSED_RAW = 200.0
CONTROL_PERIOD_S = 0.1
OPEN_SETTLE_S = 0.5
CLOSED_SETTLE_S = 2.0
STEADY_WINDOW_S = 1.0
BACKING_DEPTH_M = 0.100
FACE_HALF_SIZE_XZ_M = (0.020, 0.022)
FACE_MASS_KG = 0.006
FACE_INERTIA_KG_M2 = (0.676e-6, 0.676e-6, 0.676e-6)
MAX_PENETRATION_M = 0.001
MIN_BILATERAL_FRACTION = 0.95
MIN_EXACT_COUNT_SYMMETRY_FRACTION = 0.90
MIN_NORMAL_ALIGNMENT = 0.99
MAX_PLACEMENT_ERROR_M = 1e-9
GRIPPER_ACTUATOR = "gripper_actuator"
LEFT_JOINT = "left_driver_joint"
RIGHT_JOINT = "right_driver_joint"
GRIPPER_EQUALITY = "symmetric_gripper"
PAIR_SOLREF = (0.004, 1.0)
PAIR_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
PAD_SPECS = {
    "left_finger_pad_1": {"side": "left", "friction": 0.7},
    "left_finger_pad_2": {"side": "left", "friction": 0.6},
    "right_finger_pad_1": {"side": "right", "friction": 0.7},
    "right_finger_pad_2": {"side": "right", "friction": 0.6},
}
FACE_SPECS = {
    "left": {
        "body": "left_grip_force_backing",
        "geom": "left_grip_force_face",
        "surface_sign": 1.0,
        "pads": ("left_finger_pad_1", "left_finger_pad_2"),
    },
    "right": {
        "body": "right_grip_force_backing",
        "geom": "right_grip_force_face",
        "surface_sign": -1.0,
        "pads": ("right_finger_pad_1", "right_finger_pad_2"),
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--gripper-force-limit",
        type=float,
        default=50.0,
        help="Exact symmetric actuator-space force limit; diagnostic XML only.",
    )
    parser.add_argument(
        "--closing-rate-raw-per-s",
        type=float,
        default=DEFAULT_GRIPPER_CLOSING_RATE_RAW_PER_S,
    )
    parser.add_argument(
        "--opening-rate-raw-per-s",
        type=float,
        default=DEFAULT_GRIPPER_OPENING_RATE_RAW_PER_S,
        help="Recorded for the shared policy-facing tuning protocol.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Compile and inspect all seven placements without calling mj_step.",
    )
    return parser


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *command], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = int(mujoco.mj_name2id(model, kind, name))
    if value < 0:
        raise RuntimeError(f"Required MuJoCo object is absent: {name}")
    return value


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, object_id: int) -> str:
    value = mujoco.mj_id2name(model, kind, object_id)
    return str(value) if value is not None else f"unnamed_{int(object_id)}"


def _set_absolute_meshdir(root: ET.Element) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("MJCF compiler element is absent")
    compiler.set(
        "meshdir",
        str(
            PROJECT_ROOT / "third_party/xarm_ros2/xarm_description/meshes/xarm6/visual"
        ),
    )


def _face_center_y(width_mm: int, surface_sign: float) -> float:
    half_width = width_mm / 2000.0
    return surface_sign * (half_width - BACKING_DEPTH_M / 2.0)


def build_runtime_model_xml(*, gripper_force_limit: float = 50.0) -> str:
    """Return exact production hand mechanics plus two fixed backing faces."""

    if not math.isfinite(gripper_force_limit) or gripper_force_limit <= 0.0:
        raise ValueError("gripper_force_limit must be finite and positive")

    tree = ET.parse(BASE_MODEL_PATH)
    root = tree.getroot()
    _set_absolute_meshdir(root)

    actuator = root.find(f".//actuator/*[@name='{GRIPPER_ACTUATOR}']")
    if actuator is None:
        raise RuntimeError(f"Menagerie actuator is absent: {GRIPPER_ACTUATOR}")
    actuator.set("forcerange", f"{-gripper_force_limit:g} {gripper_force_limit:g}")

    for pad_name in PAD_SPECS:
        if root.find(f".//geom[@name='{pad_name}']") is None:
            raise RuntimeError(f"Menagerie fingertip pad is absent: {pad_name}")

    worldbody = root.find("worldbody")
    contact = root.find("contact")
    if worldbody is None or contact is None:
        raise RuntimeError("MJCF worldbody or contact element is absent")
    for side, spec in FACE_SPECS.items():
        body = ET.SubElement(worldbody, "body", name=spec["body"], pos="0 0 -1")
        ET.SubElement(
            body,
            "inertial",
            pos="0 0 0",
            mass=str(FACE_MASS_KG),
            diaginertia=" ".join(str(value) for value in FACE_INERTIA_KG_M2),
        )
        ET.SubElement(
            body,
            "geom",
            name=spec["geom"],
            type="box",
            pos=f"0 {_face_center_y(WIDTHS_MM[0], spec['surface_sign'])} 0",
            size=(
                f"{FACE_HALF_SIZE_XZ_M[0]} {BACKING_DEPTH_M / 2.0} "
                f"{FACE_HALF_SIZE_XZ_M[1]}"
            ),
            material="object_material",
            contype="0",
            conaffinity="0",
            friction="1.2 0.01 0.001",
            rgba="0.25 0.75 0.25 0.35",
        )
        for pad_name in spec["pads"]:
            pad_friction = PAD_SPECS[pad_name]["friction"]
            ET.SubElement(
                contact,
                "pair",
                name=f"{side}_grip_force_pair_{pad_name}",
                geom1=pad_name,
                geom2=spec["geom"],
                condim="3",
                friction=f"{pad_friction} {pad_friction} 0.005 0.0001 0.0001",
                solref=" ".join(str(value) for value in PAIR_SOLREF),
                solimp=" ".join(str(value) for value in PAIR_SOLIMP),
                margin="0",
                gap="0",
            )
    return ET.tostring(root, encoding="unicode")


def _geom_signature(model: mujoco.MjModel, geom_id: int) -> dict[str, Any]:
    return {
        "name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
        "size_m": model.geom_size[geom_id].tolist(),
        "pos_m": model.geom_pos[geom_id].tolist(),
        "friction": model.geom_friction[geom_id].tolist(),
        "condim": int(model.geom_condim[geom_id]),
        "solref": model.geom_solref[geom_id].tolist(),
        "solimp": model.geom_solimp[geom_id].tolist(),
        "margin_m": float(model.geom_margin[geom_id]),
        "gap_m": float(model.geom_gap[geom_id]),
        "contype": int(model.geom_contype[geom_id]),
        "conaffinity": int(model.geom_conaffinity[geom_id]),
    }


def _model_signature(model: mujoco.MjModel) -> dict[str, Any]:
    actuator = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)
    left_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, LEFT_JOINT)
    right_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, RIGHT_JOINT)
    equality = _id(model, mujoco.mjtObj.mjOBJ_EQUALITY, GRIPPER_EQUALITY)
    left_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    pad_ids = [_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in PAD_SPECS]
    left_dof = int(model.jnt_dofadr[left_joint])
    right_dof = int(model.jnt_dofadr[right_joint])
    equality_active = (
        model.eq_active0 if hasattr(model, "eq_active0") else model.eq_active
    )
    faces: dict[str, Any] = {}
    for side, spec in FACE_SPECS.items():
        body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, spec["body"])
        geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, spec["geom"])
        faces[side] = {
            "fixed": int(model.body_jntnum[body_id]) == 0,
            "body_pos_m": model.body_pos[body_id].tolist(),
            "body_quat_wxyz": model.body_quat[body_id].tolist(),
            "mass_kg": float(model.body_mass[body_id]),
            "inertia_kg_m2": model.body_inertia[body_id].tolist(),
            "surface_sign": spec["surface_sign"],
            "geom": _geom_signature(model, geom_id),
        }
    diagnostic_face_ids = {
        _id(model, mujoco.mjtObj.mjOBJ_GEOM, spec["geom"])
        for spec in FACE_SPECS.values()
    }
    pair_rows = []
    for pair_id in range(int(model.npair)):
        geom1 = int(model.pair_geom1[pair_id])
        geom2 = int(model.pair_geom2[pair_id])
        if not ({geom1, geom2} & diagnostic_face_ids):
            continue
        pair_rows.append(
            {
                "geom1": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                "geom2": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                "condim": int(model.pair_dim[pair_id]),
                "friction": model.pair_friction[pair_id].tolist(),
                "solref": model.pair_solref[pair_id].tolist(),
                "solimp": model.pair_solimp[pair_id].tolist(),
                "margin_m": float(model.pair_margin[pair_id]),
                "gap_m": float(model.pair_gap[pair_id]),
            }
        )
    return {
        "simulation": {
            "timestep_s": float(model.opt.timestep),
            "integrator": int(model.opt.integrator),
            "solver": int(model.opt.solver),
            "iterations": int(model.opt.iterations),
            "ls_iterations": (
                int(model.opt.ls_iterations)
                if hasattr(model.opt, "ls_iterations")
                else None
            ),
            "cone": int(model.opt.cone),
            "impratio": float(model.opt.impratio),
            "gravity_mps2": model.opt.gravity.tolist(),
        },
        "mechanism": {
            "tendon_count": int(model.ntendon),
            "equality_count": int(model.neq),
            "equality_active_at_reset": int(equality_active[equality]),
        },
        "actuator": {
            "trntype": int(model.actuator_trntype[actuator]),
            "dyntype": int(model.actuator_dyntype[actuator]),
            "gaintype": int(model.actuator_gaintype[actuator]),
            "biastype": int(model.actuator_biastype[actuator]),
            "trnid": model.actuator_trnid[actuator].tolist(),
            "gear": model.actuator_gear[actuator].tolist(),
            "gainprm": model.actuator_gainprm[actuator].tolist(),
            "biasprm": model.actuator_biasprm[actuator].tolist(),
            "ctrlrange": model.actuator_ctrlrange[actuator].tolist(),
            "forcerange": model.actuator_forcerange[actuator].tolist(),
        },
        "finger_joints": {
            "left_range_rad": model.jnt_range[left_joint].tolist(),
            "right_range_rad": model.jnt_range[right_joint].tolist(),
            "left_damping": float(model.dof_damping[left_dof]),
            "right_damping": float(model.dof_damping[right_dof]),
            "left_frictionloss": float(model.dof_frictionloss[left_dof]),
            "right_frictionloss": float(model.dof_frictionloss[right_dof]),
            "left_armature": float(model.dof_armature[left_dof]),
            "right_armature": float(model.dof_armature[right_dof]),
        },
        "finger_bodies": {
            "left_mass_kg": float(model.body_mass[left_body]),
            "right_mass_kg": float(model.body_mass[right_body]),
            "left_inertia_kg_m2": model.body_inertia[left_body].tolist(),
            "right_inertia_kg_m2": model.body_inertia[right_body].tolist(),
        },
        "equality": {
            "type": int(model.eq_type[equality]),
            "obj1id": int(model.eq_obj1id[equality]),
            "obj2id": int(model.eq_obj2id[equality]),
            "data": model.eq_data[equality].tolist(),
            "solref": model.eq_solref[equality].tolist(),
            "solimp": model.eq_solimp[equality].tolist(),
        },
        "pads": [_geom_signature(model, value) for value in pad_ids],
        "faces": faces,
        "diagnostic_contact_pairs": sorted(
            pair_rows, key=lambda value: (value["geom1"], value["geom2"])
        ),
    }


def _controlled_invariant_hash(model: mujoco.MjModel) -> str:
    signature = _model_signature(model)
    for face in signature["faces"].values():
        face["geom"]["pos_m"][1] = "INTENTIONAL_WIDTH_VARIABLE"
    return hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def close_command_schedule(
    *,
    closing_rate_raw_per_s: float = DEFAULT_GRIPPER_CLOSING_RATE_RAW_PER_S,
) -> list[float]:
    if not math.isfinite(closing_rate_raw_per_s) or closing_rate_raw_per_s <= 0.0:
        raise ValueError("closing_rate_raw_per_s must be finite and positive")
    maximum_step_raw = closing_rate_raw_per_s * CONTROL_PERIOD_S
    values: list[float] = []
    value = OPEN_RAW
    while value > CLOSED_RAW:
        value = max(CLOSED_RAW, value - maximum_step_raw)
        values.append(value)
    return values


def _runtime_ids(model: mujoco.MjModel) -> dict[str, Any]:
    left_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, LEFT_JOINT)
    right_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, RIGHT_JOINT)
    pads: dict[str, list[int]] = {"left": [], "right": []}
    for name, spec in PAD_SPECS.items():
        pads[spec["side"]].append(_id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
    faces = {
        side: {
            "body": _id(model, mujoco.mjtObj.mjOBJ_BODY, spec["body"]),
            "geom": _id(model, mujoco.mjtObj.mjOBJ_GEOM, spec["geom"]),
            "surface_sign": float(spec["surface_sign"]),
        }
        for side, spec in FACE_SPECS.items()
    }
    allowed_pairs = {
        frozenset((pad_id, faces[side]["geom"])): side
        for side, pad_ids in pads.items()
        for pad_id in pad_ids
    }
    return {
        "actuator": _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR),
        "left_joint": left_joint,
        "right_joint": right_joint,
        "left_qpos": int(model.jnt_qposadr[left_joint]),
        "right_qpos": int(model.jnt_qposadr[right_joint]),
        "left_dof": int(model.jnt_dofadr[left_joint]),
        "right_dof": int(model.jnt_dofadr[right_joint]),
        "faces": faces,
        "fixture_geom_ids": {value["geom"] for value in faces.values()},
        "pads": pads,
        "allowed_pairs": allowed_pairs,
        "keyframe": _id(model, mujoco.mjtObj.mjOBJ_KEY, "home"),
    }


def _position_fixture(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: dict[str, Any],
    width_mm: int,
) -> dict[str, Any]:
    mujoco.mj_resetDataKeyframe(model, data, ids["keyframe"])
    mujoco.mj_forward(model, data)
    pad_ids = [*ids["pads"]["left"], *ids["pads"]["right"]]
    center = np.mean(data.geom_xpos[pad_ids], axis=0)
    pad_frames = [np.asarray(data.geom_xmat[value]).reshape(3, 3) for value in pad_ids]
    frame = pad_frames[0]
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, frame.reshape(-1))
    for side, face in ids["faces"].items():
        model.body_pos[face["body"]] = center
        model.body_quat[face["body"]] = quaternion
        model.geom_pos[face["geom"], 1] = _face_center_y(width_mm, face["surface_sign"])
    mujoco.mj_setConst(model, data)
    mujoco.mj_resetDataKeyframe(model, data, ids["keyframe"])
    mujoco.mj_forward(model, data)

    axis = np.asarray(data.geom_xmat[ids["faces"]["left"]["geom"]]).reshape(3, 3)[:, 1]
    surfaces: dict[str, np.ndarray] = {}
    expected: dict[str, np.ndarray] = {}
    errors: list[float] = []
    for side, face in ids["faces"].items():
        sign = face["surface_sign"]
        surfaces[side] = np.asarray(
            data.geom_xpos[face["geom"]], dtype=np.float64
        ) + sign * axis * float(model.geom_size[face["geom"], 1])
        expected[side] = center + sign * axis * (width_mm / 2000.0)
        errors.append(float(np.max(np.abs(surfaces[side] - expected[side]))))
    midpoint = (surfaces["left"] + surfaces["right"]) / 2.0
    separation = float(np.dot(surfaces["left"] - surfaces["right"], axis))
    errors.extend(
        [
            float(np.max(np.abs(midpoint - center))),
            abs(separation - width_mm / 1000.0),
        ]
    )
    return {
        "width_mm": width_mm,
        "center_world_m": center.tolist(),
        "closing_axis_world": axis.tolist(),
        "left_surface_world_m": surfaces["left"].tolist(),
        "right_surface_world_m": surfaces["right"].tolist(),
        "surface_midpoint_world_m": midpoint.tolist(),
        "surface_separation_m": separation,
        "backing_depth_m": BACKING_DEPTH_M,
        "max_symmetry_placement_error_m": max(errors),
        "tolerance_m": MAX_PLACEMENT_ERROR_M,
        "passed": max(errors) <= MAX_PLACEMENT_ERROR_M,
    }


def _equality_qfrc(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[list[float], np.ndarray]:
    equality_rows = [
        row
        for row in range(int(data.nefc))
        if int(data.efc_type[row]) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    ]
    selected = np.zeros(int(data.nefc), dtype=np.float64)
    if equality_rows:
        selected[equality_rows] = data.efc_force[equality_rows]
    qfrc = np.zeros(int(model.nv), dtype=np.float64)
    if int(data.nefc):
        mujoco.mj_mulJacTVec(model, data, qfrc, selected)
    return [float(data.efc_force[row]) for row in equality_rows], qfrc


def _contact_metrics(
    model: mujoco.MjModel, data: mujoco.MjData, ids: dict[str, Any]
) -> dict[str, Any]:
    values = {
        side: {
            "count": 0,
            "normal_n": 0.0,
            "tangential_n": 0.0,
            "penetration_m": 0.0,
            "alignments": [],
        }
        for side in ("left", "right")
    }
    force = np.zeros(6, dtype=np.float64)
    unintended: list[dict[str, Any]] = []
    fixture_axis = np.asarray(
        data.geom_xmat[ids["faces"]["left"]["geom"]], dtype=np.float64
    ).reshape(3, 3)[:, 1]
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        pair = frozenset((geom1, geom2))
        if not (pair & ids["fixture_geom_ids"]):
            continue
        side = ids["allowed_pairs"].get(pair)
        if side is None:
            unintended.append(
                {
                    "geom1": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                    "geom2": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                    "distance_m": float(contact.dist),
                }
            )
            continue
        mujoco.mj_contactForce(model, data, contact_index, force)
        normal_world = np.asarray(contact.frame[:3], dtype=np.float64)
        alignment = abs(float(np.dot(normal_world, fixture_axis)))
        values[side]["count"] += 1
        values[side]["normal_n"] += abs(float(force[0]))
        values[side]["tangential_n"] += float(np.linalg.norm(force[1:3]))
        values[side]["penetration_m"] = max(
            values[side]["penetration_m"], max(0.0, -float(contact.dist))
        )
        values[side]["alignments"].append(alignment)
    left = values["left"]
    right = values["right"]
    normal_total = float(left["normal_n"] + right["normal_n"])
    alignments = [*left["alignments"], *right["alignments"]]
    return {
        "left_count": int(left["count"]),
        "right_count": int(right["count"]),
        "total_count": int(left["count"] + right["count"]),
        "bilateral": bool(left["count"] and right["count"]),
        "exact_count_symmetry": int(left["count"] == right["count"]),
        "count_asymmetry": abs(int(left["count"]) - int(right["count"])),
        "normal_left_n": float(left["normal_n"]),
        "normal_right_n": float(right["normal_n"]),
        "normal_total_n": normal_total,
        "normal_force_symmetry": (
            1.0 - abs(float(left["normal_n"] - right["normal_n"])) / normal_total
            if normal_total > 0.0
            else 0.0
        ),
        "normal_alignment_min": min(alignments, default=0.0),
        "normal_alignment_left_min": min(left["alignments"], default=0.0),
        "normal_alignment_right_min": min(right["alignments"], default=0.0),
        "tangential_left_n": float(left["tangential_n"]),
        "tangential_right_n": float(right["tangential_n"]),
        "tangential_total_n": float(left["tangential_n"] + right["tangential_n"]),
        "penetration_left_m": float(left["penetration_m"]),
        "penetration_right_m": float(right["penetration_m"]),
        "penetration_max_m": max(
            float(left["penetration_m"]), float(right["penetration_m"])
        ),
        "unintended_fixture_contact_count": len(unintended),
        "unintended_fixture_contacts": unintended,
    }


def _sample(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: dict[str, Any],
    *,
    width_mm: int,
    raw_command: float,
) -> dict[str, Any]:
    actuator = ids["actuator"]
    left_dof = ids["left_dof"]
    right_dof = ids["right_dof"]
    equality_force, equality_qfrc = _equality_qfrc(model, data)
    moment_storage = np.asarray(data.actuator_moment, dtype=np.float64)
    if moment_storage.ndim == 2:
        moment = moment_storage[actuator]
    else:
        # MuJoCo 3.10 stores actuator moments as sparse rows.
        moment = np.zeros(model.nv, dtype=np.float64)
        start = int(data.moment_rowadr[actuator])
        count = int(data.moment_rownnz[actuator])
        columns = np.asarray(
            data.moment_colind[start : start + count], dtype=np.int64
        )
        moment[columns] = moment_storage[start : start + count]
    pad_centers = [
        np.mean(data.geom_xpos[ids["pads"][side]], axis=0) for side in ("left", "right")
    ]
    actuator_length = float(data.actuator_length[actuator])
    ctrl = float(data.ctrl[actuator])
    return {
        "schema_version": "xarm_menagerie_grip_force_width_sample_v3",
        "sim_time_s": float(data.time),
        "width_mm": width_mm,
        "command_raw": raw_command,
        "menagerie_ctrl": ctrl,
        "actuator": {
            "ctrl": ctrl,
            "tendon_length_rad": actuator_length,
            "tendon_velocity_radps": float(data.actuator_velocity[actuator]),
            "unloaded_equilibrium_length_rad": float(
                -model.actuator_gainprm[actuator, 0]
                * ctrl
                / model.actuator_biasprm[actuator, 1]
            ),
            "equilibrium_length_error_rad": float(
                -model.actuator_gainprm[actuator, 0]
                * ctrl
                / model.actuator_biasprm[actuator, 1]
                - actuator_length
            ),
            "affine_formula_force_actuator_space": float(
                model.actuator_gainprm[actuator, 0] * ctrl
                + model.actuator_biasprm[actuator, 1] * actuator_length
                + model.actuator_biasprm[actuator, 2] * data.actuator_velocity[actuator]
            ),
            "force_actuator_space": float(data.actuator_force[actuator]),
            "moment_left": float(moment[left_dof]),
            "moment_right": float(moment[right_dof]),
            "moment_max_abs": float(np.max(np.abs(moment), initial=0.0)),
            "qfrc_left": float(data.qfrc_actuator[left_dof]),
            "qfrc_right": float(data.qfrc_actuator[right_dof]),
        },
        "fingers": {
            "left_driver_qpos_rad": float(data.qpos[ids["left_qpos"]]),
            "right_driver_qpos_rad": float(data.qpos[ids["right_qpos"]]),
            "left_driver_qvel_radps": float(data.qvel[left_dof]),
            "right_driver_qvel_radps": float(data.qvel[right_dof]),
            "pad_center_distance_m": float(
                np.linalg.norm(pad_centers[0] - pad_centers[1])
            ),
            "realized_opening_m": measure_fingertip_aperture_m(model, data),
        },
        "contacts": _contact_metrics(model, data, ids),
        "constraints": {
            "equality_row_force": equality_force,
            "equality_qfrc_left": float(equality_qfrc[left_dof]),
            "equality_qfrc_right": float(equality_qfrc[right_dof]),
            "total_qfrc_constraint_left": float(data.qfrc_constraint[left_dof]),
            "total_qfrc_constraint_right": float(data.qfrc_constraint[right_dof]),
        },
    }


def evaluate_trial_validity(
    rows: list[dict[str, Any]], placement: dict[str, Any]
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot validate an empty steady-state window")
    bilateral_fraction = float(
        np.mean([bool(row["contacts"]["bilateral"]) for row in rows])
    )
    count_symmetry_fraction = float(
        np.mean([bool(row["contacts"]["exact_count_symmetry"]) for row in rows])
    )
    maximum_penetration = max(
        float(row["contacts"]["penetration_max_m"]) for row in rows
    )
    maximum_unintended = max(
        int(row["contacts"]["unintended_fixture_contact_count"]) for row in rows
    )
    minimum_alignment = min(
        float(row["contacts"]["normal_alignment_min"]) for row in rows
    )
    gates = {
        "bilateral_contact_fraction": {
            "observed": bilateral_fraction,
            "minimum": MIN_BILATERAL_FRACTION,
            "passed": bilateral_fraction >= MIN_BILATERAL_FRACTION,
        },
        "exact_contact_count_symmetry_fraction": {
            "observed": count_symmetry_fraction,
            "minimum": MIN_EXACT_COUNT_SYMMETRY_FRACTION,
            "passed": count_symmetry_fraction >= MIN_EXACT_COUNT_SYMMETRY_FRACTION,
        },
        "maximum_penetration_m": {
            "observed": maximum_penetration,
            "maximum": MAX_PENETRATION_M,
            "passed": maximum_penetration <= MAX_PENETRATION_M,
        },
        "unintended_fixture_contact_count": {
            "observed": maximum_unintended,
            "maximum": 0,
            "passed": maximum_unintended == 0,
        },
        "minimum_contact_normal_axis_alignment": {
            "observed": minimum_alignment,
            "minimum": MIN_NORMAL_ALIGNMENT,
            "passed": minimum_alignment >= MIN_NORMAL_ALIGNMENT,
        },
        "symmetric_fixture_placement_error_m": {
            "observed": float(placement["max_symmetry_placement_error_m"]),
            "maximum": MAX_PLACEMENT_ERROR_M,
            "passed": bool(placement["passed"]),
        },
    }
    failed = [name for name, value in gates.items() if not value["passed"]]
    return {
        "schema_version": "xarm_menagerie_grip_force_width_validity_v3",
        "passed": not failed,
        "failed_gates": failed,
        "gates": gates,
    }


def validate_runtime_model(*, gripper_force_limit: float = 50.0) -> dict[str, Any]:
    """Compile and inspect all placements; never call mj_step."""

    production_hash = _sha256(BASE_MODEL_PATH)
    xml = build_runtime_model_xml(gripper_force_limit=gripper_force_limit)
    model = mujoco.MjModel.from_xml_string(xml)
    signature = _model_signature(model)
    actuator = signature["actuator"]
    simulation = signature["simulation"]
    errors: list[str] = []
    if simulation["cone"] != int(mujoco.mjtCone.mjCONE_ELLIPTIC):
        errors.append("LOCAL cone is not elliptic")
    if not math.isclose(simulation["impratio"], 10.0, abs_tol=1e-12):
        errors.append("LOCAL impratio is not 10")
    if not math.isclose(simulation["timestep_s"], 0.002, abs_tol=1e-12):
        errors.append("timestep is not 0.002 s")
    if not math.isclose(actuator["gainprm"][0], 120.0, abs_tol=1e-12):
        errors.append("LOCAL position actuator gain is not 120")
    if not np.allclose(actuator["biasprm"][:3], [0.0, -120.0, 0.0], atol=1e-12):
        errors.append("LOCAL position actuator bias changed")
    if not np.allclose(actuator["ctrlrange"], [0.005, 0.85], atol=1e-12):
        errors.append("actuator ctrlrange changed")
    expected_forcerange = [-gripper_force_limit, gripper_force_limit]
    if not np.allclose(actuator["forcerange"], expected_forcerange, atol=1e-12):
        errors.append(
            "actuator forcerange does not match exact diagnostic request: "
            f"actual={actuator['forcerange']}, expected={expected_forcerange}"
        )
    for side, face in signature["faces"].items():
        if not face["fixed"]:
            errors.append(f"{side} diagnostic face is not fixed")
        if not math.isclose(face["mass_kg"], FACE_MASS_KG, abs_tol=1e-12):
            errors.append(f"{side} diagnostic face mass changed")
        if not np.allclose(face["inertia_kg_m2"], FACE_INERTIA_KG_M2, atol=1e-15):
            errors.append(f"{side} diagnostic face inertia changed")
        if face["geom"]["contype"] != 0 or face["geom"]["conaffinity"] != 0:
            errors.append(f"{side} face is not isolated from automatic collisions")
        if not np.allclose(
            face["geom"]["size_m"],
            [FACE_HALF_SIZE_XZ_M[0], BACKING_DEPTH_M / 2, FACE_HALF_SIZE_XZ_M[1]],
            atol=1e-12,
        ):
            errors.append(f"{side} backing size changed")
    pairs = signature["diagnostic_contact_pairs"]
    if len(pairs) != 4:
        errors.append(f"expected four explicit pad-face pairs, got {len(pairs)}")
    for pair in pairs:
        if pair["condim"] != 3:
            errors.append("diagnostic contact pair condim changed")
        pad_name = pair["geom1"] if pair["geom1"] in PAD_SPECS else pair["geom2"]
        expected_friction = PAD_SPECS[pad_name]["friction"]
        if not np.allclose(
            pair["friction"],
            [expected_friction, expected_friction, 0.005, 0.0001, 0.0001],
            atol=1e-12,
        ):
            errors.append("diagnostic contact pair friction differs from Menagerie pad")
        if not np.allclose(pair["solref"], PAIR_SOLREF, atol=1e-12):
            errors.append("diagnostic contact pair solref changed")
        if not np.allclose(pair["solimp"], PAIR_SOLIMP, atol=1e-12):
            errors.append("diagnostic contact pair solimp changed")
        if pair["margin_m"] != 0.0 or pair["gap_m"] != 0.0:
            errors.append("diagnostic contact pair margin/gap changed")
    if _sha256(BASE_MODEL_PATH) != production_hash:
        errors.append("production model changed while building runtime XML")

    placements: list[dict[str, Any]] = []
    hashes: list[str] = []
    no_step_sample: dict[str, Any] | None = None
    for width_mm in WIDTHS_MM:
        width_model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(width_model)
        runtime_ids = _runtime_ids(width_model)
        placement = _position_fixture(width_model, data, runtime_ids, width_mm)
        placements.append(placement)
        hashes.append(_controlled_invariant_hash(width_model))
        if not placement["passed"]:
            errors.append(f"{width_mm} mm fixture placement is not symmetric")
        if width_mm == WIDTHS_MM[0]:
            data.ctrl[runtime_ids["actuator"]] = raw_gripper_to_menagerie_ctrl(
                CLOSED_RAW, load_camera_config(DEFAULT_CAMERA_CONFIG_PATH)
            )
            mujoco.mj_forward(width_model, data)
            no_step_sample = _sample(
                width_model,
                data,
                runtime_ids,
                width_mm=width_mm,
                raw_command=CLOSED_RAW,
            )
    if len(set(hashes)) != 1:
        errors.append("a forbidden compiled-model invariant changes with width")
    if no_step_sample is None:
        raise RuntimeError("No-step sample was not constructed")
    return {
        "schema_version": "xarm_menagerie_grip_force_model_validation_v3",
        "passed": not errors,
        "errors": errors,
        "production_model_path": str(BASE_MODEL_PATH),
        "production_model_sha256": production_hash,
        "runtime_xml_sha256": hashlib.sha256(xml.encode("utf-8")).hexdigest(),
        "mj_step_calls": 0,
        "controlled_invariants_identical_across_widths": len(set(hashes)) == 1,
        "placements": placements,
        "no_step_probe": {
            "sim_time_s": no_step_sample["sim_time_s"],
            "actuator_moment_left": no_step_sample["actuator"]["moment_left"],
            "actuator_moment_right": no_step_sample["actuator"]["moment_right"],
            "equality_row_count": len(
                no_step_sample["constraints"]["equality_row_force"]
            ),
        },
        "signature": signature,
    }


def _step_for(model: mujoco.MjModel, data: mujoco.MjData, duration_s: float) -> None:
    steps = int(round(duration_s / float(model.opt.timestep)))
    if not math.isclose(steps * float(model.opt.timestep), duration_s, abs_tol=1e-12):
        raise RuntimeError(
            f"Duration is not an integer timestep multiple: {duration_s}"
        )
    for _ in range(steps):
        mujoco.mj_step(model, data)


def _run_width(
    model_path: Path,
    output: Path,
    width_mm: int,
    *,
    closing_rate_raw_per_s: float,
) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    ids = _runtime_ids(model)
    placement = _position_fixture(model, data, ids, width_mm)
    config = load_camera_config(DEFAULT_CAMERA_CONFIG_PATH)
    data.ctrl[ids["actuator"]] = raw_gripper_to_menagerie_ctrl(OPEN_RAW, config)
    _step_for(model, data, OPEN_SETTLE_S)
    for raw_command in close_command_schedule(
        closing_rate_raw_per_s=closing_rate_raw_per_s
    ):
        data.ctrl[ids["actuator"]] = raw_gripper_to_menagerie_ctrl(raw_command, config)
        _step_for(model, data, CONTROL_PERIOD_S)

    trial_dir = output / "trials" / f"width_{width_mm:02d}mm"
    trial_dir.mkdir(parents=True, exist_ok=False)
    trace_path = trial_dir / "steady_state_trace.jsonl"
    settle_steps = int(round(CLOSED_SETTLE_S / float(model.opt.timestep)))
    steady_steps = int(round(STEADY_WINDOW_S / float(model.opt.timestep)))
    rows: list[dict[str, Any]] = []
    for step_index in range(settle_steps):
        mujoco.mj_step(model, data)
        if step_index >= settle_steps - steady_steps:
            rows.append(
                _sample(
                    model,
                    data,
                    ids,
                    width_mm=width_mm,
                    raw_command=CLOSED_RAW,
                )
            )
    trace_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    validity = evaluate_trial_validity(rows, placement)
    (trial_dir / "validity.json").write_text(
        json.dumps(validity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "width_mm": width_mm,
        "sample_count": len(rows),
        "steady_window_s": STEADY_WINDOW_S,
        "trace": str(trace_path),
        "validity_path": str(trial_dir / "validity.json"),
        "validity": validity,
        "fixture_placement": placement,
        "final_sim_time_s": float(data.time),
        "controlled_invariant_sha256": _controlled_invariant_hash(model),
    }


def _validate_output(path: Path | None) -> Path:
    if path is None:
        raise ValueError("--output-root is required unless --validate-only is used")
    output = path.expanduser().resolve()
    if output == ALLOWED_OUTPUT_ROOT or ALLOWED_OUTPUT_ROOT not in output.parents:
        raise ValueError(f"--output-root must be a child of {ALLOWED_OUTPUT_ROOT}")
    if output.exists():
        raise FileExistsError(f"Refusing existing output root: {output}")
    return output


def main() -> None:
    args = _parser().parse_args()
    if args.opening_rate_raw_per_s <= 0.0:
        raise ValueError("--opening-rate-raw-per-s must be positive")
    validation = validate_runtime_model(gripper_force_limit=args.gripper_force_limit)
    if not validation["passed"]:
        raise RuntimeError(f"Runtime model validation failed: {validation['errors']}")
    if args.validate_only:
        if args.output_root is not None:
            raise ValueError("--validate-only does not accept --output-root")
        print(json.dumps(validation, indent=2, sort_keys=True))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("The width sweep must run inside a Slurm allocation")

    output = _validate_output(args.output_root)
    output.mkdir(parents=True, exist_ok=False)
    model_path = output / "models" / "split_pad_backed_faces.xml"
    model_path.parent.mkdir(parents=True, exist_ok=False)
    model_path.write_text(
        build_runtime_model_xml(gripper_force_limit=args.gripper_force_limit),
        encoding="utf-8",
    )
    (output / "model_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config = load_camera_config(DEFAULT_CAMERA_CONFIG_PATH)
    manifest = {
        "schema_version": "xarm_menagerie_grip_force_vs_width_v3",
        "created_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
        "repository": str(PROJECT_ROOT),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "git_status_short": _git(["status", "--short"]),
        "mujoco_version": mujoco.__version__,
        "production_model_path": str(BASE_MODEL_PATH),
        "production_model_sha256": _sha256(BASE_MODEL_PATH),
        "runtime_model_path": str(model_path),
        "runtime_model_sha256": _sha256(model_path),
        "widths_mm": list(WIDTHS_MM),
        "open_raw": OPEN_RAW,
        "closed_raw": CLOSED_RAW,
        "closed_menagerie_ctrl": raw_gripper_to_menagerie_ctrl(CLOSED_RAW, config),
        "gripper_force_limit_actuator_space": args.gripper_force_limit,
        "closing_rate_raw_per_s": args.closing_rate_raw_per_s,
        "opening_rate_raw_per_s": args.opening_rate_raw_per_s,
        "maximum_close_step_raw": (args.closing_rate_raw_per_s * CONTROL_PERIOD_S),
        "control_period_s": CONTROL_PERIOD_S,
        "open_settle_s": OPEN_SETTLE_S,
        "closed_settle_s": CLOSED_SETTLE_S,
        "steady_window_s": STEADY_WINDOW_S,
        "mj_step_calls_per_control_period": int(
            round(
                CONTROL_PERIOD_S / validation["signature"]["simulation"]["timestep_s"]
            )
        ),
        "fixture": {
            "kind": "two separately world-fixed and inward-backed contact faces",
            "backing_depth_m": BACKING_DEPTH_M,
            "face_half_size_xz_m": list(FACE_HALF_SIZE_XZ_M),
            "face_separation_is_only_independent_variable": True,
            "collision_isolation": "four explicit same-side Menagerie-pad-to-face pairs",
        },
        "validity_gates": {
            "minimum_bilateral_fraction": MIN_BILATERAL_FRACTION,
            "minimum_exact_count_symmetry_fraction": MIN_EXACT_COUNT_SYMMETRY_FRACTION,
            "maximum_penetration_m": MAX_PENETRATION_M,
            "maximum_unintended_fixture_contact_count": 0,
            "minimum_normal_axis_alignment": MIN_NORMAL_ALIGNMENT,
            "maximum_symmetric_placement_error_m": MAX_PLACEMENT_ERROR_M,
        },
        "unchanged": [
            "home arm pose and production arm control",
            "raw close command and UFACTORY four-bar conversion",
            "Menagerie affine actuator, fixed tendon, and equality/connect constraints",
            "Menagerie dual-pad-per-finger geometry",
            "effective friction, condim, margin, and gap",
            "pyramidal cone and impratio=1",
            "timestep, integrator, solver, solref, and solimp",
            "gravity",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    trials: list[dict[str, Any]] = []
    try:
        for width_mm in WIDTHS_MM:
            trials.append(
                _run_width(
                    model_path,
                    output,
                    width_mm,
                    closing_rate_raw_per_s=args.closing_rate_raw_per_s,
                )
            )
        if len({trial["controlled_invariant_sha256"] for trial in trials}) != 1:
            raise RuntimeError(
                "A forbidden compiled-model invariant changed across widths"
            )
        invalid = [
            trial["width_mm"] for trial in trials if not trial["validity"]["passed"]
        ]
        if invalid:
            raise RuntimeError(
                f"Fail-closed validity gate rejected widths; no force analysis allowed: {invalid}"
            )
    except BaseException as exc:
        (output / "results.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": repr(exc),
                    "force_metrics_authorized": False,
                    "trials": trials,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    (output / "results.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "force_metrics_authorized": True,
                "controlled_invariants_identical": True,
                "all_widths_valid": True,
                "trials": trials,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_root": str(output)}, indent=2))


if __name__ == "__main__":
    main()
