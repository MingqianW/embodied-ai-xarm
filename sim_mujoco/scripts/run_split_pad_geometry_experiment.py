#!/usr/bin/env python3
"""Run the paired current-versus-split-pad geometry experiment on Slurm."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
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

from sim_mujoco.scripts.run_contact_model_realism_regression import (  # noqa: E402
    PLACE_TASK,
    TASKS,
    _run_placing,
    _run_pushing,
)
from sim_mujoco.scripts.run_scripted_gripper_slip_experiments import (  # noqa: E402
    BASE_MODEL_PATH,
    _run_trial,
    _sha256,
)


ALLOWED_OUTPUT_ROOT = Path("/work/nvme/bfmk/mw89")
PROTOCOLS = ("suspended_grasp", "pushing", "placing_release")
SEEDS = (50000, 50001, 50002)
PAD_SPECS = {
    "left_fingertip_pad": {"y": -0.0075, "body": "left_finger"},
    "right_fingertip_pad": {"y": 0.0075, "body": "right_finger"},
}
PAD_HALF_SIZE_M = (0.016, 0.003, 0.009)
PAD_Z_CENTERS_M = (0.043, 0.061)
PAD_MASS_KG = 0.005


def geometry_conditions() -> list[dict[str, Any]]:
    return [
        {
            "name": "production_single_pad",
            "condition": "A",
            "force_multiplier": 1.0,
            "friction_multiplier": 1.0,
            "cone": "pyramidal",
            "impratio": 1.0,
        },
        {
            "name": "diagnostic_split_pad",
            "condition": "B",
            "force_multiplier": 1.0,
            "friction_multiplier": 1.0,
            "cone": "pyramidal",
            "impratio": 1.0,
            "geometry_variant": "split_pad_two_zone_same_envelope",
        },
    ]


def experiment_matrix(seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        tasks = (PLACE_TASK,) if protocol == "placing_release" else TASKS
        for task in tasks:
            for seed in seeds:
                for setting in geometry_conditions():
                    rows.append(
                        {
                            "protocol": protocol,
                            "task": task,
                            "seed": seed,
                            "condition": setting["condition"],
                            "setting": setting["name"],
                        }
                    )
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--seed", action="append", dest="seeds", type=int, required=True
    )
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    return parser


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *command], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate_args(args: argparse.Namespace) -> Path:
    output = args.output_root.expanduser().resolve()
    if output == ALLOWED_OUTPUT_ROOT or ALLOWED_OUTPUT_ROOT not in output.parents:
        raise ValueError(f"--output-root must be a child of {ALLOWED_OUTPUT_ROOT}")
    if output.exists():
        raise FileExistsError(f"Refusing existing output root: {output}")
    if args.hold_seconds != 5.0:
        raise ValueError("This fixed experiment requires --hold-seconds 5")
    if tuple(args.seeds) != SEEDS:
        raise ValueError(f"This fixed experiment requires seeds in order {SEEDS}")
    return output


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


def build_split_pad_model(target: Path) -> dict[str, Any]:
    """Write the runtime-only B model; never mutate the production source."""

    tree = ET.parse(BASE_MODEL_PATH)
    root = tree.getroot()
    _set_absolute_meshdir(root)
    for pad_name, expected in PAD_SPECS.items():
        body = root.find(f".//body[@name='{expected['body']}']")
        if body is None:
            raise RuntimeError(f"Finger body is absent: {expected['body']}")
        lower = body.find(f"geom[@name='{pad_name}']")
        if lower is None:
            raise RuntimeError(f"Fingertip pad is absent: {pad_name}")
        original_size = tuple(float(value) for value in lower.get("size", "").split())
        original_pos = tuple(float(value) for value in lower.get("pos", "").split())
        original_mass = float(lower.get("mass", "nan"))
        if original_size != (0.016, 0.003, 0.018):
            raise RuntimeError(f"Unexpected pad size for {pad_name}: {original_size}")
        if not math.isclose(original_pos[1], expected["y"], abs_tol=1e-12):
            raise RuntimeError(
                f"Unexpected pad y position for {pad_name}: {original_pos}"
            )
        if not math.isclose(original_pos[2], 0.052, abs_tol=1e-12):
            raise RuntimeError(
                f"Unexpected pad z position for {pad_name}: {original_pos}"
            )
        if not math.isclose(original_mass, 0.01, abs_tol=1e-12):
            raise RuntimeError(f"Unexpected pad mass for {pad_name}: {original_mass}")

        lower.set("size", "0.016 0.003 0.009")
        lower.set("pos", f"0 {expected['y']} {PAD_Z_CENTERS_M[0]}")
        lower.set("mass", "0.005")
        upper = ET.fromstring(ET.tostring(lower, encoding="unicode"))
        upper.set("name", f"{pad_name}_upper")
        upper.set("pos", f"0 {expected['y']} {PAD_Z_CENTERS_M[1]}")
        body.append(upper)

    if target.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic model: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tree.write(target, encoding="utf-8", xml_declaration=True)
    return {
        "production_model_path": str(BASE_MODEL_PATH),
        "production_model_sha256": _sha256(BASE_MODEL_PATH),
        "diagnostic_model_path": str(target),
        "diagnostic_model_sha256": _sha256(target),
        "production_model_file_modified": False,
        "geometry_variant": "split_pad_two_zone_same_envelope",
        "per_finger": {
            "original_half_size_m": [0.016, 0.003, 0.018],
            "original_center_z_m": 0.052,
            "original_mass_kg": 0.01,
            "split_half_size_m": list(PAD_HALF_SIZE_M),
            "split_center_z_m": list(PAD_Z_CENTERS_M),
            "split_mass_kg_each": PAD_MASS_KG,
            "preserved_outer_envelope_m": [0.032, 0.006, 0.036],
            "preserved_total_mass_kg": 0.01,
        },
    }


def _named_geoms(model: mujoco.MjModel) -> dict[str, int]:
    result: dict[str, int] = {}
    for geom_id in range(int(model.ngeom)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is not None:
            result[str(name)] = geom_id
    return result


def _body_values(model: mujoco.MjModel, name: str) -> dict[str, list[float] | float]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise RuntimeError(f"Missing body: {name}")
    return {
        "mass_kg": float(model.body_mass[body_id]),
        "inertia_kg_m2": model.body_inertia[body_id].tolist(),
        "ipos_m": model.body_ipos[body_id].tolist(),
    }


def _geom_values(model: mujoco.MjModel, geom_id: int) -> dict[str, Any]:
    return {
        "size_m": model.geom_size[geom_id].tolist(),
        "pos_m": model.geom_pos[geom_id].tolist(),
        "friction": model.geom_friction[geom_id].tolist(),
        "condim": int(model.geom_condim[geom_id]),
        "solref": model.geom_solref[geom_id].tolist(),
        "solimp": model.geom_solimp[geom_id].tolist(),
        "margin_m": float(model.geom_margin[geom_id]),
        "gap_m": float(model.geom_gap[geom_id]),
        "type": int(model.geom_type[geom_id]),
        "contype": int(model.geom_contype[geom_id]),
        "conaffinity": int(model.geom_conaffinity[geom_id]),
    }


def _model_invariants(model: mujoco.MjModel) -> dict[str, Any]:
    names = _named_geoms(model)
    non_pad_geoms = {
        name: _geom_values(model, geom_id)
        for name, geom_id in sorted(names.items())
        if not name.startswith(("left_fingertip_pad", "right_fingertip_pad"))
    }
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_actuator"
    )
    equality_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, "symmetric_gripper"
    )
    left_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger_slide"
    )
    right_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "right_finger_slide"
    )
    if min(actuator_id, equality_id, left_joint_id, right_joint_id) < 0:
        raise RuntimeError("Required gripper mechanism element is absent")
    return {
        "simulation": {
            "cone": int(model.opt.cone),
            "impratio": float(model.opt.impratio),
            "solver": int(model.opt.solver),
            "timestep_s": float(model.opt.timestep),
            "integrator": int(model.opt.integrator),
            "iterations": int(model.opt.iterations),
            "tolerance": float(model.opt.tolerance),
        },
        "actuator": {
            "gainprm": model.actuator_gainprm[actuator_id].tolist(),
            "biasprm": model.actuator_biasprm[actuator_id].tolist(),
            "gear": model.actuator_gear[actuator_id].tolist(),
            "ctrlrange": model.actuator_ctrlrange[actuator_id].tolist(),
            "forcerange": model.actuator_forcerange[actuator_id].tolist(),
        },
        "finger_joints": {
            "left_range": model.jnt_range[left_joint_id].tolist(),
            "right_range": model.jnt_range[right_joint_id].tolist(),
            "left_damping": float(model.dof_damping[model.jnt_dofadr[left_joint_id]]),
            "right_damping": float(model.dof_damping[model.jnt_dofadr[right_joint_id]]),
            "left_armature": float(model.dof_armature[model.jnt_dofadr[left_joint_id]]),
            "right_armature": float(
                model.dof_armature[model.jnt_dofadr[right_joint_id]]
            ),
        },
        "gripper_equality": {
            "solref": model.eq_solref[equality_id].tolist(),
            "solimp": model.eq_solimp[equality_id].tolist(),
        },
        "finger_bodies": {
            "left": _body_values(model, "left_finger"),
            "right": _body_values(model, "right_finger"),
        },
        "non_pad_geoms": non_pad_geoms,
    }


def _numerically_identical(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _numerically_identical(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _numerically_identical(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def validate_split_pad_model(diagnostic_path: Path) -> dict[str, Any]:
    """Compile A and B and fail closed unless topology is the only difference."""

    production = mujoco.MjModel.from_xml_path(str(BASE_MODEL_PATH))
    diagnostic = mujoco.MjModel.from_xml_path(str(diagnostic_path))
    production_invariants = _model_invariants(production)
    diagnostic_invariants = _model_invariants(diagnostic)
    if not _numerically_identical(production_invariants, diagnostic_invariants):
        raise RuntimeError("Split-pad model changed a non-topology invariant")

    names = _named_geoms(diagnostic)
    pad_validation: dict[str, dict[str, dict[str, Any]]] = {}
    for pad_name, expected in PAD_SPECS.items():
        values: dict[str, dict[str, Any]] = {}
        for suffix, z_value in (
            ("", PAD_Z_CENTERS_M[0]),
            ("_upper", PAD_Z_CENTERS_M[1]),
        ):
            name = f"{pad_name}{suffix}"
            geom_id = names.get(name)
            if geom_id is None:
                raise RuntimeError(f"Diagnostic pad is absent: {name}")
            geom = _geom_values(diagnostic, geom_id)
            expected_pos = [0.0, expected["y"], z_value]
            if not np.allclose(geom["size_m"], PAD_HALF_SIZE_M, rtol=0.0, atol=1e-12):
                raise RuntimeError(f"Diagnostic pad size mismatch: {name}")
            if not np.allclose(geom["pos_m"], expected_pos, rtol=0.0, atol=1e-12):
                raise RuntimeError(f"Diagnostic pad position mismatch: {name}")
            values[name] = geom
        pad_validation[pad_name] = values

    return {
        "passed": True,
        "validation_kind": "compiled_model_topology_only",
        "production_ngeom": int(production.ngeom),
        "diagnostic_ngeom": int(diagnostic.ngeom),
        "added_geom_count": int(diagnostic.ngeom - production.ngeom),
        "invariants_identical": True,
        "finger_body_mass_and_inertia_identical": True,
        "diagnostic_pads": pad_validation,
    }


def _run_condition(
    output: Path,
    *,
    protocol: str,
    task: str,
    seed: int,
    setting: dict[str, Any],
    model_path: Path,
    reference: tuple[dict[str, Any], Any] | None,
    hold_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    state_reference = None if reference is None else reference[0]
    plan_reference = None if reference is None else reference[1]
    trial_dir = (
        output / "trials" / (f"{protocol}_{task}_seed{seed}_{setting['condition']}")
    )
    trial_dir.mkdir(parents=True, exist_ok=False)
    if protocol == "suspended_grasp":
        return _run_trial(
            output_dir=trial_dir,
            task=task,
            seed=seed,
            hold_kind="suspended",
            hold_seconds=hold_seconds,
            setting=setting,
            model_path=model_path,
            record_video=False,
            initial_state_reference=state_reference,
            oracle_plan_reference=plan_reference,
            protocol=protocol,
        )
    if protocol == "pushing":
        return _run_pushing(
            trial_dir,
            task=task,
            seed=seed,
            setting=setting,
            state_reference=state_reference,
            plan_reference=plan_reference,
            model_path=model_path,
        )
    if protocol == "placing_release":
        return _run_placing(
            trial_dir,
            seed=seed,
            setting=setting,
            state_reference=state_reference,
            plan_reference=plan_reference,
            model_path=model_path,
        )
    raise ValueError(f"Unknown protocol: {protocol}")


def main() -> None:
    args = _parser().parse_args()
    output = _validate_args(args)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "The split-pad experiment must run inside a Slurm allocation"
        )

    output.mkdir(parents=True, exist_ok=False)
    diagnostic_model = output / "models" / "diagnostic_split_pad.xml"
    descriptor = build_split_pad_model(diagnostic_model)
    validation = validate_split_pad_model(diagnostic_model)
    (output / "geometry_model_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    matrix = experiment_matrix(args.seeds)
    manifest = {
        "schema_version": "xarm_split_pad_geometry_experiment_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "suite": "split_pad_geometry",
        "conditions": geometry_conditions(),
        "protocols": list(PROTOCOLS),
        "tasks": list(TASKS),
        "place_task": PLACE_TASK,
        "seeds": args.seeds,
        "trial_count": len(matrix),
        "matrix": matrix,
        "runtime_model": {**descriptor, "validation": validation},
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
        "repository": str(PROJECT_ROOT),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "git_status_short": _git(["status", "--short"]),
        "mujoco_version": mujoco.__version__,
        "python": sys.version,
        "paired_design": {
            "reference_condition": "A",
            "state_pairing": "exact non-warm-start MuJoCo state SHA-256",
            "action_pairing": "condition A plan/actions reused byte-for-byte in B",
            "changes_only": ["fingertip contact topology"],
            "required_unchanged": [
                "pyramidal cone",
                "impratio=1",
                "friction",
                "kp=500",
                "actuator/transmission",
                "solver",
                "timestep",
                "solref",
                "solimp",
                "object mass/inertia",
            ],
        },
        "primary_metrics": [
            "relative_grasp_slip",
            "stable_hold_rate",
            "contact_loss",
            "left_right_contact_count_and_manifold_symmetry",
            "contact_positions",
            "normal_tangential_forces",
            "penetration_depth_and_duration",
            "pushing_displacement",
            "release_latency",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    models = {"A": BASE_MODEL_PATH, "B": diagnostic_model}
    results: list[dict[str, Any]] = []
    references: dict[tuple[str, str, int], tuple[dict[str, Any], Any]] = {}
    try:
        for protocol in PROTOCOLS:
            tasks = (PLACE_TASK,) if protocol == "placing_release" else TASKS
            for task in tasks:
                for seed in args.seeds:
                    key = (protocol, task, seed)
                    for setting in geometry_conditions():
                        result, state, plan = _run_condition(
                            output,
                            protocol=protocol,
                            task=task,
                            seed=seed,
                            setting=setting,
                            model_path=models[setting["condition"]],
                            reference=references.get(key),
                            hold_seconds=args.hold_seconds,
                        )
                        if setting["condition"] == "A":
                            references[key] = (state, plan)
                        results.append(result)
    except BaseException as exc:
        (output / "results.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "suite": "split_pad_geometry",
                    "error": repr(exc),
                    "completed_trial_count": len(results),
                    "trials": results,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise

    complete = {
        "status": "complete",
        "suite": "split_pad_geometry",
        "trial_count": len(results),
        "trials": results,
    }
    (output / "results.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(complete, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
