#!/usr/bin/env python3
"""Validate Menagerie xArm gripper direction, aperture, symmetry, and mapping."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
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

from simulation.robot.model import XARM_FOUR_BAR_JOINT_NAMES  # noqa: E402
from simulation.robot.gripper import measure_fingertip_aperture_m  # noqa: E402
from simulation.robot.gripper import read_raw_gripper_position  # noqa: E402
from simulation.robot.gripper import set_raw_gripper_configuration  # noqa: E402
from simulation.robot.gripper_mapping import raw_hardware_to_actuator_ctrl_rad  # noqa: E402
from simulation.resources import DEFAULT_CAMERA_CONFIG_PATH  # noqa: E402
from simulation.resources import DEFAULT_MODEL_PATH  # noqa: E402
from simulation.configuration import load_simulation_config  # noqa: E402


ALLOWED_OUTPUT_ROOT = Path("/work/nvme/bfmk/mw89")
RAW_COMMANDS = (50, 100, 200, 300, 400, 500, 600, 700, 800, 845)
OFFICIAL_REFERENCE_COMMANDS = (0, 850)
SETTLE_SECONDS = 2.0


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = int(mujoco.mj_name2id(model, kind, name))
    if value < 0:
        raise RuntimeError(f"Required MuJoCo object is absent: {name}")
    return value


def _joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qpos[int(model.jnt_qposadr[joint])])


def static_validation() -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH))
    data = mujoco.MjData(model)
    config = load_simulation_config(DEFAULT_CAMERA_CONFIG_PATH)
    actuator = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_actuator")
    for name in XARM_FOUR_BAR_JOINT_NAMES:
        _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    pads = [
        f"{side}_finger_pad_{index}" for side in ("left", "right") for index in (1, 2)
    ]
    for name in pads:
        _id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    for name in ("base_camera", "wrist_camera"):
        _id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)

    kinematic_rows = []
    for raw in (
        *OFFICIAL_REFERENCE_COMMANDS[:1],
        *RAW_COMMANDS,
        OFFICIAL_REFERENCE_COMMANDS[1],
    ):
        set_raw_gripper_configuration(
            model,
            data,
            raw,
            config,
            operational_bounds=False,
        )
        data.ctrl[actuator] = raw_hardware_to_actuator_ctrl_rad(
            raw,
            config,
            operational_bounds=False,
        )
        mujoco.mj_forward(model, data)
        equality_rows = np.asarray(data.efc_type) == int(
            mujoco.mjtConstraint.mjCNSTR_EQUALITY
        )
        kinematic_rows.append(
            {
                "raw": raw,
                "ctrl": float(data.ctrl[actuator]),
                "left_driver_rad": _joint_qpos(model, data, "left_driver_joint"),
                "right_driver_rad": _joint_qpos(model, data, "right_driver_joint"),
                "aperture_mm": 1000.0 * measure_fingertip_aperture_m(model, data),
                "maximum_equality_residual": float(
                    np.max(np.abs(np.asarray(data.efc_pos)[equality_rows]), initial=0.0)
                ),
            }
        )
    apertures = [row["aperture_mm"] for row in kinematic_rows]
    errors = []
    if model.nu != 7:
        errors.append(f"expected 7 actuators, found {model.nu}")
    if model.ntendon != 1:
        errors.append(f"expected 1 tendon, found {model.ntendon}")
    if model.neq != 3:
        errors.append(f"expected 3 equalities, found {model.neq}")
    if not np.all(np.diff(apertures) > 0.0):
        errors.append("kinematic aperture is not strictly increasing with raw command")
    return {
        "schema_version": "xarm_menagerie_gripper_static_validation_v1",
        "passed": not errors,
        "errors": errors,
        "mj_step_calls": 0,
        "model": str(DEFAULT_MODEL_PATH),
        "mujoco_version": mujoco.__version__,
        "actuator_count": int(model.nu),
        "gripper_joint_names": list(XARM_FOUR_BAR_JOINT_NAMES),
        "gripper_actuator_name": "gripper_actuator",
        "actuator_ctrlrange": model.actuator_ctrlrange[actuator].tolist(),
        "actuator_forcerange": model.actuator_forcerange[actuator].tolist(),
        "left_driver_range": model.jnt_range[
            _id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")
        ].tolist(),
        "right_driver_range": model.jnt_range[
            _id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
        ].tolist(),
        "equality_count": int(model.neq),
        "tendon_count": int(model.ntendon),
        "gripper_pad_count": len(pads),
        "gripper_pad_names": pads,
        "kinematic_mapping": kinematic_rows,
    }


def _step_for(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> None:
    count = round(seconds / float(model.opt.timestep))
    for _ in range(count):
        mujoco.mj_step(model, data)


def _close_renderer(renderer: Any) -> None:
    """Release rendering resources across the MuJoCo 2.x/3.x Python APIs."""
    close = getattr(renderer, "close", None)
    if callable(close):
        close()
        return
    renderer._mjr_context.free()
    renderer._gl_context.free()


def _dynamic_trial(
    model: mujoco.MjModel,
    config: dict[str, Any],
    *,
    ctrl: float,
    initial_raw: float,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    key = _id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key)
    set_raw_gripper_configuration(model, data, initial_raw, config)
    data.ctrl[6] = float(ctrl)
    mujoco.mj_forward(model, data)
    start_aperture = measure_fingertip_aperture_m(model, data)
    _step_for(model, data, SETTLE_SECONDS)
    left = _joint_qpos(model, data, "left_driver_joint")
    right = _joint_qpos(model, data, "right_driver_joint")
    warning_count = sum(int(value.number) for value in data.warning)
    return {
        "ctrl": float(ctrl),
        "initial_raw": float(initial_raw),
        "initial_aperture_mm": 1000.0 * start_aperture,
        "left_driver_rad": left,
        "right_driver_rad": right,
        "driver_symmetry_error_rad": abs(left - right),
        "measured_aperture_mm": 1000.0 * measure_fingertip_aperture_m(model, data),
        "reconstructed_raw": read_raw_gripper_position(model, data, config),
        "actuator_force_actuator_space": float(data.actuator_force[6]),
        "warning_count": warning_count,
        "finite": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
    }


def run_dynamic(output_root: Path) -> dict[str, Any]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Dynamic gripper validation requires a Slurm allocation")
    output = output_root.expanduser().resolve()
    if output == ALLOWED_OUTPUT_ROOT or ALLOWED_OUTPUT_ROOT not in output.parents:
        raise ValueError(f"--output-root must be below {ALLOWED_OUTPUT_ROOT}")
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH))
    config = load_simulation_config(DEFAULT_CAMERA_CONFIG_PATH)
    static = static_validation()
    direction = [
        _dynamic_trial(model, config, ctrl=0.0, initial_raw=425.0),
        _dynamic_trial(model, config, ctrl=0.8, initial_raw=425.0),
    ]
    sweep = []
    for raw in RAW_COMMANDS:
        row = _dynamic_trial(
            model,
            config,
            ctrl=raw_hardware_to_actuator_ctrl_rad(raw, config),
            initial_raw=845.0,
        )
        row["project_raw_command"] = raw
        row["raw_round_trip_error"] = row["reconstructed_raw"] - raw
        sweep.append(row)
    renderer = mujoco.Renderer(model, width=224, height=224)
    render_data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, render_data, _id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    )
    mujoco.mj_forward(model, render_data)
    rendered = {}
    try:
        for camera in ("base_camera", "wrist_camera"):
            renderer.update_scene(render_data, camera=camera)
            image = np.asarray(renderer.render())
            rendered[camera] = {
                "shape": list(image.shape),
                "finite": bool(np.isfinite(image).all()),
            }
    finally:
        _close_renderer(renderer)
    result = {
        "schema_version": "xarm_menagerie_gripper_dynamic_validation_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "static": static,
        "direction": direction,
        "raw_sweep": sweep,
        "camera_render": rendered,
        "measured_direction": (
            "ctrl_0_open_ctrl_255_closed"
            if direction[0]["measured_aperture_mm"]
            > direction[1]["measured_aperture_mm"]
            else "unexpected"
        ),
        "maximum_round_trip_error_raw": max(
            abs(float(row["raw_round_trip_error"])) for row in sweep
        ),
        "maximum_driver_symmetry_error_rad": max(
            float(row["driver_symmetry_error_rad"]) for row in sweep
        ),
    }
    (output / "open_close_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.validate_only:
        if args.output_root is not None:
            raise ValueError("--validate-only does not accept --output-root")
        result = static_validation()
    else:
        if args.output_root is None:
            raise ValueError("--output-root is required")
        result = run_dynamic(args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("passed", result.get("static", {}).get("passed", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
