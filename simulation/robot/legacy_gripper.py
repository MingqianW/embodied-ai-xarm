"""Frozen mappings used only by historical gripper diagnostic models.

Production simulation uses direct-angle control from
``simulation.robot.gripper``. These helpers preserve split-slide and affine
Menagerie experiments without widening the canonical API.
"""

from __future__ import annotations

from typing import Any, Mapping

import mujoco
import numpy as np

from simulation.robot.gripper_mapping import driver_angle_rad_to_raw_hardware
from simulation.robot.gripper_mapping import raw_hardware_to_aperture_m
from simulation.robot.gripper_mapping import raw_hardware_to_driver_angle_rad


LEGACY_AFFINE_CTRL_MIN = 0.0
LEGACY_AFFINE_CTRL_MAX = 255.0
LEGACY_AFFINE_GAIN = 0.333
LEGACY_AFFINE_LENGTH_BIAS = -100.0
LEGACY_AFFINE_VELOCITY_BIAS = -10.0
LEGACY_FOUR_BAR_JOINTS = (
    "left_driver_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_driver_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)
LEGACY_LEFT_SLIDE = "left_finger_slide"
LEGACY_RIGHT_SLIDE = "right_finger_slide"
LEGACY_SIM_PAD_INNER_OFFSET_M = 0.0025
LEGACY_SIM_SLIDE_MIN_M = 0.006
LEGACY_SIM_SLIDE_MAX_M = 0.047


def _mapping(config_or_mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return config_or_mapping.get("gripper_mapping", config_or_mapping)


def raw_hardware_to_legacy_slide_m(
    raw_hardware_units: float,
    config_or_mapping: Mapping[str, Any],
) -> float:
    mapping = _mapping(config_or_mapping)
    aperture = raw_hardware_to_aperture_m(raw_hardware_units, mapping)
    slide = aperture / 2.0 + float(
        mapping.get("sim_pad_inner_offset_m", LEGACY_SIM_PAD_INNER_OFFSET_M)
    )
    return float(
        np.clip(
            slide,
            float(mapping.get("sim_slide_min_m", LEGACY_SIM_SLIDE_MIN_M)),
            float(mapping.get("sim_slide_max_m", LEGACY_SIM_SLIDE_MAX_M)),
        )
    )


def legacy_slide_m_to_raw_hardware(
    slide_m: float,
    config_or_mapping: Mapping[str, Any],
) -> float:
    mapping = _mapping(config_or_mapping)
    target = float(
        np.clip(
            slide_m,
            float(mapping.get("sim_slide_min_m", LEGACY_SIM_SLIDE_MIN_M)),
            float(mapping.get("sim_slide_max_m", LEGACY_SIM_SLIDE_MAX_M)),
        )
    )
    low = float(mapping["raw_closed"])
    high = float(mapping["raw_open"])
    for _ in range(48):
        midpoint = (low + high) / 2.0
        if raw_hardware_to_legacy_slide_m(midpoint, mapping) < target:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def driver_angle_rad_to_legacy_affine_ctrl(driver_angle_rad: float) -> float:
    desired = -LEGACY_AFFINE_LENGTH_BIAS * float(driver_angle_rad) / LEGACY_AFFINE_GAIN
    return float(np.clip(desired, LEGACY_AFFINE_CTRL_MIN, LEGACY_AFFINE_CTRL_MAX))


def raw_hardware_to_legacy_affine_ctrl(
    raw_hardware_units: float,
    config_or_mapping: Mapping[str, Any],
    *,
    operational_bounds: bool = True,
) -> float:
    return driver_angle_rad_to_legacy_affine_ctrl(
        raw_hardware_to_driver_angle_rad(
            raw_hardware_units,
            config_or_mapping,
            operational_bounds=operational_bounds,
        )
    )


def legacy_affine_ctrl_to_raw_hardware(
    ctrl: float,
    config_or_mapping: Mapping[str, Any],
    *,
    operational_bounds: bool = True,
) -> float:
    limited = float(np.clip(ctrl, LEGACY_AFFINE_CTRL_MIN, LEGACY_AFFINE_CTRL_MAX))
    angle = -LEGACY_AFFINE_GAIN * limited / LEGACY_AFFINE_LENGTH_BIAS
    return driver_angle_rad_to_raw_hardware(
        angle,
        config_or_mapping,
        operational_bounds=operational_bounds,
    )


def set_legacy_slide_configuration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    raw_hardware_units: float,
    config_or_mapping: Mapping[str, Any],
) -> None:
    slide = raw_hardware_to_legacy_slide_m(raw_hardware_units, config_or_mapping)
    for name in (LEGACY_LEFT_SLIDE, LEGACY_RIGHT_SLIDE):
        identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if identifier < 0:
            raise RuntimeError(f"Legacy gripper joint not found: {name}")
        data.qpos[int(model.jnt_qposadr[identifier])] = slide


def read_legacy_slide_raw_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config_or_mapping: Mapping[str, Any],
) -> float:
    identifier = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, LEGACY_LEFT_SLIDE
    )
    if identifier < 0:
        raise RuntimeError(f"Legacy gripper joint not found: {LEGACY_LEFT_SLIDE}")
    slide = float(data.qpos[int(model.jnt_qposadr[identifier])])
    return legacy_slide_m_to_raw_hardware(slide, config_or_mapping)


def measure_legacy_fingertip_aperture_m(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> float:
    left_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "left_fingertip_pad"
    )
    right_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "right_fingertip_pad"
    )
    if min(left_id, right_id) < 0:
        raise RuntimeError("Legacy fingertip pad pair not found")
    center_separation = abs(
        float(data.geom_xpos[left_id, 1] - data.geom_xpos[right_id, 1])
    )
    return float(
        center_separation
        - float(model.geom_size[left_id, 1])
        - float(model.geom_size[right_id, 1])
    )
