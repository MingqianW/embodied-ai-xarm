from __future__ import annotations

import math
from typing import Any

import numpy as np


# Official xArm gripper linkage dimensions from xarm_gripper.urdf.xacro.
OUTER_JOINT_Y_M = 0.035
FINGER_JOINT_Y_M = 0.035465
FINGER_JOINT_Z_M = 0.042039
FINGER_INNER_SURFACE_OFFSET_M = 0.0260032024
DRIVER_MAX_POSITION = 850.0
DRIVER_UNITS_PER_RADIAN = 1000.0


def real_fingertip_aperture_m(raw_value: float, mapping: dict[str, Any]) -> float:
    raw = float(np.clip(raw_value, float(mapping["raw_closed"]), float(mapping["raw_open"])))
    drive_angle = (float(mapping.get("driver_max_position", DRIVER_MAX_POSITION)) - raw) / float(
        mapping.get("driver_units_per_radian", DRIVER_UNITS_PER_RADIAN)
    )
    finger_origin_y = (
        OUTER_JOINT_Y_M
        + math.cos(drive_angle) * FINGER_JOINT_Y_M
        - math.sin(drive_angle) * FINGER_JOINT_Z_M
    )
    return 2.0 * (finger_origin_y - FINGER_INNER_SURFACE_OFFSET_M)


def raw_gripper_to_sim_slide(raw_value: float, config: dict[str, Any]) -> float:
    mapping = config["gripper_mapping"]
    if mapping.get("model") == "menagerie_xarm7_four_bar":
        raw = float(np.clip(raw_value, float(mapping["raw_closed"]), float(mapping["raw_open"])))
        driver_angle = (
            float(mapping.get("driver_max_position", DRIVER_MAX_POSITION)) - raw
        ) / float(mapping.get("driver_units_per_radian", DRIVER_UNITS_PER_RADIAN))
        return float(np.clip(driver_angle, float(mapping["sim_joint_min_rad"]), float(mapping["sim_joint_max_rad"])))
    aperture = real_fingertip_aperture_m(raw_value, mapping)
    pad_inner_offset = float(mapping["sim_pad_inner_offset_m"])
    slide = aperture / 2.0 + pad_inner_offset
    return float(
        np.clip(
            slide,
            float(mapping["sim_slide_min_m"]),
            float(mapping["sim_slide_max_m"]),
        )
    )


def sim_slide_to_raw_gripper(slide_value: float, config: dict[str, Any]) -> float:
    mapping = config["gripper_mapping"]
    if mapping.get("model") == "menagerie_xarm7_four_bar":
        driver_angle = float(np.clip(slide_value, float(mapping["sim_joint_min_rad"]), float(mapping["sim_joint_max_rad"])))
        raw = float(mapping.get("driver_max_position", DRIVER_MAX_POSITION)) - driver_angle * float(
            mapping.get("driver_units_per_radian", DRIVER_UNITS_PER_RADIAN)
        )
        return float(np.clip(raw, float(mapping["raw_closed"]), float(mapping["raw_open"])))
    target = float(
        np.clip(
            slide_value,
            float(mapping["sim_slide_min_m"]),
            float(mapping["sim_slide_max_m"]),
        )
    )
    low = float(mapping["raw_closed"])
    high = float(mapping["raw_open"])
    for _ in range(48):
        midpoint = (low + high) / 2.0
        if raw_gripper_to_sim_slide(midpoint, config) < target:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0
