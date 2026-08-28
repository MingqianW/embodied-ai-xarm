"""Project-raw conversions for the LOCAL xArm four-bar gripper.

The external project convention remains 50 (practical closed) through 845
(practical open).  UFACTORY's linkage reference is 0 closed through 850 open.
The canonical LOCAL MJCF position-controls the ``gripper_split`` tendon in
driver-angle radians (0.005 open through 0.85 closed).  Compatibility helpers
for Delta's older affine Menagerie and frozen split-pad diagnostics remain
available, but canonical runtime control follows the actuator in the model.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np


# UFACTORY-derived linkage quantities already used by this project.
OUTER_JOINT_Y_M = 0.035
FINGER_JOINT_Y_M = 0.035465
FINGER_JOINT_Z_M = 0.042039
FINGER_INNER_SURFACE_OFFSET_M = 0.0260032024
DRIVER_MAX_POSITION = 850.0
DRIVER_UNITS_PER_RADIAN = 1000.0

# Exact current Menagerie hand.xml parameters at revision da76818e... .
MENAGERIE_CTRL_MIN = 0.0
MENAGERIE_CTRL_MAX = 255.0
MENAGERIE_DRIVER_MIN_RAD = 0.0
MENAGERIE_DRIVER_MAX_RAD = 0.85
MENAGERIE_GAIN = 0.333
MENAGERIE_LENGTH_BIAS = -100.0
MENAGERIE_VELOCITY_BIAS = -10.0
MENAGERIE_PAD_GEOMS = (
    "left_finger_pad_1",
    "left_finger_pad_2",
    "right_finger_pad_1",
    "right_finger_pad_2",
)
MENAGERIE_GRIPPER_JOINTS = (
    "left_driver_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_driver_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)

LEGACY_LEFT_SLIDE = "left_finger_slide"
LEGACY_RIGHT_SLIDE = "right_finger_slide"
LEGACY_PAD_GEOMS = (
    "left_fingertip_pad",
    "left_fingertip_pad_upper",
    "right_fingertip_pad",
    "right_fingertip_pad_upper",
)
LEGACY_SIM_PAD_INNER_OFFSET_M = 0.0025
LEGACY_SIM_SLIDE_MIN_M = 0.006
LEGACY_SIM_SLIDE_MAX_M = 0.047


def _mapping(config_or_mapping: dict[str, Any]) -> dict[str, Any]:
    return dict(config_or_mapping.get("gripper_mapping", config_or_mapping))


def is_menagerie_gripper(model: mujoco.MjModel) -> bool:
    """Return whether *model* exposes the Menagerie driver linkage."""

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint") >= 0


def raw_gripper_to_driver_angle(
    raw_value: float,
    config_or_mapping: dict[str, Any],
    *,
    operational_bounds: bool = True,
) -> float:
    """Map external raw position to UFACTORY/Menagerie driver angle.

    Raw position is linear in driver angle, not in physical fingertip aperture.
    """

    mapping = _mapping(config_or_mapping)
    low = float(mapping["raw_closed"]) if operational_bounds else 0.0
    high = (
        float(mapping["raw_open"])
        if operational_bounds
        else float(mapping.get("driver_max_position", DRIVER_MAX_POSITION))
    )
    raw = float(np.clip(raw_value, low, high))
    return (
        float(mapping.get("driver_max_position", DRIVER_MAX_POSITION)) - raw
    ) / float(mapping.get("driver_units_per_radian", DRIVER_UNITS_PER_RADIAN))


def driver_angle_to_raw_gripper(
    driver_angle_rad: float,
    config_or_mapping: dict[str, Any],
    *,
    operational_bounds: bool = True,
) -> float:
    mapping = _mapping(config_or_mapping)
    raw = float(mapping.get("driver_max_position", DRIVER_MAX_POSITION)) - float(
        mapping.get("driver_units_per_radian", DRIVER_UNITS_PER_RADIAN)
    ) * float(driver_angle_rad)
    low = float(mapping["raw_closed"]) if operational_bounds else 0.0
    high = (
        float(mapping["raw_open"])
        if operational_bounds
        else float(mapping.get("driver_max_position", DRIVER_MAX_POSITION))
    )
    return float(np.clip(raw, low, high))


def real_fingertip_aperture_m(
    raw_value: float,
    config_or_mapping: dict[str, Any],
    *,
    operational_bounds: bool = True,
) -> float:
    """Analytic UFACTORY four-bar inner aperture for a raw command."""

    drive_angle = raw_gripper_to_driver_angle(
        raw_value,
        config_or_mapping,
        operational_bounds=operational_bounds,
    )
    finger_origin_y = (
        OUTER_JOINT_Y_M
        + math.cos(drive_angle) * FINGER_JOINT_Y_M
        - math.sin(drive_angle) * FINGER_JOINT_Z_M
    )
    return 2.0 * (finger_origin_y - FINGER_INNER_SURFACE_OFFSET_M)


def raw_gripper_to_sim_slide(
    raw_value: float,
    config_or_mapping: dict[str, Any],
) -> float:
    """Map project raw state to the configured simulation coordinate.

    Despite the historical function name, the canonical LOCAL four-bar model
    returns its driver angle. Frozen split-pad diagnostics still receive a
    physical slide coordinate.
    """

    mapping = _mapping(config_or_mapping)
    if "sim_joint_min_rad" in mapping:
        angle = raw_gripper_to_driver_angle(raw_value, mapping)
        return float(
            np.clip(
                angle,
                float(mapping["sim_joint_min_rad"]),
                float(mapping["sim_joint_max_rad"]),
            )
        )
    aperture = real_fingertip_aperture_m(raw_value, mapping)
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


def sim_slide_to_raw_gripper(
    slide_value: float,
    config_or_mapping: dict[str, Any],
) -> float:
    """Invert :func:`raw_gripper_to_sim_slide`."""

    mapping = _mapping(config_or_mapping)
    if "sim_joint_min_rad" in mapping:
        angle = float(
            np.clip(
                slide_value,
                float(mapping["sim_joint_min_rad"]),
                float(mapping["sim_joint_max_rad"]),
            )
        )
        return driver_angle_to_raw_gripper(angle, mapping)
    target = float(
        np.clip(
            slide_value,
            float(mapping.get("sim_slide_min_m", LEGACY_SIM_SLIDE_MIN_M)),
            float(mapping.get("sim_slide_max_m", LEGACY_SIM_SLIDE_MAX_M)),
        )
    )
    low = float(mapping["raw_closed"])
    high = float(mapping["raw_open"])
    for _ in range(48):
        midpoint = (low + high) / 2.0
        if raw_gripper_to_sim_slide(midpoint, mapping) < target:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def driver_angle_to_menagerie_ctrl(driver_angle_rad: float) -> float:
    """Return zero-velocity affine-actuator ctrl for a desired tendon length.

    Menagerie uses ``force = 0.333*ctrl - 100*length - 10*velocity``.
    With driver equality active, split-tendon length is the common driver
    angle.  Therefore the unloaded equilibrium ctrl is ``100*angle/0.333``.
    The result is clipped to Menagerie's exact 0..255 control range.
    """

    desired = -MENAGERIE_LENGTH_BIAS * float(driver_angle_rad) / MENAGERIE_GAIN
    return float(np.clip(desired, MENAGERIE_CTRL_MIN, MENAGERIE_CTRL_MAX))


def raw_gripper_to_menagerie_ctrl(
    raw_value: float,
    config_or_mapping: dict[str, Any],
    *,
    operational_bounds: bool = True,
) -> float:
    mapping = _mapping(config_or_mapping)
    angle = raw_gripper_to_driver_angle(
        raw_value,
        mapping,
        operational_bounds=operational_bounds,
    )
    if "sim_joint_min_rad" in mapping:
        return float(
            np.clip(
                angle,
                float(mapping["sim_joint_min_rad"]),
                float(mapping["sim_joint_max_rad"]),
            )
        )
    return driver_angle_to_menagerie_ctrl(angle)


def menagerie_ctrl_to_raw_target(
    ctrl: float,
    config_or_mapping: dict[str, Any],
    *,
    operational_bounds: bool = True,
) -> float:
    mapping = _mapping(config_or_mapping)
    if "sim_joint_min_rad" in mapping:
        angle = float(
            np.clip(
                ctrl,
                float(mapping["sim_joint_min_rad"]),
                float(mapping["sim_joint_max_rad"]),
            )
        )
    else:
        ctrl_limited = float(np.clip(ctrl, MENAGERIE_CTRL_MIN, MENAGERIE_CTRL_MAX))
        angle = -MENAGERIE_GAIN * ctrl_limited / MENAGERIE_LENGTH_BIAS
    return driver_angle_to_raw_gripper(
        angle,
        mapping,
        operational_bounds=operational_bounds,
    )


def menagerie_state_to_raw_gripper(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config_or_mapping: dict[str, Any],
    *,
    operational_bounds: bool = True,
) -> float:
    angles = []
    for name in ("left_driver_joint", "right_driver_joint"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Menagerie gripper joint not found: {name}")
        angles.append(float(data.qpos[int(model.jnt_qposadr[joint_id])]))
    return driver_angle_to_raw_gripper(
        float(np.mean(angles)),
        config_or_mapping,
        operational_bounds=operational_bounds,
    )


def gripper_state_to_raw(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config_or_mapping: dict[str, Any],
) -> float:
    """Return either supported gripper state in the project raw convention."""

    if is_menagerie_gripper(model):
        return menagerie_state_to_raw_gripper(model, data, config_or_mapping)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, LEGACY_LEFT_SLIDE)
    if joint_id < 0:
        raise RuntimeError("Neither Menagerie nor legacy gripper joints were found")
    slide = float(data.qpos[int(model.jnt_qposadr[joint_id])])
    return sim_slide_to_raw_gripper(slide, config_or_mapping)


def set_menagerie_gripper_configuration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    raw_value: float,
    config_or_mapping: dict[str, Any],
    *,
    operational_bounds: bool = True,
) -> None:
    """Set the four-bar reset configuration without changing its mechanics."""

    angle = raw_gripper_to_driver_angle(
        raw_value,
        config_or_mapping,
        operational_bounds=operational_bounds,
    )
    mapping = _mapping(config_or_mapping)
    joint_names = (
        ("left_driver_joint", "right_driver_joint")
        if "sim_joint_min_rad" in mapping
        else MENAGERIE_GRIPPER_JOINTS
    )
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Menagerie gripper joint not found: {name}")
        data.qpos[int(model.jnt_qposadr[joint_id])] = angle


def set_gripper_configuration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    raw_value: float,
    config_or_mapping: dict[str, Any],
) -> None:
    """Set a reset configuration for Menagerie or frozen legacy diagnostics."""

    if is_menagerie_gripper(model):
        set_menagerie_gripper_configuration(model, data, raw_value, config_or_mapping)
        return
    slide = raw_gripper_to_sim_slide(raw_value, config_or_mapping)
    for name in (LEGACY_LEFT_SLIDE, LEGACY_RIGHT_SLIDE):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Legacy gripper joint not found: {name}")
        data.qpos[int(model.jnt_qposadr[joint_id])] = slide


def _geom_projected_half_extent(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    axis_world: np.ndarray,
) -> float:
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    sizes = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    return float(np.sum(np.abs(rotation.T @ axis_world) * sizes))


def measure_fingertip_aperture_m(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> float:
    """Measure the gap between actual paired pad box surfaces."""

    if not is_menagerie_gripper(model):
        left_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "left_fingertip_pad"
        )
        right_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "right_fingertip_pad"
        )
        if min(left_id, right_id) < 0:
            raise RuntimeError("Legacy fingertip pad pair not found")
        # The legacy slides move along world/tool Y and box half-size[1] is
        # the inward pad thickness in the validated frozen model.
        center_separation = abs(
            float(data.geom_xpos[left_id, 1] - data.geom_xpos[right_id, 1])
        )
        return float(
            center_separation
            - float(model.geom_size[left_id, 1])
            - float(model.geom_size[right_id, 1])
        )

    base_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "gripper_base",
    )
    if base_id < 0:
        base_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "xarm_gripper_base_link",
        )
    if base_id < 0:
        raise RuntimeError("Four-bar gripper base body not found")
    base_rotation = np.asarray(data.xmat[base_id], dtype=np.float64).reshape(3, 3)
    closing_axis = base_rotation[:, 1]
    gaps = []
    for index in (1, 2):
        left_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"left_finger_pad_{index}"
        )
        right_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"right_finger_pad_{index}"
        )
        if min(left_id, right_id) < 0:
            raise RuntimeError(f"Menagerie fingertip pad pair {index} not found")
        center_separation = abs(
            float(
                np.dot(data.geom_xpos[left_id] - data.geom_xpos[right_id], closing_axis)
            )
        )
        gap = (
            center_separation
            - _geom_projected_half_extent(model, data, left_id, closing_axis)
            - _geom_projected_half_extent(model, data, right_id, closing_axis)
        )
        gaps.append(gap)
    return float(np.mean(gaps))
