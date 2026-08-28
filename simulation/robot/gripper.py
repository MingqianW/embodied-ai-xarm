"""Canonical xArm6 four-bar gripper operations on a MuJoCo model."""

from __future__ import annotations

from typing import Any, Mapping

import mujoco
import numpy as np

from simulation.robot.gripper_mapping import driver_angle_rad_to_raw_hardware
from simulation.robot.gripper_mapping import raw_hardware_to_actuator_ctrl_rad
from simulation.robot.gripper_mapping import raw_hardware_to_driver_angle_rad
from simulation.robot.model import GRIPPER_DRIVER_JOINT_NAMES
from simulation.robot.model import body_id
from simulation.robot.model import joint_id


FINGER_PAD_PAIRS = (
    ("left_finger_pad_1", "right_finger_pad_1"),
    ("left_finger_pad_2", "right_finger_pad_2"),
)


def has_xarm_four_bar_gripper(model: mujoco.MjModel) -> bool:
    return all(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
        for name in GRIPPER_DRIVER_JOINT_NAMES
    )


def read_raw_gripper_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config_or_mapping: Mapping[str, Any],
    *,
    operational_bounds: bool = True,
) -> float:
    if not has_xarm_four_bar_gripper(model):
        raise RuntimeError("Canonical xArm four-bar gripper joints were not found")
    angles = [
        float(data.qpos[int(model.jnt_qposadr[joint_id(model, name)])])
        for name in GRIPPER_DRIVER_JOINT_NAMES
    ]
    return driver_angle_rad_to_raw_hardware(
        float(np.mean(angles)),
        config_or_mapping,
        operational_bounds=operational_bounds,
    )


def set_raw_gripper_configuration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    raw_hardware_units: float,
    config_or_mapping: Mapping[str, Any],
    *,
    operational_bounds: bool = True,
) -> None:
    if not has_xarm_four_bar_gripper(model):
        raise RuntimeError("Canonical xArm four-bar gripper joints were not found")
    angle = raw_hardware_to_driver_angle_rad(
        raw_hardware_units,
        config_or_mapping,
        operational_bounds=operational_bounds,
    )
    for name in GRIPPER_DRIVER_JOINT_NAMES:
        identifier = joint_id(model, name)
        data.qpos[int(model.jnt_qposadr[identifier])] = angle


def actuator_ctrl_from_raw_hardware(
    raw_hardware_units: float,
    config_or_mapping: Mapping[str, Any],
) -> float:
    return raw_hardware_to_actuator_ctrl_rad(raw_hardware_units, config_or_mapping)


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
    """Measure the inner gap between canonical paired pad surfaces."""

    if not has_xarm_four_bar_gripper(model):
        raise RuntimeError("Canonical xArm four-bar gripper joints were not found")
    try:
        base_identifier = body_id(model, "gripper_base")
    except RuntimeError:
        base_identifier = body_id(model, "xarm_gripper_base_link")
    base_rotation = np.asarray(
        data.xmat[base_identifier], dtype=np.float64
    ).reshape(3, 3)
    closing_axis = base_rotation[:, 1]
    gaps = []
    for left_name, right_name in FINGER_PAD_PAIRS:
        left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, left_name)
        right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, right_name)
        if min(left_id, right_id) < 0:
            raise RuntimeError(f"Fingertip pad pair not found: {left_name}, {right_name}")
        center_separation = abs(
            float(
                np.dot(
                    data.geom_xpos[left_id] - data.geom_xpos[right_id], closing_axis
                )
            )
        )
        gaps.append(
            center_separation
            - _geom_projected_half_extent(model, data, left_id, closing_axis)
            - _geom_projected_half_extent(model, data, right_id, closing_axis)
        )
    return float(np.mean(gaps))
