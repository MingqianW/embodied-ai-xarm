"""Unit-explicit conversion for the canonical direct-angle xArm gripper."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


OUTER_JOINT_Y_M = 0.035
FINGER_JOINT_Y_M = 0.035465
FINGER_JOINT_Z_M = 0.042039
FINGER_INNER_SURFACE_OFFSET_M = 0.0260032024
DEFAULT_DRIVER_MAX_HARDWARE_UNITS = 850.0
DEFAULT_HARDWARE_UNITS_PER_RADIAN = 1000.0


@dataclass(frozen=True)
class GripperMapping:
    """Mapping between real xArm units and simulation driver-angle radians."""

    raw_closed: float
    raw_open: float
    driver_max_position: float = DEFAULT_DRIVER_MAX_HARDWARE_UNITS
    driver_units_per_radian: float = DEFAULT_HARDWARE_UNITS_PER_RADIAN
    sim_joint_min_rad: float = 0.005
    sim_joint_max_rad: float = 0.85

    @classmethod
    def from_config(
        cls,
        config_or_mapping: Mapping[str, Any] | "GripperMapping",
    ) -> "GripperMapping":
        if isinstance(config_or_mapping, cls):
            return config_or_mapping
        mapping = config_or_mapping.get("gripper_mapping", config_or_mapping)
        return cls(
            raw_closed=float(mapping["raw_closed"]),
            raw_open=float(mapping["raw_open"]),
            driver_max_position=float(
                mapping.get("driver_max_position", DEFAULT_DRIVER_MAX_HARDWARE_UNITS)
            ),
            driver_units_per_radian=float(
                mapping.get(
                    "driver_units_per_radian", DEFAULT_HARDWARE_UNITS_PER_RADIAN
                )
            ),
            sim_joint_min_rad=float(mapping.get("sim_joint_min_rad", 0.005)),
            sim_joint_max_rad=float(mapping.get("sim_joint_max_rad", 0.85)),
        )

    def __post_init__(self) -> None:
        values = (
            self.raw_closed,
            self.raw_open,
            self.driver_max_position,
            self.driver_units_per_radian,
            self.sim_joint_min_rad,
            self.sim_joint_max_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Gripper mapping values must be finite")
        if self.raw_closed >= self.raw_open:
            raise ValueError("raw_closed must be less than raw_open")
        if self.driver_units_per_radian <= 0.0:
            raise ValueError("driver_units_per_radian must be positive")
        if self.sim_joint_min_rad >= self.sim_joint_max_rad:
            raise ValueError("sim_joint_min_rad must be less than sim_joint_max_rad")


def raw_hardware_to_driver_angle_rad(
    raw_hardware_units: float,
    config_or_mapping: Mapping[str, Any] | GripperMapping,
    *,
    operational_bounds: bool = True,
) -> float:
    """Convert real xArm position units to four-bar driver radians."""

    mapping = GripperMapping.from_config(config_or_mapping)
    low = mapping.raw_closed if operational_bounds else 0.0
    high = mapping.raw_open if operational_bounds else mapping.driver_max_position
    raw = float(np.clip(raw_hardware_units, low, high))
    return (mapping.driver_max_position - raw) / mapping.driver_units_per_radian


def driver_angle_rad_to_raw_hardware(
    driver_angle_rad: float,
    config_or_mapping: Mapping[str, Any] | GripperMapping,
    *,
    operational_bounds: bool = True,
) -> float:
    """Convert four-bar driver radians to real xArm position units."""

    mapping = GripperMapping.from_config(config_or_mapping)
    raw = (
        mapping.driver_max_position
        - mapping.driver_units_per_radian * float(driver_angle_rad)
    )
    low = mapping.raw_closed if operational_bounds else 0.0
    high = mapping.raw_open if operational_bounds else mapping.driver_max_position
    return float(np.clip(raw, low, high))


def raw_hardware_to_aperture_m(
    raw_hardware_units: float,
    config_or_mapping: Mapping[str, Any] | GripperMapping,
    *,
    operational_bounds: bool = True,
) -> float:
    """Return analytic inner fingertip aperture in meters."""

    angle = raw_hardware_to_driver_angle_rad(
        raw_hardware_units,
        config_or_mapping,
        operational_bounds=operational_bounds,
    )
    finger_origin_y = (
        OUTER_JOINT_Y_M
        + math.cos(angle) * FINGER_JOINT_Y_M
        - math.sin(angle) * FINGER_JOINT_Z_M
    )
    return 2.0 * (finger_origin_y - FINGER_INNER_SURFACE_OFFSET_M)


def raw_hardware_to_actuator_ctrl_rad(
    raw_hardware_units: float,
    config_or_mapping: Mapping[str, Any] | GripperMapping,
    *,
    operational_bounds: bool = True,
) -> float:
    """Convert real xArm units to canonical direct-angle actuator control."""

    mapping = GripperMapping.from_config(config_or_mapping)
    angle = raw_hardware_to_driver_angle_rad(
        raw_hardware_units,
        mapping,
        operational_bounds=operational_bounds,
    )
    return float(np.clip(angle, mapping.sim_joint_min_rad, mapping.sim_joint_max_rad))


def actuator_ctrl_rad_to_raw_hardware(
    actuator_ctrl_rad: float,
    config_or_mapping: Mapping[str, Any] | GripperMapping,
    *,
    operational_bounds: bool = True,
) -> float:
    """Invert canonical direct-angle actuator control to real xArm units."""

    mapping = GripperMapping.from_config(config_or_mapping)
    angle = float(
        np.clip(
            actuator_ctrl_rad,
            mapping.sim_joint_min_rad,
            mapping.sim_joint_max_rad,
        )
    )
    return driver_angle_rad_to_raw_hardware(
        angle,
        mapping,
        operational_bounds=operational_bounds,
    )
