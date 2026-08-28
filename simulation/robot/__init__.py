"""xArm6 model, mapping, gripper, and control primitives."""

from simulation.robot.gripper import actuator_ctrl_from_raw_hardware
from simulation.robot.gripper import measure_fingertip_aperture_m
from simulation.robot.gripper import read_raw_gripper_position
from simulation.robot.gripper import set_raw_gripper_configuration
from simulation.robot.gripper_mapping import GripperMapping
from simulation.robot.gripper_mapping import actuator_ctrl_rad_to_raw_hardware
from simulation.robot.gripper_mapping import driver_angle_rad_to_raw_hardware
from simulation.robot.gripper_mapping import raw_hardware_to_actuator_ctrl_rad
from simulation.robot.gripper_mapping import raw_hardware_to_driver_angle_rad

__all__ = [
    "GripperMapping",
    "actuator_ctrl_from_raw_hardware",
    "actuator_ctrl_rad_to_raw_hardware",
    "driver_angle_rad_to_raw_hardware",
    "measure_fingertip_aperture_m",
    "raw_hardware_to_actuator_ctrl_rad",
    "raw_hardware_to_driver_angle_rad",
    "read_raw_gripper_position",
    "set_raw_gripper_configuration",
]
