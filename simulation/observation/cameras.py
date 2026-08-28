"""Camera calibration, geometry, and RGB rendering for MuJoCo."""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from simulation.robot.model import BASE_CAMERA_NAME
from simulation.robot.model import WRIST_CAMERA_NAME
from simulation.robot.model import body_id
from simulation.robot.model import camera_id


def camera_axes(
    position: list[float],
    target: list[float],
    roll_deg: float = 0.0,
) -> np.ndarray:
    position_array = np.asarray(position, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    forward = target_array - position_array
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        raise ValueError("Camera position and target cannot be identical")
    forward /= norm
    camera_z = -forward
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(up, camera_z))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    camera_x = np.cross(up, camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    roll = math.radians(float(roll_deg))
    rolled_x = math.cos(roll) * camera_x + math.sin(roll) * camera_y
    rolled_y = -math.sin(roll) * camera_x + math.cos(roll) * camera_y
    return np.column_stack((rolled_x, rolled_y, camera_z))


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(
        quaternion,
        np.asarray(matrix, dtype=np.float64).reshape(-1),
    )
    return quaternion


def set_camera_parameters(
    model: mujoco.MjModel,
    name: str,
    parameters: dict[str, Any],
) -> None:
    identifier = camera_id(model, name)
    model.cam_pos[identifier] = np.asarray(parameters["position"], dtype=np.float64)
    rotation = camera_axes(
        parameters["position"],
        parameters["target"],
        float(parameters.get("roll_deg", 0.0)),
    )
    model.cam_quat[identifier] = matrix_to_quaternion(rotation)
    model.cam_fovy[identifier] = float(parameters["fovy_deg"])


def apply_camera_calibration(
    model: mujoco.MjModel,
    config: dict[str, Any],
) -> None:
    for name in (BASE_CAMERA_NAME, WRIST_CAMERA_NAME):
        if name not in config:
            raise KeyError(f"Missing {name} in camera calibration config")
        set_camera_parameters(model, name, config[name])


def render_rgb(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_name: str,
) -> np.ndarray:
    renderer.update_scene(data, camera=camera_name)
    return np.asarray(renderer.render(), dtype=np.uint8).copy()


def body_camera_visibility(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    render_config: dict[str, Any],
    *,
    camera_name: str = WRIST_CAMERA_NAME,
    margin: float = 0.9,
) -> dict[str, Any]:
    camera_identifier = camera_id(model, camera_name)
    body_identifier = body_id(model, body_name)
    relative = np.asarray(
        data.xpos[body_identifier] - data.cam_xpos[camera_identifier]
    )
    rotation = np.asarray(data.cam_xmat[camera_identifier]).reshape(3, 3)
    camera_position = rotation.T @ relative
    depth = float(-camera_position[2])
    aspect = float(render_config["native_width"]) / float(
        render_config["native_height"]
    )
    half_height = depth * math.tan(
        math.radians(float(model.cam_fovy[camera_identifier])) / 2.0
    )
    half_width = aspect * half_height
    normalized_x = (
        float(camera_position[0] / half_width) if half_width > 0.0 else math.inf
    )
    normalized_y = (
        float(camera_position[1] / half_height) if half_height > 0.0 else math.inf
    )
    visible = bool(
        depth > 0.0
        and abs(normalized_x) <= float(margin)
        and abs(normalized_y) <= float(margin)
    )
    return {
        "body": body_name,
        "camera": camera_name,
        "visible": visible,
        "normalized_xy": [normalized_x, normalized_y],
        "depth_m": depth,
        "margin": float(margin),
    }
