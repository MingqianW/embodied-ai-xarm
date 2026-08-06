from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from sim_mujoco.paths import (
    active_model_path,
    camera_config_path,
    repository_root,
    simulation_root,
)

from sim_mujoco.joint_mapping import mujoco_qpos_to_raw_arm_state
from policy_runtime.image_preprocessing import (
    ImagePreprocessingConfig,
    image_diagnostics,
    preprocess_policy_image,
)
from policy_runtime.observation_builder import validate_policy_observation
from sim_mujoco.gripper_mapping import (
    raw_gripper_to_sim_slide,
    sim_slide_to_raw_gripper,
)

PROJECT_ROOT = repository_root()
SIM_ROOT = simulation_root()
DEFAULT_MODEL_PATH = active_model_path()
DEFAULT_CAMERA_CONFIG_PATH = camera_config_path()

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
GRIPPER_LEFT_JOINT = "left_finger_slide"
GRIPPER_RIGHT_JOINT = "right_finger_slide"
BASE_CAMERA = "base_camera"
WRIST_CAMERA = "wrist_camera"
DEFAULT_PROMPT = "pick up the object"


@dataclass
class SimulationContext:
    model: mujoco.MjModel
    data: mujoco.MjData
    renderer: mujoco.Renderer
    config: dict[str, Any]
    model_path: Path
    camera_config_path: Path

    def close(self) -> None:
        close = getattr(self.renderer, "close", None)
        if close is not None:
            close()


def load_camera_config(path: Path = DEFAULT_CAMERA_CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Camera calibration config is not a mapping: {path}")
    return value


def camera_axes(position: list[float], target: list[float], roll_deg: float = 0.0) -> np.ndarray:
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
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=np.float64).reshape(-1))
    return quat


def set_camera_parameters(model: mujoco.MjModel, camera_name: str, parameters: dict[str, Any]) -> None:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise RuntimeError(f"Camera not found: {camera_name}")
    model.cam_pos[camera_id] = np.asarray(parameters["position"], dtype=np.float64)
    rotation = camera_axes(
        parameters["position"],
        parameters["target"],
        float(parameters.get("roll_deg", 0.0)),
    )
    model.cam_quat[camera_id] = matrix_to_quaternion(rotation)
    model.cam_fovy[camera_id] = float(parameters["fovy_deg"])


def apply_camera_calibration(model: mujoco.MjModel, config: dict[str, Any]) -> None:
    for camera_name in (BASE_CAMERA, WRIST_CAMERA):
        if camera_name not in config:
            raise KeyError(f"Missing {camera_name} in camera calibration config")
        set_camera_parameters(model, camera_name, config[camera_name])


def load_simulation(
    model_path: Path = DEFAULT_MODEL_PATH,
    camera_config_path: Path = DEFAULT_CAMERA_CONFIG_PATH,
) -> SimulationContext:
    config = load_camera_config(camera_config_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    apply_camera_calibration(model, config)
    data = mujoco.MjData(model)
    render_config = config["render"]
    renderer = mujoco.Renderer(
        model,
        height=int(render_config["native_height"]),
        width=int(render_config["native_width"]),
    )
    return SimulationContext(
        model=model,
        data=data,
        renderer=renderer,
        config=config,
        model_path=model_path,
        camera_config_path=camera_config_path,
    )


def initialize_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    keyframe_name: str = "home",
    settle_steps: int = 500,
) -> None:
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    if keyframe_id < 0:
        raise RuntimeError(f"Keyframe not found: {keyframe_name}")
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)


def joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"Joint not found: {joint_name}")
    return float(data.qpos[model.jnt_qposadr[joint_id]])


def gripper_sim_to_raw(half_width: float, config: dict[str, Any]) -> float:
    return sim_slide_to_raw_gripper(half_width, config)


def gripper_raw_to_sim(raw_value: float, config: dict[str, Any]) -> float:
    return raw_gripper_to_sim_slide(raw_value, config)


def get_robot_state(model: mujoco.MjModel, data: mujoco.MjData, config: dict[str, Any]) -> np.ndarray:
    mujoco_arm_qpos = np.asarray(
        [joint_qpos(model, data, joint_name) for joint_name in ARM_JOINT_NAMES],
        dtype=np.float64,
    )
    arm_state = mujoco_qpos_to_raw_arm_state(mujoco_arm_qpos).astype(np.float32)
    gripper_raw = gripper_sim_to_raw(joint_qpos(model, data, GRIPPER_LEFT_JOINT), config)
    return np.concatenate([arm_state, np.asarray([gripper_raw], dtype=np.float32)]).astype(np.float32)


def render_native_rgb(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_name: str,
) -> np.ndarray:
    renderer.update_scene(data, camera=camera_name)
    return np.asarray(renderer.render(), dtype=np.uint8).copy()


def policy_image(native_rgb: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    render_config = config["render"]
    return preprocess_policy_image(
        native_rgb,
        ImagePreprocessingConfig(
            height=int(render_config["policy_height"]),
            width=int(render_config["policy_width"]),
            input_color_order="RGB",
        ),
    )


def render_model_inputs(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    base_native = render_native_rgb(renderer, data, BASE_CAMERA)
    wrist_native = render_native_rgb(renderer, data, WRIST_CAMERA)
    return policy_image(base_native, config), policy_image(wrist_native, config)


def build_openpi_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    config: dict[str, Any],
    prompt: str = DEFAULT_PROMPT,
) -> dict[str, Any]:
    base_image, wrist_image = render_model_inputs(renderer, data, config)
    state = get_robot_state(model, data, config)
    observation = {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": str(prompt),
    }
    validate_openpi_observation(observation)
    return observation


def validate_openpi_observation(observation: dict[str, Any]) -> None:
    validate_policy_observation(observation)


def arm_joint_limits(model: mujoco.MjModel) -> np.ndarray:
    limits = []
    for joint_name in ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"Joint not found: {joint_name}")
        limits.append(model.jnt_range[joint_id])
    return np.asarray(limits, dtype=np.float32)


def arm_actuator_ctrl_limits(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(model.actuator_ctrlrange[:6], dtype=np.float32)


def gripper_actuator_ctrl_limits(model: mujoco.MjModel) -> tuple[float, float]:
    low, high = model.actuator_ctrlrange[6]
    return float(low), float(high)
