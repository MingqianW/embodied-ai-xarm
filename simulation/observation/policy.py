"""Adapt simulated RGB/state observations to the OpenPI policy contract."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from policy_runtime.image_preprocessing import ImagePreprocessingConfig
from policy_runtime.image_preprocessing import image_diagnostics
from policy_runtime.image_preprocessing import preprocess_policy_image
from policy_runtime.observation_builder import validate_policy_observation
from simulation.observation.cameras import render_rgb
from simulation.observation.state import get_robot_state
from simulation.robot.model import BASE_CAMERA_NAME
from simulation.robot.model import WRIST_CAMERA_NAME


DEFAULT_PROMPT = "pick up the object"


def policy_image(native_rgb: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    render = config["render"]
    return preprocess_policy_image(
        native_rgb,
        ImagePreprocessingConfig(
            height=int(render["policy_height"]),
            width=int(render["policy_width"]),
            input_color_order="RGB",
        ),
    )


def render_policy_images(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    base = render_rgb(renderer, data, BASE_CAMERA_NAME)
    wrist = render_rgb(renderer, data, WRIST_CAMERA_NAME)
    return policy_image(base, config), policy_image(wrist, config)


def build_policy_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    config: dict[str, Any],
    prompt: str = DEFAULT_PROMPT,
) -> dict[str, Any]:
    """Build the exact four-key OpenPI observation payload."""

    base_image, wrist_image = render_policy_images(renderer, data, config)
    observation = {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/state": get_robot_state(model, data, config),
        "prompt": str(prompt),
    }
    validate_policy_observation(observation)
    return observation


__all__ = [
    "DEFAULT_PROMPT",
    "build_policy_observation",
    "image_diagnostics",
    "policy_image",
    "render_policy_images",
]
