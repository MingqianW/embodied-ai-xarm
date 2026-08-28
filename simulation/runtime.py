"""Construction and lifetime of a compiled MuJoCo simulation context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco

from simulation.configuration import load_simulation_config
from simulation.observation.cameras import apply_camera_calibration
from simulation.resources import camera_config_path as default_camera_config_path
from simulation.resources import gripper_config_path as default_gripper_config_path
from simulation.resources import model_path as default_model_path


@dataclass
class SimulationContext:
    model: mujoco.MjModel
    data: mujoco.MjData
    renderer: mujoco.Renderer
    config: dict[str, Any]
    model_path: Path
    camera_config_path: Path
    gripper_config_path: Path

    def close(self) -> None:
        close = getattr(self.renderer, "close", None)
        if close is not None:
            close()


def load_simulation(
    model_path: Path | None = None,
    camera_config_path: Path | None = None,
    gripper_config_path: Path | None = None,
) -> SimulationContext:
    resolved_model = (
        default_model_path() if model_path is None else Path(model_path).resolve()
    )
    resolved_camera = (
        default_camera_config_path()
        if camera_config_path is None
        else Path(camera_config_path).resolve()
    )
    resolved_gripper = (
        default_gripper_config_path()
        if gripper_config_path is None
        else Path(gripper_config_path).resolve()
    )
    config = load_simulation_config(resolved_camera, resolved_gripper)
    model = mujoco.MjModel.from_xml_path(str(resolved_model))
    apply_camera_calibration(model, config)
    data = mujoco.MjData(model)
    render = config["render"]
    renderer = mujoco.Renderer(
        model,
        height=int(render["native_height"]),
        width=int(render["native_width"]),
    )
    return SimulationContext(
        model=model,
        data=data,
        renderer=renderer,
        config=config,
        model_path=resolved_model,
        camera_config_path=resolved_camera,
        gripper_config_path=resolved_gripper,
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
