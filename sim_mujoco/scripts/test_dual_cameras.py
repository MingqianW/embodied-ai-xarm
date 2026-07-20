from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "sim_mujoco"
    / "assets"
    / "xarm6"
    / "xarm6_pick_scene.xml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "sim_mujoco"
    / "outputs"
)

CAMERA_NAMES = [
    "base_camera",
    "wrist_camera",
]


def render_camera(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_name: str,
) -> np.ndarray:
    renderer.update_scene(
        data,
        camera=camera_name,
    )

    image = renderer.render()

    return np.asarray(
        image,
        dtype=np.uint8,
    ).copy()


def main() -> None:
    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )
    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    if home_id < 0:
        raise RuntimeError("Home keyframe not found.")

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_id,
    )

    # Hold the arm at home while the object settles.
    for _ in range(500):
        mujoco.mj_step(model, data)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    renderer = mujoco.Renderer(
        model,
        height=224,
        width=224,
    )

    try:
        for camera_name in CAMERA_NAMES:
            image = render_camera(
                renderer,
                data,
                camera_name,
            )

            output_path = (
                OUTPUT_DIR
                / f"{camera_name}.png"
            )

            Image.fromarray(image).save(
                output_path
            )

            print(
                camera_name,
                image.shape,
                image.dtype,
                "->",
                output_path,
            )
    finally:
        renderer.close()


if __name__ == "__main__":
    main()