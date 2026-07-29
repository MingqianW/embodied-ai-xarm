from __future__ import annotations

from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "sim_mujoco" / "scenes" / "minimal.xml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_mujoco.paths import mujoco_output_root


OUTPUT_PATH = mujoco_output_root() / "test_camera.png"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Simulate one second.
    number_of_steps = int(1.0 / model.opt.timestep)
    for _ in range(number_of_steps):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(
        model,
        height=480,
        width=640,
    )

    try:
        renderer.update_scene(
            data,
            camera="test_camera",
        )
        image_rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()
    finally:
        renderer.close()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(
        str(OUTPUT_PATH),
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
    )

    if not success:
        raise RuntimeError(f"Failed to save rendered image: {OUTPUT_PATH}")

    print("Model:", MODEL_PATH)
    print("Simulation timestep:", model.opt.timestep)
    print("Rendered image shape:", image_rgb.shape)
    print("Saved image:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
