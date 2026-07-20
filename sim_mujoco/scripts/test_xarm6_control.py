from __future__ import annotations

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "sim_mujoco"
    / "assets"
    / "xarm6"
    / "xarm6_arm.xml"
)

HOME_QPOS = np.array(
    [
        0.0,
        -0.6,
        -1.2,
        0.0,
        1.8,
        0.0,
    ],
    dtype=np.float64,
)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    keyframe_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    if keyframe_id < 0:
        raise RuntimeError("Home keyframe was not found.")

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        keyframe_id,
    )

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            target = HOME_QPOS.copy()

            # Slowly oscillate joint 1 by ±0.3 rad.
            target[0] += 0.3 * math.sin(0.5 * data.time)

            data.ctrl[:] = target

            # Advance 20 ms of simulation per viewer update.
            number_of_steps = max(
                1,
                int(0.02 / model.opt.timestep),
            )

            for _ in range(number_of_steps):
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()