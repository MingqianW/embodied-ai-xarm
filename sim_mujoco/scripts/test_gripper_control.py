from __future__ import annotations

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
    / "xarm6_pick_scene.xml"
)

HOME_ARM = np.array(
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

OPEN_HALF_WIDTH = 0.040
CLOSED_HALF_WIDTH = 0.000


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
    mujoco.mj_forward(model, data)

    print("Actuators:")

    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )
        print(f"  {actuator_id}: {name}")

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:
        while viewer.is_running():
            phase = data.time % 6.0

            if phase < 3.0:
                gripper_target = OPEN_HALF_WIDTH
            else:
                gripper_target = CLOSED_HALF_WIDTH

            data.ctrl[:6] = HOME_ARM
            data.ctrl[6] = gripper_target

            steps_per_update = max(
                1,
                int(0.02 / model.opt.timestep),
            )

            for _ in range(steps_per_update):
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()