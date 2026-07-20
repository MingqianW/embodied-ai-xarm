from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from openpi_client import image_tools


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "sim_mujoco"
    / "assets"
    / "xarm6"
    / "xarm6_pick_scene.xml"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "sim_mujoco"
    / "outputs"
    / "local_observation.npz"
)


GRIPPER_RAW_CLOSED = 50.0
GRIPPER_RAW_OPEN = 845.0
SIM_HALF_WIDTH_OPEN = 0.040


def joint_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
) -> float:
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id < 0:
        raise RuntimeError(
            f"Joint not found: {joint_name}"
        )

    qpos_address = model.jnt_qposadr[joint_id]

    return float(data.qpos[qpos_address])


def simulated_gripper_to_raw(
    half_width: float,
) -> float:
    normalized = np.clip(
        half_width / SIM_HALF_WIDTH_OPEN,
        0.0,
        1.0,
    )

    return float(
        GRIPPER_RAW_CLOSED
        + normalized
        * (
            GRIPPER_RAW_OPEN
            - GRIPPER_RAW_CLOSED
        )
    )


def render(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_name: str,
) -> np.ndarray:
    renderer.update_scene(
        data,
        camera=camera_name,
    )

    image = np.asarray(
        renderer.render(),
        dtype=np.uint8,
    ).copy()

    return image_tools.resize_with_pad(
        image,
        224,
        224,
    )


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

    for _ in range(500):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(
        model,
        height=224,
        width=224,
    )

    try:
        base_image = render(
            renderer,
            data,
            "base_camera",
        )

        wrist_image = render(
            renderer,
            data,
            "wrist_camera",
        )
    finally:
        renderer.close()

    arm_state = np.array(
        [
            joint_qpos(model, data, "joint1"),
            joint_qpos(model, data, "joint2"),
            joint_qpos(model, data, "joint3"),
            joint_qpos(model, data, "joint4"),
            joint_qpos(model, data, "joint5"),
            joint_qpos(model, data, "joint6"),
        ],
        dtype=np.float32,
    )

    gripper_half_width = joint_qpos(
        model,
        data,
        "left_finger_slide",
    )

    gripper_raw = simulated_gripper_to_raw(
        gripper_half_width
    )

    state = np.concatenate(
        [
            arm_state,
            np.array(
                [gripper_raw],
                dtype=np.float32,
            ),
        ]
    )

    observation = {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": "pick up the red cube",
    }

    print("\nObservation validation")
    print("----------------------")

    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            print(
                key,
                "shape=",
                value.shape,
                "dtype=",
                value.dtype,
            )
        else:
            print(key, "=", value)

    print("\nState:")
    print(state)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_PATH,
        base_image=base_image,
        wrist_image=wrist_image,
        state=state,
        prompt=np.array(
            observation["prompt"]
        ),
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()