from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from simulation.configuration import load_simulation_config
from simulation.observation.cameras import body_camera_visibility
from simulation.robot.gripper import actuator_ctrl_from_raw_hardware
from simulation.robot.gripper import has_xarm_four_bar_gripper
from simulation.robot.gripper import set_raw_gripper_configuration
from simulation.robot.joint_mapping import raw_arm_state_to_mujoco_qpos
from simulation.robot.legacy_gripper import legacy_slide_m_to_raw_hardware
from simulation.robot.legacy_gripper import raw_hardware_to_legacy_slide_m
from simulation.robot.legacy_gripper import set_legacy_slide_configuration
from simulation.robot.model import ARM_JOINT_NAMES
from simulation.robot.model import arm_joint_limits
from simulation.scene.objects import _body_id
from simulation.scene.objects import _freejoint_dof_address
from simulation.scene.objects import _freejoint_qpos_address
from simulation.scene.objects import _set_body_enabled
from simulation.scene.objects import _yaw_quaternion
from simulation.scene.runtime import TaskSceneRuntime
from simulation.scene.tasks import TASK_CONFIG_PATH
from simulation.scene.tasks import load_task_scene_config
from simulation.scene.tasks import resolve_task


def configure_task_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    task: str,
    seed: int | None,
    object_xy_range: float,
    object_yaw_range_deg: float,
    joint_noise: float,
    scene_variant: str = "clean",
    settle_steps: int = 500,
    config_path: Path = TASK_CONFIG_PATH,
    gripper_config: dict[str, Any] | None = None,
) -> tuple[TaskSceneRuntime, dict[str, Any]]:
    config = load_task_scene_config(config_path)
    effective_gripper_config = (
        load_simulation_config() if gripper_config is None else gripper_config
    )
    task_name, spec = resolve_task(task, config_path)
    catalog = config["catalog"]
    runtime = TaskSceneRuntime(model, data, task_name, spec, catalog)
    rng = np.random.default_rng(seed)
    if scene_variant not in {"clean", "distractors"}:
        raise ValueError(
            f"scene_variant must be 'clean' or 'distractors', got {scene_variant!r}"
        )

    free_bodies = tuple(catalog.get("free_bodies") or ())
    fixed_bodies = tuple(catalog.get("fixed_bodies") or ())
    poses = catalog.get("poses") or {}
    active_bodies = set(spec.get("active_bodies") or ())
    distractor_bodies = (
        tuple(spec.get("distractor_bodies") or ())
        if scene_variant == "distractors"
        else ()
    )
    active_bodies.update(distractor_bodies)

    for body_name in free_bodies:
        if body_name not in poses:
            raise ValueError(f"Missing catalog pose for {body_name}")
        qpos_addr = _freejoint_qpos_address(model, body_name)
        data.qpos[qpos_addr : qpos_addr + 3] = np.asarray(
            poses[body_name],
            dtype=np.float64,
        )
        data.qpos[qpos_addr + 3 : qpos_addr + 7] = _yaw_quaternion(0.0)
    shuffle_bodies = tuple(spec.get("shuffle_bodies") or ())
    if seed is not None and shuffle_bodies:
        slots = [
            np.asarray(
                data.qpos[
                    _freejoint_qpos_address(model, body_name) : _freejoint_qpos_address(
                        model, body_name
                    )
                    + 2
                ],
                dtype=np.float64,
            ).copy()
            for body_name in shuffle_bodies
        ]
        for body_name, slot_index in zip(shuffle_bodies, rng.permutation(len(slots))):
            qpos_addr = _freejoint_qpos_address(model, body_name)
            data.qpos[qpos_addr : qpos_addr + 2] = slots[int(slot_index)]
    distractor_slots = tuple(catalog.get("distractor_slots") or ())
    if len(distractor_slots) < len(distractor_bodies):
        raise ValueError(
            f"Need at least {len(distractor_bodies)} distractor slots, "
            f"found {len(distractor_slots)}"
        )
    if distractor_bodies:
        slot_order = rng.permutation(len(distractor_slots))
        for body_name, slot_index in zip(distractor_bodies, slot_order):
            if body_name not in free_bodies:
                raise ValueError(
                    f"Distractor body must be a catalog free body: {body_name}"
                )
            qpos_addr = _freejoint_qpos_address(model, body_name)
            slot_xy = np.asarray(
                distractor_slots[int(slot_index)],
                dtype=np.float64,
            )
            if slot_xy.shape != (2,):
                raise ValueError(f"Distractor slot must have two values, got {slot_xy}")
            data.qpos[qpos_addr : qpos_addr + 2] = slot_xy
    for body_name in fixed_bodies:
        if body_name not in poses:
            raise ValueError(f"Missing catalog pose for {body_name}")
        model.body_pos[_body_id(model, body_name)] = np.asarray(
            poses[body_name],
            dtype=np.float64,
        )

    for body_name in (*free_bodies, *fixed_bodies):
        _set_body_enabled(
            model,
            body_name,
            body_name in active_bodies,
        )
    for body_name in free_bodies:
        if body_name not in active_bodies:
            qpos_addr = _freejoint_qpos_address(model, body_name)
            data.qpos[qpos_addr + 2] = -1.0

    if "initial_arm_qpos" in spec:
        initial_arm_raw = np.asarray(spec["initial_arm_qpos"], dtype=np.float64)
        if initial_arm_raw.shape != (6,):
            raise ValueError(f"{task_name}.initial_arm_qpos must contain 6 values")
        initial_arm = raw_arm_state_to_mujoco_qpos(initial_arm_raw)
        for index, joint_name in enumerate(ARM_JOINT_NAMES):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            data.qpos[int(model.jnt_qposadr[joint_id])] = initial_arm[index]
            data.ctrl[index] = initial_arm[index]
        gripper_raw = float(spec.get("initial_gripper_raw", 845.0))
        if (
            not has_xarm_four_bar_gripper(model)
            and spec["success"]["type"] == "place_in_ring"
        ):
            slide = float(spec.get("initial_gripper_sim_half_width", 0.012))
            gripper_raw = legacy_slide_m_to_raw_hardware(
                slide, effective_gripper_config
            )
        if has_xarm_four_bar_gripper(model):
            set_raw_gripper_configuration(
                model, data, gripper_raw, effective_gripper_config
            )
            data.ctrl[6] = actuator_ctrl_from_raw_hardware(
                gripper_raw, effective_gripper_config
            )
        else:
            set_legacy_slide_configuration(
                model, data, gripper_raw, effective_gripper_config
            )
            data.ctrl[6] = raw_hardware_to_legacy_slide_m(
                gripper_raw, effective_gripper_config
            )

    initial_tcp_to_object = spec.get("initial_tcp_to_object")
    scene_delta = np.zeros(2, dtype=np.float64)
    if seed is not None:
        scene_delta = rng.uniform(
            -float(object_xy_range), float(object_xy_range), size=2
        )
    yaw_values: dict[str, float] = {}
    for body_name in spec.get("randomizable_bodies") or ():
        yaw = 0.0
        if seed is not None:
            yaw = math.radians(
                float(
                    rng.uniform(
                        -float(object_yaw_range_deg), float(object_yaw_range_deg)
                    )
                )
            )
        yaw_values[body_name] = yaw
        if body_name in free_bodies:
            qpos_addr = _freejoint_qpos_address(model, body_name)
            data.qpos[qpos_addr : qpos_addr + 2] += scene_delta
            data.qpos[qpos_addr + 3 : qpos_addr + 7] = _yaw_quaternion(yaw)
        else:
            body_id = _body_id(model, body_name)
            model.body_pos[body_id, :2] += scene_delta

    limits = arm_joint_limits(model)
    joint_values = []
    for index, joint_name in enumerate(ARM_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_addr = int(model.jnt_qposadr[joint_id])
        noisy = float(data.qpos[qpos_addr])
        if seed is not None:
            noisy += float(rng.normal(0.0, float(joint_noise)))
        clamped = float(np.clip(noisy, limits[index, 0], limits[index, 1]))
        data.qpos[qpos_addr] = clamped
        data.ctrl[index] = clamped
        joint_values.append(clamped)

    mujoco.mj_forward(model, data)
    # Apply the one-time free-body initialization only after the randomized arm
    # pose is final.  This implements T_world_object = T_world_tcp @
    # T_tcp_object and avoids arm noise introducing an unrecorded pose jump.
    if initial_tcp_to_object is not None:
        if task_name != "place_red_pepper_in_ring":
            raise ValueError("initial_tcp_to_object is supported only for Place")
        if spec.get("object_identity") != runtime.target_body:
            raise ValueError("Place object identity must equal target_body")
        translation = np.asarray(
            initial_tcp_to_object.get("translation_m"), dtype=np.float64
        )
        quaternion = np.asarray(
            initial_tcp_to_object.get("quaternion_wxyz"), dtype=np.float64
        )
        if translation.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("Invalid initial TCP-to-object transform")
        quaternion_norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(quaternion_norm) or quaternion_norm <= 0.0:
            raise ValueError(
                "Initial TCP-to-object quaternion must be finite and nonzero"
            )
        quaternion = quaternion / quaternion_norm
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "tool_center_point"
        )
        tcp_position = np.asarray(data.site_xpos[site_id], dtype=np.float64)
        tcp_rotation = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(
            3, 3
        )
        relative_rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(relative_rotation, quaternion)
        world_rotation = tcp_rotation @ relative_rotation.reshape(3, 3)
        world_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(world_quaternion, world_rotation.reshape(-1))
        qpos_addr = _freejoint_qpos_address(model, runtime.target_body)
        dof_addr = _freejoint_dof_address(model, runtime.target_body)
        data.qpos[qpos_addr : qpos_addr + 3] = tcp_position + tcp_rotation @ translation
        data.qpos[qpos_addr + 3 : qpos_addr + 7] = world_quaternion
        data.qvel[dof_addr : dof_addr + 6] = 0.0
        mujoco.mj_forward(model, data)

    for _ in range(int(settle_steps)):
        mujoco.mj_step(model, data)

    wrist_visibility = {
        body_name: body_camera_visibility(
            model, data, body_name, effective_gripper_config["render"]
        )
        for body_name in spec.get("wrist_visible_bodies") or (runtime.target_body,)
    }
    hidden_bodies = [
        body_name
        for body_name, visibility in wrist_visibility.items()
        if not visibility["visible"]
    ]
    if hidden_bodies:
        hidden = ", ".join(hidden_bodies)
        raise RuntimeError(
            f"{task_name} initial wrist view does not contain required bodies: {hidden}"
        )

    runtime.initial_target_z = float(runtime._target_pose()[2])
    active_positions = {
        body_name: np.asarray(
            data.xpos[_body_id(model, body_name)], dtype=np.float64
        ).tolist()
        for body_name in active_bodies
    }
    initial_conditions = {
        "task": task_name,
        "prompt": runtime.prompt,
        "seed": seed,
        "active_bodies": sorted(active_bodies),
        "scene_variant": scene_variant,
        "distractor_bodies": list(distractor_bodies),
        "target_body": runtime.target_body,
        "scene_xy_delta": scene_delta.tolist(),
        "object_yaws": yaw_values,
        "initial_body_positions": active_positions,
        "initial_joint_positions": joint_values,
        "wrist_visibility": wrist_visibility,
        "initial_target_z": runtime.initial_target_z,
        "object_identity": spec.get("object_identity", runtime.target_body),
        "initial_tcp_to_object": initial_tcp_to_object,
    }
    target_position = runtime._target_pose()
    initial_conditions.update(
        {
            "initial_object_x": float(target_position[0]),
            "initial_object_y": float(target_position[1]),
            "initial_object_z": float(target_position[2]),
            "initial_object_yaw": float(yaw_values.get(runtime.target_body, 0.0)),
        }
    )
    return runtime, initial_conditions
