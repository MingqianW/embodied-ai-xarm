from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from sim_mujoco.joint_mapping import raw_arm_state_to_mujoco_qpos
from sim_mujoco.paths import task_config_path
from sim_mujoco.remote_policy_observation import (
    ARM_JOINT_NAMES,
    DEFAULT_CAMERA_CONFIG_PATH,
    arm_joint_limits,
    gripper_raw_to_sim,
)


TASK_CONFIG_PATH = task_config_path()
TABLE_TOP_Z = 0.05
WRIST_CAMERA_NAME = "wrist_camera"
WRIST_VISIBILITY_MARGIN = 0.9


def _normalized_name(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def load_task_scene_config(path: Path = TASK_CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("tasks"), dict):
        raise ValueError(f"Invalid task scene config: {path}")
    return config


def task_names(path: Path = TASK_CONFIG_PATH) -> tuple[str, ...]:
    return tuple(load_task_scene_config(path)["tasks"])


def resolve_task(task: str, path: Path = TASK_CONFIG_PATH) -> tuple[str, dict[str, Any]]:
    config = load_task_scene_config(path)
    requested = _normalized_name(task)
    for task_name, spec in config["tasks"].items():
        candidates = [task_name, spec.get("prompt", ""), *(spec.get("aliases") or [])]
        if requested in {_normalized_name(candidate) for candidate in candidates}:
            resolved = dict(spec)
            resolved["name"] = task_name
            return task_name, resolved
    available = ", ".join(config["tasks"])
    raise ValueError(f"Unknown MuJoCo task {task!r}. Available tasks: {available}")


def _body_id(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Task scene body not found: {body_name}")
    return int(body_id)


def _freejoint_id(model: mujoco.MjModel, body_name: str) -> int:
    body_id = _body_id(model, body_name)
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise RuntimeError(f"Task scene body does not have a freejoint: {body_name}")
    return joint_id


def _freejoint_qpos_address(model: mujoco.MjModel, body_name: str) -> int:
    return int(model.jnt_qposadr[_freejoint_id(model, body_name)])


def _freejoint_dof_address(model: mujoco.MjModel, body_name: str) -> int:
    return int(model.jnt_dofadr[_freejoint_id(model, body_name)])


def _set_body_enabled(model: mujoco.MjModel, body_name: str, enabled: bool) -> None:
    body_id = _body_id(model, body_name)
    geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
    alpha = 1.0 if enabled else 0.0
    for geom_id in geom_ids:
        material_id = int(model.geom_matid[geom_id])
        if enabled and material_id >= 0:
            model.geom_rgba[geom_id] = model.mat_rgba[material_id]
        model.geom_rgba[geom_id, 3] = alpha
        model.geom_contype[geom_id] = 1 if enabled else 0
        model.geom_conaffinity[geom_id] = 1 if enabled else 0


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.asarray(
        [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
        dtype=np.float64,
    )


def body_camera_visibility(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    *,
    camera_name: str = WRIST_CAMERA_NAME,
    margin: float = WRIST_VISIBILITY_MARGIN,
) -> dict[str, Any]:
    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name,
    )
    if camera_id < 0:
        raise RuntimeError(f"Camera not found: {camera_name}")

    body_id = _body_id(model, body_name)
    relative = np.asarray(data.xpos[body_id] - data.cam_xpos[camera_id])
    rotation = np.asarray(data.cam_xmat[camera_id]).reshape(3, 3)
    camera_position = rotation.T @ relative
    depth = float(-camera_position[2])

    camera_config = load_gripper_config()
    render_config = camera_config["render"]
    aspect = float(render_config["native_width"]) / float(
        render_config["native_height"]
    )
    half_height = depth * math.tan(
        math.radians(float(model.cam_fovy[camera_id])) / 2.0
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


@dataclass
class TaskSceneRuntime:
    model: mujoco.MjModel
    data: mujoco.MjData
    task_name: str
    spec: dict[str, Any]
    catalog: dict[str, Any]
    released: bool = False
    success_streak: int = 0
    initial_target_z: float = 0.0

    @property
    def prompt(self) -> str:
        return str(self.spec["prompt"])

    @property
    def target_body(self) -> str:
        return str(self.spec["target_body"])

    def _target_pose(self) -> np.ndarray:
        body_name = self.target_body
        if self.spec["success"]["type"] == "place_in_ring" and not self.released:
            body_name = "held_red_pepper"
        return np.asarray(self.data.xpos[_body_id(self.model, body_name)], dtype=np.float64)

    def adjust_observation(self, observation: dict[str, Any]) -> None:
        if (
            self.spec["success"]["type"] == "place_in_ring"
            and not self.released
            and "initial_gripper_raw" in self.spec
        ):
            observation["observation/state"][6] = float(
                self.spec["initial_gripper_raw"]
            )

    def physical_gripper_target(
        self,
        gripper_raw_target: float,
        default_sim_target: float,
    ) -> float:
        if self.spec["success"]["type"] != "place_in_ring":
            return float(default_sim_target)
        held_opening = float(
            self.spec.get("initial_gripper_sim_half_width", default_sim_target)
        )
        if not self.released:
            return held_opening
        if float(gripper_raw_target) >= float(self.spec["release_gripper_raw"]):
            return max(
                float(default_sim_target),
                float(self.spec.get("release_min_sim_half_width", held_opening)),
            )
        return float(default_sim_target)

    def release_if_requested(self, gripper_raw_target: float) -> bool:
        threshold = self.spec.get("release_gripper_raw")
        if threshold is None or self.released or float(gripper_raw_target) < float(threshold):
            return False

        held_body_id = _body_id(self.model, "held_red_pepper")
        object_addr = _freejoint_qpos_address(self.model, "red_pepper")
        dof_addr = _freejoint_dof_address(self.model, "red_pepper")
        self.data.qpos[object_addr : object_addr + 3] = self.data.xpos[held_body_id]
        self.data.qpos[object_addr + 3 : object_addr + 7] = self.data.xquat[held_body_id]
        self.data.qvel[dof_addr : dof_addr + 6] = 0.0
        _set_body_enabled(self.model, "held_red_pepper", False)
        _set_body_enabled(self.model, "red_pepper", True)
        self.released = True
        mujoco.mj_forward(self.model, self.data)
        return True

    def metrics(self) -> dict[str, Any]:
        target_position = self._target_pose()
        success_config = self.spec["success"]
        success_type = str(success_config["type"])
        metrics: dict[str, Any] = {
            "task": self.task_name,
            "success_type": success_type,
            "target_body": self.target_body,
            "target_position": target_position.tolist(),
            "released": self.released,
        }
        if success_type == "lift":
            lift_height = float(target_position[2] - self.initial_target_z)
            metrics["lift_height_m"] = lift_height
            metrics["instant_success"] = lift_height >= float(success_config["lift_height_m"])
        elif success_type == "place_in_ring":
            ring_position = np.asarray(
                self.data.xpos[_body_id(self.model, str(success_config["ring_body"]))],
                dtype=np.float64,
            )
            ring_distance = float(np.linalg.norm(target_position[:2] - ring_position[:2]))
            metrics["ring_position"] = ring_position.tolist()
            metrics["ring_xy_distance_m"] = ring_distance
            metrics["height_above_table_m"] = float(target_position[2] - TABLE_TOP_Z)
            metrics["instant_success"] = bool(
                self.released
                and ring_distance <= float(success_config["ring_radius_m"])
                and target_position[2] - TABLE_TOP_Z
                <= float(success_config["max_height_above_table_m"])
            )
        else:
            raise ValueError(f"Unsupported success type: {success_type}")
        metrics["required_success_streak"] = int(
            success_config.get("sustained_policy_steps", 1)
        )
        metrics["success_streak"] = self.success_streak
        metrics["task_success"] = bool(
            self.success_streak >= metrics["required_success_streak"]
        )
        return metrics

    def update_success(self) -> dict[str, Any]:
        metrics = self.metrics()
        if metrics["instant_success"]:
            self.success_streak += 1
        else:
            self.success_streak = 0
        metrics["success_streak"] = self.success_streak
        metrics["task_success"] = bool(
            self.success_streak >= metrics["required_success_streak"]
        )
        return metrics


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
) -> tuple[TaskSceneRuntime, dict[str, Any]]:
    config = load_task_scene_config(config_path)
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
                    _freejoint_qpos_address(model, body_name) :
                    _freejoint_qpos_address(model, body_name) + 2
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
                raise ValueError(
                    f"Distractor slot must have two values, got {slot_xy}"
                )
            data.qpos[qpos_addr : qpos_addr + 2] = slot_xy
    for body_name in fixed_bodies:
        if body_name not in poses:
            raise ValueError(f"Missing catalog pose for {body_name}")
        model.body_pos[_body_id(model, body_name)] = np.asarray(
            poses[body_name],
            dtype=np.float64,
        )

    for body_name in (*free_bodies, *fixed_bodies):
        _set_body_enabled(model, body_name, body_name in active_bodies)
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
        gripper_sim = float(
            spec.get(
                "initial_gripper_sim_half_width",
                gripper_raw_to_sim(gripper_raw, load_gripper_config()),
            )
        )
        for joint_name in ("left_finger_slide", "right_finger_slide"):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            data.qpos[int(model.jnt_qposadr[joint_id])] = gripper_sim
        data.ctrl[6] = gripper_sim

    scene_delta = np.zeros(2, dtype=np.float64)
    if seed is not None:
        scene_delta = rng.uniform(-float(object_xy_range), float(object_xy_range), size=2)
    yaw_values: dict[str, float] = {}
    for body_name in spec.get("randomizable_bodies") or ():
        yaw = 0.0
        if seed is not None:
            yaw = math.radians(
                float(rng.uniform(-float(object_yaw_range_deg), float(object_yaw_range_deg)))
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
    for _ in range(int(settle_steps)):
        mujoco.mj_step(model, data)

    wrist_visibility = {
        body_name: body_camera_visibility(model, data, body_name)
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
        body_name: np.asarray(data.xpos[_body_id(model, body_name)], dtype=np.float64).tolist()
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


def load_gripper_config() -> dict[str, Any]:
    with DEFAULT_CAMERA_CONFIG_PATH.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or "gripper_mapping" not in config:
        raise ValueError(f"Missing gripper_mapping in {DEFAULT_CAMERA_CONFIG_PATH}")
    return config
