from __future__ import annotations

import math

import mujoco
import numpy as np


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


def _set_body_enabled(
    model: mujoco.MjModel,
    body_name: str,
    enabled: bool,
) -> None:
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
