"""Physics-cadence ground-truth logging for xArm gripper-slip experiments.

This module is diagnostic-only. It neither changes control targets nor exposes
MuJoCo ground truth to a learned policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

from sim_mujoco.gripper_mapping import (
    driver_angle_to_raw_gripper,
    is_menagerie_gripper,
    measure_fingertip_aperture_m,
    sim_slide_to_raw_gripper,
)


LEFT_FINGER_BODY = "left_finger"
RIGHT_FINGER_BODY = "right_finger"
LEFT_PAD_GEOM = "left_finger_pad_1"
RIGHT_PAD_GEOM = "right_finger_pad_1"
TABLE_GEOM = "table"
TCP_SITE = "tool_center_point"
GRIPPER_ACTUATOR = "gripper_actuator"
LEFT_DRIVER = "left_driver_joint"
RIGHT_DRIVER = "right_driver_joint"


@dataclass(frozen=True)
class CommandContext:
    """Command metadata held constant for one physics interval."""

    source: str
    stage: str
    action_step: int
    inference_index: int = -1
    chunk_index: int = -1
    action_index_in_chunk: int = -1
    gripper_network_normalized: float | None = None
    gripper_returned_raw: float | None = None
    gripper_clamped_raw: float | None = None
    gripper_ctrl: float | None = None
    network_action: list[float] | None = None
    returned_action: list[float] | None = None
    arm_target_clamped_rad: list[float] | None = None
    ctrl_target: list[float] | None = None


@dataclass(frozen=True)
class DiagnosticEvent:
    event: str
    sim_time_s: float
    sample_index: int
    details: dict[str, Any]


def inverse_quantile_normalize(
    value: float,
    *,
    q01: float,
    q99: float,
) -> float:
    """Invert OpenPI quantile unnormalization for a scalar action.

    OpenPI uses ``(x + 1) / 2 * (q99 - q01 + 1e-6) + q01``.
    """

    values = (float(value), float(q01), float(q99))
    if not all(math.isfinite(item) for item in values) or q99 <= q01:
        raise ValueError("value, q01, and q99 must be finite with q99 > q01")
    return 2.0 * (float(value) - float(q01)) / (float(q99) - float(q01) + 1e-6) - 1.0


def reconstruct_network_action(
    returned_absolute_action: np.ndarray,
    observation_state: np.ndarray,
    *,
    q01: np.ndarray,
    q99: np.ndarray,
) -> np.ndarray:
    """Reconstruct the 7D pre-denormalization xArm network action.

    The local OpenPI adapter returns absolute arm targets after adding the
    current state, while the gripper remains absolute. Subtracting the current
    arm state and inverting quantile normalization recovers the model-space
    action up to normal floating-point roundoff.
    """

    action = np.asarray(returned_absolute_action, dtype=np.float64)
    state = np.asarray(observation_state, dtype=np.float64)
    lower = np.asarray(q01, dtype=np.float64)
    upper = np.asarray(q99, dtype=np.float64)
    if action.shape != (7,) or state.shape != (7,):
        raise ValueError("returned action and observation state must have shape (7,)")
    if lower.shape != (7,) or upper.shape != (7,):
        raise ValueError("q01 and q99 must have shape (7,)")
    transformed = action.copy()
    transformed[:6] -= state[:6]
    if (
        not np.isfinite(transformed).all()
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
    ):
        raise ValueError("action, state, and quantiles must be finite")
    if np.any(upper <= lower):
        raise ValueError("every q99 value must exceed q01")
    return 2.0 * (transformed - lower) / (upper - lower + 1e-6) - 1.0


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise RuntimeError(f"Required MuJoCo object not found: {name}")
    return value


def _name(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    object_id: int,
    fallback: str,
) -> str:
    value = mujoco.mj_id2name(model, object_type, int(object_id))
    return fallback if value is None else str(value)


def _object_kinematics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_type: mujoco.mjtObj,
    object_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.zeros(6, dtype=np.float64)
    acceleration = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, object_type, object_id, velocity, 0)
    mujoco.mj_objectAcceleration(model, data, object_type, object_id, acceleration, 0)
    # MuJoCo spatial vectors are [angular, linear].
    return velocity, acceleration


def _contact_record(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contact_index: int,
) -> dict[str, Any]:
    contact = data.contact[contact_index]
    geom1 = int(contact.geom1)
    geom2 = int(contact.geom2)
    body1 = int(model.geom_bodyid[geom1])
    body2 = int(model.geom_bodyid[geom2])
    force = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(model, data, contact_index, force)
    frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
    force_world = frame.T @ force[:3]
    torque_world = frame.T @ force[3:]
    return {
        "contact_index": int(contact_index),
        "geom1": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1, f"geom_{geom1}"),
        "geom2": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2, f"geom_{geom2}"),
        "body1": _name(model, mujoco.mjtObj.mjOBJ_BODY, body1, f"body_{body1}"),
        "body2": _name(model, mujoco.mjtObj.mjOBJ_BODY, body2, f"body_{body2}"),
        "position_world_m": np.asarray(contact.pos, dtype=np.float64).tolist(),
        "normal_world": frame[0].tolist(),
        "frame_world_to_contact": frame.tolist(),
        "distance_m": float(contact.dist),
        "condim": int(contact.dim),
        "friction": np.asarray(contact.friction, dtype=np.float64).tolist(),
        "force_contact_n": force[:3].tolist(),
        "torque_contact_nm": force[3:].tolist(),
        "force_world_n": force_world.tolist(),
        "torque_world_nm": torque_world.tolist(),
        "normal_force_n": abs(float(force[0])),
        "tangential_force_n": float(np.linalg.norm(force[1:3])),
        "contact_torque_nm": float(np.linalg.norm(force[3:])),
    }


def _has_pair(record: dict[str, Any], first: str, second: str) -> bool:
    return (record["body1"] == first and record["body2"] == second) or (
        record["body1"] == second and record["body2"] == first
    )


def _has_geom(record: dict[str, Any], name: str) -> bool:
    return record["geom1"] == name or record["geom2"] == name


def _has_fingertip_pad(record: dict[str, Any]) -> bool:
    return any(
        str(record[key]).startswith(
            (
                "left_finger_pad_",
                "right_finger_pad_",
                "left_fingertip_pad",
                "right_fingertip_pad",
            )
        )
        for key in ("geom1", "geom2")
    )


class PhysicsTraceRecorder:
    """Record nested JSONL samples and event markers at physics cadence."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        target_body: str,
        camera_config: dict[str, Any],
        initial_target_z_m: float,
        trial: dict[str, Any],
    ) -> None:
        self.model = model
        self.data = data
        self.target_body = str(target_body)
        self.camera_config = camera_config
        self.initial_target_z_m = float(initial_target_z_m)
        self.trial = dict(trial)
        self.rows: list[dict[str, Any]] = []
        self.events: list[DiagnosticEvent] = []
        self.target_body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, target_body)
        self.tcp_site_id = _id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        self.gripper_actuator_id = _id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR
        )
        self.menagerie = is_menagerie_gripper(model)
        self.affine_menagerie = bool(
            self.menagerie
            and float(model.actuator_ctrlrange[self.gripper_actuator_id, 1]) > 1.0
        )
        left_joint = LEFT_DRIVER if self.menagerie else "left_finger_slide"
        right_joint = RIGHT_DRIVER if self.menagerie else "right_finger_slide"
        left_pad = LEFT_PAD_GEOM if self.menagerie else "left_fingertip_pad"
        right_pad = RIGHT_PAD_GEOM if self.menagerie else "right_fingertip_pad"
        self.left_joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, left_joint)
        self.right_joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, right_joint)
        self.left_pad_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, left_pad)
        self.right_pad_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, right_pad)
        self._reference_relative_position: np.ndarray | None = None
        self._previous_command: CommandContext | None = None
        self._seen: set[str] = set()
        self._maximum_returned_gripper_raw: float | None = None

    def set_target_body(self, target_body: str) -> None:
        """Follow a logical object when LOCAL swaps held and free bodies."""

        target_body = str(target_body)
        if target_body == self.target_body:
            return
        self.target_body = target_body
        self.target_body_id = _id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, target_body
        )

    def _event(self, event: str, details: dict[str, Any] | None = None) -> None:
        if event in self._seen:
            return
        self._seen.add(event)
        self.events.append(
            DiagnosticEvent(
                event=event,
                sim_time_s=float(self.data.time),
                sample_index=len(self.rows),
                details={} if details is None else dict(details),
            )
        )

    def sample(self, command: CommandContext) -> dict[str, Any]:
        model = self.model
        data = self.data
        # Required by mj_objectAcceleration: populate constraint-aware body
        # accelerations for the just-completed physics state.
        mujoco.mj_rnePostConstraint(model, data)
        contacts = [
            _contact_record(model, data, index) for index in range(int(data.ncon))
        ]
        left_target = [
            row
            for row in contacts
            if _has_pair(row, self.target_body, LEFT_FINGER_BODY)
        ]
        right_target = [
            row
            for row in contacts
            if _has_pair(row, self.target_body, RIGHT_FINGER_BODY)
        ]
        target_table = [
            row
            for row in contacts
            if _has_geom(row, TABLE_GEOM)
            and (row["body1"] == self.target_body or row["body2"] == self.target_body)
        ]
        fingertip_table = [
            row
            for row in contacts
            if _has_geom(row, TABLE_GEOM) and _has_fingertip_pad(row)
        ]

        object_position = np.asarray(
            data.xpos[self.target_body_id], dtype=np.float64
        ).copy()
        object_quaternion = np.asarray(
            data.xquat[self.target_body_id], dtype=np.float64
        ).copy()
        tcp_position = np.asarray(
            data.site_xpos[self.tcp_site_id], dtype=np.float64
        ).copy()
        tcp_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(
            tcp_quaternion, np.asarray(data.site_xmat[self.tcp_site_id])
        )
        object_velocity, object_acceleration = _object_kinematics(
            model, data, mujoco.mjtObj.mjOBJ_BODY, self.target_body_id
        )
        tcp_velocity, tcp_acceleration = _object_kinematics(
            model, data, mujoco.mjtObj.mjOBJ_SITE, self.tcp_site_id
        )
        relative_position = tcp_position - object_position
        bilateral = bool(left_target and right_target)
        if self._reference_relative_position is None and bilateral:
            self._reference_relative_position = relative_position.copy()
        relative_delta = (
            None
            if self._reference_relative_position is None
            else relative_position - self._reference_relative_position
        )

        left_qpos_address = int(model.jnt_qposadr[self.left_joint_id])
        right_qpos_address = int(model.jnt_qposadr[self.right_joint_id])
        left_dof_address = int(model.jnt_dofadr[self.left_joint_id])
        right_dof_address = int(model.jnt_dofadr[self.right_joint_id])
        left_driver = float(data.qpos[left_qpos_address])
        right_driver = float(data.qpos[right_qpos_address])
        pad_center_distance = float(
            np.linalg.norm(
                data.geom_xpos[self.left_pad_id] - data.geom_xpos[self.right_pad_id]
            )
        )
        pad_gap = measure_fingertip_aperture_m(model, data)
        actuator_force = float(data.actuator_force[self.gripper_actuator_id])
        force_limit = float(
            np.max(np.abs(model.actuator_forcerange[self.gripper_actuator_id]))
        )
        moment_storage = np.asarray(data.actuator_moment, dtype=np.float64)
        if moment_storage.ndim == 2:
            actuator_moment = moment_storage[self.gripper_actuator_id]
        else:
            # MuJoCo 3.10 stores actuator moments as sparse rows.
            actuator_moment = np.zeros(model.nv, dtype=np.float64)
            start = int(data.moment_rowadr[self.gripper_actuator_id])
            count = int(data.moment_rownnz[self.gripper_actuator_id])
            columns = np.asarray(
                data.moment_colind[start : start + count], dtype=np.int64
            )
            actuator_moment[columns] = moment_storage[start : start + count]

        target_contacts = [*left_target, *right_target]
        all_target_contacts = [
            value
            for value in contacts
            if value["body1"] == self.target_body or value["body2"] == self.target_body
        ]
        left_normal_sum = float(sum(row["normal_force_n"] for row in left_target))
        right_normal_sum = float(sum(row["normal_force_n"] for row in right_target))
        left_tangential_sum = float(
            sum(row["tangential_force_n"] for row in left_target)
        )
        right_tangential_sum = float(
            sum(row["tangential_force_n"] for row in right_target)
        )
        normal_sum = float(sum(row["normal_force_n"] for row in target_contacts))
        tangential_sum = float(
            sum(row["tangential_force_n"] for row in target_contacts)
        )
        target_table_normal_sum = float(
            sum(row["normal_force_n"] for row in target_table)
        )
        target_table_tangential_sum = float(
            sum(row["tangential_force_n"] for row in target_table)
        )
        all_target_normal_sum = float(
            sum(row["normal_force_n"] for row in all_target_contacts)
        )
        all_target_tangential_sum = float(
            sum(row["tangential_force_n"] for row in all_target_contacts)
        )
        warning_count = sum(int(value.number) for value in data.warning)
        warning_lastinfo = [
            int(value.lastinfo) for value in data.warning if int(value.number) > 0
        ]
        minimum_contact_distance = min(
            (float(value["distance_m"]) for value in target_contacts),
            default=None,
        )
        minimum_table_distance = min(
            (float(value["distance_m"]) for value in target_table),
            default=None,
        )
        minimum_all_target_distance = min(
            (float(value["distance_m"]) for value in all_target_contacts),
            default=None,
        )
        contact_count_total = len(left_target) + len(right_target)
        contact_count_difference = abs(len(left_target) - len(right_target))
        row = {
            "schema_version": "xarm_gripper_physics_trace_v4",
            "sample_index": len(self.rows),
            "sim_time_s": float(data.time),
            "trial": self.trial,
            "command": asdict(command),
            "actuator": {
                "ctrl": float(data.ctrl[self.gripper_actuator_id]),
                "force_actuator_space": actuator_force,
                "force_limit_actuator_space": force_limit,
                "force_fraction": abs(actuator_force) / force_limit
                if force_limit > 0.0
                else None,
                "moment": actuator_moment.tolist(),
                "qfrc_actuator_left": float(data.qfrc_actuator[left_dof_address]),
                "qfrc_actuator_right": float(data.qfrc_actuator[right_dof_address]),
                "qfrc_constraint_left": float(data.qfrc_constraint[left_dof_address]),
                "qfrc_constraint_right": float(data.qfrc_constraint[right_dof_address]),
            },
            "fingers": {
                "representation": "menagerie_linkage"
                if self.menagerie
                else "legacy_slide",
                "left_driver_qpos_rad": left_driver if self.menagerie else None,
                "right_driver_qpos_rad": right_driver if self.menagerie else None,
                "left_driver_qvel_radps": (
                    float(data.qvel[left_dof_address]) if self.menagerie else None
                ),
                "right_driver_qvel_radps": (
                    float(data.qvel[right_dof_address]) if self.menagerie else None
                ),
                "left_slide_qpos_m": None if self.menagerie else left_driver,
                "right_slide_qpos_m": None if self.menagerie else right_driver,
                "left_slide_qvel_mps": (
                    None if self.menagerie else float(data.qvel[left_dof_address])
                ),
                "right_slide_qvel_mps": (
                    None if self.menagerie else float(data.qvel[right_dof_address])
                ),
                "left_raw_equivalent": float(
                    driver_angle_to_raw_gripper(left_driver, self.camera_config)
                    if self.menagerie
                    else sim_slide_to_raw_gripper(left_driver, self.camera_config)
                ),
                "right_raw_equivalent": float(
                    driver_angle_to_raw_gripper(right_driver, self.camera_config)
                    if self.menagerie
                    else sim_slide_to_raw_gripper(right_driver, self.camera_config)
                ),
                "pad_center_distance_m": pad_center_distance,
                "pad_surface_gap_m": pad_gap,
                "opening_width_m": pad_gap,
            },
            "object": {
                "position_m": object_position.tolist(),
                "quaternion_wxyz": object_quaternion.tolist(),
                "angular_velocity_world_radps": object_velocity[:3].tolist(),
                "linear_velocity_world_mps": object_velocity[3:].tolist(),
                "angular_acceleration_world_radps2": object_acceleration[:3].tolist(),
                "linear_acceleration_world_mps2": object_acceleration[3:].tolist(),
                "lift_height_m": float(object_position[2] - self.initial_target_z_m),
            },
            "tcp": {
                "position_m": tcp_position.tolist(),
                "quaternion_wxyz": tcp_quaternion.tolist(),
                "angular_velocity_world_radps": tcp_velocity[:3].tolist(),
                "linear_velocity_world_mps": tcp_velocity[3:].tolist(),
                "angular_acceleration_world_radps2": tcp_acceleration[:3].tolist(),
                "linear_acceleration_world_mps2": tcp_acceleration[3:].tolist(),
            },
            "relative": {
                "tcp_minus_object_position_m": relative_position.tolist(),
                "delta_from_bilateral_grasp_m": None
                if relative_delta is None
                else relative_delta.tolist(),
                "vertical_slip_m": (
                    None if relative_delta is None else float(relative_delta[2])
                ),
                "vertical_slip_velocity_mps": float(
                    tcp_velocity[5] - object_velocity[5]
                ),
                "downward_slip_m": None
                if relative_delta is None
                else max(0.0, float(relative_delta[2])),
                "drift_m": None
                if relative_delta is None
                else float(np.linalg.norm(relative_delta)),
            },
            "contacts": {
                "all": contacts,
                "left_target_count": len(left_target),
                "right_target_count": len(right_target),
                "target_gripper_contact_count": len(target_contacts),
                "left_target_contact_positions_world_m": [
                    value["position_world_m"] for value in left_target
                ],
                "right_target_contact_positions_world_m": [
                    value["position_world_m"] for value in right_target
                ],
                "left_target_contact_geom_pairs": [
                    [value["geom1"], value["geom2"]] for value in left_target
                ],
                "right_target_contact_geom_pairs": [
                    [value["geom1"], value["geom2"]] for value in right_target
                ],
                "left_right_contact_count_difference": contact_count_difference,
                "left_right_contact_count_symmetry": (
                    None
                    if contact_count_total == 0
                    else 1.0 - contact_count_difference / contact_count_total
                ),
                "bilateral": bilateral,
                "target_table_count": len(target_table),
                "fingertip_table_count": len(fingertip_table),
                "target_gripper_normal_sum_n": normal_sum,
                "target_gripper_tangential_sum_n": tangential_sum,
                "target_table_normal_sum_n": target_table_normal_sum,
                "target_table_tangential_sum_n": target_table_tangential_sum,
                "all_target_normal_sum_n": all_target_normal_sum,
                "all_target_tangential_sum_n": all_target_tangential_sum,
                "left_target_normal_sum_n": left_normal_sum,
                "right_target_normal_sum_n": right_normal_sum,
                "left_target_tangential_sum_n": left_tangential_sum,
                "right_target_tangential_sum_n": right_tangential_sum,
                "target_gripper_tangential_to_normal": (
                    tangential_sum / normal_sum if normal_sum > 0.0 else None
                ),
                "minimum_target_contact_distance_m": minimum_contact_distance,
                "maximum_target_penetration_m": (
                    None
                    if minimum_contact_distance is None
                    else max(0.0, -minimum_contact_distance)
                ),
                "maximum_target_table_penetration_m": (
                    None
                    if minimum_table_distance is None
                    else max(0.0, -minimum_table_distance)
                ),
                "maximum_all_target_penetration_m": (
                    None
                    if minimum_all_target_distance is None
                    else max(0.0, -minimum_all_target_distance)
                ),
            },
            "simulation": {
                "solver_iterations": int(data.solver_iter),
                "solver_nnz": int(data.solver_nnz),
                "solver_fwdinv": np.asarray(
                    data.solver_fwdinv, dtype=np.float64
                ).tolist(),
                "warning_count": warning_count,
                "warning_lastinfo": warning_lastinfo,
                "maximum_abs_qvel": float(np.max(np.abs(data.qvel))),
                "maximum_abs_qacc": float(np.max(np.abs(data.qacc))),
            },
        }

        if target_contacts:
            self._event("grasp_contact_onset", {"contact_count": len(target_contacts)})
        if bilateral:
            self._event(
                "bilateral_grasp",
                {"reference_relative_position_m": relative_position.tolist()},
            )
            self._event(
                "grasp_establishment",
                {
                    "criterion": "sustained classification is performed in postprocessing"
                },
            )
        if float(object_position[2] - self.initial_target_z_m) >= 0.001:
            self._event(
                "object_leaving_table",
                {"lift_height_m": float(object_position[2] - self.initial_target_z_m)},
            )
        if float(object_position[2] - self.initial_target_z_m) >= 0.005:
            self._event(
                "lift_onset",
                {"lift_height_m": float(object_position[2] - self.initial_target_z_m)},
            )
        if relative_delta is not None and float(relative_delta[2]) >= 0.002:
            self._event("slip_onset", {"downward_slip_m": float(relative_delta[2])})
        if "bilateral_grasp" in self._seen and not target_contacts:
            self._event("contact_loss")
        if "lift_onset" in self._seen and target_table:
            self._event("table_impact", {"contact_count": len(target_table)})
        if force_limit > 0.0 and abs(actuator_force) >= 0.99 * force_limit:
            self._event(
                "actuator_force_saturation",
                {"force_actuator_space": actuator_force},
            )
        target_ctrl = command.gripper_ctrl
        unloaded_target_angle = (
            None
            if target_ctrl is None
            else (
                0.333 * float(target_ctrl) / 100.0
                if self.affine_menagerie
                else float(target_ctrl)
            )
        )
        if (
            unloaded_target_angle is not None
            and abs(left_driver - unloaded_target_angle) >= 0.001
            and abs(float(data.qvel[left_dof_address])) <= 0.001
            and force_limit > 0.0
            and abs(actuator_force) >= 0.99 * force_limit
        ):
            self._event(
                "gripper_stall",
                {
                    (
                        "driver_tracking_error_rad"
                        if self.menagerie
                        else "slide_tracking_error_m"
                    ): left_driver - unloaded_target_angle
                },
            )
        if self._previous_command is not None:
            previous = self._previous_command
            if (
                command.inference_index != previous.inference_index
                and command.inference_index >= 0
            ):
                details: dict[str, Any] = {
                    "previous_inference_index": previous.inference_index,
                    "inference_index": command.inference_index,
                }
                if (
                    command.gripper_returned_raw is not None
                    and previous.gripper_returned_raw is not None
                ):
                    details["gripper_command_jump_raw"] = float(
                        command.gripper_returned_raw
                    ) - float(previous.gripper_returned_raw)
                self._event(f"chunk_boundary_{command.inference_index}", details)
            if (
                command.gripper_returned_raw is not None
                and previous.gripper_returned_raw is not None
            ):
                delta = float(command.gripper_returned_raw) - float(
                    previous.gripper_returned_raw
                )
                if delta >= 10.0:
                    self._event("gripper_reopen_command", {"delta_raw": delta})
                elif delta <= -10.0 and "gripper_reopen_command" in self._seen:
                    self._event("gripper_relock_command", {"delta_raw": delta})
        if command.gripper_returned_raw is not None:
            returned_raw = float(command.gripper_returned_raw)
            self._maximum_returned_gripper_raw = (
                returned_raw
                if self._maximum_returned_gripper_raw is None
                else max(self._maximum_returned_gripper_raw, returned_raw)
            )
            if self._maximum_returned_gripper_raw - returned_raw >= 50.0:
                self._event(
                    "gripper_closure",
                    {
                        "maximum_prior_or_current_raw": self._maximum_returned_gripper_raw,
                        "current_raw": returned_raw,
                    },
                )
        self._previous_command = command
        self.rows.append(row)
        return row

    def write(self, output_dir: Path) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        trace_path = output / "physics_trace.jsonl"
        events_path = output / "events.json"
        trial_path = output / "trial.json"
        for path in (trace_path, events_path, trial_path):
            if path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite diagnostic artifact: {path}"
                )
        if self.rows and "maximum_lift" not in self._seen:
            maximum = max(
                self.rows,
                key=lambda row: float(row["object"]["lift_height_m"]),
            )
            self._seen.add("maximum_lift")
            self.events.append(
                DiagnosticEvent(
                    event="maximum_lift",
                    sim_time_s=float(maximum["sim_time_s"]),
                    sample_index=int(maximum["sample_index"]),
                    details={
                        "lift_height_m": float(maximum["object"]["lift_height_m"])
                    },
                )
            )
        if (
            self.rows
            and "lift_onset" in self._seen
            and "table_impact" in self._seen
            and "object_drop" not in self._seen
        ):
            lift_time = next(
                event.sim_time_s for event in self.events if event.event == "lift_onset"
            )
            impact = next(
                (
                    row
                    for row in self.rows
                    if int(row["contacts"]["target_table_count"]) > 0
                    and float(row["object"]["lift_height_m"]) < 0.005
                    and float(row["sim_time_s"]) >= lift_time
                ),
                self.rows[-1],
            )
            self._seen.add("object_drop")
            self.events.append(
                DiagnosticEvent(
                    event="object_drop",
                    sim_time_s=float(impact["sim_time_s"]),
                    sample_index=int(impact["sample_index"]),
                    details={"criterion": "target returned to table after lift onset"},
                )
            )
        temporary = trace_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(trace_path)
        _write_json(events_path, [asdict(event) for event in self.events])
        _write_json(trial_path, self.trial)
        return {"trace": trace_path, "events": events_path, "trial": trial_path}


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value
