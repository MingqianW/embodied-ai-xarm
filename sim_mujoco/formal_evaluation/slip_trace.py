"""Opt-in physics-cadence diagnostics for TCP-relative object slip.

The values in this module come only from MuJoCo ground truth and are never
added to policy observations.  The recorder is deliberately inert unless
``XARM_SLIP_TRACE=1`` is present in the environment.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any
from typing import Mapping

import mujoco
import numpy as np

from sim_mujoco.collision import collision_diagnostics
from sim_mujoco.collision import target_gripper_contact_count
from sim_mujoco.remote_policy_observation import get_robot_state


SLIP_TRACE_ENV = "XARM_SLIP_TRACE"
POST_SUCCESS_SECONDS_ENV = "XARM_SLIP_TRACE_POST_SUCCESS_SECONDS"
DIAGNOSTIC_LATCH_RAW_ENV = "XARM_SLIP_DIAGNOSTIC_LATCH_RAW"
DEFAULT_POST_SUCCESS_SECONDS = 2.0

SLIP_TRACE_FIELDS = (
    "sim_time_s",
    "policy_step",
    "executed_action_index",
    "action_index_in_chunk",
    "object_x_m",
    "object_y_m",
    "object_z_m",
    "tcp_x_m",
    "tcp_y_m",
    "tcp_z_m",
    "relative_x_m",
    "relative_y_m",
    "relative_z_m",
    "relative_3d_drift_m",
    "relative_downward_slip_m",
    "relative_reference_established",
    "target_gripper_contact_count",
    "left_finger_target_contact_count",
    "right_finger_target_contact_count",
    "gripper_raw_command",
    "gripper_raw_command_clamped",
    "gripper_ctrl_target",
    "actual_gripper_state",
    "fingertip_table_contact",
    "left_finger_table_contact",
    "right_finger_table_contact",
    "target_table_contact",
    "fingertip_table_max_normal_force_n",
    "left_finger_table_max_normal_force_n",
    "right_finger_table_max_normal_force_n",
    "target_gripper_max_normal_force_n",
    "fingertip_table_min_distance_m",
    "target_linear_velocity_x_mps",
    "target_linear_velocity_y_mps",
    "target_linear_velocity_z_mps",
    "target_linear_speed_mps",
    "target_vertical_velocity_mps",
    "tcp_vertical_velocity_mps",
    "original_v1_success_reached",
    "post_success_diagnostic",
)


@dataclass(frozen=True)
class SlipTraceSettings:
    enabled: bool
    post_success_seconds: float = 0.0
    diagnostic_latch_raw: float | None = None

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> SlipTraceSettings:
        values = os.environ if environ is None else environ
        enabled = values.get(SLIP_TRACE_ENV) == "1"
        if not enabled:
            return cls(enabled=False)
        seconds = float(
            values.get(POST_SUCCESS_SECONDS_ENV, DEFAULT_POST_SUCCESS_SECONDS)
        )
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError(
                f"{POST_SUCCESS_SECONDS_ENV} must be finite and non-negative"
            )
        latch_value = values.get(DIAGNOSTIC_LATCH_RAW_ENV)
        latch_raw = None if latch_value is None else float(latch_value)
        if latch_raw is not None and (
            not math.isfinite(latch_raw) or not 50.0 <= latch_raw <= 845.0
        ):
            raise ValueError(f"{DIAGNOSTIC_LATCH_RAW_ENV} must be in [50, 845]")
        return cls(
            enabled=True,
            post_success_seconds=seconds,
            diagnostic_latch_raw=latch_raw,
        )


def relative_slip_metrics(
    relative_offset: np.ndarray,
    reference_relative_offset: np.ndarray | None,
) -> tuple[float | None, float | None]:
    """Return 3D drift and positive object-down/TCP-relative slip.

    Both inputs use ``tcp_position - object_position``.  Increasing relative
    Z therefore means that the object moved downward with respect to the TCP.
    """

    if reference_relative_offset is None:
        return None, None
    relative_delta = np.asarray(relative_offset, dtype=np.float64) - np.asarray(
        reference_relative_offset, dtype=np.float64
    )
    return float(np.linalg.norm(relative_delta)), max(0.0, float(relative_delta[2]))


def _named_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Required MuJoCo object not found: {name}")
    return int(object_id)


def _body_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.asarray(data.xpos[body_id], dtype=np.float64).copy()


def _tcp_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    site_id = _named_id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_center_point")
    return np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()


def _free_body_linear_velocity(
    model: mujoco.MjModel, data: mujoco.MjData, body_name: str
) -> np.ndarray:
    body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise RuntimeError(f"Expected a free target body: {body_name}")
    address = int(model.jnt_dofadr[joint_id])
    return np.asarray(data.qvel[address : address + 3], dtype=np.float64).copy()


def _contact_has_body(contact: Mapping[str, Any], body_name: str) -> bool:
    return contact.get("body1") == body_name or contact.get("body2") == body_name


def _contact_has_geom(contact: Mapping[str, Any], geom_name: str) -> bool:
    return contact.get("geom1") == geom_name or contact.get("geom2") == geom_name


def _contact_has_pad_side(contact: Mapping[str, Any], side: str) -> bool:
    prefixes = (f"{side}_finger_pad_", f"{side}_fingertip_pad")
    return str(contact.get("geom1", "")).startswith(prefixes) or str(
        contact.get("geom2", "")
    ).startswith(prefixes)


def _contact_pair_bodies(contact: Mapping[str, Any], first: str, second: str) -> bool:
    return (contact.get("body1") == first and contact.get("body2") == second) or (
        contact.get("body1") == second and contact.get("body2") == first
    )


def _maximum_normal_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contacts: list[dict[str, Any]],
) -> float:
    maximum = 0.0
    force = np.zeros(6, dtype=np.float64)
    for contact in contacts:
        mujoco.mj_contactForce(model, data, int(contact["contact_index"]), force)
        maximum = max(maximum, abs(float(force[0])))
    return maximum


class SlipTraceRecorder:
    """Collect and atomically write one episode's physics-cadence trace."""

    def __init__(self, *, output_dir: Path, target_body: str) -> None:
        self.output_path = Path(output_dir) / "slip_trace.csv"
        self.target_body = str(target_body)
        self.rows: list[dict[str, Any]] = []
        self.reference_relative_offset: np.ndarray | None = None
        self._previous_tcp_position: np.ndarray | None = None
        self._previous_sim_time_s: float | None = None

    def set_target_body(self, target_body: str) -> None:
        """Follow a one-time logical-object body swap without resetting drift."""

        self.target_body = str(target_body)

    def sample(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_config: dict[str, Any],
        policy_step: int,
        executed_action_index: int,
        action_index_in_chunk: int,
        gripper_raw_command: float,
        gripper_raw_command_clamped: float,
        gripper_ctrl_target: float,
        collision: dict[str, Any] | None = None,
        original_v1_success_reached: bool,
        post_success_diagnostic: bool,
    ) -> None:
        if collision is None:
            collision = collision_diagnostics(model, data)
        contacts = list(collision.get("contacts") or ())
        object_position = _body_position(model, data, self.target_body)
        tcp_position = _tcp_position(model, data)
        relative_offset = tcp_position - object_position
        contact_count = target_gripper_contact_count(collision, self.target_body)
        if self.reference_relative_offset is None and contact_count > 0:
            self.reference_relative_offset = relative_offset.copy()
        relative_3d_drift, relative_downward_slip = relative_slip_metrics(
            relative_offset, self.reference_relative_offset
        )

        left_target = [
            row
            for row in contacts
            if _contact_pair_bodies(row, self.target_body, "left_finger")
        ]
        right_target = [
            row
            for row in contacts
            if _contact_pair_bodies(row, self.target_body, "right_finger")
        ]
        left_table = [
            row
            for row in contacts
            if _contact_has_geom(row, "table") and _contact_has_pad_side(row, "left")
        ]
        right_table = [
            row
            for row in contacts
            if _contact_has_geom(row, "table") and _contact_has_pad_side(row, "right")
        ]
        target_table = [
            row
            for row in contacts
            if _contact_has_geom(row, "table")
            and _contact_has_body(row, self.target_body)
        ]
        fingertip_table = [*left_table, *right_table]
        target_gripper = [*left_target, *right_target]
        target_velocity = _free_body_linear_velocity(model, data, self.target_body)
        sim_time = float(data.time)
        tcp_vertical_velocity: float | None = None
        if (
            self._previous_tcp_position is not None
            and self._previous_sim_time_s is not None
        ):
            dt = sim_time - self._previous_sim_time_s
            if dt > 0.0:
                tcp_vertical_velocity = float(
                    (tcp_position[2] - self._previous_tcp_position[2]) / dt
                )
        self._previous_tcp_position = tcp_position.copy()
        self._previous_sim_time_s = sim_time

        row = {
            "sim_time_s": sim_time,
            "policy_step": int(policy_step),
            "executed_action_index": int(executed_action_index),
            "action_index_in_chunk": int(action_index_in_chunk),
            "object_x_m": float(object_position[0]),
            "object_y_m": float(object_position[1]),
            "object_z_m": float(object_position[2]),
            "tcp_x_m": float(tcp_position[0]),
            "tcp_y_m": float(tcp_position[1]),
            "tcp_z_m": float(tcp_position[2]),
            "relative_x_m": float(relative_offset[0]),
            "relative_y_m": float(relative_offset[1]),
            "relative_z_m": float(relative_offset[2]),
            "relative_3d_drift_m": relative_3d_drift,
            "relative_downward_slip_m": relative_downward_slip,
            "relative_reference_established": self.reference_relative_offset
            is not None,
            "target_gripper_contact_count": int(contact_count),
            "left_finger_target_contact_count": len(left_target),
            "right_finger_target_contact_count": len(right_target),
            "gripper_raw_command": float(gripper_raw_command),
            "gripper_raw_command_clamped": float(gripper_raw_command_clamped),
            "gripper_ctrl_target": float(gripper_ctrl_target),
            "actual_gripper_state": float(
                get_robot_state(model, data, camera_config)[6]
            ),
            "fingertip_table_contact": bool(fingertip_table),
            "left_finger_table_contact": bool(left_table),
            "right_finger_table_contact": bool(right_table),
            "target_table_contact": bool(target_table),
            "fingertip_table_max_normal_force_n": _maximum_normal_force(
                model, data, fingertip_table
            ),
            "left_finger_table_max_normal_force_n": _maximum_normal_force(
                model, data, left_table
            ),
            "right_finger_table_max_normal_force_n": _maximum_normal_force(
                model, data, right_table
            ),
            "target_gripper_max_normal_force_n": _maximum_normal_force(
                model, data, target_gripper
            ),
            "fingertip_table_min_distance_m": min(
                (float(item["distance_m"]) for item in fingertip_table), default=None
            ),
            "target_linear_velocity_x_mps": float(target_velocity[0]),
            "target_linear_velocity_y_mps": float(target_velocity[1]),
            "target_linear_velocity_z_mps": float(target_velocity[2]),
            "target_linear_speed_mps": float(np.linalg.norm(target_velocity)),
            "target_vertical_velocity_mps": float(target_velocity[2]),
            "tcp_vertical_velocity_mps": tcp_vertical_velocity,
            "original_v1_success_reached": bool(original_v1_success_reached),
            "post_success_diagnostic": bool(post_success_diagnostic),
        }
        self.rows.append(row)

    def write(self) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SLIP_TRACE_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.output_path)
        return self.output_path
