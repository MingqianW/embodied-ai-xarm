"""Formal task metrics over simulator ground truth, never policy inputs."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import mujoco
import numpy as np

from simulation.physics.collision import collision_diagnostics
from simulation.physics.collision import target_gripper_contact_count
from evaluation.sim.config import FormalProtocol
from simulation.observation.state import get_robot_state
from simulation.robot.gripper import actuator_ctrl_from_raw_hardware
from simulation.scene import TABLE_TOP_Z
from simulation.scene import TaskSceneRuntime
from simulation.configuration import load_simulation_config


def _body_id(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Body not found: {body_name}")
    return int(body_id)


def _body_position(
    model: mujoco.MjModel, data: mujoco.MjData, body_name: str
) -> np.ndarray:
    return np.asarray(data.xpos[_body_id(model, body_name)], dtype=np.float64).copy()


def _body_velocity(
    model: mujoco.MjModel, data: mujoco.MjData, body_name: str
) -> tuple[float, float]:
    body_id = _body_id(model, body_name)
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise RuntimeError(
            f"Expected a free body for velocity diagnostics: {body_name}"
        )
    address = int(model.jnt_dofadr[joint_id])
    velocity = np.asarray(data.qvel[address : address + 6], dtype=np.float64)
    return float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))


def _tcp_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_center_point")
    if site_id < 0:
        raise RuntimeError("tool_center_point site not found")
    return np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()


def target_table_contact(collision: dict[str, Any], target_body: str) -> bool:
    return any(
        (row.get("body1") == target_body or row.get("body2") == target_body)
        and (row.get("geom1") == "table" or row.get("geom2") == "table")
        for row in collision.get("contacts") or ()
    )


def validate_initial_place_grasp(
    *,
    runtime: TaskSceneRuntime,
    initial_conditions: dict[str, Any],
    protocol: FormalProtocol,
) -> dict[str, Any]:
    """Physically hold the reset grasp and reject an unstable start state.

    This is a reset-time validation only. It neither reads policy output nor
    constrains the pepper after policy control begins.
    """

    model, data = runtime.model, runtime.data
    target = runtime.active_target_body
    start_object = _body_position(model, data, target)
    start_tcp = _tcp_position(model, data)
    initial_arm = np.asarray(
        initial_conditions["initial_joint_positions"], dtype=np.float64
    )
    if initial_arm.shape != (6,) or not np.isfinite(initial_arm).all():
        raise ValueError("Place reset has invalid initial arm hold target")
    held_raw = float(runtime.spec["initial_gripper_raw"])
    held_ctrl = actuator_ctrl_from_raw_hardware(held_raw, load_simulation_config())
    samples: list[dict[str, Any]] = []
    for _ in range(protocol.placement_initial_validation_checks):
        data.ctrl[:6] = initial_arm
        data.ctrl[6] = held_ctrl
        physics_steps = max(
            1, round(protocol.placement_initial_validation_dt_s / model.opt.timestep)
        )
        for _ in range(physics_steps):
            mujoco.mj_step(model, data)
        collision = collision_diagnostics(model, data)
        object_position = _body_position(model, data, target)
        tcp_position = _tcp_position(model, data)
        samples.append(
            {
                "object_position_m": object_position.tolist(),
                "tcp_position_m": tcp_position.tolist(),
                "relative_drift_m": float(
                    np.linalg.norm(
                        (object_position - tcp_position) - (start_object - start_tcp)
                    )
                ),
                "height_above_table_m": float(object_position[2] - TABLE_TOP_Z),
                "gripper_contact_count": target_gripper_contact_count(
                    collision, target
                ),
                "table_contact": target_table_contact(collision, target),
                "forbidden_collision": bool(collision["forbidden"]),
                "finite": bool(
                    np.isfinite(object_position).all()
                    and np.isfinite(tcp_position).all()
                ),
            }
        )
    maximum_drift = max(
        (sample["relative_drift_m"] for sample in samples), default=float("inf")
    )
    failure_reason = None
    if len(samples) != protocol.placement_initial_validation_checks:
        failure_reason = "initial_grasp_incomplete_validation"
    elif not all(sample["finite"] for sample in samples):
        failure_reason = "initial_grasp_nonfinite"
    elif any(sample["forbidden_collision"] for sample in samples):
        failure_reason = "initial_grasp_forbidden_collision"
    elif any(sample["table_contact"] for sample in samples):
        failure_reason = "initial_grasp_table_contact"
    elif any(
        sample["height_above_table_m"]
        < protocol.placement_initial_min_height_above_table_m
        for sample in samples
    ):
        failure_reason = "initial_grasp_height_below_minimum"
    elif maximum_drift > protocol.placement_initial_max_relative_drift_m:
        failure_reason = "initial_grasp_excessive_relative_drift"
    elif any(
        sample["gripper_contact_count"]
        < protocol.placement_initial_min_gripper_contacts
        for sample in samples
    ):
        failure_reason = "initial_grasp_contact_not_retained"
    return {
        "validated": failure_reason is None,
        "failure_reason": failure_reason,
        "checks": samples,
        "maximum_relative_drift_m": maximum_drift,
        "initial_object_position_m": start_object.tolist(),
        "initial_tcp_position_m": start_tcp.tolist(),
        "initial_gripper_raw": held_raw,
        "initial_gripper_ctrl": held_ctrl,
        "permanent_attachment": False,
    }


@dataclass
class PickSuccess:
    initial_height_m: float
    lift_threshold_m: float
    meaningful_lift_threshold_m: float
    required_checks: int
    post_success_hold_checks: int = 0
    max_post_success_drop_m: float = 0.0
    success_streak: int = 0
    max_height_m: float = field(init=False)
    check_count: int = 0
    max_success_confirmation_count: int = 0
    first_meaningful_lift_policy_step: int | None = None
    first_success_height_policy_step: int | None = None
    peak_lift_policy_step: int | None = None
    target_gripper_contact_ever: bool = False
    target_gripper_contact_check_count: int = 0
    target_gripper_contact_count_total: int = 0
    first_target_gripper_contact_policy_step: int | None = None
    last_target_gripper_contact_policy_step: int | None = None
    provisional_success_active: bool = False
    provisional_success_ever: bool = False
    first_provisional_success_policy_step: int | None = None
    post_success_hold_check_count: int = 0
    max_post_success_hold_check_count: int = 0
    post_success_hold_failure_count: int = 0
    current_post_success_hold_peak_lift_m: float | None = None
    post_success_max_downward_slip_m: float = 0.0
    post_success_slip_ever: bool = False
    first_post_success_slip_policy_step: int | None = None

    def __post_init__(self) -> None:
        self.max_height_m = self.initial_height_m

    def _record_height(self, *, height_m: float, policy_step: int) -> float:
        previous_max_height = self.max_height_m
        self.max_height_m = max(self.max_height_m, float(height_m))
        if self.max_height_m > previous_max_height:
            self.peak_lift_policy_step = policy_step
        return float(height_m - self.initial_height_m)

    def _start_post_success_hold(self, *, lift_m: float, policy_step: int) -> None:
        self.provisional_success_active = True
        self.provisional_success_ever = True
        if self.first_provisional_success_policy_step is None:
            self.first_provisional_success_policy_step = policy_step
        self.post_success_hold_check_count = 0
        self.current_post_success_hold_peak_lift_m = lift_m

    def _fail_post_success_hold(self, *, lift_m: float, policy_step: int) -> None:
        peak = self.current_post_success_hold_peak_lift_m
        drop = 0.0 if peak is None else max(0.0, peak - lift_m)
        self.post_success_max_downward_slip_m = max(
            self.post_success_max_downward_slip_m, drop
        )
        self.post_success_hold_failure_count += 1
        self.post_success_slip_ever = True
        if self.first_post_success_slip_policy_step is None:
            self.first_post_success_slip_policy_step = policy_step
        self.provisional_success_active = False
        self.post_success_hold_check_count = 0
        self.current_post_success_hold_peak_lift_m = None
        self.success_streak = 0

    def observe_post_success_hold(self, height_m: float) -> None:
        """Monitor every physics step while a provisional pick success is held."""

        if not self.provisional_success_active:
            return
        lift = self._record_height(height_m=height_m, policy_step=self.check_count)
        peak = self.current_post_success_hold_peak_lift_m
        if peak is None:
            peak = lift
        peak = max(peak, lift)
        self.current_post_success_hold_peak_lift_m = peak
        downward_slip = max(0.0, peak - lift)
        self.post_success_max_downward_slip_m = max(
            self.post_success_max_downward_slip_m, downward_slip
        )
        if lift < self.lift_threshold_m or downward_slip > self.max_post_success_drop_m:
            self._fail_post_success_hold(lift_m=lift, policy_step=self.check_count)

    def update(self, height_m: float, gripper_contact_count: int) -> dict[str, Any]:
        policy_step = self.check_count
        lift = self._record_height(height_m=height_m, policy_step=policy_step)
        max_lift = self.max_height_m - self.initial_height_m
        if (
            lift >= self.meaningful_lift_threshold_m
            and self.first_meaningful_lift_policy_step is None
        ):
            self.first_meaningful_lift_policy_step = policy_step
        instant = lift >= self.lift_threshold_m
        if instant and self.first_success_height_policy_step is None:
            self.first_success_height_policy_step = policy_step
        task_success = False
        if self.provisional_success_active:
            self.observe_post_success_hold(height_m)
            if self.provisional_success_active:
                self.post_success_hold_check_count += 1
                self.max_post_success_hold_check_count = max(
                    self.max_post_success_hold_check_count,
                    self.post_success_hold_check_count,
                )
                task_success = (
                    self.post_success_hold_check_count >= self.post_success_hold_checks
                )
        if not self.provisional_success_active and not task_success:
            self.success_streak = self.success_streak + 1 if instant else 0
            self.max_success_confirmation_count = max(
                self.max_success_confirmation_count, self.success_streak
            )
            if self.success_streak >= self.required_checks:
                if self.post_success_hold_checks == 0:
                    task_success = True
                else:
                    self._start_post_success_hold(lift_m=lift, policy_step=policy_step)
        contact_count = int(gripper_contact_count)
        if contact_count > 0:
            self.target_gripper_contact_ever = True
            self.target_gripper_contact_check_count += 1
            self.target_gripper_contact_count_total += contact_count
            if self.first_target_gripper_contact_policy_step is None:
                self.first_target_gripper_contact_policy_step = policy_step
            self.last_target_gripper_contact_policy_step = policy_step
        self.check_count += 1
        return {
            "success_type": "pick_lift",
            "target_initial_height_m": self.initial_height_m,
            "target_height_m": float(height_m),
            "target_max_height_m": self.max_height_m,
            "max_lift_m": max_lift,
            "final_lift_m": lift,
            "drop_from_peak_m": max(0.0, max_lift - lift),
            "lift_height_m": lift,
            "lift_threshold_m": self.lift_threshold_m,
            "meaningful_lift_diagnostic_threshold_m": self.meaningful_lift_threshold_m,
            "success_confirmation_count": self.success_streak,
            "max_success_confirmation_count": self.max_success_confirmation_count,
            "required_success_confirmation_count": self.required_checks,
            "post_success_hold_enabled": self.post_success_hold_checks > 0,
            "provisional_success_ever": self.provisional_success_ever,
            "provisional_success_active": self.provisional_success_active,
            "first_provisional_success_policy_step": self.first_provisional_success_policy_step,
            "post_success_hold_check_count": self.post_success_hold_check_count,
            "max_post_success_hold_check_count": self.max_post_success_hold_check_count,
            "required_post_success_hold_checks": self.post_success_hold_checks,
            "current_post_success_hold_peak_lift_m": self.current_post_success_hold_peak_lift_m,
            "post_success_max_downward_slip_m": self.post_success_max_downward_slip_m,
            "max_post_success_drop_m": self.max_post_success_drop_m,
            "post_success_slip_ever": self.post_success_slip_ever,
            "first_post_success_slip_policy_step": self.first_post_success_slip_policy_step,
            "post_success_hold_failure_count": self.post_success_hold_failure_count,
            "ever_meaningful_lift": self.first_meaningful_lift_policy_step is not None,
            "ever_reached_success_height": self.first_success_height_policy_step
            is not None,
            "first_meaningful_lift_policy_step": self.first_meaningful_lift_policy_step,
            "first_success_height_policy_step": self.first_success_height_policy_step,
            "peak_lift_policy_step": self.peak_lift_policy_step,
            "target_gripper_contact_ever": self.target_gripper_contact_ever,
            "target_gripper_contact_count": contact_count,
            "target_gripper_contact_check_count": self.target_gripper_contact_check_count,
            "target_gripper_contact_count_total": self.target_gripper_contact_count_total,
            "first_target_gripper_contact_policy_step": self.first_target_gripper_contact_policy_step,
            "last_target_gripper_contact_policy_step": self.last_target_gripper_contact_policy_step,
            "instant_success": instant,
            "task_success": task_success,
        }


@dataclass
class PlacementSuccess:
    protocol: FormalProtocol
    success_streak: int = 0
    check_count: int = 0
    max_success_confirmation_count: int = 0
    ever_release_confirmed: bool = False
    ever_containment_confirmed: bool = False
    ever_height_confirmed: bool = False
    ever_stability_confirmed: bool = False
    ever_instant_success: bool = False
    min_xy_distance_m: float = float("inf")
    max_containment_margin_m: float = float("-inf")
    min_linear_speed_after_release_mps: float | None = None
    min_angular_speed_after_release_radps: float | None = None

    def update(
        self,
        *,
        pepper_position_m: np.ndarray,
        ring_position_m: np.ndarray,
        pepper_linear_speed_mps: float,
        pepper_angular_speed_radps: float,
        pepper_gripper_distance_m: float,
        gripper_contact_count: int,
        gripper_raw: float,
        release_requested: bool,
    ) -> dict[str, Any]:
        xy_distance = float(np.linalg.norm(pepper_position_m[:2] - ring_position_m[:2]))
        xyz_distance = float(np.linalg.norm(pepper_position_m - ring_position_m))
        height_above_table = float(pepper_position_m[2] - TABLE_TOP_Z)
        limit = self.protocol.placement_max_center_distance_m
        containment_margin = limit - xy_distance
        release_confirmed = bool(
            release_requested
            and gripper_raw >= self.protocol.placement_release_gripper_raw
            and gripper_contact_count == 0
            and pepper_gripper_distance_m
            >= self.protocol.placement_min_gripper_distance_m
        )
        contained = xy_distance <= limit
        height_ok = (
            self.protocol.placement_min_height_above_table_m
            <= height_above_table
            <= self.protocol.placement_max_height_above_table_m
        )
        stable = bool(
            np.isfinite(pepper_linear_speed_mps)
            and np.isfinite(pepper_angular_speed_radps)
            and pepper_linear_speed_mps <= self.protocol.placement_max_linear_speed_mps
            and pepper_angular_speed_radps
            <= self.protocol.placement_max_angular_speed_radps
        )
        instant = release_confirmed and contained and height_ok and stable
        self.success_streak = self.success_streak + 1 if instant else 0
        self.max_success_confirmation_count = max(
            self.max_success_confirmation_count, self.success_streak
        )
        self.ever_release_confirmed = self.ever_release_confirmed or release_confirmed
        self.ever_containment_confirmed = self.ever_containment_confirmed or contained
        self.ever_height_confirmed = self.ever_height_confirmed or height_ok
        self.ever_stability_confirmed = self.ever_stability_confirmed or stable
        self.ever_instant_success = self.ever_instant_success or instant
        self.min_xy_distance_m = min(self.min_xy_distance_m, xy_distance)
        self.max_containment_margin_m = max(
            self.max_containment_margin_m, containment_margin
        )
        if release_confirmed:
            if self.min_linear_speed_after_release_mps is None:
                self.min_linear_speed_after_release_mps = float(pepper_linear_speed_mps)
                self.min_angular_speed_after_release_radps = float(
                    pepper_angular_speed_radps
                )
            else:
                self.min_linear_speed_after_release_mps = min(
                    self.min_linear_speed_after_release_mps,
                    float(pepper_linear_speed_mps),
                )
                self.min_angular_speed_after_release_radps = min(
                    self.min_angular_speed_after_release_radps,
                    float(pepper_angular_speed_radps),
                )
        self.check_count += 1
        return {
            "success_type": "stable_place_in_ring",
            "pepper_ring_xy_distance_m": xy_distance,
            "pepper_ring_xyz_distance_m": xyz_distance,
            "pepper_height_above_table_m": height_above_table,
            "pepper_linear_speed_mps": float(pepper_linear_speed_mps),
            "pepper_angular_speed_radps": float(pepper_angular_speed_radps),
            "pepper_gripper_distance_m": float(pepper_gripper_distance_m),
            "gripper_contact_count": int(gripper_contact_count),
            "gripper_raw": float(gripper_raw),
            "release_requested": bool(release_requested),
            "release_confirmed": release_confirmed,
            "containment_limit_m": limit,
            "containment_margin_m": containment_margin,
            "containment_confirmed": contained,
            "height_confirmed": height_ok,
            "stability_confirmed": stable,
            "success_confirmation_count": self.success_streak,
            "max_success_confirmation_count": self.max_success_confirmation_count,
            "required_success_confirmation_count": self.protocol.placement_success_checks,
            "ever_release_confirmed": self.ever_release_confirmed,
            "ever_containment_confirmed": self.ever_containment_confirmed,
            "ever_height_confirmed": self.ever_height_confirmed,
            "ever_stability_confirmed": self.ever_stability_confirmed,
            "ever_instant_success": self.ever_instant_success,
            "min_xy_distance_m": self.min_xy_distance_m,
            "max_containment_margin_m": self.max_containment_margin_m,
            "min_linear_speed_after_release_mps": self.min_linear_speed_after_release_mps,
            "min_angular_speed_after_release_radps": self.min_angular_speed_after_release_radps,
            "placement_min_height_above_table_m": self.protocol.placement_min_height_above_table_m,
            "placement_max_height_above_table_m": self.protocol.placement_max_height_above_table_m,
            "placement_max_linear_speed_mps": self.protocol.placement_max_linear_speed_mps,
            "placement_max_angular_speed_radps": self.protocol.placement_max_angular_speed_radps,
            "placement_min_gripper_distance_m": self.protocol.placement_min_gripper_distance_m,
            "placement_release_gripper_raw": self.protocol.placement_release_gripper_raw,
            "instant_success": instant,
            "task_success": self.success_streak
            >= self.protocol.placement_success_checks,
        }


class FormalTaskEvaluator:
    """Task-specific scoring using simulator state, never policy inputs."""

    def __init__(
        self,
        runtime: TaskSceneRuntime,
        protocol: FormalProtocol,
        camera_config: dict[str, Any],
    ) -> None:
        self.runtime = runtime
        self.protocol = protocol
        self.camera_config = camera_config
        success_type = str(runtime.spec["success"]["type"])
        if success_type == "lift":
            self.metric: PickSuccess | PlacementSuccess = PickSuccess(
                initial_height_m=float(runtime.initial_target_z),
                lift_threshold_m=protocol.pick_lift_height_m,
                meaningful_lift_threshold_m=protocol.pick_meaningful_lift_diagnostic_m,
                required_checks=protocol.pick_success_checks,
                post_success_hold_checks=protocol.pick_post_success_hold_checks,
                max_post_success_drop_m=protocol.pick_max_post_success_drop_m,
            )
        elif success_type == "place_in_ring":
            self.metric = PlacementSuccess(protocol)
        else:
            raise ValueError(f"Unsupported formal task success type: {success_type}")

    def observe_post_success_hold(self) -> None:
        """Continuously reject a provisional pick that slips during its hold window."""

        if not isinstance(self.metric, PickSuccess):
            return
        target = _body_position(
            self.runtime.model, self.runtime.data, self.runtime.target_body
        )
        self.metric.observe_post_success_hold(float(target[2]))

    def update(self) -> dict[str, Any]:
        model, data, target_body = (
            self.runtime.model,
            self.runtime.data,
            self.runtime.target_body,
        )
        target = _body_position(model, data, target_body)
        if isinstance(self.metric, PickSuccess):
            collision = collision_diagnostics(model, data)
            return {
                "target_body": target_body,
                "target_position_m": target.tolist(),
                **self.metric.update(
                    float(target[2]),
                    target_gripper_contact_count(collision, target_body),
                ),
            }
        ring_body = str(self.runtime.spec["success"]["ring_body"])
        ring = _body_position(model, data, ring_body)
        collision = collision_diagnostics(model, data)
        linear_speed, angular_speed = _body_velocity(model, data, target_body)
        tcp = _tcp_position(model, data)
        gripper_raw = float(get_robot_state(model, data, self.camera_config)[6])
        return {
            "target_body": target_body,
            "target_position_m": target.tolist(),
            "ring_position_m": ring.tolist(),
            **self.metric.update(
                pepper_position_m=target,
                ring_position_m=ring,
                pepper_linear_speed_mps=linear_speed,
                pepper_angular_speed_radps=angular_speed,
                pepper_gripper_distance_m=float(np.linalg.norm(target - tcp)),
                gripper_contact_count=target_gripper_contact_count(
                    collision, target_body
                ),
                gripper_raw=gripper_raw,
                release_requested=self.runtime.released,
            ),
        }
