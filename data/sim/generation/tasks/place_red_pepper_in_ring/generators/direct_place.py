"""Default direct-place generator, including the historic held-pepper check."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import mujoco
import numpy as np

from data.sim.generation.acceptance import simulation_is_finite
from data.sim.generation.core.generator import (
    ControllerEpisodeGenerator,
    GeneratorContext,
    GeneratorInitialization,
    RejectedEpisodeGenerator,
)
from data.sim.generation.oracle import PlaceOracleConfig, PlaceRedPepperOracleController
from data.sim.generation.stability import StabilitySample, evaluate_place_initial_grasp
from data.sim.generation.state_conversion import policy_state_from_mujoco
from simulation.observation.cameras import render_rgb
from simulation.physics.collision import target_gripper_contact_count
from simulation.scene import TABLE_TOP_Z


PLACE_VARIANT_OVERRIDE_FIELDS = frozenset(
    {
        "preplace_offset_xy_m",
        "preplace_pepper_height_m",
        "max_action_steps",
    }
)


def _body_pose(environment, body_name: str):
    body_id = mujoco.mj_name2id(environment.context.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return (
        np.asarray(environment.context.data.xpos[body_id], dtype=np.float64).copy(),
        np.asarray(environment.context.data.xquat[body_id], dtype=np.float64).copy(),
    )


def _tcp_pose(environment):
    site_id = mujoco.mj_name2id(environment.context.model, mujoco.mjtObj.mjOBJ_SITE, "tool_center_point")
    rotation = np.asarray(environment.context.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    return np.asarray(environment.context.data.site_xpos[site_id], dtype=np.float64).copy(), quaternion


def _table_contact(collision, target_body: str) -> bool:
    return any(
        (row.get("body1") == target_body or row.get("body2") == target_body)
        and (row.get("geom1") == "table" or row.get("geom2") == "table")
        for row in collision.get("contacts") or ()
    )


def _validate_initial_grasp(context: GeneratorContext) -> GeneratorInitialization:
    environment = context.environment
    config = context.pipeline_config
    runtime = environment.task_runtime
    assert runtime is not None
    initial_object, initial_object_quaternion = _body_pose(environment, runtime.active_target_body)
    initial_tcp, initial_tcp_quaternion = _tcp_pose(environment)
    ring_position, _ = _body_pose(environment, str(runtime.spec["success"]["ring_body"]))
    action = policy_state_from_mujoco(environment.context.model, environment.context.data).astype(np.float32)
    initial_arm_target = np.asarray(environment.initial_conditions["initial_joint_positions"], dtype=np.float32)
    if initial_arm_target.shape != (6,) or not np.isfinite(initial_arm_target).all():
        raise ValueError("Place initial arm target must be finite with shape (6,)")
    action[:6] = initial_arm_target
    action[6] = float(runtime.spec["initial_gripper_raw"])
    samples: list[StabilitySample] = []
    frames = {"realsense_0": [], "realsense_1": [], "realsense_2": []}
    for _ in range(config.place_initial.steps):
        environment.apply_action(action)
        environment.step_physics(config.place_initial.action_dt_s)
        collision = environment.safety_diagnostics()["collision"]
        object_position, _ = _body_pose(environment, runtime.active_target_body)
        tcp_position, _ = _tcp_pose(environment)
        samples.append(
            StabilitySample(
                simulation_time_s=float(environment.context.data.time),
                object_position_m=tuple(float(value) for value in object_position),
                tcp_position_m=tuple(float(value) for value in tcp_position),
                finite=bool(simulation_is_finite(environment) and np.isfinite(object_position).all() and np.isfinite(tcp_position).all()),
                table_contact=_table_contact(collision, runtime.active_target_body),
                forbidden_collision=bool(collision.get("forbidden")),
                inside_ring=float(np.linalg.norm(object_position[:2] - ring_position[:2])) <= config.place.ring_radius_m,
                gripper_contact_count=target_gripper_contact_count(collision, runtime.active_target_body),
            )
        )
        for raw_name, camera_name in (("realsense_0", "base_camera"), ("realsense_1", "wrist_camera"), ("realsense_2", "overview_camera")):
            frames[raw_name].append(render_rgb(environment.context.renderer, environment.context.data, camera_name).copy())
    metadata = evaluate_place_initial_grasp(
        samples,
        config=config.place_initial,
        table_top_z_m=TABLE_TOP_Z,
        initial_object_position_m=initial_object,
        initial_tcp_position_m=initial_tcp,
    )
    metadata.update(
        {
            "initial_tcp_position_m": initial_tcp.tolist(),
            "initial_tcp_orientation": initial_tcp_quaternion.tolist(),
            "initial_pepper_position_m": initial_object.tolist(),
            "initial_pepper_orientation": initial_object_quaternion.tolist(),
            "initial_pepper_to_tcp_transform": {
                "translation_m": (initial_object - initial_tcp).tolist(),
                "configured_translation_m": list(config.place_initial.tcp_to_pepper_translation_m),
                "configured_quaternion_wxyz": list(config.place_initial.tcp_to_pepper_quaternion_wxyz),
            },
            "initial_gripper_raw": float(runtime.spec["initial_gripper_raw"]),
            "initial_arm_hold_target": initial_arm_target.tolist(),
            "initialization_frames_recorded": 0,
            "object_identity": runtime.active_target_body,
            "released_object_identity": runtime.target_body,
            "release_uses_held_body_swap": True,
            "permanent_attachment": False,
        }
    )
    return GeneratorInitialization(
        success=bool(metadata["initial_grasp_success"]),
        metadata=metadata,
        diagnostic_frames=frames,
        failure_reason=None if metadata["initial_grasp_success"] else str(metadata["initial_grasp_failure_reason"]),
    )


def _create(
    context: GeneratorContext,
    *,
    generator_id: str,
    oracle_overrides: Mapping[str, Any] | None = None,
):
    initialization = _validate_initial_grasp(context)
    if not initialization.success:
        return RejectedEpisodeGenerator(
            generator_id=generator_id,
            initialization=initialization,
        )
    config = context.pipeline_config.place
    overrides = dict(oracle_overrides or {})
    unknown = set(overrides) - PLACE_VARIANT_OVERRIDE_FIELDS
    if unknown:
        raise ValueError(f"Unsupported Place variant overrides: {sorted(unknown)}")
    values = asdict(
        PlaceOracleConfig(
            action_dt_s=config.action_dt_s,
            verify_steps=config.steps,
            ring_radius_m=config.ring_radius_m,
            maximum_height_above_table_m=config.maximum_height_above_table_m,
            maximum_final_speed_mps=config.maximum_final_speed_mps,
            velocity_fit_samples=config.velocity_fit_samples,
        )
    )
    values.update(overrides)
    controller = PlaceRedPepperOracleController(
        context.environment,
        PlaceOracleConfig(**values),
    )
    return ControllerEpisodeGenerator(
        controller,
        generator_id=generator_id,
        kind="place",
        initialization=initialization,
    )


def create(context: GeneratorContext):
    return _create(context, generator_id="direct_place")


def create_left_approach_v1(context: GeneratorContext):
    """Approach the ring from a left-front preplace waypoint before centering."""

    return _create(
        context,
        generator_id="direct_place_left_approach_v1",
        oracle_overrides={
            "preplace_offset_xy_m": (-0.025, 0.02),
            "preplace_pepper_height_m": 0.22,
            "max_action_steps": 280,
        },
    )


def create_right_approach_v1(context: GeneratorContext):
    """Approach the ring from a right-rear preplace waypoint before centering."""

    return _create(
        context,
        generator_id="direct_place_right_approach_v1",
        oracle_overrides={
            "preplace_offset_xy_m": (0.025, -0.02),
            "preplace_pepper_height_m": 0.22,
            "max_action_steps": 280,
        },
    )
