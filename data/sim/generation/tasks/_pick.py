"""Shared implementation for task-owned default Pick generators."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Mapping

from data.sim.generation.core.generator import ControllerEpisodeGenerator, GeneratorContext
from data.sim.generation.oracle import OracleConfig, ScriptedOracleController


PICK_VARIANT_OVERRIDE_FIELDS = frozenset(
    {
        "max_joint_step_rad",
        "lift_max_joint_step_rad",
        "gripper_closing_rate_raw_per_s",
        "gripper_opening_rate_raw_per_s",
        "pregrasp_clearance_from_object_m",
        "lift_clearance_from_object_m",
        "pregrasp_offset_xy_m",
        "approach_waypoint_offset_xy_m",
        "lift_offset_xy_m",
        "tcp_yaw_offset_deg",
        "hold_steps",
        "max_action_steps",
        "approach_speed_scale",
        "lift_speed_scale",
    }
)


PICK_GEOMETRY_PROFILES: dict[str, dict[str, Any]] = {
    "side_approach_v1": {
        "pregrasp_offset_xy_m": (0.0, 0.025),
        "max_action_steps": 280,
    },
    "opposite_side_approach_v1": {
        "pregrasp_offset_xy_m": (0.0, -0.025),
        "max_action_steps": 280,
    },
    "yaw15_v1": {
        "tcp_yaw_offset_deg": 15.0,
        "max_action_steps": 280,
    },
    "yaw_minus15_v1": {
        "tcp_yaw_offset_deg": -15.0,
        "max_action_steps": 280,
    },
    "diagonal_approach_v1": {
        "pregrasp_offset_xy_m": (0.02, -0.02),
        "max_action_steps": 280,
    },
    "waypoint_lift_v1": {
        "approach_waypoint_offset_xy_m": (-0.025, 0.02),
        "lift_offset_xy_m": (0.01, -0.01),
        "max_action_steps": 320,
    },
    "opposite_waypoint_lift_v1": {
        "approach_waypoint_offset_xy_m": (0.025, -0.02),
        "lift_offset_xy_m": (-0.01, 0.01),
        "max_action_steps": 320,
    },
}


def create_scripted_pick(
    context: GeneratorContext,
    *,
    generator_id: str,
    oracle_overrides: Mapping[str, Any] | None = None,
) -> ControllerEpisodeGenerator:
    """Create a task-owned Pick variant over the shared oracle state machine.

    Task identity, gripper contract, action cadence, and stable-grasp
    acceptance remain centrally owned by the generation plan. Variants may
    alter only documented trajectory timing and geometric approach parameters.
    """

    task = context.task
    config = context.pipeline_config
    overrides = dict(oracle_overrides or {})
    overrides.update(context.trajectory_parameters or {})
    unknown = set(overrides) - PICK_VARIANT_OVERRIDE_FIELDS
    if unknown:
        raise ValueError(
            f"Unsupported Pick variant overrides: {sorted(unknown)}"
        )
    values = asdict(
        OracleConfig(
            task=task.task_id,
            action_dt_s=config.pick.action_dt_s,
            closed_gripper_raw=float(task.closed_gripper_raw),
            grasp_tcp_offset_from_object_m=float(task.grasp_tcp_offset_from_object_m),
        )
    )
    values.update(
        {
            "verify_steps": config.pick.steps,
            "verification_entry_lift_height_m": config.pick.entry_lift_height_m,
            "verification_minimum_lift_height_m": config.pick.minimum_lift_height_m,
            "maximum_relative_downward_slip_m": config.pick.maximum_relative_downward_slip_m,
            "maximum_final_relative_downward_slip_m": config.pick.maximum_final_relative_downward_slip_m,
            "maximum_final_downward_speed_mps": config.pick.maximum_final_downward_speed_mps,
            "maximum_grasp_region_delta_m": config.pick.maximum_grasp_region_delta_m,
            "velocity_fit_samples": config.pick.velocity_fit_samples,
        }
    )
    # Speed scales are expressed in the profile but map to controller joint increments.
    for scale, target in (("approach_speed_scale", "max_joint_step_rad"), ("lift_speed_scale", "lift_max_joint_step_rad")):
        if scale in overrides:
            overrides[target] = float(values[target]) * float(overrides.pop(scale))
    values.update(overrides)
    return ControllerEpisodeGenerator(
        ScriptedOracleController(context.environment, OracleConfig(**values)),
        generator_id=generator_id,
        kind="pick",
        trajectory_metadata={
            "seed": context.seed,
            "scene_seed": context.scene_seed,
            "resolved_parameters": dict(context.trajectory_parameters or {}),
        },
    )


def create_geometric_pick(
    context: GeneratorContext,
    *,
    generator_id: str,
    profile: str,
) -> ControllerEpisodeGenerator:
    """Create a registered geometric Pick profile for a task-owned factory."""

    try:
        overrides = PICK_GEOMETRY_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown Pick geometry profile: {profile!r}") from exc
    sampled = dict(context.trajectory_parameters or {})
    # A per-member distribution may vary a member's magnitude, but it cannot
    # cross the geometric side/yaw that gives that member its identity.
    for name in ("pregrasp_offset_xy_m", "approach_waypoint_offset_xy_m", "lift_offset_xy_m", "tcp_yaw_offset_deg"):
        if name not in overrides or name not in sampled:
            continue
        baseline, candidate = overrides[name], sampled[name]
        baseline_values = (baseline,) if isinstance(baseline, (int, float)) else baseline
        candidate_values = (candidate,) if isinstance(candidate, (int, float)) else candidate
        if len(baseline_values) != len(candidate_values) or any(
            (abs(float(base)) < 1e-12 and abs(float(value)) > 1e-12)
            or (abs(float(base)) >= 1e-12 and float(base) * float(value) <= 0.0)
            for base, value in zip(baseline_values, candidate_values, strict=True)
        ):
            raise ValueError(f"{generator_id} trajectory parameters crossed its fixed motion-family direction")
    return create_scripted_pick(
        replace(context, trajectory_parameters=sampled),
        generator_id=generator_id,
        oracle_overrides=overrides,
    )
