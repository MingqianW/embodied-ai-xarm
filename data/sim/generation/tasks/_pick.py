"""Shared implementation for task-owned default Pick generators."""

from __future__ import annotations

from dataclasses import asdict
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
    }
)


PICK_GEOMETRY_PROFILES: dict[str, dict[str, Any]] = {
    "side_approach_v1": {
        "pregrasp_offset_xy_m": (0.0, 0.025),
        "max_action_steps": 280,
    },
    "yaw15_v1": {
        "tcp_yaw_offset_deg": 15.0,
        "max_action_steps": 280,
    },
    "waypoint_lift_v1": {
        "approach_waypoint_offset_xy_m": (-0.025, 0.02),
        "lift_offset_xy_m": (0.01, -0.01),
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
    values.update(overrides)
    return ControllerEpisodeGenerator(
        ScriptedOracleController(context.environment, OracleConfig(**values)),
        generator_id=generator_id,
        kind="pick",
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
    return create_scripted_pick(
        context,
        generator_id=generator_id,
        oracle_overrides=overrides,
    )
