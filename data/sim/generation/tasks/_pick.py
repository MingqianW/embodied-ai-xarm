"""Shared implementation for task-owned default Pick generators."""

from __future__ import annotations

from dataclasses import asdict

from data.sim.generation.core.generator import ControllerEpisodeGenerator, GeneratorContext
from data.sim.generation.oracle import OracleConfig, ScriptedOracleController


def create_scripted_pick(context: GeneratorContext, *, generator_id: str) -> ControllerEpisodeGenerator:
    task = context.task
    config = context.pipeline_config
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
    return ControllerEpisodeGenerator(
        ScriptedOracleController(context.environment, OracleConfig(**values)),
        generator_id=generator_id,
        kind="pick",
    )
