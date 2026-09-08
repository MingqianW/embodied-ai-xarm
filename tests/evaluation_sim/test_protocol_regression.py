from __future__ import annotations

from pathlib import Path

from data.sim.generation.config import load_pipeline_config
from evaluation.sim.config import load_protocol
from simulation.runtime import initialize_scene
from simulation.runtime import load_simulation
from simulation.scene import configure_task_scene


def test_six_task_formal_v3_uses_the_generation_scene_profile() -> None:
    protocol = load_protocol()
    assert [(task.task_id, task.prompt) for task in protocol.tasks] == [
        ("red_pepper", "pick up the red pepper"),
        ("blue_block", "pick up the blue block"),
        ("red_block", "pick up the red block"),
        ("smallest_block", "pick up the smallest block"),
        ("largest_block", "pick up the largest block"),
        ("place_red_pepper_in_ring", "place the red pepper in the ring"),
    ]
    assert protocol.seeds == tuple(range(50018, 50038))
    assert (
        protocol.execute_chunk_steps,
        protocol.policy_action_horizon,
        protocol.max_policy_steps,
        protocol.control_duration_s,
        protocol.expected_physics_timestep_s,
    ) == (5, 10, 50, 0.1, 0.002)
    assert protocol.scene_profile == "clean_wide_v4"
    assert (
        protocol.object_xy_range_m,
        protocol.object_yaw_range_deg,
        protocol.joint_noise_rad,
        protocol.layout_profile,
    ) == (0.10, 10.0, 0.005, "wide_independent_workspace_v1")
    assert (
        protocol.pick_lift_height_m,
        protocol.pick_meaningful_lift_diagnostic_m,
        protocol.pick_success_checks,
        protocol.pick_post_success_hold_checks,
        protocol.pick_max_post_success_drop_m,
    ) == (0.05, 0.005, 3, 3, 0.005)
    assert (
        protocol.placement_initial_validation_checks,
        protocol.placement_initial_validation_dt_s,
        protocol.placement_initial_max_relative_drift_m,
        protocol.placement_initial_min_height_above_table_m,
        protocol.placement_initial_min_gripper_contacts,
    ) == (10, 0.1, 0.005, 0.04, 1)
    assert (
        protocol.placement_ring_inner_radius_m,
        protocol.placement_pepper_effective_radius_m,
        protocol.placement_containment_tolerance_m,
        protocol.placement_min_height_above_table_m,
        protocol.placement_max_height_above_table_m,
        protocol.placement_max_linear_speed_mps,
        protocol.placement_max_angular_speed_radps,
        protocol.placement_min_gripper_distance_m,
        protocol.placement_release_gripper_raw,
        protocol.placement_success_checks,
    ) == (0.053, 0.022, 0.002, 0.005, 0.04, 0.01, 0.25, 0.045, 650.0, 3)
    assert protocol.video_policy == "category_representative"
    assert protocol.representatives_per_category == 1
    assert protocol.periodic_video_every == 5
    assert protocol.fail_on_invalid is True
    assert protocol.rng_salt == "xarm-pi05-formal-evaluation-v3"


def test_formal_v3_resolves_the_same_profile_as_v4_generation() -> None:
    generation = load_pipeline_config(
        Path("configs/data/sim/generation/clean_multitask_stable_v4_10x_real.yaml")
    )
    evaluation = load_protocol()
    assert (
        evaluation.scene_profile,
        evaluation.object_xy_range_m,
        evaluation.object_yaw_range_deg,
        evaluation.joint_noise_rad,
        evaluation.layout_profile,
    ) == (
        generation.scene_profile,
        generation.object_xy_range_m,
        generation.object_yaw_range_deg,
        generation.joint_noise_rad,
        generation.layout_profile,
    )


def test_formal_v3_fixed_seeds_have_valid_wide_profile_resets() -> None:
    protocol = load_protocol()
    for task in protocol.tasks:
        for seed in protocol.seeds:
            context = load_simulation(
                protocol.robot_xml_path, protocol.camera_config_path
            )
            try:
                initialize_scene(context.model, context.data, settle_steps=0)
                configure_task_scene(
                    context.model,
                    context.data,
                    task=task.task_id,
                    seed=seed,
                    object_xy_range=protocol.object_xy_range_m,
                    object_yaw_range_deg=protocol.object_yaw_range_deg,
                    joint_noise=protocol.joint_noise_rad,
                    layout_profile=protocol.layout_profile,
                    settle_steps=500,
                    config_path=protocol.task_scene_config_path,
                )
            finally:
                context.close()
