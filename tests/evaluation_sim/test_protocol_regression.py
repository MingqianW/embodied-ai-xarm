from __future__ import annotations

from evaluation.sim.config import load_protocol


def test_six_task_formal_v2_semantics_are_frozen() -> None:
    protocol = load_protocol()
    assert [(task.task_id, task.prompt) for task in protocol.tasks] == [
        ("red_pepper", "pick up the red pepper"),
        ("blue_block", "pick up the blue block"),
        ("red_block", "pick up the red block"),
        ("smallest_block", "pick up the smallest block"),
        ("largest_block", "pick up the largest block"),
        ("place_red_pepper_in_ring", "place the red pepper in the ring"),
    ]
    assert protocol.seeds == tuple(range(50000, 50020))
    assert (
        protocol.execute_chunk_steps,
        protocol.policy_action_horizon,
        protocol.max_policy_steps,
        protocol.control_duration_s,
        protocol.expected_physics_timestep_s,
    ) == (5, 10, 50, 0.1, 0.002)
    assert (
        protocol.object_xy_range_m,
        protocol.object_yaw_range_deg,
        protocol.joint_noise_rad,
    ) == (0.03, 15.0, 0.01)
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
    assert protocol.rng_salt == "xarm-pi05-formal-evaluation-v1"

