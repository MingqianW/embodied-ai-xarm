"""Original exhausted paired member must pass without changing acceptance."""
import os
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytestmark = pytest.mark.skipif(not os.environ.get("SLURM_JOB_ID"), reason="Physical/render regression requires Slurm allocation")

from data.sim.generation.acceptance import simulation_is_finite, update_task_success
from data.sim.generation.config import load_pipeline_config
from data.sim.generation.core.generator import GeneratorContext
from data.sim.generation.core.registry import create_generator
from data.sim.generation.trajectory import scene_seed as paired_scene_seed, trajectory_seed as paired_trajectory_seed, resolve_parameters
from simulation.environment import MuJoCoEnvironment


@pytest.mark.parametrize("retry_index", range(4))
def test_exhausted_right_front_place_trajectory_settles_with_original_check(retry_index):
    path = Path(__file__).resolve().parents[2] / "configs/data/sim/generation/clean_multitask_paired_trajectory_v1.yaml"
    config = load_pipeline_config(path)
    task = next(task for task in config.tasks if task.task_id == "place_red_pepper_in_ring")
    member = "direct_place_right_front_approach_v1"
    scene_seed = paired_scene_seed(task.base_seed, task.task_id, 1)
    trajectory_seed = paired_trajectory_seed(task.base_seed, task.task_id, 1, member, retry_index)
    parameters = resolve_parameters(config.trajectory_profile, task_id=task.task_id, member_id=member, seed=trajectory_seed)
    with MuJoCoEnvironment(task=task.task_id, prompt=task.prompt, camera_config_path=config.camera_config,
                          task_scene_config_path=config.task_scene_config, layout_profile=config.layout_profile,
                          object_xy_range=config.object_xy_range_m, object_yaw_range_deg=config.object_yaw_range_deg,
                          joint_noise=config.joint_noise_rad, scene_variant="clean") as environment:
        environment.reset(seed=scene_seed, build_policy_observation=False)
        generator = create_generator(GeneratorContext(environment, config, task, 11, retry_index,
                                                      trajectory_seed, scene_seed, parameters))
        assert generator.initialization.success
        assert generator.initialization.metadata["initial_grasp_validation_steps_executed"] == 10
        assert generator.initialization.metadata["release_uses_held_body_swap"] is False
        while not generator.terminal:
            assert environment.task_runtime.free_place_grasp
            assert environment.task_runtime.active_target_body == "red_pepper"
            action = generator.next_action()
            if action is None:
                break
            environment.apply_action(action)
            environment.step_physics(0.1)
            generator.notify_post_step(task_metrics=update_task_success(environment),
                                       collision=environment.safety_diagnostics()["collision"],
                                       simulation_finite=simulation_is_finite(environment))
        stability = generator.stability_metadata()
        assert generator.accepted(), stability
        assert stability["place_verification_steps_executed"] == 20
        assert stability["place_verification_duration_s"] == pytest.approx(2.0)
        assert stability["estimated_final_object_speed_mps"] <= config.place.maximum_final_speed_mps
