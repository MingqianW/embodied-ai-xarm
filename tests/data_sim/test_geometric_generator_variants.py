from __future__ import annotations

from pathlib import Path

import pytest

from data.sim.generation.acceptance import simulation_is_finite, update_task_success
from data.sim.generation.config import load_pipeline_config
from data.sim.generation.core.generator import GeneratorContext
from data.sim.generation.core.registry import resolve_generator
from simulation.environment import MuJoCoEnvironment


CONFIG_PATH = Path("configs/data/sim/generation/clean_multitask_stable_v3.yaml")
PICK_TASK_IDS = (
    "red_block",
    "blue_block",
    "red_pepper",
    "smallest_block",
    "largest_block",
)
PICK_VARIANTS = (
    (
        "scripted_pick_side_approach_v1",
        {"pregrasp_offset_xy_m": [0.0, 0.025]},
    ),
    (
        "scripted_pick_yaw15_v1",
        {"tcp_yaw_offset_deg": 15.0},
    ),
    (
        "scripted_pick_waypoint_lift_v1",
        {
            "approach_waypoint_offset_xy_m": [-0.025, 0.02],
            "lift_offset_xy_m": [0.01, -0.01],
        },
    ),
)
PLACE_VARIANTS = (
    (
        "direct_place_left_approach_v1",
        {"preplace_offset_xy_m": [-0.025, 0.02]},
    ),
    (
        "direct_place_right_approach_v1",
        {"preplace_offset_xy_m": [0.025, -0.02]},
    ),
)


VARIANTS = tuple(
    (task_id, generator_id, expected_geometry)
    for task_id in PICK_TASK_IDS
    for generator_id, expected_geometry in PICK_VARIANTS
) + tuple(
    ("place_red_pepper_in_ring", generator_id, expected_geometry)
    for generator_id, expected_geometry in PLACE_VARIANTS
)


@pytest.mark.parametrize(("task_id", "generator_id", "expected_geometry"), VARIANTS)
def test_geometric_variants_record_and_complete(
    task_id: str,
    generator_id: str,
    expected_geometry: dict[str, object],
) -> None:
    config = load_pipeline_config(CONFIG_PATH)
    task = next(item for item in config.tasks if item.task_id == task_id)
    seed = (
        task.base_seed + 3
        if task_id == "place_red_pepper_in_ring"
        else task.base_seed
    )
    with MuJoCoEnvironment(
        task=task.task_id,
        prompt=task.prompt,
        settle_steps=0,
        object_xy_range=config.object_xy_range_m,
        object_yaw_range_deg=config.object_yaw_range_deg,
        joint_noise=config.joint_noise_rad,
    ) as environment:
        environment.reset(seed=seed, build_policy_observation=False)
        generator = resolve_generator(task.task_id, generator_id)(
            GeneratorContext(
                environment=environment,
                pipeline_config=config,
                task=task,
                requested_episode_index=0,
                retry_index=0,
                seed=seed,
            )
        )
        geometry = generator.plan_metadata()["geometry"]
        for key, value in expected_geometry.items():
            assert geometry[key] == value

        metrics = environment.task_runtime.metrics()
        while not generator.terminal:
            action = generator.next_action()
            if action is None:
                break
            environment.apply_action(action)
            environment.step_physics(1.0 / config.action_hz)
            metrics = update_task_success(environment)
            generator.notify_post_step(
                task_metrics=metrics,
                collision=environment.safety_diagnostics()["collision"],
                simulation_finite=simulation_is_finite(environment),
            )

        assert generator.accepted()
        assert generator.failure_reason is None
