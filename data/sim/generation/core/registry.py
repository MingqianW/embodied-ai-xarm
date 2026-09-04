"""Explicit task/generator registry; task identity remains in data.common."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from data.common.task_identity import TASK_BY_ID


GeneratorFactory = Callable[[Any], Any]

# Values are lazy import targets so config parsing does not initialize MuJoCo.
_REGISTRY: dict[str, dict[str, tuple[str, bool]]] = {
    "red_block": {
        "scripted_pick": (
            "data.sim.generation.tasks.red_block.generators.scripted_pick:create",
            True,
        ),
        "scripted_pick_side_approach_v1": (
            "data.sim.generation.tasks.red_block.generators.geometric:create_side_approach",
            False,
        ),
        "scripted_pick_yaw15_v1": (
            "data.sim.generation.tasks.red_block.generators.geometric:create_yaw15",
            False,
        ),
        "scripted_pick_waypoint_lift_v1": (
            "data.sim.generation.tasks.red_block.generators.geometric:create_waypoint_lift",
            False,
        ),
    },
    "blue_block": {
        "scripted_pick": (
            "data.sim.generation.tasks.blue_block.generators.scripted_pick:create",
            True,
        ),
        "scripted_pick_side_approach_v1": (
            "data.sim.generation.tasks.blue_block.generators.geometric:create_side_approach",
            False,
        ),
        "scripted_pick_yaw15_v1": (
            "data.sim.generation.tasks.blue_block.generators.geometric:create_yaw15",
            False,
        ),
        "scripted_pick_waypoint_lift_v1": (
            "data.sim.generation.tasks.blue_block.generators.geometric:create_waypoint_lift",
            False,
        ),
    },
    "red_pepper": {
        "scripted_pick": (
            "data.sim.generation.tasks.red_pepper.generators.scripted_pick:create",
            True,
        ),
        "scripted_pick_side_approach_v1": (
            "data.sim.generation.tasks.red_pepper.generators.geometric:create_side_approach",
            False,
        ),
        "scripted_pick_yaw15_v1": (
            "data.sim.generation.tasks.red_pepper.generators.geometric:create_yaw15",
            False,
        ),
        "scripted_pick_waypoint_lift_v1": (
            "data.sim.generation.tasks.red_pepper.generators.geometric:create_waypoint_lift",
            False,
        ),
    },
    "smallest_block": {
        "scripted_pick": (
            "data.sim.generation.tasks.smallest_block.generators.scripted_pick:create",
            True,
        ),
        "scripted_pick_side_approach_v1": (
            "data.sim.generation.tasks.smallest_block.generators.geometric:create_side_approach",
            False,
        ),
        "scripted_pick_yaw15_v1": (
            "data.sim.generation.tasks.smallest_block.generators.geometric:create_yaw15",
            False,
        ),
        "scripted_pick_waypoint_lift_v1": (
            "data.sim.generation.tasks.smallest_block.generators.geometric:create_waypoint_lift",
            False,
        ),
    },
    "largest_block": {
        "scripted_pick": (
            "data.sim.generation.tasks.largest_block.generators.scripted_pick:create",
            True,
        ),
        "scripted_pick_side_approach_v1": (
            "data.sim.generation.tasks.largest_block.generators.geometric:create_side_approach",
            False,
        ),
        "scripted_pick_yaw15_v1": (
            "data.sim.generation.tasks.largest_block.generators.geometric:create_yaw15",
            False,
        ),
        "scripted_pick_waypoint_lift_v1": (
            "data.sim.generation.tasks.largest_block.generators.geometric:create_waypoint_lift",
            False,
        ),
    },
    "place_red_pepper_in_ring": {
        "direct_place": (
            "data.sim.generation.tasks.place_red_pepper_in_ring.generators.direct_place:create",
            True,
        ),
        "direct_place_left_approach_v1": (
            "data.sim.generation.tasks.place_red_pepper_in_ring.generators.direct_place:create_left_approach_v1",
            False,
        ),
        "direct_place_right_approach_v1": (
            "data.sim.generation.tasks.place_red_pepper_in_ring.generators.direct_place:create_right_approach_v1",
            False,
        ),
    },
}


def generator_ids_for_task(task_id: str) -> tuple[str, ...]:
    if task_id not in TASK_BY_ID:
        raise ValueError(f"Unknown canonical task: {task_id!r}")
    return tuple(_REGISTRY[task_id])


def default_generator_id(task_id: str) -> str:
    candidates = [generator_id for generator_id, (_, is_default) in _REGISTRY[task_id].items() if is_default]
    if len(candidates) != 1:
        raise RuntimeError(f"Task {task_id!r} must have exactly one default generator")
    return candidates[0]


def resolve_generator(task_id: str, generator_id: str) -> GeneratorFactory:
    if task_id not in TASK_BY_ID:
        raise ValueError(f"Unknown canonical task: {task_id!r}")
    try:
        target, _ = _REGISTRY[task_id][generator_id]
    except KeyError as exc:
        choices = ", ".join(generator_ids_for_task(task_id)) or "(none)"
        raise ValueError(f"Unknown generator {generator_id!r} for {task_id!r}; choose: {choices}") from exc
    module_name, attribute = target.split(":", 1)
    factory = getattr(import_module(module_name), attribute)
    return factory


def create_generator(context: Any) -> Any:
    generator_id = context.task.generator_for_episode(context.requested_episode_index)
    return resolve_generator(context.task.task_id, generator_id)(context)
