"""Explicit task/generator registry; task identity remains in data.common."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from data.common.task_identity import TASK_BY_ID


GeneratorFactory = Callable[[Any], Any]

# Values are lazy import targets so config parsing does not initialize MuJoCo.
_PICK_GEOMETRY_FACTORIES = {
    "scripted_pick_side_approach_v1": "create_side_approach",
    "scripted_pick_opposite_side_approach_v1": "create_opposite_side_approach",
    "scripted_pick_yaw15_v1": "create_yaw15",
    "scripted_pick_yaw_minus15_v1": "create_yaw_minus15",
    "scripted_pick_diagonal_approach_v1": "create_diagonal_approach",
    "scripted_pick_waypoint_lift_v1": "create_waypoint_lift",
    "scripted_pick_opposite_waypoint_lift_v1": "create_opposite_waypoint_lift",
}


def _pick_generator_registry(task_id: str) -> dict[str, tuple[str, bool]]:
    package = f"data.sim.generation.tasks.{task_id}.generators"
    return {
        "scripted_pick": (f"{package}.scripted_pick:create", True),
        **{
            generator_id: (f"{package}.geometric:{factory}", False)
            for generator_id, factory in _PICK_GEOMETRY_FACTORIES.items()
        },
    }


_REGISTRY: dict[str, dict[str, tuple[str, bool]]] = {
    "red_block": _pick_generator_registry("red_block"),
    "blue_block": _pick_generator_registry("blue_block"),
    "red_pepper": _pick_generator_registry("red_pepper"),
    "smallest_block": _pick_generator_registry("smallest_block"),
    "largest_block": _pick_generator_registry("largest_block"),
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
        "direct_place_high_center_v1": (
            "data.sim.generation.tasks.place_red_pepper_in_ring.generators.direct_place:create_high_center_v1",
            False,
        ),
        "direct_place_left_rear_approach_v1": (
            "data.sim.generation.tasks.place_red_pepper_in_ring.generators.direct_place:create_left_rear_approach_v1",
            False,
        ),
        "direct_place_right_front_approach_v1": (
            "data.sim.generation.tasks.place_red_pepper_in_ring.generators.direct_place:create_right_front_approach_v1",
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
