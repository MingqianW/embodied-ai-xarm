"""Canonical task IDs, prompts, aliases, and task-required scene objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    prompt: str
    aliases: tuple[str, ...]
    required_active_objects: tuple[str, ...]
    kind: str


TASKS: tuple[TaskDefinition, ...] = (
    TaskDefinition(
        "red_pepper",
        "pick up the red pepper",
        ("pick_up_the_red_pepper", "red pepper"),
        ("red_pepper",),
        "pick",
    ),
    TaskDefinition(
        "blue_block",
        "pick up the blue block",
        ("pick_up_the_blue_block", "pick up the light blue block", "blue block"),
        ("blue_block",),
        "pick",
    ),
    TaskDefinition(
        "red_block",
        "pick up the red block",
        ("pick_up_the_red_block", "red block"),
        ("object",),
        "pick",
    ),
    TaskDefinition(
        "smallest_block",
        "pick up the smallest block",
        ("pick_up_the_smallest_block", "smallest block", "smallest"),
        ("small_block", "large_block"),
        "pick",
    ),
    TaskDefinition(
        "largest_block",
        "pick up the largest block",
        ("pick_up_the_largest_block", "largest block", "largest"),
        ("small_block", "large_block"),
        "pick",
    ),
    TaskDefinition(
        "place_red_pepper_in_ring",
        "place the red pepper in the ring",
        ("place_the_red_pepper_in_the_ring",),
        ("red_pepper", "ring"),
        "place",
    ),
)

TASK_BY_ID = {task.task_id: task for task in TASKS}


def _normalized(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def resolve_task_id(value: str) -> str:
    requested = _normalized(value)
    for definition in TASKS:
        candidates = (definition.task_id, definition.prompt, *definition.aliases)
        if requested in {_normalized(candidate) for candidate in candidates}:
            return definition.task_id
    raise ValueError(f"Unknown task or prompt: {value!r}")


def canonical_prompt(value: str) -> str:
    return TASK_BY_ID[resolve_task_id(value)].prompt


def task_definition(value: str) -> TaskDefinition:
    return TASK_BY_ID[resolve_task_id(value)]
