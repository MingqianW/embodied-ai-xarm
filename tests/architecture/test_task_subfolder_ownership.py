"""Repository-wide enforcement for canonical task-folder ownership."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from data.common.task_identity import TASKS
from data.sim.generation.core import registry


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = ROOT / "data" / "sim" / "generation" / "tasks"
PROTOCOL_ROOT = ROOT / "configs" / "evaluation" / "sim" / "protocols"
SCENE_CONFIG = ROOT / "simulation" / "config" / "task_scenes.yaml"


def _canonical_ids() -> list[str]:
    return [task.task_id for task in TASKS]


def test_every_canonical_subtask_has_one_owned_generator_folder() -> None:
    for task_id in _canonical_ids():
        folder = TASK_ROOT / task_id / "generators"
        assert folder.is_dir(), f"Missing task folder: {folder.relative_to(ROOT)}"
        for generator_id, (target, _) in registry._REGISTRY[task_id].items():
            module_name, _ = target.split(":", 1)
            expected_prefix = f"data.sim.generation.tasks.{task_id}.generators."
            assert module_name.startswith(expected_prefix), (task_id, generator_id, module_name)
            module_path = ROOT / Path(*module_name.split(".")).with_suffix(".py")
            assert module_path.is_file(), f"Missing task generator module: {module_path.relative_to(ROOT)}"


def test_repository_task_consumers_use_the_canonical_subtask_set() -> None:
    canonical_ids = _canonical_ids()
    scenes = yaml.safe_load(SCENE_CONFIG.read_text(encoding="utf-8"))["tasks"]
    assert set(scenes) == set(canonical_ids)
    for path in sorted(PROTOCOL_ROOT.glob("*.json")):
        protocol = json.loads(path.read_text(encoding="utf-8"))
        if "tasks" in protocol:
            assert [task["task_id"] for task in protocol["tasks"]] == canonical_ids, path.name
