from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from data.common.task_identity import TASKS
from data.sim.generation.config import GeneratorPlan, load_pipeline_config
from data.sim.generation.core.generator import ControllerEpisodeGenerator
from data.sim.generation.core import registry
from data.sim.generation.core.registry import default_generator_id, resolve_generator


V3 = Path("configs/data/sim/generation/clean_multitask_stable_v3.yaml")
V4 = Path("configs/data/sim/generation/clean_multitask_stable_v4_10x_real.yaml")


def test_every_canonical_task_resolves_one_default_generator() -> None:
    for task in TASKS:
        generator_id = default_generator_id(task.task_id)
        assert callable(resolve_generator(task.task_id, generator_id))


def test_every_canonical_task_has_its_own_generator_subfolder() -> None:
    task_root = Path("data/sim/generation/tasks")
    for task in TASKS:
        generators = task_root / task.task_id / "generators"
        assert generators.is_dir(), f"Missing generator subfolder for {task.task_id}"
        assert any(generators.glob("*.py")), f"Missing generator implementation for {task.task_id}"


def test_generator_names_are_scoped_to_the_canonical_task() -> None:
    red_factory = resolve_generator("red_block", "scripted_pick")
    blue_factory = resolve_generator("blue_block", "scripted_pick")
    assert red_factory.__module__ != blue_factory.__module__
    with pytest.raises(ValueError, match="Unknown generator"):
        resolve_generator("red_block", "direct_place")


def test_v3_and_v4_implicit_legacy_configs_resolve_to_defaults() -> None:
    for path in (V3, V4):
        config = load_pipeline_config(path)
        assert all(
            task.generators == (GeneratorPlan(default_generator_id(task.task_id), task.episodes),)
            for task in config.tasks
        )


def test_exact_multi_generator_allocations_are_task_local_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_pipeline_config(V3)
    task = next(task for task in config.tasks if task.task_id == "red_block")
    original = dict(registry._REGISTRY[task.task_id])
    monkeypatch.setitem(
        registry._REGISTRY,
        task.task_id,
        {**original, "scripted_pick_replay": original["scripted_pick"]},
    )
    raw = yaml.safe_load(V3.read_text(encoding="utf-8"))
    raw["tasks"]["red_block"]["generators"] = {
        "scripted_pick": {"episodes": 10},
        "scripted_pick_replay": {"episodes": 15},
    }
    config_path = tmp_path / "multi_generator.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    allocation = next(task for task in load_pipeline_config(config_path).tasks if task.task_id == "red_block")
    assert [allocation.generator_for_episode(index) for index in (0, 9, 10, 24)] == [
        "scripted_pick",
        "scripted_pick",
        "scripted_pick_replay",
        "scripted_pick_replay",
    ]


def test_generator_boundary_preserves_canonical_hardware_raw_gripper() -> None:
    class Controller:
        terminal = False
        stage = "MOVE"
        failure_reason = None
        plan = type("Plan", (), {"to_json": lambda self: {}})()

        def next_action(self):
            self.terminal = True
            return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 492.58]

        def notify_post_step(self, **_):
            return None

        def stability_metadata(self):
            return {"stable_grasp_success": True}

        def transition_log(self):
            return []

    action = ControllerEpisodeGenerator(Controller(), generator_id="unit", kind="pick").next_action()
    assert action is not None and action.shape == (7,)
    assert action[6] == pytest.approx(492.58)
