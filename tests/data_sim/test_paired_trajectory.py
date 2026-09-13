from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.sim.generation.audit import audit_raw
from data.sim.generation.collection import _paired_member_episode_index
from data.sim.generation.config import load_pipeline_config
from data.sim.generation.trajectory import (
    parse_trajectory_profile,
    resolve_parameters,
    scene_seed,
    trajectory_seed,
)


CONFIG = Path("configs/data/sim/generation/clean_multitask_paired_trajectory_v1.yaml")
LEGACY_CONFIG = Path("configs/data/sim/generation/clean_multitask_stable_v4_10x_real.yaml")


def _profile() -> dict[str, object]:
    return {
        "name": "test-v1",
        "version": 1,
        "defaults": {},
        "tasks": {
            "red_block": {
                "parameters": {"tcp_yaw_offset_deg": {"fixed": 5.0}},
                "members": {
                    "scripted_pick": {
                        "parameters": {
                            "pregrasp_offset_xy_m": {
                                "truncated_normal": {
                                    "mean": [0.01, 0.0], "std": [0.002, 0.002],
                                    "min": [0.005, -0.004], "max": [0.015, 0.004],
                                }
                            }
                        }
                    }
                },
            }
        },
    }


def test_paired_config_allocates_every_enabled_member_for_every_scene() -> None:
    config = load_pipeline_config(CONFIG)
    assert config.generation_mode == "paired_scene_groups"
    assert config.scenes_per_task == 2
    assert all(task.episodes == 2 * len(task.generators) for task in config.tasks)
    red = next(task for task in config.tasks if task.task_id == "red_block")
    assert [_paired_member_episode_index(red, 1, index) for index in range(len(red.generators))] == list(range(1, 16, 2))


def test_profile_precedence_and_deterministic_bounded_sampling() -> None:
    profile = parse_trajectory_profile(
        _profile(), task_ids={"red_block"}, member_ids={"red_block": {"scripted_pick"}}
    )
    first = resolve_parameters(profile, task_id="red_block", member_id="scripted_pick", seed=17)
    second = resolve_parameters(profile, task_id="red_block", member_id="scripted_pick", seed=17)
    assert first == second
    assert first["tcp_yaw_offset_deg"] == 5.0
    assert 0.005 <= first["pregrasp_offset_xy_m"][0] <= 0.015
    assert -0.004 <= first["pregrasp_offset_xy_m"][1] <= 0.004


@pytest.mark.parametrize(
    "profile",
    [
        {"name": "bad", "version": 1, "defaults": {"unknown": 1.0}, "tasks": {}},
        {"name": "bad", "version": 1, "defaults": {}, "tasks": {"unknown": {}}},
        {"name": "bad", "version": 1, "defaults": {"tcp_yaw_offset_deg": {"uniform": {"min": 10, "max": -10}}}, "tasks": {}},
    ],
)
def test_profile_rejects_unknown_names_and_invalid_bounds(profile: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parse_trajectory_profile(profile, task_ids={"red_block"}, member_ids={"red_block": {"scripted_pick"}})


def test_paired_seeds_are_independent_of_member_order_and_additions() -> None:
    before = trajectory_seed(10, "red_block", 4, "scripted_pick", 2)
    after = trajectory_seed(10, "red_block", 4, "scripted_pick", 2)
    assert before == after
    assert scene_seed(10, "red_block", 4) == scene_seed(10, "red_block", 4)
    assert before != trajectory_seed(10, "red_block", 4, "another_member", 2)


def test_incomplete_paired_groups_are_excluded_before_conversion_audit(tmp_path: Path) -> None:
    config = load_pipeline_config(CONFIG)
    (tmp_path / "collection_manifest.json").write_text(
        json.dumps({"complete": False, "completed": [], "scene_groups": []}), encoding="utf-8"
    )
    (tmp_path / "collection_summary.json").write_text(
        json.dumps({"complete": False, "incomplete_scene_groups": [{"scene_group_id": "red_block:scene_00000"}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Incomplete paired"):
        audit_raw(config, tmp_path, decode_all_images=False)


def test_existing_v4_configuration_remains_legacy() -> None:
    config = load_pipeline_config(LEGACY_CONFIG)
    assert config.generation_mode == "legacy_episodes"
    assert config.scenes_per_task is None
    assert config.trajectory_profile is None
