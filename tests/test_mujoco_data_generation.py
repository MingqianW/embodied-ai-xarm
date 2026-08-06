from __future__ import annotations

import json
import csv
from dataclasses import replace
from pathlib import Path

import pytest

from sim_mujoco.data_generation.collection import resolve_seed
from sim_mujoco.data_generation.config import OutputRoots, load_pipeline_config
from sim_mujoco.data_generation import conversion
from sim_mujoco.data_generation.manifest import atomic_write_json, initial_manifest
from sim_mujoco.data_generation.registry import (
    TASKS,
    canonical_prompt,
    resolve_task_id,
)
from sim_mujoco.data_generation import safety
from sim_mujoco.data_generation.stability import (
    StabilitySample,
    evaluate_pick_stability,
    evaluate_place_initial_grasp,
    evaluate_place_stability,
)
from sim_mujoco.task_scenes import load_task_scene_config


CONFIG_PATH = Path(
    "sim_mujoco/config/data_generation/clean_multitask_stable_v3.yaml"
)


def _config():
    return load_pipeline_config(CONFIG_PATH)


def _sample(
    index: int,
    *,
    object_xyz: tuple[float, float, float] = (0.0, 0.0, 0.15),
    tcp_xyz: tuple[float, float, float] = (0.0, 0.0, 0.25),
    **kwargs,
) -> StabilitySample:
    return StabilitySample(
        simulation_time_s=(index + 1) * 0.1,
        object_position_m=object_xyz,
        tcp_position_m=tcp_xyz,
        **kwargs,
    )


def _pick_result(samples: list[StabilitySample]):
    config = _config().pick
    return evaluate_pick_stability(
        samples,
        config=config,
        initial_object_z_m=0.10,
        verification_start_object_position_m=(0.0, 0.0, 0.15),
        verification_start_tcp_position_m=(0.0, 0.0, 0.25),
    )


class TestTaskRegistryAndConfig:
    def test_exact_six_ids_prompts_and_counts(self) -> None:
        config = _config()
        assert [task.task_id for task in config.tasks] == [task.task_id for task in TASKS]
        assert [task.prompt for task in config.tasks] == [task.prompt for task in TASKS]
        assert {task.task_id: task.episodes for task in config.tasks} == {
            "red_pepper": 50,
            "blue_block": 25,
            "red_block": 25,
            "smallest_block": 25,
            "largest_block": 25,
            "place_red_pepper_in_ring": 50,
        }
        assert config.total_episodes == 200
        assert all(task.distractor_episodes == 0 for task in config.tasks)
        assert config.distractor_count == 0

    def test_prompt_contract_and_alias_normalization(self) -> None:
        prompts = [task.prompt for task in TASKS]
        assert len(prompts) == len(set(prompts)) == 6
        assert all("_" not in prompt for prompt in prompts)
        assert all(task.prompt.startswith("pick up ") for task in TASKS if task.kind == "pick")
        assert canonical_prompt("pick_up_the_red_block") == "pick up the red block"
        assert canonical_prompt("pick_up_the_smallest_block") == "pick up the smallest block"
        assert canonical_prompt("pick_up_the_largest_block") == "pick up the largest block"
        assert resolve_task_id("place_the_red_pepper_in_the_ring") == "place_red_pepper_in_ring"
        assert canonical_prompt("smallest_block") != canonical_prompt("largest_block")

    def test_clean_scene_required_objects_remain_active(self) -> None:
        config = _config()
        by_id = {task.task_id: task.required_active_objects for task in config.tasks}
        assert by_id["smallest_block"] == ("small_block", "large_block")
        assert by_id["largest_block"] == ("small_block", "large_block")
        assert by_id["place_red_pepper_in_ring"] == ("red_pepper", "ring")
        scenes = load_task_scene_config()["tasks"]
        for task in config.tasks:
            assert tuple(scenes[task.task_id]["active_bodies"]) == task.required_active_objects

    def test_output_paths_are_exact_and_overwrite_defaults_false(self) -> None:
        config = _config()
        assert config.overwrite_existing_outputs is False
        assert set(vars(config.outputs).values()) == safety.AUTHORIZED_ROOTS

    def test_seed_resolution_is_deterministic_and_task_separated(self) -> None:
        config = _config()
        first = config.tasks[0]
        second = config.tasks[1]
        assert resolve_seed(first, 4, 3, config.seed_retry_stride) == 103004
        assert resolve_seed(first, 4, 3, config.seed_retry_stride) == resolve_seed(
            first, 4, 3, config.seed_retry_stride
        )
        assert resolve_seed(first, 0, 0, config.seed_retry_stride) != resolve_seed(
            second, 0, 0, config.seed_retry_stride
        )


class TestOutputSafetyAndManifest:
    def test_rejects_non_authorized_parent_sibling_root_and_empty(self) -> None:
        for value in (
            "",
            "/",
            "/work/nvme/bfmk/mw89",
            "/work/nvme/bfmk/mw89/mujoco_datasets/raw/sibling",
        ):
            with pytest.raises(ValueError):
                safety.validate_authorized_root(Path(value))

    def test_exact_scoped_overwrite_and_symlink_rejection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        authorized = tmp_path / "authorized"
        authorized.mkdir()
        (authorized / "old.txt").write_text("preserved in inventory only", encoding="utf-8")
        monkeypatch.setattr(safety, "AUTHORIZED_ROOTS", frozenset({authorized}))
        monkeypatch.setattr(safety, "LOG_PARENT", tmp_path / "logs")
        monkeypatch.setattr(safety.shutil, "chown", lambda *args, **kwargs: None)
        monkeypatch.setattr(safety.shutil, "which", lambda name: None)
        with pytest.raises(ValueError):
            safety.replace_authorized_roots(
                [authorized], overwrite=False, git_sha="abc", config_path=CONFIG_PATH
            )
        result = safety.replace_authorized_roots(
            [authorized], overwrite=True, git_sha="abc", config_path=CONFIG_PATH
        )
        assert json.loads((authorized / "OVERWRITE_MARKER.json").read_text())["git_sha"] == "abc"
        assert Path(result["preoverwrite_inventory"]).is_file()
        assert "old.txt" not in {path.name for path in authorized.iterdir()}

        authorized_link = tmp_path / "authorized_link"
        target = tmp_path / "real"
        target.mkdir()
        authorized_link.symlink_to(target, target_is_directory=True)
        monkeypatch.setattr(safety, "AUTHORIZED_ROOTS", frozenset({authorized_link}))
        with pytest.raises(ValueError, match="symbolic link"):
            safety.validate_authorized_root(authorized_link)

    def test_atomic_manifest_write_and_partial_default(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        manifest = initial_manifest("v3", {"total_target_episodes": 200})
        assert manifest["complete"] is False
        atomic_write_json(path, manifest)
        assert json.loads(path.read_text()) == manifest
        assert not list(tmp_path.glob("*.tmp"))


class TestPickStability:
    def test_stable_full_window_passes_with_required_metadata(self) -> None:
        result = _pick_result([_sample(index) for index in range(20)])
        assert result["stable_grasp_success"] is True
        assert result["verification_steps_executed"] == 20
        assert result["verification_duration_s"] == pytest.approx(2.0)
        required = {
            "initial_object_z_m",
            "verification_start_object_z_m",
            "verification_start_tcp_z_m",
            "verification_start_tcp_to_object_offset_m",
            "peak_lift_height_m",
            "minimum_verification_lift_height_m",
            "final_lift_height_m",
            "maximum_relative_downward_slip_m",
            "final_relative_downward_slip_m",
            "estimated_final_object_vertical_velocity_mps",
            "verification_steps_required",
            "verification_steps_executed",
            "verification_duration_s",
            "stable_grasp_success",
            "stable_grasp_failure_reason",
        }
        assert required <= set(result)

    @pytest.mark.parametrize(
        ("samples", "reason"),
        [
            (
                [_sample(i, object_xyz=(0.0, 0.0, 0.15 if i < 5 else 0.12)) for i in range(20)],
                "stable_grasp_lift_below_minimum",
            ),
            (
                [_sample(i, object_xyz=(0.0, 0.0, 0.145), tcp_xyz=(0.0, 0.0, 0.26)) for i in range(20)],
                "stable_grasp_excessive_relative_slip",
            ),
            (
                [
                    _sample(
                        i,
                        object_xyz=(
                            0.0,
                            0.0,
                            0.155 if i < 10 else 0.155 - (i - 9) * 0.0012,
                        ),
                        tcp_xyz=(
                            0.0,
                            0.0,
                            (0.155 if i < 10 else 0.155 - (i - 9) * 0.0012)
                            + 0.1,
                        ),
                    )
                    for i in range(20)
                ],
                "stable_grasp_downward_motion",
            ),
            ([_sample(i) for i in range(19)], "stable_grasp_incomplete_verification"),
            ([_sample(i, table_contact=(i == 7)) for i in range(20)], "stable_grasp_table_contact"),
            ([_sample(i, finite=(i != 9)) for i in range(20)], "stable_grasp_non_finite"),
        ],
    )
    def test_rejects_false_positive_sequences(
        self, samples: list[StabilitySample], reason: str
    ) -> None:
        result = _pick_result(samples)
        assert result["stable_grasp_success"] is False
        assert result["stable_grasp_failure_reason"] == reason

    def test_all_five_pick_tasks_share_validator_contract(self) -> None:
        pick_ids = [task.task_id for task in TASKS if task.kind == "pick"]
        assert pick_ids == [
            "red_pepper",
            "blue_block",
            "red_block",
            "smallest_block",
            "largest_block",
        ]
        assert _config().pick.steps == 20


class TestPlaceValidation:
    def _initial(self, samples: list[StabilitySample]):
        return evaluate_place_initial_grasp(
            samples, config=_config().place_initial, table_top_z_m=0.05
        )

    def test_stably_held_in_air_passes_and_metadata_is_complete(self) -> None:
        result = self._initial([_sample(index) for index in range(10)])
        assert result["initial_grasp_success"] is True
        assert result["initial_grasp_validation_steps_executed"] == 10
        assert result["initial_grasp_validation_duration_s"] == pytest.approx(1.0)
        assert result["initial_grasp_max_relative_drift_m"] == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("samples", "reason"),
        [
            ([_sample(i, object_xyz=(0.0, 0.0, 0.05)) for i in range(10)], "initial_place_grasp_unstable"),
            ([_sample(i, inside_ring=True) for i in range(10)], "initial_place_grasp_inside_ring"),
            ([_sample(i, object_xyz=(0.0, 0.0, 0.15 - i * 0.012)) for i in range(10)], "initial_place_grasp_unstable"),
            ([_sample(i, object_xyz=(0.0, 0.0, 0.15 + i * 0.0007)) for i in range(10)], "initial_place_grasp_excessive_drift"),
            ([_sample(i) for i in range(9)], "initial_place_grasp_incomplete_validation"),
        ],
    )
    def test_initial_grasp_failures(
        self, samples: list[StabilitySample], reason: str
    ) -> None:
        result = self._initial(samples)
        assert result["initial_grasp_success"] is False
        assert result["initial_grasp_failure_reason"] == reason

    def test_held_pepper_cannot_count_as_place_success(self) -> None:
        samples = [
            _sample(i, inside_ring=True, released=False, retreat_detected=True)
            for i in range(20)
        ]
        result = evaluate_place_stability(samples, config=_config().place)
        assert result["stable_place_success"] is False
        assert result["stable_place_failure_reason"] == "stable_place_release_not_detected"

    def test_released_settled_pepper_full_window_passes(self) -> None:
        samples = [
            _sample(i, inside_ring=True, released=True, retreat_detected=True)
            for i in range(20)
        ]
        result = evaluate_place_stability(samples, config=_config().place)
        assert result["stable_place_success"] is True
        assert result["place_verification_steps_executed"] == 20
        assert result["place_verification_duration_s"] == pytest.approx(2.0)

    def test_place_uses_one_free_body_and_no_fake_attachment(self) -> None:
        scene = load_task_scene_config()["tasks"]["place_red_pepper_in_ring"]
        assert scene["target_body"] == scene["object_identity"] == "red_pepper"
        assert scene["active_bodies"] == ["red_pepper", "ring"]
        assert "held_red_pepper" not in scene["active_bodies"]
        assert scene["initial_tcp_to_object"]["translation_m"] == [0.0, 0.0, -0.027]


class TestConversionContract:
    def test_conversion_emits_only_canonical_clean_accepted_frames(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = _config()
        raw = tmp_path / "raw"
        output = tmp_path / "converted"
        raw.mkdir()
        config = replace(
            base,
            outputs=OutputRoots(
                raw=raw,
                converted=output,
                smoke=base.outputs.smoke,
                log=base.outputs.log,
            ),
        )
        log = tmp_path / "log"
        log.mkdir()
        (log / "RAW_DATASET_AUDIT.json").write_text(
            json.dumps({"status": "RAW_PASS"}), encoding="utf-8"
        )
        config = replace(
            config,
            outputs=replace(config.outputs, log=log),
        )
        completed = []
        global_index = 0
        expected_counts: dict[str, int] = {}
        for task in config.tasks:
            expected_counts[task.task_id] = task.episodes
            for requested_index in range(task.episodes):
                relative = Path("accepted") / task.task_id / f"episode_{requested_index:03d}"
                episode = raw / relative
                episode.mkdir(parents=True)
                validation = (
                    {
                        "place_initial_grasp": {
                            "initial_grasp_success": True,
                            "initialization_frames_recorded": 0,
                        },
                        "stable_place": {"stable_place_success": True},
                    }
                    if task.task_id == "place_red_pepper_in_ring"
                    else {"stable_grasp": {"stable_grasp_success": True}}
                )
                (episode / "meta.json").write_text(
                    json.dumps(
                        {
                            "task": task.prompt,
                            "task_id": task.task_id,
                            "task_prompt": task.prompt,
                            "simulation": {"validation": validation},
                        }
                    ),
                    encoding="utf-8",
                )
                fieldnames = [
                    *(f"j{index}_rad" for index in range(1, 7)),
                    "gripper_mm",
                    "realsense_0_file",
                    "realsense_1_file",
                ]
                with (episode / "robot_log.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as stream:
                    writer = csv.DictWriter(stream, fieldnames=fieldnames)
                    writer.writeheader()
                    for row_index in range(2):
                        writer.writerow(
                            {
                                **{f"j{index}_rad": row_index for index in range(1, 7)},
                                "gripper_mm": 250 + row_index,
                                "realsense_0_file": f"base_{row_index}.png",
                                "realsense_1_file": f"wrist_{row_index}.png",
                            }
                        )
                completed.append(
                    {
                        "task_id": task.task_id,
                        "task_prompt": task.prompt,
                        "requested_episode_index": requested_index,
                        "global_episode_index": global_index,
                        "scene_variant": "clean",
                        "path": relative.as_posix(),
                    }
                )
                global_index += 1
        (raw / "collection_manifest.json").write_text(
            json.dumps({"complete": True, "completed": completed}), encoding="utf-8"
        )
        (raw / "collection_summary.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "total_accepted_episodes": 200,
                    "total_distractor_episodes": 0,
                    "accepted_counts_by_task": expected_counts,
                }
            ),
            encoding="utf-8",
        )
        failed = raw / "failed_attempts" / "red_block" / "never_convert"
        failed.mkdir(parents=True)
        (failed / "failure.json").write_text("{}", encoding="utf-8")

        def fake_replace(*args, **kwargs):
            output.mkdir()
            (output / "OVERWRITE_MARKER.json").write_text(
                json.dumps({"overwritten_utc": "test"}), encoding="utf-8"
            )
            return {"timestamp": "test", "paths": [str(output)]}

        captured: dict[str, object] = {}

        def fake_writer(records_by_episode, **kwargs):
            captured["records"] = records_by_episode
            captured["kwargs"] = kwargs
            (output / "meta").mkdir()
            return {"written_episodes": len(records_by_episode), "written_frames": 200}

        monkeypatch.setattr(conversion, "replace_authorized_roots", fake_replace)
        monkeypatch.setattr(conversion, "write_xarm_lerobot_dataset", fake_writer)
        result = conversion.convert_dataset(config, raw, output, overwrite=True)
        records = captured["records"]
        assert isinstance(records, list) and len(records) == 200
        flattened = [record for episode in records for record in episode]
        assert len(flattened) == 200
        assert {record["task_id"] for record in flattened} == {
            task.task_id for task in config.tasks
        }
        assert all(record["task"] == canonical_prompt(record["task_id"]) for record in flattened)
        assert all("_" not in record["task"] for record in flattened)
        assert all(record["state"].shape == (7,) for record in flattened)
        assert all(record["actions"].shape == (7,) for record in flattened)
        assert [item["episode_index"] for item in result["episodes"]] == list(range(200))
        assert result["total_distractor_episodes"] == 0
        assert result["task_index_order"] == [task.task_id for task in config.tasks]
        assert not any("failed_attempts" in str(record["image"]) for record in flattened)
