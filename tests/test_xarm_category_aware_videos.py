from __future__ import annotations

from pathlib import Path

import pytest

from sim_mujoco.formal_evaluation.human_review import build_manifest
from sim_mujoco.formal_evaluation.outputs import read_json
from sim_mujoco.formal_evaluation.outputs import write_json
from sim_mujoco.formal_evaluation.representative_videos import index_json_path
from sim_mujoco.formal_evaluation.representative_videos import load_representative_index
from sim_mujoco.formal_evaluation.representative_videos import retain_video_bundle
from sim_mujoco.formal_evaluation.representative_videos import validate_category_video_coverage


def _result(*, model: str, task: str, seed: int, category: str) -> dict[str, object]:
    success, valid = category == "SUCCESS", category != "INVALID"
    failure = None if success or not valid else category
    return {
        "schema_version": "xarm-formal-episode-v2",
        "evaluation_protocol_version": "test",
        "timestamp_utc": "2026-08-08T00:00:00+00:00",
        "model": {
            "model_id": model,
            "training_config": "config",
            "checkpoint_root": "/work/checkpoints",
            "manager_step": 15000,
            "resolved_manager_directory": "/work/checkpoints/15000",
            "norm_asset_id": "norm",
        },
        "episode": {
            "task": task,
            "prompt": "pick up the red block",
            "seed": seed,
            "success": success,
            "valid": valid,
            "termination_reason": "task_success" if success else "max_policy_steps",
            "invalid_reason": None if valid else "policy_error",
            "failure_category": failure,
            "failure_reason": None if failure is None else "diagnosed failure",
            "failure_stage": None if failure is None else "lift",
            "policy_steps": 1,
            "executed_actions": 5,
        },
        "metrics": {},
        "failure_diagnostics": None if failure is None else {"observed": {}, "thresholds": {}},
        "safety": {},
        "initial_state": {},
        "final_state": {},
        "provenance": {
            "protocol_sha256": "protocol",
            "model_spec_sha256": "model",
            "openpi_git_commit": "openpi",
            "embodied_ai_xarm_git_commit": "embodied",
            "paths": {},
        },
        "artifacts": {},
    }


def _temporary_bundle(path: Path) -> tuple[Path, dict[str, object]]:
    path.mkdir(parents=True)
    combined = path / "combined.mp4"
    base = path / "base_camera.mp4"
    combined.write_bytes(b"combined")
    base.write_bytes(b"base")
    return path, {
        "video_frames": 1,
        "video_fps": 30,
        "video_paths": {"combined": str(combined), "base_camera": str(base)},
        "combined_video_path": str(combined),
    }


def _episode(
    root: Path, *, model: str = "A", task: str = "red_block", seed: int = 50000, category: str = "SUCCESS"
) -> tuple[Path, Path, Path, dict[str, object], dict[str, object]]:
    episode_root = root / "models" / model / "tasks" / task / f"seed_{seed}"
    result_path = episode_root / "result.json"
    result = _result(model=model, task=task, seed=seed, category=category)
    write_json(result_path, result)
    temporary, metadata = _temporary_bundle(episode_root / "temporary_video")
    return root / "models" / model, result_path, temporary, result, metadata


def _retain(root: Path, **kwargs: object) -> tuple[dict[str, object], Path, dict[str, object]]:
    model_root, result_path, temporary, result, metadata = _episode(root, **kwargs)
    artifacts = retain_video_bundle(
        model_root=model_root,
        result_json_path=result_path,
        result=result,
        temporary_video_dir=temporary,
        temporary_metadata=metadata,
        video_policy="category_representative",
    )
    result["artifacts"] = artifacts
    write_json(result_path, result)
    return artifacts, result_path, result


def test_first_success_and_failure_categories_preserve_independent_representatives(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    success, _, _ = _retain(root, category="SUCCESS", seed=50002)
    failure, _, _ = _retain(root, category="PICK_PARTIAL_LIFT", seed=50003)
    assert success["video_retention"]["status"] == "preserved_as_representative"
    assert failure["video_retention"]["status"] == "preserved_as_representative"
    index = load_representative_index(model_root=root / "models" / "A", model_id="A")
    assert {(row["category"], row["seed"]) for row in index["records"]} == {
        ("SUCCESS", 50002),
        ("PICK_PARTIAL_LIFT", 50003),
    }


def test_second_same_category_is_discarded_and_temporary_bundle_is_cleaned(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    _retain(root, category="PICK_PARTIAL_LIFT", seed=50001)
    artifacts, result_path, result = _retain(root, category="PICK_PARTIAL_LIFT", seed=50002)
    assert artifacts["video_retention"]["status"] == "discarded_after_classification"
    assert not (result_path.parent / "temporary_video").exists()
    assert result_path.is_file()  # Results are never deleted with temporary videos.
    assert read_json(result_path)["episode"]["failure_category"] == result["episode"]["failure_category"]
    index = load_representative_index(model_root=root / "models" / "A", model_id="A")
    assert index["records"][0]["seed"] == 50001


def test_lower_seed_later_replaces_higher_seed_only_after_new_bundle_finalizes(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    _, high_result_path, _ = _retain(root, category="PICK_DROPPED_AFTER_LIFT", seed=50010)
    high_bundle = root / "models" / "A" / "representative_videos" / "red_block" / "PICK_DROPPED_AFTER_LIFT" / "seed_50010"
    assert high_bundle.is_dir()
    _, low_result_path, _ = _retain(root, category="PICK_DROPPED_AFTER_LIFT", seed=50005)
    low_bundle = root / "models" / "A" / "representative_videos" / "red_block" / "PICK_DROPPED_AFTER_LIFT" / "seed_50005"
    assert low_bundle.is_dir()
    assert not high_bundle.exists()
    assert high_result_path.is_file()
    assert low_result_path.is_file()
    assert read_json(high_result_path)["artifacts"]["video_retention"]["status"] == "superseded_by_lower_seed_representative"


def test_task_model_and_invalid_categories_do_not_share_representatives(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    _retain(root, model="A", task="red_block", category="SUCCESS")
    _retain(root, model="A", task="red_pepper", category="SUCCESS")
    invalid, _, _ = _retain(root, model="A", task="red_block", seed=50001, category="INVALID")
    _retain(root, model="B", task="red_block", category="SUCCESS")
    assert invalid["representative_video"]["category"] == "INVALID"
    a_index = load_representative_index(model_root=root / "models" / "A", model_id="A")
    b_index = load_representative_index(model_root=root / "models" / "B", model_id="B")
    assert len(a_index["records"]) == 3
    assert len(b_index["records"]) == 1


def test_resume_keeps_existing_lower_seed_representative_and_index_order_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    _retain(root, category="PICK_TIMEOUT_OTHER", seed=50002)
    discarded, _, _ = _retain(root, category="PICK_TIMEOUT_OTHER", seed=50003)
    _retain(root, category="SUCCESS", seed=50001)
    assert discarded["video_retention"]["status"] == "discarded_after_classification"
    index = read_json(index_json_path(root / "models" / "A"))
    assert index["records"] == sorted(index["records"], key=lambda row: (row["task"], row["category"], row["seed"]))


def test_representative_index_rejects_duplicate_task_category_records(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    _retain(root, category="PICK_TIMEOUT_OTHER", seed=50002)
    path = index_json_path(root / "models" / "A")
    index = read_json(path)
    index["records"].append(dict(index["records"][0]))
    write_json(path, index)
    with pytest.raises(ValueError, match="duplicate task/category"):
        load_representative_index(model_root=root / "models" / "A", model_id="A")


def test_coverage_validator_detects_a_missing_observed_category(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    _retain(root, category="SUCCESS", seed=50000)
    model_root, _, _, _, _ = _episode(root, category="PICK_PARTIAL_LIFT", seed=50001)
    report = validate_category_video_coverage(root)
    task_report = report["model_task_reports"][0]
    assert model_root.is_dir()
    assert not report["coverage_complete"]
    assert task_report["missing_categories"] == ["PICK_PARTIAL_LIFT"]


def test_all_video_policy_retains_every_episode_and_representative_review_reads_index(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    model_root, result_path, temporary, result, metadata = _episode(root, category="SUCCESS", seed=50000)
    all_artifacts = retain_video_bundle(
        model_root=model_root,
        result_json_path=result_path,
        result=result,
        temporary_video_dir=temporary,
        temporary_metadata=metadata,
        video_policy="all",
    )
    assert all_artifacts["video_retention"]["status"] == "retained_all_episode_video"
    assert Path(str(all_artifacts["combined_video_path"])).is_file()
    _retain(root, category="PICK_PARTIAL_LIFT", seed=50001)
    write_json(
        root / "protocol.json",
        {"protocol": {"tasks": [{"task_id": "red_block"}], "seed_start": 50000, "seed_count": 2}},
    )
    private, reviewer, _ = build_manifest(evaluation_root=root, review_seed=4, mode="representative")
    assert len(private["items"]) == 1
    assert private["items"][0]["automated_category"] == "PICK_PARTIAL_LIFT"
    assert set(reviewer["items"][0]) == {"prompt", "review_id", "task"}
