from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.sim.human_review import build_manifest
from evaluation.sim.human_review import load_decisions
from evaluation.sim.human_review import save_decision
from evaluation.sim.human_review import summarize_review_rows
from evaluation.sim.human_review import unblind_manifest
from evaluation.sim.human_review import write_manifest
from evaluation.sim.outputs import read_json
from evaluation.sim.outputs import write_json
from evaluation.sim.representative_videos import REPRESENTATIVE_INDEX_VERSION
from evaluation.sim.representative_videos import SELECTION_POLICY
from evaluation.sim.representative_videos import index_json_path
from evaluation.sim.tools.review_human_videos import ReviewApplication


def _result(*, model: str, task: str, seed: int, success: bool, video: Path | None) -> dict[str, object]:
    failure = None if success else "PICK_PARTIAL_LIFT"
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
            "valid": True,
            "termination_reason": "task_success" if success else "max_policy_steps",
            "invalid_reason": None,
            "failure_category": failure,
            "failure_reason": None if success else "Partial lift.",
            "failure_stage": None if success else "lift",
            "policy_steps": 1,
            "executed_actions": 5,
        },
        "metrics": {"max_lift_m": 0.06 if success else 0.02},
        "failure_diagnostics": None if success else {"observed": {}, "thresholds": {}},
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
        "artifacts": {} if video is None else {"combined_video_path": str(video)},
    }


def _evaluation_root(tmp_path: Path) -> Path:
    root = tmp_path / "evaluation"
    write_json(
        root / "protocol.json",
        {
            "protocol": {
                "tasks": [{"task_id": "red_block"}],
                "seed_start": 7,
                "seed_count": 2,
            }
        },
    )
    for model, seed, success, has_video in (
        ("A", 7, True, True),
        ("A", 8, False, False),
        ("B", 7, False, True),
        ("B", 8, False, True),
        ("C", 7, True, True),
        ("C", 8, False, True),
    ):
        video = root / "videos" / f"{model}_{seed}.mp4" if has_video else None
        if video is not None:
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"not-a-real-video")
        write_json(
            root / "models" / model / "tasks" / "red_block" / f"seed_{seed}" / "result.json",
            _result(model=model, task="red_block", seed=seed, success=success, video=video),
        )
    return root


def _write_representative_indexes(root: Path) -> None:
    """Create the category index the representative-review mode consumes."""

    by_model: dict[str, dict[tuple[str, str], dict[str, object]]] = {"A": {}, "B": {}, "C": {}}
    for result_path in sorted(root.glob("models/*/tasks/*/seed_*/result.json")):
        # The test fixture intentionally uses the same stable v2 result schema
        # as formal runs, so the index contains only an existing representative
        # video for each (model, task, category) key.
        result = read_json(result_path)
        artifacts = dict(result.get("artifacts") or {})
        video = artifacts.get("combined_video_path")
        if video is None or not Path(str(video)).is_file():
            continue
        episode = result["episode"]
        category = "SUCCESS" if episode["success"] else str(episode["failure_category"])
        model = str(result["model"]["model_id"])
        record = {
            "model": model,
            "task": str(episode["task"]),
            "category": category,
            "seed": int(episode["seed"]),
            "video_path": str(Path(str(video)).resolve()),
            "video_bundle_path": str(Path(str(video)).parent.resolve()),
            "result_json_path": str(result_path.resolve()),
            "success": bool(episode["success"]),
            "valid": bool(episode["valid"]),
            "failure_category": episode["failure_category"],
            "selection_policy": SELECTION_POLICY,
        }
        key = (str(record["task"]), str(record["category"]))
        previous = by_model[model].get(key)
        if previous is None or int(record["seed"]) < int(previous["seed"]):
            by_model[model][key] = record
    for model, records_by_category in by_model.items():
        write_json(
            index_json_path(root / "models" / model),
            {
                "schema_version": REPRESENTATIVE_INDEX_VERSION,
                "model": model,
                "selection_policy": SELECTION_POLICY,
                "records": list(records_by_category.values()),
            },
        )


def test_manifest_is_deterministic_blinded_and_reports_missing_videos(tmp_path: Path) -> None:
    root = _evaluation_root(tmp_path)
    private_one, reviewer_one, coverage_one = build_manifest(
        evaluation_root=root, review_seed=77, mode="full"
    )
    private_two, reviewer_two, coverage_two = build_manifest(
        evaluation_root=root, review_seed=77, mode="full"
    )
    assert private_one == private_two
    assert reviewer_one == reviewer_two
    assert coverage_one == coverage_two
    assert coverage_one["expected_episode_count"] == 6
    assert coverage_one["episodes_with_videos"] == 5
    assert coverage_one["episodes_without_videos"] == 1
    assert coverage_one["missing_video_episodes"][0]["model"] == "A"
    reviewer_text = str(reviewer_one)
    assert "model" not in reviewer_text
    assert "automated" not in reviewer_text
    assert "video_path" not in reviewer_text


def test_representative_mode_selects_each_automated_category(tmp_path: Path) -> None:
    root = _evaluation_root(tmp_path)
    _write_representative_indexes(root)
    private, _, _ = build_manifest(evaluation_root=root, review_seed=2, mode="representative")
    categories = {(item["model"], item["task"], item["automated_category"]) for item in private["items"]}
    assert ("A", "red_block", "SUCCESS") in categories
    assert ("B", "red_block", "PICK_PARTIAL_LIFT") in categories
    assert len(categories) == len(private["items"])


def test_decisions_resume_and_require_explicit_overwrite(tmp_path: Path) -> None:
    root = _evaluation_root(tmp_path)
    private, reviewer, coverage = build_manifest(evaluation_root=root, review_seed=3, mode="full")
    review_root = tmp_path / "review"
    write_manifest(
        output_root=review_root,
        private_manifest=private,
        reviewer_manifest=reviewer,
        coverage=coverage,
    )
    review_id = private["items"][0]["review_id"]
    save_decision(
        csv_path=review_root / "human_review.csv",
        review_ids={item["review_id"] for item in private["items"]},
        review_id=review_id,
        label="FAILURE",
        failure_reason="PARTIAL_LIFT",
        notes="first pass",
        allow_overwrite=False,
    )
    assert load_decisions(review_root / "human_review.csv")[review_id]["notes"] == "first pass"
    with pytest.raises(FileExistsError, match="explicit overwrite"):
        save_decision(
            csv_path=review_root / "human_review.csv",
            review_ids={item["review_id"] for item in private["items"]},
            review_id=review_id,
            label="SUCCESS",
            failure_reason="",
            notes="changed",
            allow_overwrite=False,
        )
    overwritten = save_decision(
        csv_path=review_root / "human_review.csv",
        review_ids={item["review_id"] for item in private["items"]},
        review_id=review_id,
        label="SUCCESS",
        failure_reason="",
        notes="changed",
        allow_overwrite=True,
    )
    assert overwritten["human_label"] == "SUCCESS"


def test_reviewer_api_item_exposes_only_blinded_fields(tmp_path: Path) -> None:
    root = _evaluation_root(tmp_path)
    private, reviewer, coverage = build_manifest(evaluation_root=root, review_seed=3, mode="full")
    review_root = tmp_path / "review"
    write_manifest(
        output_root=review_root,
        private_manifest=private,
        reviewer_manifest=reviewer,
        coverage=coverage,
    )
    item = ReviewApplication(review_root=review_root, allow_overwrite_decisions=False).next_item()
    assert set(item) == {"complete", "prompt", "review_id", "reviewed", "total", "video_url"}
    assert "model" not in item
    assert "automated_success" not in item
    assert "failure_category" not in item


def test_unblinding_agreement_and_paired_joins_are_correct(tmp_path: Path) -> None:
    root = _evaluation_root(tmp_path)
    private, _, _ = build_manifest(evaluation_root=root, review_seed=5, mode="full")
    decisions = {}
    labels = {
        ("A", 7): "SUCCESS",
        ("B", 7): "SUCCESS",  # automated failure / human success
        ("C", 7): "SUCCESS",
        ("B", 8): "FAILURE",
        ("C", 8): "UNCERTAIN",
    }
    for item in private["items"]:
        label = labels.get((item["model"], item["seed"]))
        if label:
            decisions[item["review_id"]] = {
                "review_id": item["review_id"],
                "human_label": label,
                "human_failure_reason": "" if label != "FAILURE" else "PARTIAL_LIFT",
                "notes": "",
                "review_timestamp": "2026-08-08T00:00:00+00:00",
            }
    rows = unblind_manifest(private_manifest=private, decisions=decisions, allow_incomplete=True)
    assert {(row["model"], row["seed"]) for row in rows} == set(labels)
    summary = summarize_review_rows(rows)
    agreement = summary["automatic_vs_human"]
    assert agreement["both_success"] == 2
    assert agreement["both_failure"] == 1
    assert agreement["automatic_failure_human_success"] == 1
    assert agreement["uncertain"] == 1
    paired = summary["paired_model_analysis"]
    seed_seven = next(row for row in paired["task_seed_rows"] if row["seed"] == 7)
    assert (seed_seven["A_human_label"], seed_seven["B_human_label"], seed_seven["C_human_label"]) == (
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
    )
