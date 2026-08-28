from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from evaluation.sim.config import load_protocol
from evaluation.sim.episode_runner import record_executed_target_clipping
from evaluation.sim.episode_runner import validate_formal_action_chunk
from evaluation.sim.failure_diagnosis import diagnose_episode_failure
from evaluation.common.models import ModelSpec
from evaluation.common.models import validate_abc_comparison_specs
from evaluation.common.models import validate_model_spec
from evaluation.sim.outputs import EPISODE_SCHEMA_VERSION
from evaluation.sim.outputs import LEGACY_EPISODE_SCHEMA_VERSION
from evaluation.sim.outputs import initialize_output
from evaluation.sim.outputs import upgrade_result_with_failure_diagnosis
from evaluation.sim.outputs import validate_episode_result
from evaluation.sim.rng import policy_rng_seed
from evaluation.sim.result_contract import as_common_result
from evaluation.sim.success import PickSuccess
from evaluation.sim.success import PlacementSuccess
from evaluation.sim.summary import build_video_index
from evaluation.sim.summary import summarize_results


@pytest.fixture(scope="module")
def protocol():
    return load_protocol()


def _provenance() -> dict[str, object]:
    return {
        "backend": "sim",
        "evaluation_protocol_version": "xarm-pi05-formal-evaluation-v1",
        "protocol_sha256": "protocol",
        "model_spec_sha256": "model",
        "provenance_sha256": "provenance",
        "protocol": {},
        "openpi_git_commit": "openpi",
        "embodied_ai_xarm_git_commit": "embodied",
        "paths": {},
    }


def _result(*, success: bool, valid: bool, task: str = "red_block") -> dict[str, object]:
    diagnosis = diagnose_episode_failure(
        task_id=task,
        success=success,
        valid=valid,
        metrics={"max_lift_m": 0.06, "lift_threshold_m": 0.05},
    )
    return {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "backend": "sim",
        "evaluation_protocol_version": "xarm-pi05-formal-evaluation-v1",
        "timestamp_utc": "2026-08-07T00:00:00+00:00",
        "model": {
            "model_id": "A",
            "training_config": "pi05_xarm_real50_sim50_stratified",
            "checkpoint_root": "/work/checkpoints/A",
            "manager_step": 15000,
            "resolved_manager_directory": "/work/checkpoints/A/15000",
            "norm_asset_id": "xarm_pi05_real_v3sim_1x",
        },
        "episode": {
            "task": task,
            "prompt": "pick up the red block",
            "seed": 50000,
            "success": success,
            "valid": valid,
            "termination_reason": "task_success" if success else "max_policy_steps",
            "invalid_reason": None if valid else "policy_error",
            "failure_category": diagnosis.category,
            "failure_reason": diagnosis.reason,
            "failure_stage": diagnosis.stage,
            "policy_steps": 3,
            "executed_actions": 15,
        },
        "metrics": {"max_lift_m": 0.06},
        "failure_diagnostics": diagnosis.diagnostics,
        "safety": {},
        "initial_state": {},
        "final_state": {},
        "provenance": _provenance(),
        "artifacts": {},
    }


def test_formal_result_has_backend_explicit_common_contract() -> None:
    result = as_common_result(_result(success=True, valid=True))
    assert result.run.backend == "sim"
    assert result.episode.task.task_id == "red_block"
    assert result.outcome == "success"


def test_pre_phase4_formal_result_without_backend_remains_compatible() -> None:
    document = _result(success=False, valid=True)
    document.pop("backend")
    document["provenance"].pop("backend")
    validate_episode_result(document)
    assert as_common_result(document).run.backend == "sim"


def test_protocol_has_exact_formal_control_and_prompts(protocol) -> None:
    assert (protocol.execute_chunk_steps, protocol.policy_action_horizon, protocol.max_policy_steps) == (5, 10, 50)
    assert protocol.video_policy == "category_representative"
    assert protocol.representatives_per_category == 1
    assert (protocol.pick_success_checks, protocol.pick_post_success_hold_checks) == (3, 3)
    assert protocol.pick_max_post_success_drop_m == 0.005
    assert [task.prompt for task in protocol.tasks] == [
        "pick up the red pepper",
        "pick up the blue block",
        "pick up the red block",
        "pick up the smallest block",
        "pick up the largest block",
        "place the red pepper in the ring",
    ]


def test_smoke_protocol_preserves_control_semantics_with_two_seeds() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/evaluation/sim/protocols/formal_xarm_pi05_eval_smoke_v2.json"
    smoke = load_protocol(path)
    assert smoke.seed_count == 2
    assert (smoke.execute_chunk_steps, smoke.policy_action_horizon, smoke.max_policy_steps) == (5, 10, 50)
    assert smoke.video_policy == "all"
    assert smoke.pick_post_success_hold_checks == 3


def test_all_video_protocol_is_explicit_and_has_an_isolated_output_root(protocol) -> None:
    path = Path(__file__).resolve().parents[1] / "configs/evaluation/sim/protocols/formal_xarm_pi05_eval_video_all_v2.json"
    all_video = load_protocol(path)
    assert all_video.video_policy == "all"
    assert all_video.representatives_per_category == 1
    assert all_video.output_root != protocol.output_root


def test_legacy_v1_protocol_remains_loadable_without_post_success_hold() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/evaluation/sim/protocols/formal_xarm_pi05_eval_v1.json"
    legacy = load_protocol(path)
    assert legacy.pick_post_success_hold_checks == 0
    assert legacy.pick_max_post_success_drop_m == 0.0


def test_abc_specs_use_explicit_15000_checkpoints_and_expected_norm_assets() -> None:
    specs = Path(__file__).resolve().parents[1] / "configs/evaluation/sim/models"
    values = {name: json.loads((specs / f"{name}.json").read_text(encoding="utf-8")) for name in ("A", "B", "C")}
    assert {value["manager_step"] for value in values.values()} == {15000}
    assert values["A"]["norm_asset_id"] == "xarm_pi05_real_v3sim_1x"
    assert values["B"]["norm_asset_id"] == values["C"]["norm_asset_id"] == "xarm_pi05_real_v4sim_10x"


def test_comparison_guard_requires_byte_identical_b_c_norms(tmp_path: Path) -> None:
    specs = []
    for model_id, asset, contents in (
        ("A", "xarm_pi05_real_v3sim_1x", "a"),
        ("B", "xarm_pi05_real_v4sim_10x", "bc"),
        ("C", "xarm_pi05_real_v4sim_10x", "bc"),
    ):
        root = tmp_path / model_id
        norm = root / "15000" / "assets" / asset / "norm_stats.json"
        norm.parent.mkdir(parents=True)
        norm.write_text(contents, encoding="utf-8")
        specs.append(ModelSpec(model_id, f"config_{model_id}", root, 15000, asset))
    report = validate_abc_comparison_specs(tuple(specs))
    assert report["manager_step"] == 15000
    c_norm = tmp_path / "C" / "15000" / "assets" / "xarm_pi05_real_v4sim_10x" / "norm_stats.json"
    c_norm.write_text("different", encoding="utf-8")
    with pytest.raises(ValueError, match="byte-for-byte"):
        validate_abc_comparison_specs(tuple(specs))


def test_model_spec_requires_exact_manager_params_and_embedded_norm(tmp_path: Path) -> None:
    manager = tmp_path / "run" / "15000"
    (manager / "params").mkdir(parents=True)
    (manager / "params" / "manifest.ocdbt").write_text("manifest", encoding="utf-8")
    norm = manager / "assets" / "xarm_pi05_real_v4sim_10x" / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}", encoding="utf-8")
    spec = ModelSpec("B", "pi05_xarm_real1_sim10_stratified", manager.parent, 15000, "xarm_pi05_real_v4sim_10x")
    validate_model_spec(spec)
    norm.unlink()
    with pytest.raises(FileNotFoundError):
        validate_model_spec(spec)


def test_resume_rejects_changed_model_or_protocol_provenance(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    initialize_output(output_root=root, model_id="A", provenance=_provenance(), resume=False)
    changed = {**_provenance(), "model_spec_sha256": "other"}
    with pytest.raises(ValueError, match="provenance"):
        initialize_output(output_root=root, model_id="A", provenance=changed, resume=True)


def test_policy_rng_is_stable_and_order_independent() -> None:
    later = policy_rng_seed(protocol_salt="salt", task_id="red_block", evaluation_seed=50001, policy_step=7)
    _ = [
        policy_rng_seed(protocol_salt="salt", task_id="red_pepper", evaluation_seed=50000, policy_step=step)
        for step in range(2)
    ]
    assert later == policy_rng_seed(protocol_salt="salt", task_id="red_block", evaluation_seed=50001, policy_step=7)
    assert later != policy_rng_seed(protocol_salt="salt", task_id="red_block", evaluation_seed=50002, policy_step=7)


def test_full_chunk_is_validated_but_discarded_actions_do_not_trigger_safety_rejection(protocol) -> None:
    actions = np.zeros((10, 7), dtype=np.float32)
    actions[5:, :6] = 100.0  # deliberately unsafe but never executed under c5
    full, prefix = validate_formal_action_chunk(
        actions,
        current_state=np.zeros(7, dtype=np.float32),
        joint_limits=np.tile(np.array([[-10.0, 10.0]], dtype=np.float32), (6, 1)),
        protocol=protocol,
    )
    assert full.shape == (10, 7)
    assert prefix.accepted
    actions[9, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_formal_action_chunk(
            actions,
            current_state=np.zeros(7, dtype=np.float32),
            joint_limits=np.tile(np.array([[-10.0, 10.0]], dtype=np.float32), (6, 1)),
            protocol=protocol,
        )


def test_clipping_accounting_counts_each_executed_target() -> None:
    safety = {
        "executed_action_count": 0,
        "clipped_action_count": 0,
        "per_dimension_clip_counts": [0] * 7,
        "max_requested_vs_executed_delta": 0.0,
    }
    clipped = SimpleNamespace(
        clipped=True,
        raw_action=np.array([0.1, 0, 0, 0, 0, 0, 900.0]),
        arm_target_clamped=np.array([0.05, 0, 0, 0, 0, 0]),
        gripper_raw=900.0,
        gripper_raw_clamped=845.0,
    )
    unclipped = SimpleNamespace(clipped=False)
    record_executed_target_clipping(safety, clipped)
    record_executed_target_clipping(safety, unclipped)
    record_executed_target_clipping(safety, clipped)
    assert safety["executed_action_count"] == 3
    assert safety["clipped_action_count"] == 2
    assert safety["per_dimension_clip_counts"] == [2, 0, 0, 0, 0, 0, 2]
    assert safety["max_requested_vs_executed_delta"] == 55.0


def test_pick_requires_three_initial_and_three_post_success_hold_checks(protocol) -> None:
    metric = PickSuccess(
        0.10,
        protocol.pick_lift_height_m,
        protocol.pick_meaningful_lift_diagnostic_m,
        protocol.pick_success_checks,
        protocol.pick_post_success_hold_checks,
        protocol.pick_max_post_success_drop_m,
    )
    assert not metric.update(0.151, 0)["task_success"]
    assert not metric.update(0.152, 0)["task_success"]
    provisional = metric.update(0.153, 0)
    assert provisional["provisional_success_active"]
    assert not provisional["task_success"]
    assert not metric.update(0.154, 0)["task_success"]
    assert not metric.update(0.155, 0)["task_success"]
    assert metric.update(0.156, 0)["task_success"]
    assert metric.update(0.10, 0)["success_confirmation_count"] == 0


def test_pick_post_success_downward_slip_resets_the_hold(protocol) -> None:
    metric = PickSuccess(
        0.10,
        protocol.pick_lift_height_m,
        protocol.pick_meaningful_lift_diagnostic_m,
        protocol.pick_success_checks,
        protocol.pick_post_success_hold_checks,
        protocol.pick_max_post_success_drop_m,
    )
    for height in (0.151, 0.152, 0.153):
        metric.update(height, 0)
    metric.observe_post_success_hold(0.147)  # 6 mm downward slip from the hold peak.
    result = metric.update(0.147, 0)
    assert not result["task_success"]
    assert not result["provisional_success_active"]
    assert result["post_success_slip_ever"]
    assert result["post_success_max_downward_slip_m"] == pytest.approx(0.006)


def test_placement_requires_release_containment_and_stability(protocol) -> None:
    metric = PlacementSuccess(protocol)
    common = {
        "pepper_position_m": np.array([0.01, 0.0, 0.072]),
        "ring_position_m": np.array([0.0, 0.0, 0.052]),
        "pepper_linear_speed_mps": 0.001,
        "pepper_angular_speed_radps": 0.01,
        "pepper_gripper_distance_m": 0.06,
        "gripper_contact_count": 0,
        "gripper_raw": 700.0,
        "release_requested": True,
    }
    assert not metric.update(**{**common, "gripper_contact_count": 1})["release_confirmed"]
    assert not metric.update(**{**common, "pepper_position_m": np.array([0.04, 0.0, 0.072])})["containment_confirmed"]
    assert not metric.update(**{**common, "pepper_linear_speed_mps": 0.5})["stability_confirmed"]
    assert not metric.update(**common)["task_success"]
    assert not metric.update(**common)["task_success"]
    assert metric.update(**common)["task_success"]


def test_schema_and_summary_report_invalid_denominator() -> None:
    rows = [_result(success=True, valid=True), _result(success=False, valid=True), _result(success=False, valid=False)]
    for row in rows:
        validate_episode_result(row)
    summary = summarize_results(rows)
    assert summary["attempted_episodes"] == 3
    assert summary["valid_episodes"] == 2
    assert summary["invalid_episodes"] == 1
    assert summary["success_rate_valid"] == 0.5
    assert summary["success_rate_all"] == pytest.approx(1 / 3)
    assert summary["failure_category_counts"] == {"PICK_REACHED_HEIGHT_BUT_NOT_SUSTAINED": 1}


def test_formal_pipeline_has_no_legacy_29999_selection() -> None:
    source = Path(__file__).resolve().parents[1] / "evaluation" / "sim" / "cli.py"
    assert "29999" not in source.read_text(encoding="utf-8")


def _pick_failure(*, max_lift: float, final_lift: float | None = None, streak: int = 0) -> object:
    return diagnose_episode_failure(
        task_id="red_block",
        success=False,
        valid=True,
        metrics={
            "max_lift_m": max_lift,
            "final_lift_m": max_lift if final_lift is None else final_lift,
            "lift_threshold_m": 0.05,
            "meaningful_lift_diagnostic_threshold_m": 0.005,
            "max_success_confirmation_count": streak,
            "required_success_confirmation_count": 3,
            "target_gripper_contact_ever": False,
        },
    )


def test_pick_failure_taxonomy_uses_measured_lift_history() -> None:
    no_lift = _pick_failure(max_lift=0.001)
    assert (no_lift.category, no_lift.stage) == ("PICK_NO_MEANINGFUL_LIFT", "approach")
    partial = _pick_failure(max_lift=0.041859)
    assert partial.category == "PICK_PARTIAL_LIFT"
    reached = _pick_failure(max_lift=0.06, final_lift=0.06, streak=2)
    assert reached.category == "PICK_REACHED_HEIGHT_BUT_NOT_SUSTAINED"
    dropped = _pick_failure(max_lift=0.06, final_lift=0.01)
    assert dropped.category == "PICK_DROPPED_AFTER_LIFT"
    assert "0.0600" in str(dropped.reason)


def test_pick_failure_diagnosis_reports_post_success_hold_slip() -> None:
    diagnosis = diagnose_episode_failure(
        task_id="red_block",
        success=False,
        valid=True,
        metrics={
            "max_lift_m": 0.060,
            "final_lift_m": 0.054,
            "lift_threshold_m": 0.050,
            "meaningful_lift_diagnostic_threshold_m": 0.005,
            "max_success_confirmation_count": 3,
            "required_success_confirmation_count": 3,
            "provisional_success_ever": True,
            "max_post_success_hold_check_count": 1,
            "required_post_success_hold_checks": 3,
            "post_success_hold_failure_count": 1,
            "post_success_slip_ever": True,
            "post_success_max_downward_slip_m": 0.006,
            "max_post_success_drop_m": 0.005,
        },
    )
    assert diagnosis.category == "PICK_REACHED_HEIGHT_BUT_NOT_SUSTAINED"
    assert diagnosis.stage == "hold"
    assert "0.0060" in str(diagnosis.reason)


def _placement_failure(**overrides: object) -> object:
    metrics: dict[str, object] = {
        "release_requested": True,
        "release_confirmed": True,
        "containment_confirmed": True,
        "height_confirmed": True,
        "stability_confirmed": True,
        "instant_success": False,
        "success_confirmation_count": 0,
        "max_success_confirmation_count": 0,
        "required_success_confirmation_count": 3,
        "pepper_ring_xy_distance_m": 0.01,
        "containment_limit_m": 0.029,
        "pepper_height_above_table_m": 0.02,
        "pepper_linear_speed_mps": 0.001,
        "pepper_angular_speed_radps": 0.01,
    }
    metrics.update(overrides)
    return diagnose_episode_failure(
        task_id="place_red_pepper_in_ring", success=False, valid=True, metrics=metrics
    )


def test_placement_failure_taxonomy_tracks_success_components() -> None:
    assert _placement_failure(release_confirmed=False).category == "PLACE_NOT_RELEASED"
    assert _placement_failure(containment_confirmed=False).category == "PLACE_RELEASED_OUTSIDE_RING"
    assert _placement_failure(height_confirmed=False).category == "PLACE_RELEASED_WRONG_HEIGHT"
    unstable = _placement_failure(stability_confirmed=False)
    assert unstable.category == "PLACE_RELEASED_IN_RING_BUT_UNSTABLE"
    sustained = _placement_failure(instant_success=True, max_success_confirmation_count=2)
    assert sustained.category == "PLACE_STABLE_BUT_NOT_SUSTAINED"
    assert sustained.stage == "stability"


def test_success_and_invalid_episodes_do_not_receive_task_failure_diagnoses() -> None:
    assert diagnose_episode_failure(task_id="red_block", success=True, valid=True, metrics={}).category is None
    invalid = diagnose_episode_failure(task_id="red_block", success=False, valid=False, metrics={})
    assert invalid.category is None
    row = _result(success=False, valid=False)
    assert row["episode"]["invalid_reason"] == "policy_error"
    assert row["episode"]["failure_category"] is None


def test_video_index_and_retroactive_upgrade_are_deterministic() -> None:
    first = _result(success=False, valid=True)
    first["episode"]["seed"] = 50001
    first["artifacts"] = {"combined_video_path": "/work/video_50001.mp4"}
    second = _result(success=False, valid=True)
    second["episode"]["seed"] = 50000
    second["artifacts"] = {"combined_video_path": "/work/video_50000.mp4"}
    index = build_video_index([first, second])
    assert index["selection_policy"] == "lowest_seed_per_model_task_category"
    assert index["representatives"][0]["seed"] == 50000
    legacy = _result(success=False, valid=True)
    legacy["schema_version"] = LEGACY_EPISODE_SCHEMA_VERSION
    for key in ("failure_category", "failure_reason", "failure_stage"):
        del legacy["episode"][key]
    del legacy["failure_diagnostics"]
    upgraded = upgrade_result_with_failure_diagnosis(legacy)
    assert upgraded["schema_version"] == EPISODE_SCHEMA_VERSION
    assert upgraded["episode"]["failure_category"] == "PICK_REACHED_HEIGHT_BUT_NOT_SUSTAINED"
