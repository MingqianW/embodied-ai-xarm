from __future__ import annotations

import sys

import numpy as np
import pytest

from evaluation.common.contracts import EvaluationTask
from evaluation.common.human_review import HumanReviewRecord
from evaluation.real.results import as_common_result
from evaluation.real.results import build_real_episode_result
from evaluation.real.run_policy import main
from evaluation.real.run_policy import safe_execute_actions
from evaluation.real.safety import RealExecutionAuthorization
from evaluation.real.safety import validate_real_action_chunk
from policy_runtime.safety import SafetyConfig


def _authorization(*, authorized: bool) -> RealExecutionAuthorization:
    return RealExecutionAuthorization(
        operator_present=authorized,
        workspace_clear=authorized,
        emergency_stop_accessible=authorized,
        robot_motion_confirmed=authorized,
    )


def _real_document(review: HumanReviewRecord | None = None) -> dict[str, object]:
    return build_real_episode_result(
        run_id="run-1",
        trial_id="trial-1",
        model={"model_id": "A", "checkpoint": "/external/checkpoint"},
        task=EvaluationTask("red_block", "pick up the red block"),
        execution_metadata={"observed_frames": 3},
        safety={"operator_supervised": True},
        provenance={"runtime": "external"},
        artifacts={"telemetry": "episode/robot_log.csv"},
        human_review=review,
    )


def test_real_result_remains_unreviewed_without_success_perception() -> None:
    document = _real_document()
    assert document["episode"]["automatic_success"] is None
    assert document["success_measurement"]["automatic_detector_available"] is False
    normalized = as_common_result(document)
    assert normalized.run.backend == "real"
    assert normalized.outcome == "unreviewed"


def test_real_result_uses_shared_human_review_not_telemetry() -> None:
    review = HumanReviewRecord("review-1", "FAILURE", "PARTIAL_LIFT")
    normalized = as_common_result(_real_document(review))
    assert normalized.outcome == "failure"
    assert normalized.failure_category == "PARTIAL_LIFT"


def test_real_result_rejects_fabricated_automatic_success() -> None:
    document = _real_document()
    document["episode"]["automatic_success"] = True
    with pytest.raises(ValueError, match="cannot be populated automatically"):
        as_common_result(document)


def test_real_safety_gate_rejects_before_any_robot_access() -> None:
    class RobotMustNotBeRead:
        def __getattribute__(self, name):
            raise AssertionError(f"robot was accessed: {name}")

    with pytest.raises(PermissionError, match="Real-robot motion requires"):
        safe_execute_actions(
            RobotMustNotBeRead(),
            np.zeros((1, 7), dtype=np.float32),
            authorization=_authorization(authorized=False),
        )


def test_real_action_validation_is_offline_and_clips_existing_conventions() -> None:
    result = validate_real_action_chunk(
        np.asarray([[0.2, 0, 0, 0, 0, 0, 900, 123]], dtype=np.float32),
        current_state=np.zeros(7, dtype=np.float32),
        joint_limits=np.asarray([[-1, 1]] * 6, dtype=np.float32),
        authorization=_authorization(authorized=True),
        config=SafetyConfig(max_joint_delta_rad=0.1, gripper_min=50, gripper_max=845),
    )
    assert result.accepted
    assert result.clipped
    assert result.actions.shape == (1, 7)
    assert result.actions[0, 0] == pytest.approx(0.1)
    assert result.actions[0, 6] == pytest.approx(845)


def test_real_entrypoint_refuses_hardware_by_default(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluation.real.run_policy"])
    with pytest.raises(SystemExit, match="Hardware access is disabled by default"):
        main()
