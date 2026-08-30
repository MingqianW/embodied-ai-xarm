from __future__ import annotations

import sys

import numpy as np
import pytest

from evaluation.common.contracts import EvaluationTask
from evaluation.common.human_review import HumanReviewRecord
from evaluation.real.results import as_common_result
from evaluation.real.results import build_real_episode_result
from evaluation.real.run_policy import main
from evaluation.real.run_policy import _run_hardware_session
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


def test_real_execution_uses_limit_validated_target_before_sdk_command() -> None:
    limits = np.asarray([[-1.0, 1.0]] * 6, dtype=np.float32)
    current = np.asarray([0.98, 0, 0, 0, 0, 0], dtype=np.float32)

    class FakeApi:
        def __init__(self) -> None:
            self.commands = []

        def set_servo_angle(self, **kwargs):
            self.commands.append(kwargs["angle"])
            return 0

        def set_gripper_position(self, *args, **kwargs):
            return 0

    class FakeRobot:
        def __init__(self) -> None:
            self.api = FakeApi()

        def get_current_joint(self):
            return np.rad2deg(current)

        def get_gripper_state(self):
            return 500.0

    robot = FakeRobot()
    requested = np.asarray([[1.98, 0, 0, 0, 0, 0, 500]], dtype=np.float32)
    delta_only_target = current[0] + 0.1
    assert delta_only_target > limits[0, 1]

    safe_execute_actions(
        robot,
        requested,
        max_steps=1,
        max_joint_delta=0.1,
        joint_limits=limits,
        dt=0.0001,
        authorization=_authorization(authorized=True),
    )

    assert len(robot.api.commands) == 1
    assert robot.api.commands[0][0] == pytest.approx(limits[0, 1])


class _LifecycleRobot:
    def __init__(self, events: list[str], *, cleanup_fails: bool = False) -> None:
        self.events = events
        self.cleanup_fails = cleanup_fails

    def disconnect(self) -> None:
        self.events.append("robot.disconnect")
        if self.cleanup_fails:
            raise RuntimeError("robot cleanup")


class _LifecycleCamera:
    is_ready = True
    n_cameras = 2

    def __init__(
        self,
        events: list[str],
        *,
        start_fails: bool = False,
        cleanup_fails: bool = False,
    ) -> None:
        self.events = events
        self.start_fails = start_fails
        self.cleanup_fails = cleanup_fails

    def start(self, *, wait: bool) -> None:
        self.events.append("camera.start")
        if self.start_fails:
            raise RuntimeError("camera start")

    def stop(self, *, wait: bool) -> None:
        self.events.append("camera.stop")
        if self.cleanup_fails:
            raise RuntimeError("camera cleanup")


class _TrainingConfig:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def get_config(self, name: str):
        self.events.append("training.get_config")
        return {"name": name}


class _PolicyConfig:
    def __init__(self, events: list[str], *, fails: bool = False) -> None:
        self.events = events
        self.fails = fails

    def create_trained_policy(self, config, checkpoint):
        self.events.append("policy.load")
        if self.fails:
            raise RuntimeError("policy load")
        return object()


def test_hardware_lifecycle_robot_factory_failure_acquires_nothing() -> None:
    events = []

    def robot_factory(**kwargs):
        events.append("robot.create")
        raise RuntimeError("robot create")

    with pytest.raises(RuntimeError, match="robot create"):
        _run_hardware_session(
            _TrainingConfig(events),
            _PolicyConfig(events),
            robot_factory,
            lambda: pytest.fail("camera must not be created"),
        )
    assert events == ["robot.create"]


def test_hardware_lifecycle_camera_factory_failure_cleans_robot() -> None:
    events = []

    def camera_factory():
        events.append("camera.create")
        raise RuntimeError("camera create")

    with pytest.raises(RuntimeError, match="camera create"):
        _run_hardware_session(
            _TrainingConfig(events),
            _PolicyConfig(events),
            lambda **kwargs: _LifecycleRobot(events),
            camera_factory,
        )
    assert events == ["camera.create", "robot.disconnect"]


def test_hardware_lifecycle_camera_start_failure_cleans_reverse_order() -> None:
    events = []
    with pytest.raises(RuntimeError, match="camera start"):
        _run_hardware_session(
            _TrainingConfig(events),
            _PolicyConfig(events),
            lambda **kwargs: _LifecycleRobot(events),
            lambda: _LifecycleCamera(events, start_fails=True),
        )
    assert events == ["camera.start", "camera.stop", "robot.disconnect"]


def test_hardware_lifecycle_policy_load_failure_cleans_reverse_order() -> None:
    events = []
    with pytest.raises(RuntimeError, match="policy load"):
        _run_hardware_session(
            _TrainingConfig(events),
            _PolicyConfig(events, fails=True),
            lambda **kwargs: _LifecycleRobot(events),
            lambda: _LifecycleCamera(events),
        )
    assert events[-2:] == ["camera.stop", "robot.disconnect"]


def test_hardware_lifecycle_runtime_failure_preserves_original_and_continues_cleanup() -> None:
    events = []

    def fail_runtime(robot, cameras, policy):
        events.append("runtime")
        raise RuntimeError("runtime failure")

    with pytest.raises(RuntimeError, match="runtime failure"):
        _run_hardware_session(
            _TrainingConfig(events),
            _PolicyConfig(events),
            lambda **kwargs: _LifecycleRobot(events, cleanup_fails=True),
            lambda: _LifecycleCamera(events, cleanup_fails=True),
            session_runner=fail_runtime,
        )
    assert events[-3:] == ["runtime", "camera.stop", "robot.disconnect"]


def test_real_entrypoint_refuses_hardware_by_default(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluation.real.run_policy"])
    with pytest.raises(SystemExit, match="Hardware access is disabled by default"):
        main()
