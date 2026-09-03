from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
from mujoco.glfw import glfw

from simulation.tools import teleoperate_pick


def _state() -> teleoperate_pick.TeleoperationState:
    return teleoperate_pick.TeleoperationState(
        action=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 840.0]),
        applied_action=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 840.0]),
        gripper_raw_limits=np.asarray([50.0, 845.0]),
        gripper_step_raw=25.0,
        cartesian_step_m=0.01,
        max_arm_joint_speed_rad_s=0.5,
        seed=7,
    )


def test_task_selection_accepts_number_alias_and_default() -> None:
    assert teleoperate_pick.select_task(None, input_fn=lambda _: "1") == "red_pepper"
    assert (
        teleoperate_pick.select_task(
            None,
            input_fn=lambda _: "pick up the blue block",
        )
        == "blue_block"
    )
    assert teleoperate_pick.select_task(None, input_fn=lambda _: "") == "red_block"
    assert teleoperate_pick.select_task("smallest") == "smallest_block"


def test_gripper_keys_use_raw_hardware_units_and_clip() -> None:
    state = _state()
    assert state.handle_key(glfw.KEY_O)
    assert state.action[6] == 845.0
    assert state.handle_key(glfw.KEY_C)
    assert state.action[6] == 820.0
    for _ in range(100):
        state.handle_key(glfw.KEY_C)
    assert state.action[6] == 50.0


def test_arm_targets_are_slewed_but_gripper_remains_immediate() -> None:
    applied = np.asarray([0.0, 0.1, -0.1, 0.0, 0.0, 0.0, 840.0])
    desired = np.asarray([1.0, -1.0, 1.0, -1.0, 0.5, -0.5, 50.0])

    next_action = teleoperate_pick._slew_arm_target(
        applied,
        desired,
        period_s=0.02,
        max_arm_joint_speed_rad_s=0.5,
    )

    np.testing.assert_allclose(
        next_action[:6],
        np.asarray([0.01, 0.09, -0.09, -0.01, 0.01, -0.01]),
    )
    assert next_action[6] == 50.0


def test_motion_reset_seed_status_and_task_keys() -> None:
    state = _state()
    state.handle_key(glfw.KEY_UP)
    assert len(state.motion_requests) == 1
    np.testing.assert_array_equal(
        state.motion_requests[0],
        np.asarray([1.0, 0.0, 0.0]),
    )

    state.handle_key(glfw.KEY_N)
    assert state.seed == 8
    assert state.reset_requested
    state.handle_key(glfw.KEY_M)
    assert state.status_requested
    state.handle_key(glfw.KEY_6)
    assert state.next_task == "place_red_pepper_in_ring"


def test_list_tasks_does_not_start_viewer(capsys) -> None:
    teleoperate_pick.main(["--list-tasks"])
    output = capsys.readouterr().out
    assert "red_pepper" in output
    assert "place_red_pepper_in_ring" in output


def test_one_control_step_runs_without_a_real_gui(monkeypatch) -> None:
    class FakeViewer:
        def __init__(self) -> None:
            self.cam = SimpleNamespace(type=None, fixedcamid=None)
            self.running = True

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def lock(self):
            return nullcontext()

        def is_running(self) -> bool:
            return self.running

        def sync(self) -> None:
            self.running = False

    monkeypatch.setattr(
        teleoperate_pick.mujoco.viewer,
        "launch_passive",
        lambda *args, **kwargs: FakeViewer(),
    )
    original_reset = teleoperate_pick.MuJoCoEnvironment.reset

    def reset_without_policy_observation(*args, **kwargs):
        assert kwargs["build_policy_observation"] is False
        return original_reset(*args, **kwargs)

    monkeypatch.setattr(
        teleoperate_pick.MuJoCoEnvironment,
        "reset",
        reset_without_policy_observation,
    )
    args = teleoperate_pick.parse_args(
        [
            "--task",
            "red_block",
            "--settle-steps",
            "0",
            "--object-xy-range-m",
            "0",
            "--object-yaw-range-deg",
            "0",
            "--joint-noise-rad",
            "0",
        ]
    )

    assert teleoperate_pick._run_task("red_block", args) is None
