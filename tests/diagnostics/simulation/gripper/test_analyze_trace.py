from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from evaluation.sim import slip_trace
from evaluation.sim.slip_trace import POST_SUCCESS_SECONDS_ENV
from evaluation.sim.slip_trace import DIAGNOSTIC_LATCH_RAW_ENV
from evaluation.sim.slip_trace import SLIP_TRACE_ENV
from evaluation.sim.slip_trace import SLIP_TRACE_FIELDS
from evaluation.sim.slip_trace import SlipTraceRecorder
from evaluation.sim.slip_trace import SlipTraceSettings
from evaluation.sim.slip_trace import relative_slip_metrics
from diagnostics.simulation.gripper.analyze_trace import _contact_intervals
from diagnostics.simulation.gripper.analyze_trace import _reference_summary


def test_slip_trace_is_strictly_opt_in() -> None:
    assert SlipTraceSettings.from_environment({}) == SlipTraceSettings(enabled=False)
    assert SlipTraceSettings.from_environment({SLIP_TRACE_ENV: "true"}) == SlipTraceSettings(
        enabled=False
    )
    enabled = SlipTraceSettings.from_environment({SLIP_TRACE_ENV: "1"})
    assert enabled.enabled
    assert enabled.post_success_seconds == 2.0


def test_slip_trace_post_success_duration_validation() -> None:
    settings = SlipTraceSettings.from_environment(
        {SLIP_TRACE_ENV: "1", POST_SUCCESS_SECONDS_ENV: "1.25"}
    )
    assert settings.post_success_seconds == 1.25
    with pytest.raises(ValueError, match="finite and non-negative"):
        SlipTraceSettings.from_environment(
            {SLIP_TRACE_ENV: "1", POST_SUCCESS_SECONDS_ENV: "-1"}
        )


def test_diagnostic_latch_is_opt_in_and_range_checked() -> None:
    settings = SlipTraceSettings.from_environment(
        {SLIP_TRACE_ENV: "1", DIAGNOSTIC_LATCH_RAW_ENV: "50"}
    )
    assert settings.diagnostic_latch_raw == 50.0
    assert SlipTraceSettings.from_environment(
        {DIAGNOSTIC_LATCH_RAW_ENV: "50"}
    ).diagnostic_latch_raw is None
    with pytest.raises(ValueError, match=r"\[50, 845\]"):
        SlipTraceSettings.from_environment(
            {SLIP_TRACE_ENV: "1", DIAGNOSTIC_LATCH_RAW_ENV: "49.9"}
        )


def test_relative_slip_uses_tcp_minus_object_sign_convention() -> None:
    reference = np.array([0.01, -0.02, 0.03])
    drift, downward = relative_slip_metrics(np.array([0.013, -0.024, 0.041]), reference)
    assert drift == pytest.approx(np.linalg.norm([0.003, -0.004, 0.011]))
    assert downward == pytest.approx(0.011)
    _, upward = relative_slip_metrics(np.array([0.01, -0.02, 0.02]), reference)
    assert upward == 0.0
    assert relative_slip_metrics(np.zeros(3), None) == (None, None)


def test_trace_csv_is_finalized_atomically(tmp_path: Path) -> None:
    recorder = SlipTraceRecorder(output_dir=tmp_path, target_body="object")
    row = {field: 0 for field in SLIP_TRACE_FIELDS}
    row["sim_time_s"] = 0.002
    row["relative_downward_slip_m"] = None
    recorder.rows.append(row)
    target = recorder.write()
    assert target == tmp_path / "slip_trace.csv"
    assert target.is_file()
    assert not (tmp_path / "slip_trace.csv.tmp").exists()
    with target.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["sim_time_s"] == "0.002"
    assert rows[0]["relative_downward_slip_m"] == ""


def test_trace_schema_contains_requested_grasp_and_collision_evidence() -> None:
    required = {
        "relative_3d_drift_m",
        "relative_downward_slip_m",
        "target_gripper_contact_count",
        "gripper_raw_command",
        "gripper_ctrl_target",
        "actual_gripper_state",
        "left_finger_table_contact",
        "right_finger_table_contact",
        "target_table_contact",
        "target_vertical_velocity_mps",
        "tcp_vertical_velocity_mps",
        "fingertip_table_max_normal_force_n",
    }
    assert required.issubset(SLIP_TRACE_FIELDS)


def test_recorder_reference_contact_and_per_finger_table_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    positions = {
        "object": np.array([0.4, 0.0, 0.06]),
        "tcp": np.array([0.4, 0.0, 0.10]),
    }
    monkeypatch.setattr(
        slip_trace,
        "_body_position",
        lambda _model, _data, _name: positions["object"].copy(),
    )
    monkeypatch.setattr(
        slip_trace,
        "_tcp_position",
        lambda _model, _data: positions["tcp"].copy(),
    )
    monkeypatch.setattr(
        slip_trace,
        "_free_body_linear_velocity",
        lambda _model, _data, _name: np.array([0.0, 0.0, -0.02]),
    )
    monkeypatch.setattr(
        slip_trace,
        "_maximum_normal_force",
        lambda _model, _data, contacts: 3.5 if contacts else 0.0,
    )
    monkeypatch.setattr(
        slip_trace,
        "get_robot_state",
        lambda _model, _data, _config: np.array([0, 0, 0, 0, 0, 0, 205.0]),
    )
    collision = {
        "contacts": [
            {
                "contact_index": 0,
                "body1": "object",
                "body2": "left_finger",
                "geom1": "object_geom",
                "geom2": "left_fingertip_pad",
                "distance_m": -0.0002,
            },
            {
                "contact_index": 1,
                "body1": "world",
                "body2": "left_finger",
                "geom1": "table",
                "geom2": "left_fingertip_pad",
                "distance_m": -0.0004,
            },
        ]
    }
    recorder = SlipTraceRecorder(output_dir=tmp_path, target_body="object")
    data = type("Data", (), {"time": 1.0})()
    recorder.sample(
        model=object(),
        data=data,
        camera_config={},
        policy_step=7,
        executed_action_index=35,
        action_index_in_chunk=0,
        gripper_raw_command=195.0,
        gripper_raw_command_clamped=195.0,
        gripper_ctrl_target=240.0,
        collision=collision,
        original_v1_success_reached=False,
        post_success_diagnostic=False,
    )
    first = recorder.rows[0]
    assert first["relative_3d_drift_m"] == 0.0
    assert first["relative_downward_slip_m"] == 0.0
    assert first["left_finger_target_contact_count"] == 1
    assert first["right_finger_target_contact_count"] == 0
    assert first["left_finger_table_contact"] is True
    assert first["right_finger_table_contact"] is False
    assert first["fingertip_table_max_normal_force_n"] == 3.5
    assert first["fingertip_table_min_distance_m"] == -0.0004
    assert first["actual_gripper_state"] == 205.0

    positions["object"] = np.array([0.4, 0.0, 0.055])
    data.time = 1.002
    recorder.sample(
        model=object(),
        data=data,
        camera_config={},
        policy_step=7,
        executed_action_index=35,
        action_index_in_chunk=0,
        gripper_raw_command=195.0,
        gripper_raw_command_clamped=195.0,
        gripper_ctrl_target=240.0,
        collision={"contacts": []},
        original_v1_success_reached=False,
        post_success_diagnostic=False,
    )
    second = recorder.rows[1]
    assert second["relative_downward_slip_m"] == pytest.approx(0.005)
    assert second["relative_3d_drift_m"] == pytest.approx(0.005)


def test_analysis_reference_sensitivity_uses_requested_reference_row() -> None:
    rows = [
        {
            "sim_time_s": str(index * 0.1),
            "policy_step": str(index),
            "executed_action_index": str(index),
            "relative_x_m": "0.0",
            "relative_y_m": "0.0",
            "relative_z_m": str(value),
        }
        for index, value in enumerate((0.01, 0.012, 0.018, 0.017))
    ]
    summary = _reference_summary(rows, rows[1])
    assert summary is not None
    assert summary["reference_time_s"] == pytest.approx(0.1)
    assert summary["maximum_relative_downward_slip_m"] == pytest.approx(0.006)
    assert summary["final_relative_downward_slip_m"] == pytest.approx(0.005)
    assert summary["first_relative_downward_slip_1mm_time_s"] == pytest.approx(0.2)


def test_analysis_contact_intervals_do_not_merge_separate_events() -> None:
    rows = [{"contact": value} for value in (False, True, True, False, True, False)]
    intervals = _contact_intervals(rows, lambda row: bool(row["contact"]))
    assert [len(interval) for interval in intervals] == [2, 1]
