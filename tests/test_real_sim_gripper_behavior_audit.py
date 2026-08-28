from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim_mujoco.scripts.audit_real_sim_gripper_behavior import (
    _menagerie_baseline,
    motion_segments,
    next_state_label_metrics,
)


def _states(gripper: list[float]) -> np.ndarray:
    values = np.zeros((len(gripper), 7), dtype=np.float64)
    values[:, 6] = gripper
    return values


def test_next_state_label_metrics_detects_exact_shift() -> None:
    states = _states([800.0, 700.0, 500.0, 300.0])
    labels = _states([700.0, 500.0, 300.0, 250.0])

    result = next_state_label_metrics([states], [labels])

    assert result["label_t_vs_state_t_plus_1"]["fraction_exactly_equal"] == 1.0
    assert result["label_t_vs_state_t_plus_1"]["maximum_absolute_difference_raw"] == 0.0
    assert result["same_frame"]["mean_absolute_difference_raw"] > 0.0


def test_motion_segments_uses_feedback_direction_and_merges_short_pause() -> None:
    values = np.asarray([800.0, 750.0, 700.0, 700.0, 650.0, 600.0, 600.0, 620.0, 650.0])
    times = np.arange(len(values), dtype=np.float64) * 0.1

    rows = motion_segments(
        values,
        times,
        episode_index=3,
        task="pick up the red block",
        source="real",
    )

    assert [row["direction"] for row in rows] == ["closing", "opening"]
    assert rows[0]["start_state_raw"] == 800.0
    assert rows[0]["end_state_raw"] == 600.0
    assert rows[0]["duration_s"] == 0.5
    assert rows[0]["true_command_available"] is False


def test_motion_segments_rejects_noise_only_motion() -> None:
    values = np.asarray([500.0, 500.5, 499.5, 500.2, 500.0])
    times = np.arange(len(values), dtype=np.float64) * 0.1

    assert not motion_segments(
        values,
        times,
        episode_index=0,
        task="task",
        source="real",
    )


def test_menagerie_summary_preserves_missing_penetration_as_null(
    tmp_path: Path,
) -> None:
    (tmp_path / "open_close").mkdir()
    (tmp_path / "grasp_slip/analysis_v2").mkdir(parents=True)
    (tmp_path / "open_close/open_close_validation.json").write_text(
        json.dumps(
            {
                "raw_sweep": [{"raw_round_trip_error": 1.0}],
                "measured_direction": "ctrl=0 opens; ctrl=255 closes",
                "maximum_driver_symmetry_error_rad": 1e-7,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "grasp_slip/analysis_v2/summary.json").write_text(
        json.dumps(
            [
                {
                    "hold_kind": "suspended",
                    "command_variant": "oracle_command",
                    "mechanically_stable_2mm": False,
                    "sustained_contact_loss_s": 1.0,
                    "maximum_downward_slip_m": 0.1,
                    "task": "red_block",
                    "hold_bilateral_contact_fraction": 0.0,
                    "mean_bilateral_normal_force_sum_n": None,
                    "maximum_target_penetration_m": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = _menagerie_baseline(tmp_path)

    assert result is not None
    assert (
        result["suspended_oracle"]["by_task"]["red_block"][
            "maximum_target_penetration_mm"
        ]
        is None
    )
