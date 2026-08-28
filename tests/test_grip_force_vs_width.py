from __future__ import annotations

import json
import math

from sim_mujoco.scripts.analyze_grip_force_vs_width import (
    _authorize_analysis,
    acceptance,
    summarize_values,
)
from sim_mujoco.scripts.run_grip_force_vs_width import (
    CLOSED_RAW,
    MAX_PENETRATION_M,
    OPEN_RAW,
    WIDTHS_MM,
    close_command_schedule,
    evaluate_trial_validity,
    validate_runtime_model,
)


def test_runtime_model_compiles_without_stepping_and_preserves_mechanism():
    validation = validate_runtime_model()

    assert validation["passed"] is True
    assert validation["mj_step_calls"] == 0
    signature = validation["signature"]
    assert signature["actuator"]["gainprm"][0] == 120.0
    assert signature["actuator"]["biasprm"][:3] == [0.0, -120.0, 0.0]
    assert signature["actuator"]["ctrlrange"] == [0.005, 0.85]
    assert signature["actuator"]["forcerange"] == [-50.0, 50.0]
    assert set(signature["faces"]) == {"left", "right"}
    assert all(face["fixed"] for face in signature["faces"].values())
    assert all(face["geom"]["contype"] == 0 for face in signature["faces"].values())
    assert len(signature["pads"]) == 4
    assert len(signature["diagnostic_contact_pairs"]) == 4
    assert signature["mechanism"]["tendon_count"] == 1
    assert signature["mechanism"]["equality_active_at_reset"] == 1
    assert validation["controlled_invariants_identical_across_widths"] is True
    assert [row["width_mm"] for row in validation["placements"]] == list(WIDTHS_MM)
    assert (
        max(row["max_symmetry_placement_error_m"] for row in validation["placements"])
        <= 1e-9
    )


def test_close_schedule_is_fixed_and_reaches_raw_200_once():
    schedule = close_command_schedule()

    assert math.isclose(schedule[0], OPEN_RAW - 24.4)
    assert schedule[-1] == CLOSED_RAW
    assert len(schedule) == 27
    assert schedule.count(CLOSED_RAW) == 1
    assert all(later < earlier for earlier, later in zip(schedule, schedule[1:]))


def test_runtime_force_limit_is_exact_and_changes_no_other_actuator_fields():
    baseline = validate_runtime_model(gripper_force_limit=50.0)["signature"]["actuator"]
    tuned = validate_runtime_model(gripper_force_limit=1.5)["signature"]["actuator"]

    assert tuned["forcerange"] == [-1.5, 1.5]
    for field in ("gainprm", "biasprm", "ctrlrange", "gear"):
        assert tuned[field] == baseline[field]


def _synthetic_summary(width_mm: int, force_n: float) -> dict:
    return {
        "width_mm": width_mm,
        "metrics": {
            "normal_force_total_n": {"mean": force_n},
            "actuator_force_magnitude": {"mean": force_n},
            "actuator_moment_left": {"mean": 1.0},
            "bilateral_contact_fraction": {"mean": 1.0},
            "exact_count_symmetry_fraction": {"mean": 1.0},
            "penetration_m": {"max": 0.0001},
            "unintended_fixture_contact_count": {"max": 0.0},
            "contact_normal_alignment_min": {"min": 1.0},
        },
    }


def test_acceptance_reports_valid_untuned_menagerie_width_result():
    summaries = [
        _synthetic_summary(width, force)
        for width, force in zip(
            WIDTHS_MM, (0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0), strict=True
        )
    ]

    result = acceptance(summaries)

    assert result["classification"] == "VALID_MENAGERIE_FORCE_WIDTH_RESULT"
    assert result["valid_contact_precondition"] is True
    assert result["observed"]["nondecreasing_adjacent_steps"] == 6
    assert math.isclose(result["observed"]["normal_force_25_to_55_ratio"], 1.0 / 15.0)
    assert result["observed"]["actuator_moment_relative_span"] == 0.0


def test_magnitude_summary_preserves_signed_summary_separately():
    signed = summarize_values((-2.0, -1.0))
    magnitude = summarize_values((-2.0, -1.0), magnitude=True)

    assert signed["mean"] == -1.5
    assert magnitude["mean"] == 1.5


def test_acceptance_returns_no_conclusion_when_bilateral_contact_is_invalid():
    summaries = [
        _synthetic_summary(width, force)
        for width, force in zip(
            WIDTHS_MM, (0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0), strict=True
        )
    ]
    summaries[1]["metrics"]["bilateral_contact_fraction"]["mean"] = 0.1

    result = acceptance(summaries)

    assert result["classification"] == "INVALID_CONTACT_PRECONDITION"
    assert result["valid_contact_precondition"] is False


def _valid_trace_row() -> dict:
    return {
        "contacts": {
            "bilateral": True,
            "exact_count_symmetry": 1,
            "penetration_max_m": 0.0001,
            "unintended_fixture_contact_count": 0,
            "normal_alignment_min": 1.0,
        }
    }


def test_runtime_validity_gate_accepts_only_backed_symmetric_contact():
    placement = {"passed": True, "max_symmetry_placement_error_m": 1e-12}

    result = evaluate_trial_validity([_valid_trace_row()] * 500, placement)

    assert result["passed"] is True
    assert result["failed_gates"] == []


def test_runtime_validity_gate_accepts_observed_backed_face_compliance():
    row = _valid_trace_row()
    row["contacts"]["penetration_max_m"] = 0.000669
    placement = {"passed": True, "max_symmetry_placement_error_m": 1e-17}

    result = evaluate_trial_validity([row], placement)

    assert result["passed"] is True


def test_runtime_validity_gate_rejects_penetration_or_unintended_contact():
    placement = {"passed": True, "max_symmetry_placement_error_m": 1e-12}
    bad_row = _valid_trace_row()
    bad_row["contacts"]["penetration_max_m"] = MAX_PENETRATION_M + 1e-6
    bad_row["contacts"]["unintended_fixture_contact_count"] = 1

    result = evaluate_trial_validity([bad_row], placement)

    assert result["passed"] is False
    assert set(result["failed_gates"]) == {
        "maximum_penetration_m",
        "unintended_fixture_contact_count",
    }


def test_corrected_gate_revalidation_authorizes_complete_existing_traces(tmp_path):
    trials = []
    for index, width_mm in enumerate(WIDTHS_MM):
        trace = tmp_path / f"width_{width_mm}.jsonl"
        trace.write_text(
            "".join(json.dumps(_valid_trace_row()) + "\n" for _ in range(500)),
            encoding="utf-8",
        )
        failed_gates = ["maximum_penetration_m"] if index >= 4 else []
        trials.append(
            {
                "width_mm": width_mm,
                "trace": str(trace),
                "fixture_placement": {
                    "passed": True,
                    "max_symmetry_placement_error_m": 1e-17,
                },
                "validity": {
                    "passed": not failed_gates,
                    "failed_gates": failed_gates,
                },
            }
        )
    results = {
        "status": "failed",
        "force_metrics_authorized": False,
        "error": "original 0.5 mm gate",
        "trials": trials,
    }

    trial_rows, provenance = _authorize_analysis(
        tmp_path.resolve(), results, revalidate_existing_traces=True
    )

    assert len(trial_rows) == len(WIDTHS_MM)
    assert provenance["authorization"] == "corrected_gate_revalidation"
    assert all(row["validity"]["passed"] for row in provenance["revalidated_trials"])


def test_no_contact_outcome_can_be_reported_but_not_authorized(tmp_path):
    trials = []
    for width_mm in WIDTHS_MM:
        trace = tmp_path / f"width_{width_mm}.jsonl"
        trace.write_text(
            "".join(json.dumps(_valid_trace_row()) + "\n" for _ in range(500)),
            encoding="utf-8",
        )
        failed_gates = (
            ["bilateral_contact_fraction", "minimum_contact_normal_axis_alignment"]
            if width_mm == 25
            else []
        )
        trials.append(
            {
                "width_mm": width_mm,
                "trace": str(trace),
                "validity": {
                    "passed": not failed_gates,
                    "failed_gates": failed_gates,
                },
            }
        )
    results = {
        "status": "failed",
        "force_metrics_authorized": False,
        "error": "25 mm did not establish contact",
        "trials": trials,
    }

    trial_rows, provenance = _authorize_analysis(
        tmp_path.resolve(),
        results,
        revalidate_existing_traces=False,
        report_invalid_contact_outcomes=True,
    )

    assert len(trial_rows) == len(WIDTHS_MM)
    assert provenance["authorization"] == "invalid_contact_outcome_report_only"
    assert provenance["force_width_conclusion_authorized"] is False
    assert provenance["invalid_widths"] == [25]
