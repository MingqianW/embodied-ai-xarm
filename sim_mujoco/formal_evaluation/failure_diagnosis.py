"""Evidence-backed task-failure diagnosis for formal xArm evaluation.

This module deliberately classifies only normally completed, unsuccessful
episodes. Runtime-invalid episodes retain ``invalid_reason`` and are not
represented as model-performance failures.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

FAILURE_DIAGNOSIS_VERSION = "xarm-formal-failure-diagnosis-v1"
DEFAULT_PICK_MEANINGFUL_LIFT_DIAGNOSTIC_M = 0.005


@dataclass(frozen=True)
class FailureDiagnosis:
    category: str | None
    reason: str | None
    stage: str | None
    diagnostics: dict[str, Any] | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _number(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _boolean(metrics: dict[str, Any], key: str, *, fallback: bool = False) -> bool:
    value = metrics.get(key, fallback)
    return bool(value)


def _diagnostics(*, observed: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    return {
        "diagnosis_version": FAILURE_DIAGNOSIS_VERSION,
        "observed": observed,
        "thresholds": thresholds,
    }


def _pick_diagnosis(metrics: dict[str, Any]) -> FailureDiagnosis:
    required_lift = _number(metrics, "lift_threshold_m", 0.05)
    meaningful_lift = _number(
        metrics,
        "meaningful_lift_diagnostic_threshold_m",
        DEFAULT_PICK_MEANINGFUL_LIFT_DIAGNOSTIC_M,
    )
    max_lift = _number(metrics, "max_lift_m")
    final_lift_value = metrics.get("final_lift_m", metrics.get("lift_height_m"))
    has_final_lift = final_lift_value is not None
    final_lift = _number(metrics, "final_lift_m", _number(metrics, "lift_height_m", max_lift))
    max_confirmations = int(_number(metrics, "max_success_confirmation_count", _number(metrics, "success_confirmation_count")))
    required_confirmations = int(_number(metrics, "required_success_confirmation_count", 3))
    provisional_success_ever = _boolean(metrics, "provisional_success_ever")
    max_hold_checks = int(_number(metrics, "max_post_success_hold_check_count"))
    required_hold_checks = int(_number(metrics, "required_post_success_hold_checks"))
    hold_failure_count = int(_number(metrics, "post_success_hold_failure_count"))
    post_success_slip = _boolean(metrics, "post_success_slip_ever")
    post_success_max_slip = _number(metrics, "post_success_max_downward_slip_m")
    allowed_post_success_drop = _number(metrics, "max_post_success_drop_m")
    contact_ever = _boolean(metrics, "target_gripper_contact_ever")
    drop_from_peak = max(
        0.0,
        _number(metrics, "drop_from_peak_m", max_lift - final_lift if has_final_lift else 0.0),
    )
    observed = {
        "target_initial_height_m": _number(metrics, "target_initial_height_m"),
        "target_max_height_m": _number(metrics, "target_max_height_m"),
        "max_lift_m": max_lift,
        "final_lift_m": final_lift,
        "drop_from_peak_m": drop_from_peak,
        "target_gripper_contact_ever": contact_ever,
        "target_gripper_contact_check_count": int(_number(metrics, "target_gripper_contact_check_count")),
        "first_meaningful_lift_policy_step": metrics.get("first_meaningful_lift_policy_step"),
        "first_success_height_policy_step": metrics.get("first_success_height_policy_step"),
        "peak_lift_policy_step": metrics.get("peak_lift_policy_step"),
        "max_success_confirmation_count": max_confirmations,
        "required_success_confirmation_count": required_confirmations,
        "provisional_success_ever": provisional_success_ever,
        "max_post_success_hold_check_count": max_hold_checks,
        "post_success_hold_failure_count": hold_failure_count,
        "post_success_slip_ever": post_success_slip,
        "post_success_max_downward_slip_m": post_success_max_slip,
    }
    thresholds = {
        "meaningful_lift_diagnostic_m": meaningful_lift,
        "required_lift_m": required_lift,
        "required_success_confirmation_count": required_confirmations,
        "required_post_success_hold_checks": required_hold_checks,
        "max_post_success_drop_m": allowed_post_success_drop,
    }
    diagnostics = _diagnostics(observed=observed, thresholds=thresholds)
    if max_lift < meaningful_lift:
        stage = "grasp" if contact_ever else "approach"
        contact_clause = " after target-gripper contact" if contact_ever else " and no target-gripper contact was observed"
        return FailureDiagnosis(
            "PICK_NO_MEANINGFUL_LIFT",
            f"Target reached {max_lift:.4f} m maximum lift, below the {meaningful_lift:.4f} m diagnostic lift threshold{contact_clause}.",
            stage,
            diagnostics,
        )
    if max_lift < required_lift:
        return FailureDiagnosis(
            "PICK_PARTIAL_LIFT",
            f"Target reached {max_lift:.4f} m maximum lift, below the required {required_lift:.4f} m sustained-lift threshold.",
            "lift",
            diagnostics,
        )
    if has_final_lift and drop_from_peak >= meaningful_lift and final_lift < required_lift:
        return FailureDiagnosis(
            "PICK_DROPPED_AFTER_LIFT",
            f"Target reached {max_lift:.4f} m lift then fell by {drop_from_peak:.4f} m before the episode ended.",
            "hold",
            diagnostics,
        )
    if provisional_success_ever and required_hold_checks > 0 and (
        hold_failure_count > 0 or max_hold_checks < required_hold_checks
    ):
        if post_success_slip:
            reason = (
                f"Target reached the {required_lift:.4f} m lift criterion but slipped "
                f"{post_success_max_slip:.4f} m during post-success holding; the allowed "
                f"drop is {allowed_post_success_drop:.4f} m."
            )
        else:
            reason = (
                f"Target reached the {required_lift:.4f} m lift criterion but completed only "
                f"{max_hold_checks}/{required_hold_checks} required post-success hold checks."
            )
        return FailureDiagnosis(
            "PICK_REACHED_HEIGHT_BUT_NOT_SUSTAINED",
            reason,
            "hold",
            diagnostics,
        )
    if max_confirmations < required_confirmations:
        return FailureDiagnosis(
            "PICK_REACHED_HEIGHT_BUT_NOT_SUSTAINED",
            f"Target reached the {required_lift:.4f} m lift threshold but achieved only {max_confirmations}/{required_confirmations} required consecutive confirmation checks.",
            "hold",
            diagnostics,
        )
    return FailureDiagnosis(
        "PICK_TIMEOUT_OTHER",
        "Pick episode completed without satisfying the formal sustained-lift success criterion.",
        "timeout",
        diagnostics,
    )


def _placement_diagnosis(metrics: dict[str, Any]) -> FailureDiagnosis:
    release = _boolean(
        metrics, "ever_release_confirmed", fallback=_boolean(metrics, "release_confirmed")
    )
    containment = _boolean(
        metrics, "ever_containment_confirmed", fallback=_boolean(metrics, "containment_confirmed")
    )
    height = _boolean(metrics, "ever_height_confirmed", fallback=_boolean(metrics, "height_confirmed"))
    stability = _boolean(
        metrics, "ever_stability_confirmed", fallback=_boolean(metrics, "stability_confirmed")
    )
    instant_success = _boolean(
        metrics, "ever_instant_success", fallback=_boolean(metrics, "instant_success")
    )
    release_requested = _boolean(metrics, "release_requested")
    max_confirmations = int(_number(metrics, "max_success_confirmation_count", _number(metrics, "success_confirmation_count")))
    required_confirmations = int(_number(metrics, "required_success_confirmation_count", 3))
    observed = {
        "release_requested": release_requested,
        "ever_release_confirmed": release,
        "ever_containment_confirmed": containment,
        "ever_height_confirmed": height,
        "ever_stability_confirmed": stability,
        "ever_instant_success": instant_success,
        "max_success_confirmation_count": max_confirmations,
        "required_success_confirmation_count": required_confirmations,
        "gripper_contact_count": int(_number(metrics, "gripper_contact_count")),
        "pepper_gripper_distance_m": _number(metrics, "pepper_gripper_distance_m"),
        "gripper_raw": _number(metrics, "gripper_raw"),
        "pepper_ring_xy_distance_m": _number(metrics, "pepper_ring_xy_distance_m"),
        "min_xy_distance_m": _number(metrics, "min_xy_distance_m", _number(metrics, "pepper_ring_xy_distance_m")),
        "containment_margin_m": _number(metrics, "containment_margin_m"),
        "max_containment_margin_m": _number(metrics, "max_containment_margin_m", _number(metrics, "containment_margin_m")),
        "pepper_height_above_table_m": _number(metrics, "pepper_height_above_table_m"),
        "pepper_linear_speed_mps": _number(metrics, "pepper_linear_speed_mps"),
        "pepper_angular_speed_radps": _number(metrics, "pepper_angular_speed_radps"),
        "min_linear_speed_after_release_mps": metrics.get("min_linear_speed_after_release_mps"),
        "min_angular_speed_after_release_radps": metrics.get("min_angular_speed_after_release_radps"),
    }
    thresholds = {
        "containment_limit_m": _number(metrics, "containment_limit_m"),
        "min_height_above_table_m": _number(metrics, "placement_min_height_above_table_m"),
        "max_height_above_table_m": _number(metrics, "placement_max_height_above_table_m"),
        "max_linear_speed_mps": _number(metrics, "placement_max_linear_speed_mps"),
        "max_angular_speed_radps": _number(metrics, "placement_max_angular_speed_radps"),
        "min_gripper_distance_m": _number(metrics, "placement_min_gripper_distance_m"),
        "release_gripper_raw": _number(metrics, "placement_release_gripper_raw"),
        "required_success_confirmation_count": required_confirmations,
    }
    diagnostics = _diagnostics(observed=observed, thresholds=thresholds)
    if not release:
        stage = "release" if release_requested else "transport"
        return FailureDiagnosis(
            "PLACE_NOT_RELEASED",
            "Pepper was not observably released from the gripper before episode end.",
            stage,
            diagnostics,
        )
    if not containment:
        return FailureDiagnosis(
            "PLACE_RELEASED_OUTSIDE_RING",
            f"Pepper was released but never met ring containment; best XY distance was {observed['min_xy_distance_m']:.4f} m versus {thresholds['containment_limit_m']:.4f} m limit.",
            "containment",
            diagnostics,
        )
    if not height:
        return FailureDiagnosis(
            "PLACE_RELEASED_WRONG_HEIGHT",
            "Pepper was released and contained but never met the formal placement-height range.",
            "height",
            diagnostics,
        )
    if not stability:
        return FailureDiagnosis(
            "PLACE_RELEASED_IN_RING_BUT_UNSTABLE",
            "Pepper was released, contained within the ring, and at valid height, but did not meet the formal stability limits.",
            "stability",
            diagnostics,
        )
    if instant_success and max_confirmations < required_confirmations:
        return FailureDiagnosis(
            "PLACE_STABLE_BUT_NOT_SUSTAINED",
            f"Pepper met the instantaneous placement conditions but achieved only {max_confirmations}/{required_confirmations} required consecutive confirmation checks.",
            "stability",
            diagnostics,
        )
    return FailureDiagnosis(
        "PLACE_TIMEOUT_OTHER",
        "Placement episode completed without satisfying the formal sustained-placement success criterion.",
        "timeout",
        diagnostics,
    )


def diagnose_episode_failure(
    *,
    task_id: str,
    success: bool,
    valid: bool,
    metrics: dict[str, Any],
) -> FailureDiagnosis:
    """Return a task-failure diagnosis; successes and invalids intentionally have none."""

    if success or not valid:
        return FailureDiagnosis(None, None, None, None)
    if task_id == "place_red_pepper_in_ring":
        return _placement_diagnosis(metrics)
    return _pick_diagnosis(metrics)
