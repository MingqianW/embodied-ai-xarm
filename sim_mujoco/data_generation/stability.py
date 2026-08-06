"""Canonical stable Pick, initial Place grasp, and stable Place validators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from sim_mujoco.data_generation.config import (
    PickVerificationConfig,
    PlaceInitialGraspConfig,
    PlaceVerificationConfig,
)


@dataclass(frozen=True)
class StabilitySample:
    simulation_time_s: float
    object_position_m: tuple[float, float, float]
    tcp_position_m: tuple[float, float, float]
    finite: bool = True
    table_contact: bool = False
    forbidden_collision: bool = False
    inside_ring: bool = False
    released: bool = False
    retreat_detected: bool = False


def _velocity(samples: Sequence[StabilitySample], fit_samples: int) -> float:
    count = min(len(samples), max(5, int(fit_samples)))
    selected = samples[-count:]
    times = np.asarray([sample.simulation_time_s for sample in selected], dtype=np.float64)
    heights = np.asarray([sample.object_position_m[2] for sample in selected], dtype=np.float64)
    times = times - times[0]
    if count < 2 or float(np.ptp(times)) <= 0.0:
        return float("nan")
    slope, _ = np.polyfit(times, heights, deg=1)
    return float(slope)


def _speed(samples: Sequence[StabilitySample], fit_samples: int) -> float:
    count = min(len(samples), max(5, int(fit_samples)))
    selected = samples[-count:]
    times = np.asarray([sample.simulation_time_s for sample in selected], dtype=np.float64)
    positions = np.asarray([sample.object_position_m for sample in selected], dtype=np.float64)
    times = times - times[0]
    if count < 2 or float(np.ptp(times)) <= 0.0:
        return float("nan")
    slopes = [float(np.polyfit(times, positions[:, axis], deg=1)[0]) for axis in range(3)]
    return float(np.linalg.norm(slopes))


def evaluate_pick_stability(
    samples: Sequence[StabilitySample],
    *,
    config: PickVerificationConfig,
    initial_object_z_m: float,
    verification_start_object_position_m: Sequence[float],
    verification_start_tcp_position_m: Sequence[float],
) -> dict[str, Any]:
    start_object = np.asarray(verification_start_object_position_m, dtype=np.float64)
    start_tcp = np.asarray(verification_start_tcp_position_m, dtype=np.float64)
    start_offset = start_tcp - start_object
    positions = np.asarray([sample.object_position_m for sample in samples], dtype=np.float64)
    tcp_positions = np.asarray([sample.tcp_position_m for sample in samples], dtype=np.float64)
    lifts = positions[:, 2] - float(initial_object_z_m) if len(samples) else np.asarray([])
    relative_offsets = tcp_positions - positions if len(samples) else np.empty((0, 3))
    downward_slips = (
        relative_offsets[:, 2] - start_offset[2] if len(samples) else np.asarray([])
    )
    grasp_delta = (
        np.linalg.norm(relative_offsets - start_offset, axis=1)
        if len(samples)
        else np.asarray([])
    )
    duration = (
        float(samples[-1].simulation_time_s - (samples[0].simulation_time_s - config.action_dt_s))
        if samples
        else 0.0
    )
    velocity = _velocity(samples, config.velocity_fit_samples) if samples else float("nan")
    failure = None
    if len(samples) != config.steps or duration + 1e-9 < config.steps * config.action_dt_s:
        failure = "stable_grasp_incomplete_verification"
    elif not all(sample.finite for sample in samples):
        failure = "stable_grasp_non_finite"
    elif any(sample.forbidden_collision for sample in samples):
        failure = "stable_grasp_forbidden_collision"
    elif any(sample.table_contact for sample in samples):
        failure = "stable_grasp_table_contact"
    elif float(np.min(lifts)) < config.minimum_lift_height_m:
        failure = "stable_grasp_lift_below_minimum"
    elif float(np.max(downward_slips)) > config.maximum_relative_downward_slip_m:
        failure = "stable_grasp_excessive_relative_slip"
    elif float(downward_slips[-1]) > config.maximum_final_relative_downward_slip_m:
        failure = "stable_grasp_final_relative_slip"
    elif float(np.max(grasp_delta)) > config.maximum_grasp_region_delta_m:
        failure = "stable_grasp_left_grasp_region"
    elif not np.isfinite(velocity):
        failure = "stable_grasp_non_finite"
    elif velocity < -config.maximum_final_downward_speed_mps:
        failure = "stable_grasp_downward_motion"
    return {
        "initial_object_z_m": float(initial_object_z_m),
        "verification_start_object_z_m": float(start_object[2]),
        "verification_start_tcp_z_m": float(start_tcp[2]),
        "verification_start_tcp_to_object_offset_m": start_offset.tolist(),
        "peak_lift_height_m": float(np.max(lifts)) if len(lifts) else None,
        "minimum_verification_lift_height_m": float(np.min(lifts)) if len(lifts) else None,
        "final_lift_height_m": float(lifts[-1]) if len(lifts) else None,
        "maximum_relative_downward_slip_m": (
            float(np.max(downward_slips)) if len(downward_slips) else None
        ),
        "final_relative_downward_slip_m": (
            float(downward_slips[-1]) if len(downward_slips) else None
        ),
        "estimated_final_object_vertical_velocity_mps": velocity,
        "maximum_gripper_relative_delta_m": (
            float(np.max(grasp_delta)) if len(grasp_delta) else None
        ),
        "table_contact_detected": any(sample.table_contact for sample in samples),
        "forbidden_collision_detected": any(
            sample.forbidden_collision for sample in samples
        ),
        "all_samples_finite": all(sample.finite for sample in samples),
        "verification_steps_required": config.steps,
        "verification_steps_executed": len(samples),
        "verification_duration_s": duration,
        "stable_grasp_success": failure is None,
        "stable_grasp_failure_reason": failure,
    }


def evaluate_place_initial_grasp(
    samples: Sequence[StabilitySample],
    *,
    config: PlaceInitialGraspConfig,
    table_top_z_m: float,
    initial_object_position_m: Sequence[float] | None = None,
    initial_tcp_position_m: Sequence[float] | None = None,
) -> dict[str, Any]:
    if not samples:
        initial_object = np.full(3, np.nan)
        initial_tcp = np.full(3, np.nan)
        drifts = np.asarray([])
    else:
        initial_object = np.asarray(
            initial_object_position_m
            if initial_object_position_m is not None
            else samples[0].object_position_m,
            dtype=np.float64,
        )
        initial_tcp = np.asarray(
            initial_tcp_position_m
            if initial_tcp_position_m is not None
            else samples[0].tcp_position_m,
            dtype=np.float64,
        )
        initial_offset = initial_object - initial_tcp
        offsets = np.asarray([sample.object_position_m for sample in samples]) - np.asarray(
            [sample.tcp_position_m for sample in samples]
        )
        drifts = np.linalg.norm(offsets - initial_offset, axis=1)
    duration = (
        float(samples[-1].simulation_time_s - (samples[0].simulation_time_s - config.action_dt_s))
        if samples
        else 0.0
    )
    failure = None
    if len(samples) != config.steps or duration + 1e-9 < config.steps * config.action_dt_s:
        failure = "initial_place_grasp_incomplete_validation"
    elif not all(sample.finite for sample in samples):
        failure = "initial_place_grasp_non_finite"
    elif any(sample.forbidden_collision for sample in samples):
        failure = "initial_place_grasp_forbidden_collision"
    elif any(sample.table_contact for sample in samples):
        failure = "initial_place_grasp_table_contact"
    elif any(sample.inside_ring for sample in samples):
        failure = "initial_place_grasp_inside_ring"
    elif any(
        sample.object_position_m[2] - table_top_z_m
        < config.minimum_height_above_table_m
        for sample in samples
    ):
        failure = "initial_place_grasp_unstable"
    elif float(np.max(drifts)) > config.maximum_grasp_region_delta_m:
        failure = "initial_place_grasp_left_grasp_region"
    elif float(np.max(drifts)) > config.maximum_relative_drift_m:
        failure = "initial_place_grasp_excessive_drift"
    return {
        "initial_tcp_position_m": initial_tcp.tolist(),
        "initial_pepper_position_m": initial_object.tolist(),
        "initial_pepper_to_tcp_transform": {
            "translation_m": (initial_object - initial_tcp).tolist()
        },
        "initial_grasp_validation_steps_required": config.steps,
        "initial_grasp_validation_steps_executed": len(samples),
        "initial_grasp_validation_duration_s": duration,
        "initial_grasp_max_relative_drift_m": (
            float(np.max(drifts)) if len(drifts) else None
        ),
        "initial_grasp_table_contact_detected": any(
            sample.table_contact for sample in samples
        ),
        "initial_grasp_forbidden_collision_detected": any(
            sample.forbidden_collision for sample in samples
        ),
        "initial_grasp_all_samples_finite": all(
            sample.finite for sample in samples
        ),
        "initial_grasp_success": failure is None,
        "initial_grasp_failure_reason": failure,
    }


def evaluate_place_stability(
    samples: Sequence[StabilitySample],
    *,
    config: PlaceVerificationConfig,
) -> dict[str, Any]:
    duration = (
        float(samples[-1].simulation_time_s - (samples[0].simulation_time_s - config.action_dt_s))
        if samples
        else 0.0
    )
    speed = _speed(samples, config.velocity_fit_samples) if samples else float("nan")
    failure = None
    if len(samples) != config.steps or duration + 1e-9 < config.steps * config.action_dt_s:
        failure = "stable_place_incomplete_verification"
    elif not all(sample.finite for sample in samples):
        failure = "stable_place_non_finite"
    elif any(sample.forbidden_collision for sample in samples):
        failure = "stable_place_forbidden_collision"
    elif not all(sample.released for sample in samples):
        failure = "stable_place_release_not_detected"
    elif not all(sample.retreat_detected for sample in samples):
        failure = "stable_place_retreat_not_detected"
    elif not all(sample.inside_ring for sample in samples):
        failure = "stable_place_left_ring"
    elif not np.isfinite(speed) or speed > config.maximum_final_speed_mps:
        failure = "stable_place_not_settled"
    return {
        "place_verification_steps_required": config.steps,
        "place_verification_steps_executed": len(samples),
        "place_verification_duration_s": duration,
        "estimated_final_object_speed_mps": speed,
        "forbidden_collision_detected": any(
            sample.forbidden_collision for sample in samples
        ),
        "all_samples_finite": all(sample.finite for sample in samples),
        "stable_place_success": failure is None,
        "stable_place_failure_reason": failure,
    }
